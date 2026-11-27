import BasaltCore
import Foundation
import MLX

package enum KVarNFlashDecodeMetal {
    package static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]
    package static let maxHeadDim = 256

    package struct Inputs {
        package var queries: MLXArray
        package var keyTile: KVarNEncodedTile
        package var valueTile: KVarNEncodedTile
        package var jlMatrix: MLXArray?
        package var scale: Float

        package init(
            queries: MLXArray,
            keyTile: KVarNEncodedTile,
            valueTile: KVarNEncodedTile,
            jlMatrix: MLXArray? = nil,
            scale: Float
        ) {
            self.queries = queries
            self.keyTile = keyTile
            self.valueTile = valueTile
            self.jlMatrix = jlMatrix
            self.scale = scale
        }
    }

    package static func flashDecode(_ inputs: Inputs) -> MLXArray {
        flashDecodeWithLSE(inputs).output
    }

    private static func hasFloat64(_ inputs: Inputs) -> Bool {
        let arrays: [MLXArray?] = [
            inputs.queries, inputs.jlMatrix,
            inputs.keyTile.packedCodes, inputs.keyTile.tokenScale, inputs.keyTile.channelScale,
            inputs.keyTile.qjlBits, inputs.keyTile.qjlNorm,
            inputs.valueTile.packedCodes, inputs.valueTile.tokenScale, inputs.valueTile.channelScale,
        ]
        return arrays.contains(where: { $0?.dtype == .float64 })
    }

    package static func flashDecodeWithLSE(_ inputs: Inputs) -> (output: MLXArray, lse: MLXArray) {
        if Device.defaultDevice() == .gpu, hasFloat64(inputs) {
            return Device.withDefaultDevice(.cpu) { flashDecodeWithLSE(inputs) }
        }
        let origType = inputs.queries.dtype
        let d = inputs.keyTile.headDim
        let t = inputs.keyTile.tokens
        precondition(
            d > 0 && d.isMultiple(of: 32) && d <= maxHeadDim
                && inputs.valueTile.headDim == d && inputs.valueTile.tokens == t)
        let heads = inputs.queries.size / d
        precondition(inputs.queries.size == heads * d)
        guard heads > 0, t > 0 else {
            return (
                MLXArray.zeros(inputs.queries.shape, dtype: origType),
                MLXArray.full([heads], values: MLXArray(-Float.infinity)))
        }
        guard Device.defaultDevice() == .gpu, supportedDTypes.contains(origType) else {
            let (hostOut, hostLSE) = cpuFlashDecodeWithLSE(inputs)
            return (MLXArray(hostOut, inputs.queries.shape).asType(origType), MLXArray(hostLSE, [heads]))
        }
        return runMetalFlashDecode(inputs, heads: heads, tokens: t, headDim: d, origType: origType)
    }

    private static func runMetalFlashDecode(
        _ inputs: Inputs, heads: Int, tokens: Int, headDim: Int, origType: DType
    ) -> (output: MLXArray, lse: MLXArray) {
        let ones = MLXArray.ones([headDim], dtype: .float32)
        let qRot = MACaQFusedKernels.blockHadamard(
            inputs.queries.asType(.float32).reshaped(heads, headDim),
            scaleVector: ones, blockSize: 32, isInverse: false)
        let hasQJL = inputs.keyTile.qjlBits != nil && inputs.keyTile.qjlNorm != nil && inputs.jlMatrix != nil
        let qJL: MLXArray = if hasQJL, let sJL = inputs.jlMatrix {
            matmul(qRot, sJL.asType(.float32).T)
        } else {
            MLXArray([Float(0), 0])
        }
        let dummyU32 = MLXArray([UInt32(0), 0])
        let dummyF32 = MLXArray([Float(0), 0])
        let qjlBits = inputs.keyTile.qjlBits ?? dummyU32
        let qjlNorm = inputs.keyTile.qjlNorm ?? dummyF32
        let params = MLXArray([inputs.scale, KVarNTileCodec.qjlScaleFactor / Float(headDim)])
        let shapes = MLXArray([Int32(heads), Int32(tokens), Int32(headDim)])
        let outs = flashDecodeKernel(
            [
                qRot.asType(.float32),
                qJL.asType(.float32),
                inputs.keyTile.packedCodes,
                inputs.keyTile.tokenScale.asType(.float32),
                inputs.keyTile.channelScale.asType(.float32),
                qjlBits,
                qjlNorm.asType(.float32),
                inputs.valueTile.packedCodes,
                inputs.valueTile.tokenScale.asType(.float32),
                inputs.valueTile.channelScale.asType(.float32),
                params,
                shapes,
            ],
            template: [
                ("T", origType),
                ("BLOCKS", headDim / 32),
                ("HAS_QJL", hasQJL),
            ],
            grid: (32, heads, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[heads, headDim], [heads]],
            outputDTypes: [origType, .float32]
        )
        return (outs[0].reshaped(inputs.queries.shape), outs[1])
    }

    package static func cpuFlashDecode(_ inputs: Inputs) -> [Float] {
        cpuFlashDecodeWithLSE(inputs).output
    }

    package static func cpuFlashDecodeWithLSE(_ inputs: Inputs) -> (output: [Float], lse: [Float]) {
        if Device.defaultDevice() == .gpu, hasFloat64(inputs) {
            return Device.withDefaultDevice(.cpu) { cpuFlashDecodeWithLSE(inputs) }
        }
        let d = inputs.keyTile.headDim
        let t = inputs.keyTile.tokens
        guard d > 0, inputs.valueTile.headDim == d, inputs.valueTile.tokens == t else {
            return ([], [])
        }
        let heads = inputs.queries.size / d
        guard heads > 0, t > 0 else {
            return ([Float](repeating: 0, count: heads * d), [Float](repeating: -.infinity, count: heads))
        }
        let qHost = inputs.queries.asType(.float32).asArray(Float.self)
        let jlHost = inputs.jlMatrix?.asType(.float32).asArray(Float.self)
        let vUnp = PVCodeGrid.unpackCodes(inputs.valueTile.packedCodes, bits: 2).asType(.int32).asArray(Int32.self)
        let vSTok = inputs.valueTile.tokenScale.asType(.float32).asArray(Float.self)
        let vSChan = inputs.valueTile.channelScale.asType(.float32).asArray(Float.self)
        let ones = [Float](repeating: 1.0, count: 32)

        var out = [Float](repeating: 0, count: heads * d)
        var lse = [Float](repeating: -.infinity, count: heads)
        for h in 0 ..< heads {
            let qVec = Array(qHost[(h * d) ..< ((h + 1) * d)])
            let rawScores = KVarNTileCodec.cpuEstimateScores(query: qVec, tile: inputs.keyTile, jlMatrixHost: jlHost)
            let scaled = rawScores.map { $0 * inputs.scale }
            if scaled.contains(where: { $0.isNaN || $0 == .infinity }) {
                for c in 0 ..< d { out[h * d + c] = .nan }
                lse[h] = .nan
                continue
            }
            let mMax = scaled.max() ?? -.infinity
            guard mMax > -.infinity else { continue }
            let weights = scaled.map { $0 > -.infinity ? expf($0 - mMax) : Float(0) }
            let wSum = weights.reduce(0, +)
            guard wSum > 0 else { continue }
            lse[h] = mMax + logf(wSum)
            var vRotAcc = [Float](repeating: 0, count: d)
            for tok in 0 ..< t {
                let p = weights[tok] / wSum
                let st = vSTok[tok]
                for c in 0 ..< d {
                    let code = Int(vUnp[tok * d + c])
                    vRotAcc[c] += p * (KVarNTileCodec.lloydMax2Bit[code] * vSChan[c] * st)
                }
            }
            let vUnrot = BasaltMetalSubstrate.cpuHadamardBlock(vRotAcc, scale: ones, blockSize: 32)
            for c in 0 ..< d {
                out[h * d + c] = vUnrot[c]
            }
        }
        return (out, lse)
    }
}

private let flashDecodeSource = """
    constexpr float LM2[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};
    const int H = int(shapes[0]);
    const int TLEN = int(shapes[1]);
    const int D = int(shapes[2]);
    const int lane = int(thread_index_in_simdgroup);
    const int h = int(threadgroup_position_in_grid.y);
    if (h >= H) { return; }
    const float attn_scale = params[0];
    const float qjl_coef = params[1];
    const int words_k = D >> 4;
    const int words_qjl = D >> 5;
    const size_t q_base = size_t(h) * size_t(D);

    float q_w[BLOCKS];
    float q_j[BLOCKS];
    float v_acc[BLOCKS];
    for (int b = 0; b < BLOCKS; ++b) {
        const int col = b * 32 + lane;
        q_w[b] = q_rot[q_base + size_t(col)] * k_s_chan[col];
        q_j[b] = HAS_QJL ? q_jl[q_base + size_t(col)] : 0.0f;
        v_acc[b] = 0.0f;
    }

    float m_run = -INFINITY;
    float l_run = 0.0f;
    for (int tok = 0; tok < TLEN; ++tok) {
        const size_t k_base = size_t(tok) * size_t(words_k);
        float dot_part = 0.0f;
        for (int b = 0; b < BLOCKS; ++b) {
            const int col = b * 32 + lane;
            const uint code = (k_packed[k_base + size_t(col >> 4)] >> uint((col & 15) * 2)) & 3u;
            dot_part = fma(q_w[b], LM2[code], dot_part);
        }
        float score = simd_sum(dot_part) * k_s_tok[tok];
        if (HAS_QJL) {
            const size_t qjl_base = size_t(tok) * size_t(words_qjl);
            float jl_part = 0.0f;
            for (int b = 0; b < BLOCKS; ++b) {
                const uint mask = qjl_bits[qjl_base + size_t(b)];
                const bool neg = ((mask >> uint(lane)) & 1u) != 0u;
                jl_part += neg ? -q_j[b] : q_j[b];
            }
            score += qjl_coef * qjl_norm[tok] * simd_sum(jl_part);
        }
        score *= attn_scale;
        const bool bad = isnan(score) || isnan(m_run);
        const float m_new = bad ? NAN : max(m_run, score);
        const float alpha = bad ? NAN : ((m_run > -INFINITY && m_new > -INFINITY) ? exp(m_run - m_new) : 0.0f);
        const float beta = bad ? NAN : ((m_new > -INFINITY) ? exp(score - m_new) : 0.0f);
        l_run = l_run * alpha + beta;
        m_run = m_new;

        const size_t v_base = size_t(tok) * size_t(words_k);
        const float vst = beta * v_s_tok[tok];
        for (int b = 0; b < BLOCKS; ++b) {
            const int col = b * 32 + lane;
            const uint vcode = (v_packed[v_base + size_t(col >> 4)] >> uint((col & 15) * 2)) & 3u;
            v_acc[b] = fma(vst, LM2[vcode], v_acc[b] * alpha);
        }
    }

    const float inv_l = (isnan(l_run) || isnan(m_run)) ? NAN : ((l_run > 0.0f) ? (1.0f / l_run) : 0.0f);
    for (int b = 0; b < BLOCKS; ++b) {
        const int col = b * 32 + lane;
        const float v_rot = v_acc[b] * v_s_chan[col] * inv_l;
        const float v_unrot = basalt_wht32_simd(v_rot, lane);
        out[q_base + size_t(col)] = static_cast<T>(v_unrot);
    }
    if (lane == 0) {
        lse[h] = (isnan(l_run) || isnan(m_run)) ? NAN : ((l_run > 0.0f) ? (m_run + log(l_run)) : -INFINITY);
    }
    """

private let flashDecodeKernel = MLXFast.metalKernel(
    name: "kv_fused_flash_decode_kvarn_qjl",
    inputNames: [
        "q_rot", "q_jl", "k_packed", "k_s_tok", "k_s_chan",
        "qjl_bits", "qjl_norm", "v_packed", "v_s_tok", "v_s_chan", "params", "shapes",
    ],
    outputNames: ["out", "lse"],
    source: flashDecodeSource,
    header: BasaltMetalSubstrate.combinedHeader
)
