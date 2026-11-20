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

    package static func flashDecodeWithLSE(_ inputs: Inputs) -> (output: MLXArray, lse: MLXArray) {
        let origType = inputs.queries.dtype
        let d = inputs.keyTile.headDim
        let t = inputs.keyTile.tokens
        let heads = inputs.queries.size / max(1, d)
        precondition(d.isMultiple(of: 32) && d <= maxHeadDim && inputs.valueTile.tokens == t)
        let ones = MLXArray.ones([d], dtype: .float32)
        let qRot = MACaQFusedKernels.blockHadamard(
            inputs.queries.asType(.float32).reshaped(heads, d),
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
        let params = MLXArray([inputs.scale, KVarNTileCodec.qjlScaleFactor / Float(d)])
        let shapes = MLXArray([Int32(heads), Int32(t), Int32(d)])
        let outs = flashDecodeKernel(
            [
                qRot.asType(origType),
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
                ("BLOCKS", d / 32),
                ("HAS_QJL", hasQJL),
            ],
            grid: (32, heads, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[heads, d], [heads]],
            outputDTypes: [origType, .float32]
        )
        return (outs[0].reshaped(inputs.queries.shape), outs[1])
    }

    package static func cpuFlashDecode(_ inputs: Inputs) -> [Float] {
        let d = inputs.keyTile.headDim
        let t = inputs.keyTile.tokens
        let heads = inputs.queries.size / d
        let qHost = inputs.queries.asType(.float32).asArray(Float.self)
        let jlHost = inputs.jlMatrix?.asType(.float32).asArray(Float.self)
        let vUnp = PVCodeGrid.unpackCodes(inputs.valueTile.packedCodes, bits: 2).asType(.int32).asArray(Int32.self)
        let vSTok = inputs.valueTile.tokenScale.asArray(Float.self)
        let vSChan = inputs.valueTile.channelScale.asArray(Float.self)
        let ones = [Float](repeating: 1.0, count: 32)

        var out = [Float](repeating: 0, count: heads * d)
        for h in 0 ..< heads {
            let qVec = Array(qHost[(h * d) ..< ((h + 1) * d)])
            let rawScores = KVarNTileCodec.cpuEstimateScores(query: qVec, tile: inputs.keyTile, jlMatrixHost: jlHost)
            let scaled = rawScores.map { $0 * inputs.scale }
            let mMax = scaled.max() ?? 0
            let weights = scaled.map { expf($0 - mMax) }
            let wSum = max(weights.reduce(0, +), 1e-20)
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
        return out
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

    float q_w[BLOCKS];
    float q_j[BLOCKS];
    float v_acc[BLOCKS];
    for (int b = 0; b < BLOCKS; ++b) {
        const int col = b * 32 + lane;
        q_w[b] = float(q_rot[h * D + col]) * k_s_chan[col];
        q_j[b] = HAS_QJL ? q_jl[h * D + col] : 0.0f;
        v_acc[b] = 0.0f;
    }

    float m_run = -INFINITY;
    float l_run = 0.0f;
    for (int tok = 0; tok < TLEN; ++tok) {
        auto kw = k_packed + size_t(tok) * size_t(words_k);
        float dot_part = 0.0f;
        for (int b = 0; b < BLOCKS; ++b) {
            const int col = b * 32 + lane;
            const uint code = (kw[col >> 4] >> uint((col & 15) * 2)) & 3u;
            dot_part = fma(q_w[b], LM2[code], dot_part);
        }
        float score = simd_sum(dot_part) * k_s_tok[tok];
        if (HAS_QJL) {
            auto qw = qjl_bits + size_t(tok) * size_t(words_qjl);
            float jl_part = 0.0f;
            for (int b = 0; b < BLOCKS; ++b) {
                const uint mask = qw[b];
                const bool neg = ((mask >> uint(lane)) & 1u) != 0u;
                jl_part += neg ? -q_j[b] : q_j[b];
            }
            score += qjl_coef * qjl_norm[tok] * simd_sum(jl_part);
        }
        score *= attn_scale;
        const float m_new = max(m_run, score);
        const float alpha = (m_run == -INFINITY) ? 0.0f : exp(m_run - m_new);
        const float beta = exp(score - m_new);
        l_run = l_run * alpha + beta;
        m_run = m_new;

        auto vw = v_packed + size_t(tok) * size_t(words_k);
        const float vst = beta * v_s_tok[tok];
        for (int b = 0; b < BLOCKS; ++b) {
            const int col = b * 32 + lane;
            const uint vcode = (vw[col >> 4] >> uint((col & 15) * 2)) & 3u;
            v_acc[b] = fma(vst, LM2[vcode], v_acc[b] * alpha);
        }
    }

    const float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    for (int b = 0; b < BLOCKS; ++b) {
        const int col = b * 32 + lane;
        const float v_rot = v_acc[b] * v_s_chan[col] * inv_l;
        const float v_unrot = basalt_wht32_simd(v_rot, lane);
        out[size_t(h) * size_t(D) + size_t(col)] = static_cast<T>(v_unrot);
    }
    if (lane == 0) {
        lse[h] = (l_run > 0.0f) ? (m_run + log(l_run)) : -INFINITY;
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
