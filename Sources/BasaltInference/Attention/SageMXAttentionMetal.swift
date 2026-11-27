import BasaltCore
import Foundation
import MLX

package enum SageMXAttentionMetal {
    package static let uosThreshold: Float = 7.25
    package static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]
    package static let maxHeadDim = 256

    package static let onlineSoftmaxPNQHeader = """
        inline float basalt_pnq_prob8(float p) {
            const float clamped = clamp(p, 0.0f, 1.0f);
            return rint(clamped * 255.0f) * (1.0f / 255.0f);
        }
        """

    package struct Options: Sendable {
        package var scale: Float
        package var causal: Bool
        package var useUOS: Bool
        package var usePNQ: Bool
        package var meanCenter: Bool

        package init(
            scale: Float,
            causal: Bool = false,
            useUOS: Bool = true,
            usePNQ: Bool = true,
            meanCenter: Bool = true
        ) {
            self.scale = scale
            self.causal = causal
            self.useUOS = useUOS
            self.usePNQ = usePNQ
            self.meanCenter = meanCenter
        }
    }

    package static func cpuQuantizeMXBlock32(_ block: [Float], threshold: Float) -> [Float] {
        precondition(block.count == 32)
        if block.contains(where: \.isNaN) { return [Float](repeating: .nan, count: 32) }
        let maxAbs = block.map { abs($0) }.max() ?? 0
        if !(maxAbs > 0) { return [Float](repeating: 0, count: 32) }
        let raw = maxAbs / threshold
        let bits = raw.bitPattern
        var exp = Int((bits >> 23) & 255)
        if (bits & 0x7f_ffff) != 0, exp < 254 { exp += 1 }
        exp = min(max(exp, 0), 254)
        let sc = Float(bitPattern: UInt32(exp) << 23)
        let inv = sc > 0 ? (1.0 / sc) : 0
        let lut: [Float] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        return block.map { v in
            let sv = v * inv
            let sgn: Float = sv < 0 ? -1.0 : 1.0
            let a = abs(sv)
            let idx: Int
            if a >= 5.0 { idx = 7 }
            else if a >= 3.5 { idx = 6 }
            else if a >= 2.5 { idx = 5 }
            else if a >= 1.75 { idx = 4 }
            else if a >= 1.25 { idx = 3 }
            else if a >= 0.75 { idx = 2 }
            else if a >= 0.25 { idx = 1 }
            else { idx = 0 }
            return sgn * lut[idx] * sc
        }
    }

    package static func supportsMetalKernel(q: MLXArray, k: MLXArray, v: MLXArray) -> Bool {
        guard Device.defaultDevice() == .gpu,
              q.ndim >= 2, k.ndim >= 2, v.ndim >= 2,
              supportedDTypes.contains(q.dtype),
              supportedDTypes.contains(k.dtype),
              supportedDTypes.contains(v.dtype)
        else { return false }
        let (lQ, d) = (q.dim(-2), q.dim(-1))
        let lK = k.dim(-2)
        let heads = q.shape.dropLast(2).reduce(1, *)
        let kHeads = k.shape.dropLast(2).reduce(1, *)
        let vHeads = v.shape.dropLast(2).reduce(1, *)
        return heads > 0 && lQ > 0 && lK > 0 && d > 0
            && d.isMultiple(of: 32) && d <= maxHeadDim
            && kHeads == heads && vHeads == heads
            && k.dim(-1) == d && v.dim(-2) == lK && v.dim(-1) == d
    }

    package static func attention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        options: Options
    ) -> MLXArray {
        precondition(q.ndim >= 2 && k.ndim >= 2 && v.ndim >= 2)
        let origType = q.dtype
        let lQ = q.dim(-2)
        let lK = k.dim(-2)
        let d = q.dim(-1)
        let dV = v.dim(-1)
        let heads = q.shape.dropLast(2).reduce(1, *)
        let kHeads = k.shape.dropLast(2).reduce(1, *)
        let vHeads = v.shape.dropLast(2).reduce(1, *)
        precondition(kHeads == heads && vHeads == heads && k.dim(-1) == d && v.dim(-2) == lK)
        let outShape = Array(q.shape.dropLast()) + [dV]
        if [heads, lQ, lK, d, dV].contains(0) {
            return MLXArray.zeros(outShape, dtype: origType, stream: origType == .float64 ? .cpu : .default)
        }
        let shape = Shape(heads: heads, lQ: lQ, lK: lK, d: d)
        guard supportsMetalKernel(q: q, k: k, v: v) else {
            return referenceAttention(q: q, k: k, v: v, shape: shape, options: options)
        }
        return launchKernel(q: q, k: k, v: v, shape: shape, options: options)
    }
}

extension SageMXAttentionMetal {
    private static func launchKernel(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        shape: Shape,
        options: Options
    ) -> MLXArray {
        let origType = q.dtype
        let (heads, lQ, lK, d) = (shape.heads, shape.lQ, shape.lK, shape.d)
        let q3D = q.asType(.float32).reshaped(heads, lQ, d)
        let k3D = k.asType(.float32).reshaped(heads, lK, d)
        let v3D = v.asType(origType).reshaped(heads, lK, d)
        let centerK = options.meanCenter && lK > 1
        let centerQ = options.meanCenter
        let kCentered = centerK ? (k3D - mean(k3D, axis: 1, keepDims: true)) : k3D
        let qMean: MLXArray? = centerQ ? mean(q3D, axis: 1, keepDims: true) : nil
        let qCentered = qMean.map { q3D - $0 } ?? q3D
        let deltaS = qMean.map { matmul($0, kCentered.swappedAxes(-1, -2)).squeezed(axis: 1) }
            ?? MLXArray.zeros([1], dtype: .float32)

        let gridMax: Float = options.useUOS ? uosThreshold : 6.0
        let params = MLXArray([options.scale, gridMax])
        let shapes = MLXArray([Int32(heads), Int32(lQ), Int32(lK), Int32(d)])
        let out = sageMXFlashKernel(
            [qCentered, kCentered, v3D, deltaS, params, shapes],
            template: [
                ("T", origType),
                ("BLOCKS", d / 32),
                ("CAUSAL", options.causal),
                ("USE_PNQ", options.usePNQ),
                ("HAS_DELTA", centerQ),
            ],
            grid: (32, lQ, heads),
            threadGroup: (32, 1, 1),
            outputShapes: [[heads, lQ, d]],
            outputDTypes: [origType]
        )[0]
        return out.reshaped(q.shape)
    }

    private static func referenceAttention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        shape: Shape,
        options: Options
    ) -> MLXArray {
        let origType = q.dtype
        let usesFloat64 = origType == .float64 || k.dtype == .float64 || v.dtype == .float64
        if usesFloat64 && Device.defaultDevice() == .gpu {
            return Device.withDefaultDevice(.cpu) {
                referenceAttention(q: q, k: k, v: v, shape: shape, options: options)
            }
        }
        let computeType: DType = usesFloat64 ? .float64 : .float32
        let dV = v.dim(-1)
        let outShape = Array(q.shape.dropLast()) + [dV]
        let q3D = q.asType(computeType).reshaped(shape.heads, shape.lQ, shape.d)
        let k3D = k.asType(computeType).reshaped(shape.heads, shape.lK, shape.d)
        let v3D = v.asType(computeType).reshaped(shape.heads, shape.lK, dV)
        let scaleArr = MLXArray(options.scale).asType(computeType)
        let negInf = MLXArray(-Float.infinity).asType(computeType)
        let zero = MLXArray(Float(0)).asType(computeType)
        let scores = matmul(q3D, k3D.swappedAxes(-1, -2)) * scaleArr
        var allowed = scores .!= negInf
        if options.causal {
            let offset = shape.lK - shape.lQ
            let qIdx = MLXArray(Int32(offset) ..< Int32(offset + shape.lQ)).reshaped(1, shape.lQ, 1)
            let kIdx = MLXArray(Int32(0) ..< Int32(shape.lK)).reshaped(1, 1, shape.lK)
            allowed = allowed .&& (qIdx .>= kIdx)
        }
        let anyAllowed = any(allowed, axis: -1, keepDims: true)
        let safeScores = which(anyAllowed, which(allowed, scores, negInf), zero)
        let probs = which(anyAllowed, softmax(safeScores, axis: -1), zero)
        return matmul(probs, v3D).asType(origType).reshaped(outShape)
    }
}

extension SageMXAttentionMetal {
    package struct Shape: Sendable {
        package var heads: Int
        package var lQ: Int
        package var lK: Int
        package var d: Int

        package init(heads: Int, lQ: Int, lK: Int, d: Int) {
            self.heads = heads
            self.lQ = lQ
            self.lK = lK
            self.d = d
        }
    }

    package static func cpuAttention(
        q: [Float],
        k: [Float],
        v: [Float],
        shape: Shape,
        options: Options
    ) -> [Float] {
        let (heads, lQ, lK, d) = (shape.heads, shape.lQ, shape.lK, shape.d)
        let thresh: Float = options.useUOS ? uosThreshold : 6.0
        let centerK = options.meanCenter && lK > 1
        let centerQ = options.meanCenter
        var out = [Float](repeating: 0, count: heads * lQ * d)
        for h in 0 ..< heads {
            let kBase = h * lK * d
            let kMean = centerK ? cpuMeanVector(k, base: kBase, count: lK, dim: d) : [Float](repeating: 0, count: d)
            let qMean = centerQ ? cpuMeanVector(q, base: h * lQ * d, count: lQ, dim: d) : [Float](repeating: 0, count: d)
            var kQuant = [Float](repeating: 0, count: lK * d)
            var deltaS = [Float](repeating: 0, count: lK)
            for j in 0 ..< lK {
                let kc = (0 ..< d).map { k[kBase + j * d + $0] - kMean[$0] }
                if centerQ {
                    var ds: Float = 0
                    for c in 0 ..< d { ds += qMean[c] * kc[c] }
                    deltaS[j] = ds
                }
                let qk = cpuQuantizeBlocks(kc, threshold: thresh)
                for c in 0 ..< d { kQuant[j * d + c] = qk[c] }
            }
            let ctx = HeadContext(head: h, shape: shape, q: q, v: v, kQuant: kQuant, qMean: qMean, deltaS: deltaS)
            computeHeadRows(ctx: ctx, centerQ: centerQ, options: options, into: &out)
        }
        return out
    }

    private struct HeadContext {
        var head: Int
        var shape: Shape
        var q: [Float]
        var v: [Float]
        var kQuant: [Float]
        var qMean: [Float]
        var deltaS: [Float]
    }

    private static func cpuMeanVector(_ values: [Float], base: Int, count: Int, dim: Int) -> [Float] {
        var meanVec = [Float](repeating: 0, count: dim)
        let inv = 1.0 / Float(count)
        for idx in 0 ..< count {
            for c in 0 ..< dim { meanVec[c] += values[base + idx * dim + c] * inv }
        }
        return meanVec
    }

    private static func cpuQuantizeBlocks(_ centered: [Float], threshold: Float) -> [Float] {
        stride(from: 0, to: centered.count, by: 32).flatMap { b0 in
            cpuQuantizeMXBlock32(Array(centered[b0 ..< (b0 + 32)]), threshold: threshold)
        }
    }

    private static func cpuSoftmaxWeights(
        score: Float, mRun: Float, usePNQ: Bool
    ) -> (alpha: Float, pHat: Float, mNew: Float) {
        let bad = mRun.isNaN || score.isNaN
        let mNew: Float = bad ? .nan : max(mRun, score)
        let active = bad || mNew != -.infinity
        let alpha: Float = bad ? .nan : ((mRun != -.infinity && active) ? expf(mRun - mNew) : 0)
        let rawBeta: Float = bad ? .nan : ((active && score != -.infinity) ? expf(score - mNew) : 0)
        let pHat: Float = usePNQ && !rawBeta.isNaN ? (rintf(min(max(rawBeta, 0), 1) * 255.0) / 255.0) : rawBeta
        return (alpha, pHat, mNew)
    }

    private static func computeHeadRows(
        ctx: HeadContext,
        centerQ: Bool,
        options: Options,
        into out: inout [Float]
    ) {
        let (h, lQ, lK, d) = (ctx.head, ctx.shape.lQ, ctx.shape.lK, ctx.shape.d)
        let thresh: Float = options.useUOS ? uosThreshold : 6.0
        let qBase = h * lQ * d
        let vBase = h * lK * d
        for i in 0 ..< lQ {
            let qc = (0 ..< d).map { ctx.q[qBase + i * d + $0] - ctx.qMean[$0] }
            let qq = cpuQuantizeBlocks(qc, threshold: thresh)
            var mRun: Float = -.infinity
            var lRun: Float = 0
            var acc = [Float](repeating: 0, count: d)
            let maxJ = options.causal ? min(lK - 1, (lK - lQ) + i) : (lK - 1)
            guard maxJ >= 0 else { continue }
            for j in 0 ... maxJ {
                var dot: Float = 0
                for c in 0 ..< d { dot += qq[c] * ctx.kQuant[j * d + c] }
                let s = (dot + (centerQ ? ctx.deltaS[j] : 0)) * options.scale
                let (alpha, pHat, mNew) = cpuSoftmaxWeights(score: s, mRun: mRun, usePNQ: options.usePNQ)
                lRun = lRun * alpha + pHat
                mRun = mNew
                for c in 0 ..< d {
                    acc[c] = acc[c] * alpha + pHat * ctx.v[vBase + j * d + c]
                }
            }
            let inv: Float = (lRun.isNaN || mRun.isNaN) ? .nan : (lRun > 0 ? (1.0 / lRun) : 0)
            for c in 0 ..< d { out[qBase + i * d + c] = acc[c] * inv }
        }
    }
}

private let sageMXSource = """
    const int H = int(shapes[0]);
    const int LQ = int(shapes[1]);
    const int LK = int(shapes[2]);
    const int D = int(shapes[3]);
    const int lane = int(thread_index_in_simdgroup);
    const int qi = int(threadgroup_position_in_grid.y);
    const int h = int(threadgroup_position_in_grid.z);
    if (h >= H || qi >= LQ) { return; }
    const float attn_scale = params[0];
    const float grid_max = params[1];
    const size_t q_base = (size_t(h) * size_t(LQ) + size_t(qi)) * size_t(D);
    const int max_j = CAUSAL ? min(LK - 1, (LK - LQ) + qi) : (LK - 1);
    if (max_j < 0) {
        for (int b = 0; b < BLOCKS; ++b) {
            out[q_base + size_t(b * 32 + lane)] = static_cast<T>(0.0f);
        }
        return;
    }

    float q_quant[BLOCKS];
    float o_acc[BLOCKS];
    for (int b = 0; b < BLOCKS; ++b) {
        const float qv = q_c[q_base + size_t(b * 32 + lane)];
        const bool q_nan = simd_any(isnan(qv));
        const float qmax = simd_max(fabs(qv));
        if (q_nan) {
            q_quant[b] = NAN;
        } else if (qmax > 0.0f) {
            const float qsc = basalt_e8m0_scale(basalt_compute_e8m0(qmax, grid_max));
            q_quant[b] = (qsc > 0.0f) ? (basalt_e2m1_val(basalt_quant_e2m1(qv, 1.0f / qsc)) * qsc) : 0.0f;
        } else {
            q_quant[b] = 0.0f;
        }
        o_acc[b] = 0.0f;
    }

    float m_run = -INFINITY;
    float l_run = 0.0f;
    for (int j = 0; j <= max_j; ++j) {
        const size_t kv_base = (size_t(h) * size_t(LK) + size_t(j)) * size_t(D);
        float dot_part = 0.0f;
        for (int b = 0; b < BLOCKS; ++b) {
            const float kv = k_c[kv_base + size_t(b * 32 + lane)];
            const bool k_nan = simd_any(isnan(kv));
            const float kmax = simd_max(fabs(kv));
            float kq = 0.0f;
            if (k_nan) {
                kq = NAN;
            } else if (kmax > 0.0f) {
                const float ksc = basalt_e8m0_scale(basalt_compute_e8m0(kmax, grid_max));
                kq = (ksc > 0.0f) ? (basalt_e2m1_val(basalt_quant_e2m1(kv, 1.0f / ksc)) * ksc) : 0.0f;
            }
            dot_part = fma(q_quant[b], kq, dot_part);
        }
        const float ds = HAS_DELTA ? delta_s[size_t(h) * size_t(LK) + size_t(j)] : 0.0f;
        const float s = (simd_sum(dot_part) + ds) * attn_scale;
        const bool bad = isnan(m_run) || isnan(s);
        const float m_new = bad ? NAN : max(m_run, s);
        const bool active_row = bad || (m_new != -INFINITY);
        const float alpha = bad ? NAN : ((m_run != -INFINITY && active_row) ? exp(m_run - m_new) : 0.0f);
        const float raw_beta = bad ? NAN : ((active_row && s != -INFINITY) ? exp(s - m_new) : 0.0f);
        const float p_hat = (USE_PNQ && !isnan(raw_beta)) ? basalt_pnq_prob8(raw_beta) : raw_beta;
        l_run = l_run * alpha + p_hat;
        m_run = m_new;
        for (int b = 0; b < BLOCKS; ++b) {
            o_acc[b] = fma(p_hat, float(v_in[kv_base + size_t(b * 32 + lane)]), o_acc[b] * alpha);
        }
    }

    const float inv_l = (isnan(l_run) || isnan(m_run)) ? NAN : ((l_run > 0.0f) ? (1.0f / l_run) : 0.0f);
    for (int b = 0; b < BLOCKS; ++b) {
        out[q_base + size_t(b * 32 + lane)] = static_cast<T>(o_acc[b] * inv_l);
    }
    """

private let sageMXFlashKernel = MLXFast.metalKernel(
    name: "sage_mx_flash_attn_metal",
    inputNames: ["q_c", "k_c", "v_in", "delta_s", "params", "shapes"],
    outputNames: ["out"],
    source: sageMXSource,
    header: BasaltMetalSubstrate.combinedHeader + SageMXAttentionMetal.onlineSoftmaxPNQHeader
)
