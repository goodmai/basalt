import BasaltCore
import Foundation
import MLX

package enum SageMXAttentionMetal {
    package static let uosThreshold: Float = 7.25
    package static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]
    package static let maxHeadDim = 256

    package static let onlineSoftmaxPNQHeader = """
        inline uint8_t basalt_uos_e8m0(float max_abs) {
            return basalt_compute_e8m0(max_abs, 7.25f);
        }

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

    package static func attention(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        options: Options
    ) -> MLXArray {
        let origType = q.dtype
        let lQ = q.dim(-2)
        let lK = k.dim(-2)
        let d = q.dim(-1)
        let heads = q.size / max(1, lQ * d)
        precondition(d.isMultiple(of: 32) && d <= maxHeadDim && k.size == heads * lK * d)

        let q3D = q.asType(.float32).reshaped(heads, lQ, d)
        let k3D = k.asType(.float32).reshaped(heads, lK, d)
        let v3D = v.asType(origType).reshaped(heads, lK, d)

        let kMean: MLXArray = options.meanCenter ? mean(k3D, axis: 1, keepDims: true) : MLXArray.zeros([1, 1, d])
        let qMean: MLXArray = options.meanCenter ? mean(q3D, axis: 1, keepDims: true) : MLXArray.zeros([1, 1, d])
        let kCentered = k3D - kMean
        let qCentered = q3D - qMean
        let deltaS: MLXArray = options.meanCenter
            ? matmul(qMean, kCentered.swappedAxes(-1, -2)).squeezed(axis: 1)
            : MLXArray.zeros([heads, lK], dtype: .float32)

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
                ("HAS_DELTA", options.meanCenter),
            ],
            grid: (32, lQ, heads),
            threadGroup: (32, 1, 1),
            outputShapes: [[heads, lQ, d]],
            outputDTypes: [origType]
        )[0]
        return out.reshaped(q.shape)
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
        var out = [Float](repeating: 0, count: heads * lQ * d)
        for h in 0 ..< heads {
            let qBase = h * lQ * d
            let kBase = h * lK * d
            var kMean = [Float](repeating: 0, count: d)
            var qMean = [Float](repeating: 0, count: d)
            if options.meanCenter {
                for j in 0 ..< lK {
                    for c in 0 ..< d { kMean[c] += k[kBase + j * d + c] / Float(lK) }
                }
                for i in 0 ..< lQ {
                    for c in 0 ..< d { qMean[c] += q[qBase + i * d + c] / Float(lQ) }
                }
            }
            var kQuant = [Float](repeating: 0, count: lK * d)
            var deltaS = [Float](repeating: 0, count: lK)
            for j in 0 ..< lK {
                let kc = (0 ..< d).map { k[kBase + j * d + $0] - kMean[$0] }
                if options.meanCenter {
                    var ds: Float = 0
                    for c in 0 ..< d { ds += qMean[c] * kc[c] }
                    deltaS[j] = ds
                }
                let qk = stride(from: 0, to: d, by: 32).flatMap { b0 in
                    cpuQuantizeMXBlock32(Array(kc[b0 ..< (b0 + 32)]), threshold: thresh)
                }
                for c in 0 ..< d { kQuant[j * d + c] = qk[c] }
            }
            let ctx = HeadContext(head: h, shape: shape, q: q, v: v, kQuant: kQuant, qMean: qMean, deltaS: deltaS)
            computeHeadRows(ctx: ctx, options: options, into: &out)
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

    private static func computeHeadRows(
        ctx: HeadContext,
        options: Options,
        into out: inout [Float]
    ) {
        let (h, lQ, lK, d) = (ctx.head, ctx.shape.lQ, ctx.shape.lK, ctx.shape.d)
        let thresh: Float = options.useUOS ? uosThreshold : 6.0
        let qBase = h * lQ * d
        let vBase = h * lK * d
        for i in 0 ..< lQ {
            let qc = (0 ..< d).map { ctx.q[qBase + i * d + $0] - ctx.qMean[$0] }
            let qq = stride(from: 0, to: d, by: 32).flatMap { b0 in
                cpuQuantizeMXBlock32(Array(qc[b0 ..< (b0 + 32)]), threshold: thresh)
            }
            var mRun: Float = -.infinity
            var lRun: Float = 0
            var acc = [Float](repeating: 0, count: d)
            let maxJ = options.causal ? min(lK - 1, (lK - lQ) + i) : (lK - 1)
            guard maxJ >= 0 else { continue }
            for j in 0 ... maxJ {
                var dot: Float = 0
                for c in 0 ..< d { dot += qq[c] * ctx.kQuant[j * d + c] }
                let s = (dot + (options.meanCenter ? ctx.deltaS[j] : 0)) * options.scale
                let mNew = max(mRun, s)
                let rawAlpha = mRun == -.infinity ? 0 : expf(mRun - mNew)
                let rawBeta = expf(s - mNew)
                let pHat = options.usePNQ ? (rintf(min(max(rawBeta, 0), 1) * 255.0) / 255.0) : rawBeta
                lRun = lRun * rawAlpha + pHat
                mRun = mNew
                for c in 0 ..< d {
                    acc[c] = acc[c] * rawAlpha + pHat * ctx.v[vBase + j * d + c]
                }
            }
            let inv: Float = lRun > 0 ? (1.0 / lRun) : 0
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
    const int max_j = CAUSAL ? min(LK - 1, (LK - LQ) + qi) : (LK - 1);
    if (max_j < 0) {
        for (int b = 0; b < BLOCKS; ++b) {
            out[(size_t(h) * size_t(LQ) + size_t(qi)) * size_t(D) + size_t(b * 32 + lane)] = static_cast<T>(0.0f);
        }
        return;
    }

    float q_quant[BLOCKS];
    float o_acc[BLOCKS];
    const device float* q_ptr = q_c + (size_t(h) * size_t(LQ) + size_t(qi)) * size_t(D);
    for (int b = 0; b < BLOCKS; ++b) {
        const float qv = q_ptr[b * 32 + lane];
        const float qmax = simd_max(fabs(qv));
        if (qmax > 0.0f) {
            const float qsc = basalt_e8m0_scale(basalt_compute_e8m0(qmax, grid_max));
            q_quant[b] = basalt_e2m1_val(basalt_quant_e2m1(qv, 1.0f / qsc)) * qsc;
        } else {
            q_quant[b] = 0.0f;
        }
        o_acc[b] = 0.0f;
    }

    float m_run = -INFINITY;
    float l_run = 0.0f;
    for (int j = 0; j <= max_j; ++j) {
        const device float* k_ptr = k_c + (size_t(h) * size_t(LK) + size_t(j)) * size_t(D);
        const device T* v_ptr = v_in + (size_t(h) * size_t(LK) + size_t(j)) * size_t(D);
        float dot_part = 0.0f;
        for (int b = 0; b < BLOCKS; ++b) {
            const float kv = k_ptr[b * 32 + lane];
            const float kmax = simd_max(fabs(kv));
            float kq = 0.0f;
            if (kmax > 0.0f) {
                const float ksc = basalt_e8m0_scale(basalt_compute_e8m0(kmax, grid_max));
                kq = basalt_e2m1_val(basalt_quant_e2m1(kv, 1.0f / ksc)) * ksc;
            }
            dot_part = fma(q_quant[b], kq, dot_part);
        }
        float ds = HAS_DELTA ? delta_s[h * LK + j] : 0.0f;
        const float s = (simd_sum(dot_part) + ds) * attn_scale;
        const float m_new = max(m_run, s);
        const float alpha = (m_run == -INFINITY) ? 0.0f : exp(m_run - m_new);
        const float raw_beta = exp(s - m_new);
        const float p_hat = USE_PNQ ? basalt_pnq_prob8(raw_beta) : raw_beta;
        l_run = l_run * alpha + p_hat;
        m_run = m_new;
        for (int b = 0; b < BLOCKS; ++b) {
            o_acc[b] = fma(p_hat, float(v_ptr[b * 32 + lane]), o_acc[b] * alpha);
        }
    }

    const float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    device T* out_ptr = out + (size_t(h) * size_t(LQ) + size_t(qi)) * size_t(D);
    for (int b = 0; b < BLOCKS; ++b) {
        out_ptr[b * 32 + lane] = static_cast<T>(o_acc[b] * inv_l);
    }
    """

private let sageMXFlashKernel = MLXFast.metalKernel(
    name: "sage_mx_flash_attn_metal",
    inputNames: ["q_c", "k_c", "v_in", "delta_s", "params", "shapes"],
    outputNames: ["out"],
    source: sageMXSource,
    header: BasaltMetalSubstrate.combinedHeader + SageMXAttentionMetal.onlineSoftmaxPNQHeader
)
