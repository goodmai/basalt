import BasaltCore
import Foundation
import MLX

public struct FusedLogitConfidence {
    public static let dtypes: Set<DType> = [.bfloat16, .float16, .float32]
    public static let threadsPerRow = 256
    public static let simdgroupsPerRow = 8

    public let top1TokenID: MLXArray
    public let top1Logprob: MLXArray
    public let top2Margin: MLXArray
    public let entropy: MLXArray

    public init(top1TokenID: MLXArray, top1Logprob: MLXArray, top2Margin: MLXArray, entropy: MLXArray) {
        self.top1TokenID = top1TokenID
        self.top1Logprob = top1Logprob
        self.top2Margin = top2Margin
        self.entropy = entropy
    }

    public init(_ logits: MLXArray) {
        self = Self.compute(logits)
    }

    public static func compute(_ logits: MLXArray) -> FusedLogitConfidence {
        let leading = Array(logits.shape.dropLast())
        let vocab = logits.ndim > 0 ? logits.dim(-1) : 0
        let rows = vocab > 0 ? logits.size / vocab : 0
        guard rows > 0, vocab > 0 else {
            let emptyFloat = MLXArray.zeros(leading, type: Float.self)
            return FusedLogitConfidence(
                top1TokenID: MLXArray.zeros(leading, type: Int32.self),
                top1Logprob: emptyFloat,
                top2Margin: emptyFloat,
                entropy: emptyFloat)
        }
        guard Device.defaultDevice() == .gpu,
              dtypes.contains(logits.dtype),
              rows <= Int(Int32.max),
              vocab <= Int(Int32.max)
        else {
            return reference(logits)
        }
        let outputs = fusedLogitConfidenceKernel(
            [logits.reshaped(rows, vocab)],
            template: [("T", logits.dtype)],
            grid: (threadsPerRow, rows, 1),
            threadGroup: (threadsPerRow, 1, 1),
            outputShapes: [[rows], [rows], [rows], [rows]],
            outputDTypes: [.int32, .float32, .float32, .float32])
        return FusedLogitConfidence(
            top1TokenID: outputs[0].reshaped(leading),
            top1Logprob: outputs[1].reshaped(leading),
            top2Margin: outputs[2].reshaped(leading),
            entropy: outputs[3].reshaped(leading))
    }

    public static func reference(_ logits: MLXArray) -> FusedLogitConfidence {
        let leading = Array(logits.shape.dropLast())
        let vocab = logits.ndim > 0 ? logits.dim(-1) : 0
        let rows = vocab > 0 ? logits.size / vocab : 0
        guard rows > 0, vocab > 0 else {
            let emptyFloat = MLXArray.zeros(leading, type: Float.self)
            return FusedLogitConfidence(
                top1TokenID: MLXArray.zeros(leading, type: Int32.self),
                top1Logprob: emptyFloat,
                top2Margin: emptyFloat,
                entropy: emptyFloat)
        }
        let z = logits.asType(.float32, stream: logits.dtype == .float64 ? .cpu : .default).reshaped(rows, vocab)
        let max1 = z.max(axis: -1)
        let usable = logicalNot(isNaN(z).any(axis: -1)) .&& isFinite(max1)
        let safeZ = MLX.where(expandedDimensions(usable, axis: -1), z, MLXArray(Float(0)))
        let safeMax = safeZ.max(axis: -1, keepDims: true)
        let shifted = safeZ - safeMax
        let sumExp = exp(shifted).sum(axis: -1, keepDims: true)
        let logSum = log(sumExp)
        let logp = shifted - logSum
        let p = exp(logp)
        let finiteLogp = MLX.where(isFinite(logp), logp, MLXArray(Float(0)))
        let rawEntropy = maximum(-sum(p * finiteLogp, axis: -1), MLXArray(Float(0)))
        let rawTop1ID = argMax(safeZ, axis: -1).asType(.int32)
        let rawTop1Logprob = -logSum.squeezed(axis: -1)
        let rawMargin: MLXArray
        if vocab > 1 {
            let best = top(safeZ, k: 2, axis: -1)
            rawMargin = best.max(axis: -1) - best.min(axis: -1)
        } else {
            rawMargin = MLXArray.full([rows], values: MLXArray(Float.infinity))
        }
        let nan = MLXArray(Float.nan)
        return FusedLogitConfidence(
            top1TokenID: MLX.where(usable, rawTop1ID, MLXArray(Int32(0))).reshaped(leading),
            top1Logprob: MLX.where(usable, rawTop1Logprob, nan).reshaped(leading),
            top2Margin: MLX.where(usable, rawMargin, nan).reshaped(leading),
            entropy: MLX.where(usable, rawEntropy, nan).reshaped(leading))
    }

    public static func statistics(logits: MLXArray) -> (probability: Double, entropy: Double, margin: Double)? {
        guard logits.size > 0 else { return nil }
        let clean = (Device.defaultDevice() == .gpu && logits.dtype == .float64)
            ? logits.asType(.float32, stream: .cpu) : logits
        let out = compute(clean.reshaped(1, -1))
        let scalars = concatenated([out.top1Logprob, out.top2Margin, out.entropy], axis: 0).asArray(Float.self)
        guard scalars.count == 3, scalars[0].isFinite, scalars[2].isFinite, !scalars[1].isNaN else { return nil }
        let first = Foundation::exp(Double(scalars[0]))
        let second = scalars[1].isInfinite ? 0.0 : Foundation::exp(Double(scalars[0] - scalars[1]))
        guard first.isFinite, second.isFinite else { return nil }
        return (min(max(first, 0), 1), Double(scalars[2]), max(0, first - second))
    }

    public static func summarize(_ logits: MLXArray) -> ResponseConfidence? {
        let vocab = logits.ndim > 0 ? logits.dim(-1) : 0
        let rows = vocab > 0 ? logits.size / vocab : 0
        guard rows > 0, vocab > 0 else { return nil }
        let clean = (Device.defaultDevice() == .gpu && logits.dtype == .float64)
            ? logits.asType(.float32, stream: .cpu) : logits
        let out = compute(clean.reshaped(rows, vocab))
        let table = stacked([out.top1Logprob, out.top2Margin, out.entropy], axis: -1).asArray(Float.self)
        var probabilities: [Double] = []
        var entropies: [Double] = []
        var margins: [Double] = []
        probabilities.reserveCapacity(rows)
        entropies.reserveCapacity(rows)
        margins.reserveCapacity(rows)
        for row in 0..<rows {
            let base = row * 3
            let (logp, gap, ent) = (table[base], table[base + 1], table[base + 2])
            guard logp.isFinite, ent.isFinite, !gap.isNaN else { continue }
            let first = Foundation::exp(Double(logp))
            let second = gap.isInfinite ? 0.0 : Foundation::exp(Double(logp) - Double(gap))
            guard first.isFinite, second.isFinite else { continue }
            probabilities.append(min(max(first, 0), 1))
            entropies.append(Double(ent))
            margins.append(max(0, first - second))
        }
        guard var summary = ResponseConfidence.summarize(probabilities) else { return nil }
        summary.meanEntropy = entropies.isEmpty ? nil : entropies.reduce(0, +) / Double(entropies.count)
        summary.minMargin = margins.min()
        return summary
    }
}

private enum FusedLogitConfidenceMetal {
    static let source = """
        constexpr int THREADS = 256;
        constexpr int SIMDGROUPS = 8;
        constexpr int MAX_IDX = 2147483647;
        const int row = int(threadgroup_position_in_grid.y);
        const int tid = int(thread_position_in_threadgroup.x);
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(simdgroup_index_in_threadgroup);
        const int V = int(logits_shape[1]);
        const size_t row_base = size_t(row) * size_t(V);

        float m1 = -INFINITY;
        int idx1 = MAX_IDX;
        float m2 = -INFINITY;
        float m = -INFINITY;
        float s = 0.0f;
        float wy = 0.0f;
        int bad = 0;

        for (uint col = uint(tid); col < uint(V); col += uint(THREADS)) {
            const float x = float(logits[row_base + size_t(col)]);
            if (isnan(x) || x == INFINITY) {
                bad = 1;
            } else {
                if (x > m1) {
                    m2 = m1;
                    m1 = x;
                    idx1 = int(col);
                } else if (x > m2) {
                    m2 = x;
                }
                if (x > -INFINITY) {
                    if (x > m) {
                        const float d = (s > 0.0f) ? (m - x) : 0.0f;
                        const float scale = (s > 0.0f) ? metal::exp(d) : 0.0f;
                        wy = (scale > 0.0f) ? (scale * (wy + d * s)) : 0.0f;
                        s = scale * s + 1.0f;
                        m = x;
                    } else {
                        const float d = x - m;
                        const float p = metal::exp(d);
                        s += p;
                        if (p > 0.0f) { wy += p * d; }
                    }
                }
            }
        }

        const int sg_bad = simd_max(bad);
        const float sg_m1 = simd_max(m1);
        const int sg_idx1 = simd_min((m1 == sg_m1) ? idx1 : MAX_IDX);
        const float sg_m2 = simd_max((idx1 == sg_idx1) ? m2 : m1);
        const float sg_m = simd_max(m);
        const float lane_d = (s > 0.0f) ? (m - sg_m) : 0.0f;
        const float lane_f = (s > 0.0f) ? metal::exp(lane_d) : 0.0f;
        const float sg_s = simd_sum(s * lane_f);
        const float sg_wy = simd_sum((lane_f > 0.0f) ? ((wy + lane_d * s) * lane_f) : 0.0f);

        threadgroup float tg_m1[SIMDGROUPS];
        threadgroup int tg_idx1[SIMDGROUPS];
        threadgroup float tg_m2[SIMDGROUPS];
        threadgroup float tg_m[SIMDGROUPS];
        threadgroup float tg_s[SIMDGROUPS];
        threadgroup float tg_wy[SIMDGROUPS];
        threadgroup int tg_bad[SIMDGROUPS];

        if (lane == 0) {
            tg_m1[sg] = sg_m1;
            tg_idx1[sg] = sg_idx1;
            tg_m2[sg] = sg_m2;
            tg_m[sg] = sg_m;
            tg_s[sg] = sg_s;
            tg_wy[sg] = sg_wy;
            tg_bad[sg] = sg_bad;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            const float b_m1 = (lane < SIMDGROUPS) ? tg_m1[lane] : -INFINITY;
            const int b_idx1 = (lane < SIMDGROUPS) ? tg_idx1[lane] : MAX_IDX;
            const float b_m2 = (lane < SIMDGROUPS) ? tg_m2[lane] : -INFINITY;
            const float b_m = (lane < SIMDGROUPS) ? tg_m[lane] : -INFINITY;
            const float b_s = (lane < SIMDGROUPS) ? tg_s[lane] : 0.0f;
            const float b_wy = (lane < SIMDGROUPS) ? tg_wy[lane] : 0.0f;
            const int b_bad = (lane < SIMDGROUPS) ? tg_bad[lane] : 0;

            const int row_bad = simd_max(b_bad);
            const float row_m1 = simd_max(b_m1);
            const int row_idx1 = simd_min((b_m1 == row_m1) ? b_idx1 : MAX_IDX);
            const float row_m2 = simd_max((b_idx1 == row_idx1) ? b_m2 : b_m1);
            const float row_m = simd_max(b_m);
            const float bd = (b_s > 0.0f) ? (b_m - row_m) : 0.0f;
            const float bf = (b_s > 0.0f) ? metal::exp(bd) : 0.0f;
            const float row_s = simd_sum(b_s * bf);
            const float row_wy = simd_sum((bf > 0.0f) ? ((b_wy + bd * b_s) * bf) : 0.0f);

            if (lane == 0) {
                if (row_bad != 0 || !(row_s > 0.0f) || !(row_m1 > -INFINITY)) {
                    top1_id[row] = 0;
                    top1_logprob[row] = NAN;
                    top2_margin[row] = NAN;
                    entropy[row] = NAN;
                } else {
                    const float log_s = metal::log(row_s);
                    const float margin = (V > 1 && row_m2 > -INFINITY) ? (row_m1 - row_m2) : INFINITY;
                    top1_id[row] = row_idx1;
                    top1_logprob[row] = (row_m1 - row_m) - log_s;
                    top2_margin[row] = margin;
                    entropy[row] = max(0.0f, log_s - row_wy / row_s);
                }
            }
        }
        """
}

private let fusedLogitConfidenceKernel = MLXFast.metalKernel(
    name: "fused_logit_confidence",
    inputNames: ["logits"],
    outputNames: ["top1_id", "top1_logprob", "top2_margin", "entropy"],
    source: FusedLogitConfidenceMetal.source,
    header: "#include <metal_simdgroup>\n")
