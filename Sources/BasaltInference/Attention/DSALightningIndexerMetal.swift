import BasaltCore
import Foundation
import MLX

package enum DSALightningIndexerMetal {
    package static let supportedDTypes: Set<DType> = [.bfloat16, .float16, .float32]
    package static let maxLatentDim = 1024
    package static let maxRopeDim = 256

    package struct ScoreOptions: Sendable {
        package var causal: Bool
        package var qOffset: Int?
        package var useMetal: Bool

        package init(causal: Bool = true, qOffset: Int? = nil, useMetal: Bool = true) {
            self.causal = causal
            self.qOffset = qOffset
            self.useMetal = useMetal
        }
    }

    package struct SparseLatentInputs {
        package var qAbsorbed: MLXArray
        package var qPe: MLXArray?
        package var cKV: MLXArray
        package var kPe: MLXArray?
        package var topIndices: MLXArray
        package var valueHeadGate: MLXArray?

        package init(
            qAbsorbed: MLXArray,
            qPe: MLXArray? = nil,
            cKV: MLXArray,
            kPe: MLXArray? = nil,
            topIndices: MLXArray,
            valueHeadGate: MLXArray? = nil
        ) {
            self.qAbsorbed = qAbsorbed
            self.qPe = qPe
            self.cKV = cKV
            self.kPe = kPe
            self.topIndices = topIndices
            self.valueHeadGate = valueHeadGate
        }
    }

    package struct SparseLatentOptions: Sendable {
        package var scale: Float
        package var causal: Bool
        package var qOffset: Int?
        package var useMetal: Bool

        package init(scale: Float, causal: Bool = true, qOffset: Int? = nil, useMetal: Bool = true) {
            self.scale = scale
            self.causal = causal
            self.qOffset = qOffset
            self.useMetal = useMetal
        }
    }

    private static func runOnCPUIfFloat64(_ arrays: [MLXArray?], _ body: () -> MLXArray) -> MLXArray {
        guard Device.defaultDevice() == .gpu, arrays.contains(where: { $0?.dtype == .float64 }) else {
            return body()
        }
        return Device.withDefaultDevice(.cpu, body)
    }

    package static func score(
        q: MLXArray,
        k: MLXArray,
        weights: MLXArray,
        options: ScoreOptions = ScoreOptions()
    ) -> MLXArray {
        runOnCPUIfFloat64([q, k, weights]) {
            let batch = q.dim(0)
            let lQ = q.dim(1)
            let lK = k.dim(1)
            guard batch > 0, lQ > 0, lK > 0 else {
                return MLXArray.zeros([batch, lQ, lK], dtype: .float32)
            }
            if options.useMetal,
               Device.defaultDevice() == .gpu,
               q.ndim == 4, k.ndim == 3, weights.ndim == 3,
               q.dim(2) > 0, q.dim(3) > 0,
               k.shape == [batch, lK, q.dim(3)],
               weights.shape == [batch, lQ, q.dim(2)],
               supportedDTypes.contains(q.dtype),
               k.dtype == q.dtype,
               supportedDTypes.contains(weights.dtype) {
                return scoreMetal(q: q, k: k, weights: weights, options: options)
            }
            return scoreMLX(q: q, k: k, weights: weights, options: options)
        }
    }

    package static func selectTopK(
        scores: MLXArray,
        topK: Int,
        causal: Bool = true,
        qOffset: Int? = nil
    ) -> MLXArray {
        runOnCPUIfFloat64([scores]) {
            let batch = scores.dim(0)
            let lQ = scores.dim(1)
            let lK = scores.dim(2)
            let kSel = min(max(0, topK), lK)
            guard kSel > 0, lQ > 0, batch > 0 else {
                return MLXArray.zeros([batch, lQ, kSel], type: Int32.self)
            }
            guard causal else {
                return BlockSparseRoutingMetal.deterministicTopIndices(scores, k: kSel)
            }
            let effectiveOffset = qOffset ?? max(0, lK - lQ)
            let qPos = MLXArray(Int32(effectiveOffset) ..< Int32(effectiveOffset + lQ)).reshaped(1, lQ, 1)
            let kPos = MLXArray(Int32(0) ..< Int32(lK)).reshaped(1, 1, lK)
            let masked = MLX.where(kPos .<= qPos, scores.asType(.float32), MLXArray(-Float.infinity))
            let picked = BlockSparseRoutingMetal.deterministicTopIndices(masked, k: kSel)
            return MLX.where(picked .<= qPos, picked, MLXArray(Int32(-1)))
        }
    }

    package static func sparseLatentAttention(
        _ inputs: SparseLatentInputs,
        options: SparseLatentOptions
    ) -> MLXArray {
        runOnCPUIfFloat64([inputs.qAbsorbed, inputs.qPe, inputs.cKV, inputs.kPe, inputs.topIndices, inputs.valueHeadGate]) {
            let origType = inputs.qAbsorbed.dtype
            let batch = inputs.qAbsorbed.dim(0)
            let heads = inputs.qAbsorbed.dim(1)
            let lQ = inputs.qAbsorbed.dim(2)
            let dc = inputs.qAbsorbed.dim(3)
            let lK = inputs.cKV.dim(-2)
            let topK = inputs.topIndices.dim(-1)
            guard batch > 0, heads > 0, lQ > 0, dc > 0, lK > 0, topK > 0 else {
                return MLXArray.zeros(inputs.qAbsorbed.shape, dtype: origType)
            }
            if canUseSparseMetal(inputs, useMetal: options.useMetal) {
                return sparseLatentAttentionMetal(inputs, options: options)
            }
            return sparseLatentAttentionMLX(inputs, options: options)
        }
    }
}

extension DSALightningIndexerMetal {
    private static func scoreMetal(
        q: MLXArray,
        k: MLXArray,
        weights: MLXArray,
        options: ScoreOptions
    ) -> MLXArray {
        let origType = q.dtype
        let batch = q.dim(0)
        let lQ = q.dim(1)
        let hI = q.dim(2)
        let dI = q.dim(3)
        let lK = k.dim(1)
        let effectiveOffset = options.qOffset ?? max(0, lK - lQ)
        let shapes = MLXArray([
            Int32(batch), Int32(lQ), Int32(lK), Int32(hI), Int32(dI),
            Int32(options.causal ? 1 : 0), Int32(effectiveOffset),
        ])
        return dsaIndexerScoreKernel(
            [
                contiguous(q.asType(origType)),
                contiguous(k.asType(origType)),
                contiguous(weights.asType(.float32)),
                shapes,
            ],
            template: [("T", origType)],
            grid: (32, lK, batch * lQ),
            threadGroup: (32, 1, 1),
            outputShapes: [[batch, lQ, lK]],
            outputDTypes: [.float32]
        )[0]
    }

    private static func scoreMLX(
        q: MLXArray,
        k: MLXArray,
        weights: MLXArray,
        options: ScoreOptions
    ) -> MLXArray {
        let batch = q.dim(0)
        let lQ = q.dim(1)
        let hI = q.dim(2)
        let dI = q.dim(3)
        let lK = k.dim(1)
        let qFlat = q.asType(.float32).reshaped(batch, lQ * hI, dI)
        let dots = matmul(qFlat, k.asType(.float32).swappedAxes(-1, -2)).reshaped(batch, lQ, hI, lK)
        let activated = maximum(dots, MLXArray(Float(0)))
        let weighted = activated * weights.asType(.float32).expandedDimensions(axis: -1)
        let rawScores = weighted.sum(axis: 2)
        guard options.causal else { return rawScores }
        let effectiveOffset = options.qOffset ?? max(0, lK - lQ)
        let qPos = MLXArray(Int32(effectiveOffset) ..< Int32(effectiveOffset + lQ)).reshaped(1, lQ, 1)
        let kPos = MLXArray(Int32(0) ..< Int32(lK)).reshaped(1, 1, lK)
        return MLX.where(kPos .<= qPos, rawScores, MLXArray(-Float.infinity))
    }
}

extension DSALightningIndexerMetal {
    private static func canUseSparseMetal(_ inputs: SparseLatentInputs, useMetal: Bool) -> Bool {
        let origType = inputs.qAbsorbed.dtype
        let dc = inputs.qAbsorbed.dim(-1)
        let dr = inputs.qPe?.dim(-1) ?? 0
        let peDTypesOK = (inputs.qPe.map { $0.dtype == origType } ?? true)
            && (inputs.kPe.map { $0.dtype == origType } ?? true)
        let gateDTypeOK = inputs.valueHeadGate.map { supportedDTypes.contains($0.dtype) } ?? true
        return useMetal
            && Device.defaultDevice() == .gpu
            && supportedDTypes.contains(origType)
            && inputs.cKV.dtype == origType
            && peDTypesOK && gateDTypeOK
            && dc > 0 && dc <= maxLatentDim
            && dr <= maxRopeDim
    }

    private static func preparePeArgs(
        _ inputs: SparseLatentInputs,
        batch: Int,
        heads: Int,
        lQ: Int,
        lK: Int,
        origType: DType
    ) -> (dr: Int, hasPe: Bool, qPe: MLXArray, kPe: MLXArray) {
        if let qPe = inputs.qPe, let kPe = inputs.kPe, qPe.dim(-1) > 0 {
            let dr = qPe.dim(-1)
            return (
                dr, true,
                contiguous(qPe.asType(origType).reshaped(batch * heads, lQ, dr)),
                contiguous(kPe.asType(origType).reshaped(batch, lK, dr))
            )
        }
        let dummy = MLXArray.zeros([1], dtype: origType)
        return (0, false, dummy, dummy)
    }

    private static func sparseLatentAttentionMetal(
        _ inputs: SparseLatentInputs,
        options: SparseLatentOptions
    ) -> MLXArray {
        let origType = inputs.qAbsorbed.dtype
        let (batch, heads, lQ, dc) = (
            inputs.qAbsorbed.dim(0), inputs.qAbsorbed.dim(1),
            inputs.qAbsorbed.dim(2), inputs.qAbsorbed.dim(3))
        let lK = inputs.cKV.dim(-2)
        let topK = inputs.topIndices.dim(-1)
        let effectiveOffset = options.qOffset ?? max(0, lK - lQ)
        let pe = preparePeArgs(inputs, batch: batch, heads: heads, lQ: lQ, lK: lK, origType: origType)
        let hasVGate = inputs.valueHeadGate != nil
        let vGateArg = inputs.valueHeadGate.map {
            contiguous($0.asType(.float32).reshaped(batch * heads, lK))
        } ?? MLXArray.zeros([1], dtype: .float32)

        let params = MLXArray([options.scale])
        let shapes = MLXArray([
            Int32(batch), Int32(heads), Int32(lQ), Int32(lK),
            Int32(dc), Int32(pe.dr), Int32(topK),
            Int32(options.causal ? 1 : 0), Int32(effectiveOffset),
        ])
        let out = dsaSparseLatentKernel(
            [
                contiguous(inputs.qAbsorbed.asType(origType).reshaped(batch * heads, lQ, dc)),
                pe.qPe,
                contiguous(inputs.cKV.asType(origType).reshaped(batch, lK, dc)),
                pe.kPe,
                contiguous(BlockSparseRoutingMetal.sanitizeInt32Indices(inputs.topIndices, upperBound: lK).reshaped(batch, lQ, topK)),
                vGateArg,
                params,
                shapes,
            ],
            template: [
                ("T", origType),
                ("DC_BLOCKS", (dc + 31) / 32),
                ("DR_BLOCKS", pe.hasPe ? (pe.dr + 31) / 32 : 0),
                ("HAS_PE", pe.hasPe),
                ("HAS_VGATE", hasVGate),
            ],
            grid: (32, lQ, batch * heads),
            threadGroup: (32, 1, 1),
            outputShapes: [[batch * heads, lQ, dc]],
            outputDTypes: [origType]
        )[0]
        return out.reshaped(batch, heads, lQ, dc)
    }

    private static func sparseLatentAttentionMLX(
        _ inputs: SparseLatentInputs,
        options: SparseLatentOptions
    ) -> MLXArray {
        let origType = inputs.qAbsorbed.dtype
        let batch = inputs.qAbsorbed.dim(0)
        let heads = inputs.qAbsorbed.dim(1)
        let lQ = inputs.qAbsorbed.dim(2)
        let dc = inputs.qAbsorbed.dim(3)
        let lK = inputs.cKV.dim(-2)
        let topK = inputs.topIndices.dim(-1)
        let effectiveOffset = options.qOffset ?? max(0, lK - lQ)

        let idx = BlockSparseRoutingMetal.sanitizeInt32Indices(inputs.topIndices, upperBound: lK).reshaped(batch, lQ, topK)
        var valid = (idx .>= Int32(0)) .&& (idx .< Int32(lK))
        if options.causal {
            let qPos = MLXArray(Int32(effectiveOffset) ..< Int32(effectiveOffset + lQ)).reshaped(1, lQ, 1)
            valid = valid .&& (idx .<= qPos)
        }
        let safeIdx = MLX.where(valid, idx, MLXArray(Int32(0)))
        let flatIdx = stopGradient(safeIdx.reshaped(batch, lQ * topK, 1))
        let cKV3D = inputs.cKV.asType(.float32).reshaped(batch, lK, dc)
        let cGathered = takeAlong(cKV3D, flatIdx, axis: 1).reshaped(batch, 1, lQ, topK, dc)
        let cSel = MLX.where(valid.reshaped(batch, 1, lQ, topK, 1), cGathered, MLXArray(Float(0)))
        let qAbsF = inputs.qAbsorbed.asType(.float32).expandedDimensions(axis: -2)
        var logits = matmul(qAbsF, cSel.swappedAxes(-1, -2)).squeezed(axis: -2)
        if let qPe = inputs.qPe, let kPe = inputs.kPe, qPe.dim(-1) > 0 {
            let dr = qPe.dim(-1)
            let kPe3D = kPe.asType(.float32).reshaped(batch, lK, dr)
            let kPeSel = takeAlong(kPe3D, flatIdx, axis: 1).reshaped(batch, 1, lQ, topK, dr)
            logits = logits + matmul(qPe.asType(.float32).expandedDimensions(axis: -2), kPeSel.swappedAxes(-1, -2)).squeezed(axis: -2)
        }
        logits = logits * options.scale
        let validMask = valid.expandedDimensions(axis: 1)
        let activeValid = validMask .&& (logits .!= MLXArray(-Float.infinity))
        let anyValid = activeValid.any(axis: -1).expandedDimensions(axis: -1)
        let maskedLogits = MLX.where(activeValid, logits, MLXArray(-Float.infinity))
        var probs = MLX.where(anyValid, softmax(MLX.where(anyValid, maskedLogits, MLXArray(Float(0))), axis: -1), MLXArray(Float(0)))
        if let vGate = inputs.valueHeadGate {
            let g3D = vGate.asType(.float32).reshaped(batch, heads, lK)
            let gIdx = stopGradient(safeIdx.reshaped(batch, 1, lQ * topK))
            let gSel = takeAlong(g3D, gIdx, axis: 2).reshaped(batch, heads, lQ, topK)
            probs = MLX.where(validMask, probs * gSel, MLXArray(Float(0)))
        }
        return matmul(probs.expandedDimensions(axis: -2), cSel).squeezed(axis: -2).asType(origType)
    }
}

private let dsaIndexerScoreSource = """
    const int B = int(shapes[0]);
    const int LQ = int(shapes[1]);
    const int LK = int(shapes[2]);
    const int HI = int(shapes[3]);
    const int DI = int(shapes[4]);
    const int CAUSAL = int(shapes[5]);
    const int Q_OFF = int(shapes[6]);
    const int lane = int(thread_index_in_simdgroup);
    const int s = int(threadgroup_position_in_grid.y);
    const int bt = int(threadgroup_position_in_grid.z);
    if (bt >= B * LQ || s >= LK) { return; }
    const int b_idx = bt / LQ;
    const int t_idx = bt % LQ;
    const size_t out_off = size_t(bt) * size_t(LK) + size_t(s);
    if (CAUSAL != 0 && s > Q_OFF + t_idx) {
        if (lane == 0) { scores[out_off] = -INFINITY; }
        return;
    }
    auto k_ptr = k_in + (size_t(b_idx) * size_t(LK) + size_t(s)) * size_t(DI);
    auto q_base = q_in + size_t(bt) * size_t(HI) * size_t(DI);
    auto w_ptr = w_in + size_t(bt) * size_t(HI);
    float acc = 0.0f;
    for (int j = 0; j < HI; ++j) {
        auto q_ptr = q_base + size_t(j) * size_t(DI);
        float part = 0.0f;
        for (int d = lane; d < DI; d += 32) {
            part = fma(float(q_ptr[d]), float(k_ptr[d]), part);
        }
        const float dot = simd_sum(part);
        const float relu_dot = isnan(dot) ? dot : max(dot, 0.0f);
        acc = fma(w_ptr[j], relu_dot, acc);
    }
    if (lane == 0) {
        scores[out_off] = acc;
    }
    """

private let dsaSparseLatentSource = """
    const int B = int(shapes[0]);
    const int H = int(shapes[1]);
    const int LQ = int(shapes[2]);
    const int LK = int(shapes[3]);
    const int DC = int(shapes[4]);
    const int DR = int(shapes[5]);
    const int TOPK = int(shapes[6]);
    const int CAUSAL = int(shapes[7]);
    const int Q_OFF = int(shapes[8]);
    const int lane = int(thread_index_in_simdgroup);
    const int qi = int(threadgroup_position_in_grid.y);
    const int bh = int(threadgroup_position_in_grid.z);
    if (bh >= B * H || qi >= LQ) { return; }
    const int b_idx = bh / H;
    const int max_tok = (CAUSAL != 0) ? min(LK - 1, Q_OFF + qi) : (LK - 1);
    const float attn_scale = params[0];

    float q_reg[DC_BLOCKS];
    float o_acc[DC_BLOCKS];
    auto q_ptr = q_abs + (size_t(bh) * size_t(LQ) + size_t(qi)) * size_t(DC);
    for (int blk = 0; blk < DC_BLOCKS; ++blk) {
        const int col = blk * 32 + lane;
        q_reg[blk] = (col < DC) ? float(q_ptr[col]) : 0.0f;
        o_acc[blk] = 0.0f;
    }

    constexpr int DR_ARR = (DR_BLOCKS > 0) ? DR_BLOCKS : 1;
    float qpe_reg[DR_ARR];
    if (HAS_PE) {
        auto qpe_ptr = q_pe + (size_t(bh) * size_t(LQ) + size_t(qi)) * size_t(DR);
        for (int blk = 0; blk < DR_BLOCKS; ++blk) {
            const int col = blk * 32 + lane;
            qpe_reg[blk] = (col < DR) ? float(qpe_ptr[col]) : 0.0f;
        }
    }

    auto idx_ptr = top_idx + (size_t(b_idx) * size_t(LQ) + size_t(qi)) * size_t(TOPK);
    auto c_base = c_kv + size_t(b_idx) * size_t(LK) * size_t(DC);
    auto kpe_base = k_pe + size_t(b_idx) * size_t(LK) * size_t(DR);
    auto vg_ptr = v_gate + size_t(bh) * size_t(LK);

    float m_run = -INFINITY;
    float l_run = 0.0f;
    for (int j = 0; j < TOPK; ++j) {
        const int tok = int(idx_ptr[j]);
        if (tok < 0 || tok > max_tok) { continue; }
        auto c_ptr = c_base + size_t(tok) * size_t(DC);
        float dot_part = 0.0f;
        for (int blk = 0; blk < DC_BLOCKS; ++blk) {
            const int col = blk * 32 + lane;
            if (col < DC) {
                dot_part = fma(q_reg[blk], float(c_ptr[col]), dot_part);
            }
        }
        if (HAS_PE) {
            auto kpe_ptr = kpe_base + size_t(tok) * size_t(DR);
            for (int blk = 0; blk < DR_BLOCKS; ++blk) {
                const int col = blk * 32 + lane;
                if (col < DR) {
                    dot_part = fma(qpe_reg[blk], float(kpe_ptr[col]), dot_part);
                }
            }
        }
        const float s = simd_sum(dot_part) * attn_scale;
        const bool bad = isnan(m_run) || isnan(s);
        const float m_new = bad ? NAN : max(m_run, s);
        const bool active = bad || (m_new != -INFINITY);
        const float alpha = bad ? NAN : ((m_run != -INFINITY && active) ? exp(m_run - m_new) : 0.0f);
        const float beta = bad ? NAN : ((active && s != -INFINITY) ? exp(s - m_new) : 0.0f);
        l_run = l_run * alpha + beta;
        m_run = m_new;
        const float v_scale = HAS_VGATE ? vg_ptr[tok] : 1.0f;
        const float vb = beta * v_scale;
        for (int blk = 0; blk < DC_BLOCKS; ++blk) {
            const int col = blk * 32 + lane;
            if (col < DC) {
                o_acc[blk] = fma(vb, float(c_ptr[col]), o_acc[blk] * alpha);
            }
        }
    }

    const float inv_l = (isnan(l_run) || isnan(m_run)) ? NAN : ((l_run > 0.0f) ? (1.0f / l_run) : 0.0f);
    device T* out_ptr = out + (size_t(bh) * size_t(LQ) + size_t(qi)) * size_t(DC);
    for (int blk = 0; blk < DC_BLOCKS; ++blk) {
        const int col = blk * 32 + lane;
        if (col < DC) {
            out_ptr[col] = static_cast<T>(o_acc[blk] * inv_l);
        }
    }
    """

private let dsaIndexerScoreKernel = MLXFast.metalKernel(
    name: "dsa_lightning_indexer_score_metal",
    inputNames: ["q_in", "k_in", "w_in", "shapes"],
    outputNames: ["scores"],
    source: dsaIndexerScoreSource,
    header: BasaltMetalSubstrate.combinedHeader
)

private let dsaSparseLatentKernel = MLXFast.metalKernel(
    name: "dsa_sparse_latent_attn_metal",
    inputNames: ["q_abs", "q_pe", "c_kv", "k_pe", "top_idx", "v_gate", "params", "shapes"],
    outputNames: ["out"],
    source: dsaSparseLatentSource,
    header: BasaltMetalSubstrate.combinedHeader
)
