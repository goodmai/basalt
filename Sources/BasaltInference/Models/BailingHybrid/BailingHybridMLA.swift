import Foundation
import MLX
import MLXLMCommon
import MLXNN

package final class BailingHybridMLAAttention: Module {
    let layerIdx: Int
    let numHeads: Int
    let kvHeads: Int
    let headDim: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let qkHeadDim: Int
    let vHeadDim: Int
    let kvLoraRank: Int
    let gateGranularity: String?
    let scale: Float
    let rope: (any RoPELayer)?

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm?
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "indexer") var indexer: DSALightningIndexer?

    package init(_ config: BailingHybridConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.numHeads = config.numAttentionHeads
        self.kvHeads = 1
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.qkHeadDim = config.qkHeadDim
        self.headDim = config.vHeadDim
        self.vHeadDim = config.vHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.gateGranularity = config.gatedAttentionGranularity
        self.scale = Self.attentionScale(qkHeadDim: config.qkHeadDim, ropeScaling: config.ropeScaling)

        let hidden = config.hiddenSize
        let bias = config.useQKVBias
        if let qLoraRank = config.qLoraRank {
            self._qAProj.wrappedValue = Linear(hidden, qLoraRank, bias: bias)
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: config.rmsNormEps)
            self._qBProj.wrappedValue = Linear(qLoraRank, config.numAttentionHeads * config.qkHeadDim, bias: false)
        } else {
            self._qProj.wrappedValue = Linear(hidden, config.numAttentionHeads * config.qkHeadDim, bias: false)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(hidden, config.kvLoraRank + config.qkRopeHeadDim, bias: bias)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank, eps: config.rmsNormEps)
        self._kvBProj.wrappedValue = Linear(
            config.kvLoraRank,
            config.numAttentionHeads * (config.qkNopeHeadDim + config.vHeadDim),
            bias: false
        )

        if config.useOutputNorm {
            self._oNorm.wrappedValue = RMSNorm(dimensions: config.vHeadDim, eps: config.rmsNormEps)
        }
        if config.gatedAttentionGranularity == "head_wise" {
            self._gProj.wrappedValue = Linear(hidden, config.numAttentionHeads, bias: false)
        } else if config.gatedAttentionGranularity == "element_wise" {
            self._gProj.wrappedValue = Linear(hidden, config.numAttentionHeads * config.vHeadDim, bias: false)
        }
        if let indexerCfg = config.dsaIndexerConfig {
            self._indexer.wrappedValue = DSALightningIndexer(
                hiddenSize: hidden, qInputDim: config.qLoraRank, config: indexerCfg)
        }

        self._dense.wrappedValue = Linear(config.numAttentionHeads * config.vHeadDim, hidden, bias: bias)
        self.rope = Self.makeRoPE(config)
    }

    package func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        probe: (any ForwardProbe)? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let prevOffset = cache?.offset ?? 0
        let ropeOffset = cache?.ropeOffset
        let (qNope, qPe, qLatent) = projectAndRotateQuery(x, batch: batch, length: length, offset: ropeOffset)
        let (kvLatent, kPe) = projectAndRotateLatentKV(x, batch: batch, length: length, offset: ropeOffset)
        let topIndices = indexer?.selectTopK(
            x: x, qInput: qLatent, cache: cache as? DSALatentKVCache,
            causal: Self.isCausal(mask), qOffset: prevOffset
        ).indices
        let (cachedLatent, cachedKPe) = updateLatentCache(cache, kvLatent: kvLatent, kPe: kPe)
        let ctx = LatentContext(
            qNope: qNope, qPe: qPe, cachedLatent: cachedLatent,
            cachedKPe: cachedKPe, topIndices: topIndices, qOffset: prevOffset)
        let attended = attendFromLatent(ctx, mask: mask)
        return finalizeOutput(attended, input: x, batch: batch, length: length, probe: probe)
    }

    private static func makeRoPE(_ config: BailingHybridConfiguration) -> (any RoPELayer)? {
        guard config.qkRopeHeadDim > 0 else { return nil }
        return initializeRope(
            dims: config.qkRopeHeadDim,
            base: config.ropeTheta,
            traditional: config.ropeInterleave,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
    }

    private static func attentionScale(qkHeadDim: Int, ropeScaling: [String: StringOrNumber]?) -> Float {
        var scale = pow(Float(qkHeadDim), -0.5)
        guard let ropeScaling,
              let mscaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat(), mscaleAllDim != 0,
              let factor = ropeScaling["factor"]?.asFloat(), factor > 1
        else {
            return scale
        }
        let mscale = 0.1 * mscaleAllDim * log(factor) + 1.0
        scale *= mscale * mscale
        return scale
    }
}

extension BailingHybridMLAAttention {
    struct LatentContext {
        let qNope: MLXArray?
        let qPe: MLXArray?
        let cachedLatent: MLXArray
        let cachedKPe: MLXArray?
        let topIndices: MLXArray?
        let qOffset: Int
    }

    package func forwardNonAbsorbed(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        probe: (any ForwardProbe)? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let prevOffset = cache?.offset ?? 0
        let ropeOffset = cache?.ropeOffset
        let (qNope, qPe, qLatent) = projectAndRotateQuery(x, batch: batch, length: length, offset: ropeOffset)
        let (kvLatent, kPeRotated) = projectAndRotateLatentKV(x, batch: batch, length: length, offset: ropeOffset)
        if let indexer {
            let topIndices = indexer.selectTopK(
                x: x, qInput: qLatent, cache: cache as? DSALatentKVCache,
                causal: Self.isCausal(mask), qOffset: prevOffset
            ).indices
            let (cachedLatent, cachedKPe) = updateLatentCache(cache, kvLatent: kvLatent, kPe: kPeRotated)
            let ctx = LatentContext(
                qNope: qNope, qPe: qPe, cachedLatent: cachedLatent,
                cachedKPe: cachedKPe, topIndices: topIndices, qOffset: prevOffset)
            let attended = attendSparseExpanded(ctx, mask: mask)
            return finalizeOutput(attended, input: x, batch: batch, length: length, probe: probe)
        }
        let kv = kvBProj(kvLatent.squeezed(axis: 1))
            .reshaped(batch, length, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let (kNope, values) = splitExpandedKV(kv)
        let (queries, keys) = combineQK(
            qNope: qNope, qPe: qPe, kNope: kNope, kPe: kPeRotated, batch: batch, length: length)
        let attended = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask)
        return finalizeOutput(attended, input: x, batch: batch, length: length, probe: probe)
    }

    func projectAndRotateQuery(
        _ x: MLXArray,
        batch: Int,
        length: Int,
        offset: RoPEOffset?
    ) -> (qNope: MLXArray?, qPe: MLXArray?, qLatent: MLXArray?) {
        let rawQ: MLXArray
        let qLatent: MLXArray?
        if let qAProj, let qALayerNorm, let qBProj {
            let normed = qALayerNorm(qAProj(x))
            qLatent = normed
            rawQ = qBProj(normed)
        } else if let qProj {
            qLatent = nil
            rawQ = qProj(x)
        } else {
            preconditionFailure("BailingHybridMLAAttention query projection not initialized")
        }
        let shaped = rawQ.reshaped(batch, length, numHeads, qkHeadDim).transposed(0, 2, 1, 3)
        if qkRopeHeadDim == 0 {
            return (shaped, nil, qLatent)
        }
        guard let rope else { preconditionFailure("BailingHybridMLAAttention RoPE missing for positive qkRopeHeadDim") }
        if qkNopeHeadDim == 0 {
            return (nil, applyRotaryPosition(rope, to: shaped, offset: offset), qLatent)
        }
        let parts = split(shaped, indices: [qkNopeHeadDim], axis: -1)
        return (parts[0], applyRotaryPosition(rope, to: parts[1], offset: offset), qLatent)
    }

    func projectAndRotateLatentKV(
        _ x: MLXArray,
        batch: Int,
        length: Int,
        offset: RoPEOffset?
    ) -> (kvLatent: MLXArray, kPe: MLXArray?) {
        let compressed = kvAProjWithMqa(x)
        if qkRopeHeadDim == 0 {
            let kvLatent = kvALayerNorm(compressed)
                .reshaped(batch, length, 1, kvLoraRank)
                .transposed(0, 2, 1, 3)
            return (kvLatent, nil)
        }
        guard let rope else { preconditionFailure("BailingHybridMLAAttention RoPE missing for positive qkRopeHeadDim") }
        let splitCompressed = split(compressed, indices: [kvLoraRank], axis: -1)
        let kvLatent = kvALayerNorm(splitCompressed[0])
            .reshaped(batch, length, 1, kvLoraRank)
            .transposed(0, 2, 1, 3)
        let kPeRaw = splitCompressed[1]
            .reshaped(batch, length, 1, qkRopeHeadDim)
            .transposed(0, 2, 1, 3)
        return (kvLatent, applyRotaryPosition(rope, to: kPeRaw, offset: offset))
    }

    func updateLatentCache(
        _ cache: KVCache?,
        kvLatent: MLXArray,
        kPe: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        if let dsaCache = cache as? DSALatentKVCache {
            return dsaCache.updateLatent(latent: kvLatent, kPe: kPe)
        }
        guard let cache else { return (kvLatent, kPe) }
        if let kPe {
            let (lat, pe) = cache.update(keys: kvLatent, values: kPe)
            return (lat, pe)
        }
        let empty = MLXArray.zeros([kvLatent.dim(0), 1, kvLatent.dim(2), 0], dtype: kvLatent.dtype)
        let (lat, _) = cache.update(keys: kvLatent, values: empty)
        return (lat, nil)
    }
}

extension BailingHybridMLAAttention {
    func attendFromLatent(
        _ ctx: LatentContext,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        if ctx.topIndices != nil {
            if let sparse = attendSparseAbsorbed(ctx, mask: mask) {
                return sparse
            }
            return attendSparseExpanded(ctx, mask: mask)
        }
        if let absorbed = attendAbsorbed(ctx, mask: mask) {
            return absorbed
        }
        return attendExpandedFromLatent(ctx, mask: mask)
    }

    private func attendAbsorbed(
        _ ctx: LatentContext,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray? {
        guard BailingHybridMLAAbsorption.canAbsorb(kvBProj) else { return nil }
        let queries: MLXArray
        let keys: MLXArray
        if let qNope = ctx.qNope, qkNopeHeadDim > 0 {
            guard let qAbs = BailingHybridMLAAbsorption.embedQuery(
                qNope, kvBProj: kvBProj, heads: numHeads, nopeDim: qkNopeHeadDim, vDim: vHeadDim
            ) else { return nil }
            if let qPe = ctx.qPe, let cachedKPe = ctx.cachedKPe {
                queries = concatenated([qAbs, qPe], axis: -1)
                keys = concatenated([ctx.cachedLatent, cachedKPe], axis: -1)
            } else {
                queries = qAbs
                keys = ctx.cachedLatent
            }
        } else if let qPe = ctx.qPe, let cachedKPe = ctx.cachedKPe {
            queries = qPe
            keys = cachedKPe
        } else {
            return nil
        }
        let latentOut = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: ctx.cachedLatent, scale: scale, mask: mask)
        return BailingHybridMLAAbsorption.unembedOutput(
            latentOut, kvBProj: kvBProj, heads: numHeads, nopeDim: qkNopeHeadDim, vDim: vHeadDim)
    }

    private func attendSparseAbsorbed(
        _ ctx: LatentContext,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray? {
        guard let topIndices = ctx.topIndices, BailingHybridMLAAbsorption.canAbsorb(kvBProj) else { return nil }
        let qAbs: MLXArray
        if let qNope = ctx.qNope, qkNopeHeadDim > 0 {
            guard let embedded = BailingHybridMLAAbsorption.embedQuery(
                qNope, kvBProj: kvBProj, heads: numHeads, nopeDim: qkNopeHeadDim, vDim: vHeadDim
            ) else { return nil }
            qAbs = embedded
        } else if let qPe = ctx.qPe {
            qAbs = MLXArray.zeros([qPe.dim(0), qPe.dim(1), qPe.dim(2), kvLoraRank], dtype: qPe.dtype)
        } else {
            return nil
        }
        let useMetal = (indexer?.config.useMetal ?? true) && !training
        let latentOut = DSALightningIndexerMetal.sparseLatentAttention(
            DSALightningIndexerMetal.SparseLatentInputs(
                qAbsorbed: qAbs, qPe: ctx.qPe, cKV: ctx.cachedLatent,
                kPe: ctx.cachedKPe, topIndices: topIndices),
            options: DSALightningIndexerMetal.SparseLatentOptions(
                scale: scale, causal: Self.isCausal(mask), qOffset: ctx.qOffset, useMetal: useMetal))
        return BailingHybridMLAAbsorption.unembedOutput(
            latentOut, kvBProj: kvBProj, heads: numHeads, nopeDim: qkNopeHeadDim, vDim: vHeadDim)
    }

    private func attendExpandedFromLatent(
        _ ctx: LatentContext,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let batch = ctx.cachedLatent.dim(0)
        let total = ctx.cachedLatent.dim(2)
        let kv = kvBProj(ctx.cachedLatent.squeezed(axis: 1))
            .reshaped(batch, total, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let (kNope, values) = splitExpandedKV(kv)
        let (queries, keys) = combineQK(
            qNope: ctx.qNope, qPe: ctx.qPe, kNope: kNope, kPe: ctx.cachedKPe, batch: batch, length: total)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }

    private func attendSparseExpanded(
        _ ctx: LatentContext,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        guard let topIndices = ctx.topIndices else { return attendExpandedFromLatent(ctx, mask: mask) }
        let batch = ctx.cachedLatent.dim(0)
        let total = ctx.cachedLatent.dim(2)
        let kv = kvBProj(ctx.cachedLatent.squeezed(axis: 1))
            .reshaped(batch, total, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let (kNope, values) = splitExpandedKV(kv)
        let (queries, keys) = combineQK(
            qNope: ctx.qNope, qPe: ctx.qPe, kNope: kNope, kPe: ctx.cachedKPe, batch: batch, length: total)
        let lQ = queries.dim(2)
        let topK = topIndices.dim(-1)
        let idx = topIndices.asType(.int32)
        var valid = (idx .>= Int32(0)) .&& (idx .< Int32(total))
        if Self.isCausal(mask) {
            let qPos = MLXArray(Int32(ctx.qOffset) ..< Int32(ctx.qOffset + lQ)).reshaped(1, lQ, 1)
            valid = valid .&& (idx .<= qPos)
        }
        let safeIdx = MLX.where(valid, idx, MLXArray(Int32(0)))
        let flatIdx = safeIdx.reshaped(batch, 1, lQ * topK, 1)
        let kSel = takeAlong(keys, flatIdx, axis: 2).reshaped(batch, numHeads, lQ, topK, qkHeadDim)
        let vSel = takeAlong(values, flatIdx, axis: 2).reshaped(batch, numHeads, lQ, topK, vHeadDim)
        let logits = matmul(queries.asType(.float32).expandedDimensions(axis: -2), kSel.asType(.float32).swappedAxes(-1, -2))
            .squeezed(axis: -2) * scale
        let validMask = valid.expandedDimensions(axis: 1)
        let anyValid = validMask.any(axis: -1).expandedDimensions(axis: -1)
        let masked = MLX.where(validMask, logits, MLXArray(-Float.infinity))
        let probs = MLX.where(anyValid, softmax(MLX.where(anyValid, masked, MLXArray(Float(0))), axis: -1), MLXArray(Float(0)))
        return matmul(probs.expandedDimensions(axis: -2), vSel.asType(.float32)).squeezed(axis: -2).asType(queries.dtype)
    }
}

extension BailingHybridMLAAttention {
    private func splitExpandedKV(_ kv: MLXArray) -> (kNope: MLXArray?, values: MLXArray) {
        guard qkNopeHeadDim > 0 else { return (nil, kv) }
        let parts = split(kv, indices: [qkNopeHeadDim], axis: -1)
        return (parts[0], parts[1])
    }

    private func combineQK(
        qNope: MLXArray?,
        qPe: MLXArray?,
        kNope: MLXArray?,
        kPe: MLXArray?,
        batch: Int,
        length: Int
    ) -> (queries: MLXArray, keys: MLXArray) {
        if let qPe, let kPe {
            let kPeExp = broadcast(kPe, to: [batch, numHeads, length, qkRopeHeadDim])
            if let qNope, let kNope {
                return (concatenated([qNope, qPe], axis: -1), concatenated([kNope, kPeExp], axis: -1))
            }
            return (qPe, kPeExp)
        }
        guard let qNope, let kNope else {
            preconditionFailure("BailingHybridMLAAttention requires positive qkNopeHeadDim or qkRopeHeadDim")
        }
        return (qNope, kNope)
    }

    private static func isCausal(_ mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
        if case .causal = mask { return true }
        return false
    }

    func finalizeOutput(
        _ attended: MLXArray,
        input x: MLXArray,
        batch: Int,
        length: Int,
        probe: (any ForwardProbe)?
    ) -> MLXArray {
        var merged = attended.transposed(0, 2, 1, 3)
        if let oNorm {
            merged = oNorm(merged)
        }
        if let gProj {
            let gate = sigmoid(gProj(x).asType(.float32)).asType(x.dtype)
            if gateGranularity == "head_wise" {
                merged = merged * gate.expandedDimensions(axis: -1)
            } else {
                merged = merged * gate.reshaped(batch, length, numHeads, vHeadDim)
            }
        }
        if let probe {
            var perHead = merged.transposed(0, 2, 1, 3)
            if let headMask = probe.headMask(layer: layerIdx) {
                perHead = perHead * headMask.reshaped(1, numHeads, 1, 1).asType(perHead.dtype)
            }
            probe.headOutputs(layer: layerIdx, perHead)
            merged = perHead.transposed(0, 2, 1, 3)
        }
        return dense(merged.reshaped(batch, length, -1))
    }
}
