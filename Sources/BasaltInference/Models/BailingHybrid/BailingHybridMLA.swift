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
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "dense") var dense: Linear

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

        if config.gatedAttentionGranularity == "head_wise" {
            self._gProj.wrappedValue = Linear(hidden, config.numAttentionHeads, bias: false)
        } else if config.gatedAttentionGranularity == "element_wise" {
            self._gProj.wrappedValue = Linear(hidden, config.numAttentionHeads * config.vHeadDim, bias: false)
        }

        self._dense.wrappedValue = Linear(config.numAttentionHeads * config.vHeadDim, hidden, bias: bias)
        self.rope = initializeRope(
            dims: config.qkRopeHeadDim,
            base: config.ropeTheta,
            traditional: config.ropeInterleave,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
    }

    package func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        probe: (any ForwardProbe)? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let (qNope, qPe) = projectAndRotateQuery(x, batch: batch, length: length, offset: cache?.ropeOffset)
        let (kvLatent, kPe) = projectAndRotateLatentKV(x, batch: batch, length: length, offset: cache?.ropeOffset)
        let (cachedLatent, cachedKPe): (MLXArray, MLXArray)
        if let cache {
            (cachedLatent, cachedKPe) = cache.update(keys: kvLatent, values: kPe)
        } else {
            (cachedLatent, cachedKPe) = (kvLatent, kPe)
        }
        let attended = attendFromLatent(
            qNope: qNope,
            qPe: qPe,
            cachedLatent: cachedLatent,
            cachedKPe: cachedKPe,
            mask: mask
        )
        return finalizeOutput(attended, input: x, batch: batch, length: length, probe: probe)
    }

    func projectQuery(_ x: MLXArray) -> MLXArray {
        if let qAProj, let qALayerNorm, let qBProj {
            return qBProj(qALayerNorm(qAProj(x)))
        }
        guard let qProj else {
            preconditionFailure("BailingHybridMLAAttention query projection not initialized")
        }
        return qProj(x)
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
    package func forwardNonAbsorbed(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        probe: (any ForwardProbe)? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let (qNope, qPe) = projectAndRotateQuery(x, batch: batch, length: length, offset: cache?.ropeOffset)
        let (kvLatent, kPeRotated) = projectAndRotateLatentKV(x, batch: batch, length: length, offset: cache?.ropeOffset)
        let kv = kvBProj(kvLatent.squeezed(axis: 1))
            .reshaped(batch, length, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let splitKV = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let (kNope, values) = (splitKV[0], splitKV[1])
        let kPe = broadcast(kPeRotated, to: [batch, numHeads, length, qkRopeHeadDim])
        let queries = concatenated([qNope, qPe], axis: -1)
        let keys = concatenated([kNope, kPe], axis: -1)
        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        return finalizeOutput(attended, input: x, batch: batch, length: length, probe: probe)
    }

    func projectAndRotateQuery(
        _ x: MLXArray,
        batch: Int,
        length: Int,
        offset: RoPEOffset?
    ) -> (MLXArray, MLXArray) {
        let rawQ = projectQuery(x).reshaped(batch, length, numHeads, qkHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(rawQ, indices: [qkNopeHeadDim], axis: -1)
        let qPe = applyRotaryPosition(rope, to: splitQ[1], offset: offset)
        return (splitQ[0], qPe)
    }

    func projectAndRotateLatentKV(
        _ x: MLXArray,
        batch: Int,
        length: Int,
        offset: RoPEOffset?
    ) -> (MLXArray, MLXArray) {
        let compressed = kvAProjWithMqa(x)
        let splitCompressed = split(compressed, indices: [kvLoraRank], axis: -1)
        let kvLatent = kvALayerNorm(splitCompressed[0])
            .reshaped(batch, length, 1, kvLoraRank)
            .transposed(0, 2, 1, 3)
        let kPeRaw = splitCompressed[1]
            .reshaped(batch, length, 1, qkRopeHeadDim)
            .transposed(0, 2, 1, 3)
        let kPe = applyRotaryPosition(rope, to: kPeRaw, offset: offset)
        return (kvLatent, kPe)
    }
}

extension BailingHybridMLAAttention {
    func attendFromLatent(
        qNope: MLXArray,
        qPe: MLXArray,
        cachedLatent: MLXArray,
        cachedKPe: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        if let absorbed = attendAbsorbed(
            qNope: qNope,
            qPe: qPe,
            cachedLatent: cachedLatent,
            cachedKPe: cachedKPe,
            mask: mask
        ) {
            return absorbed
        }
        return attendExpandedFromLatent(
            qNope: qNope,
            qPe: qPe,
            cachedLatent: cachedLatent,
            cachedKPe: cachedKPe,
            mask: mask
        )
    }

    private func attendAbsorbed(
        qNope: MLXArray,
        qPe: MLXArray,
        cachedLatent: MLXArray,
        cachedKPe: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray? {
        guard BailingHybridMLAAbsorption.canAbsorb(kvBProj) else { return nil }
        let queries: MLXArray
        let keys: MLXArray
        if qkNopeHeadDim > 0 {
            guard let qAbsorbed = BailingHybridMLAAbsorption.embedQuery(
                qNope, kvBProj: kvBProj, heads: numHeads, nopeDim: qkNopeHeadDim, vDim: vHeadDim
            ) else { return nil }
            queries = concatenated([qAbsorbed, qPe], axis: -1)
            keys = concatenated([cachedLatent, cachedKPe], axis: -1)
        } else {
            queries = qPe
            keys = cachedKPe
        }
        let latentOut = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: cachedLatent,
            scale: scale,
            mask: mask
        )
        return BailingHybridMLAAbsorption.unembedOutput(
            latentOut, kvBProj: kvBProj, heads: numHeads, nopeDim: qkNopeHeadDim, vDim: vHeadDim
        )
    }

    private func attendExpandedFromLatent(
        qNope: MLXArray,
        qPe: MLXArray,
        cachedLatent: MLXArray,
        cachedKPe: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let batch = cachedLatent.dim(0)
        let total = cachedLatent.dim(2)
        let kv = kvBProj(cachedLatent.squeezed(axis: 1))
            .reshaped(batch, total, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let splitKV = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kPe = broadcast(cachedKPe, to: [batch, numHeads, total, qkRopeHeadDim])
        return MLXFast.scaledDotProductAttention(
            queries: concatenated([qNope, qPe], axis: -1),
            keys: concatenated([splitKV[0], kPe], axis: -1),
            values: splitKV[1],
            scale: scale,
            mask: mask
        )
    }

    func finalizeOutput(
        _ attended: MLXArray,
        input x: MLXArray,
        batch: Int,
        length: Int,
        probe: (any ForwardProbe)?
    ) -> MLXArray {
        var merged = attended.transposed(0, 2, 1, 3)
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
