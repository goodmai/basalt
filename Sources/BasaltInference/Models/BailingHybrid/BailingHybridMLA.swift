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
        self.kvHeads = config.numAttentionHeads
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

        let rawQ = projectQuery(x).reshaped(batch, length, numHeads, qkHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(rawQ, indices: [qkNopeHeadDim], axis: -1)
        let (qNope, qPeRaw) = (splitQ[0], splitQ[1])

        let compressed = kvAProjWithMqa(x)
        let splitCompressed = split(compressed, indices: [kvLoraRank], axis: -1)
        let kvLatent = kvALayerNorm(splitCompressed[0])
        let kPeRaw = splitCompressed[1].reshaped(batch, length, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        let kv = kvBProj(kvLatent)
            .reshaped(batch, length, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let splitKV = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let (kNope, values) = (splitKV[0], splitKV[1])

        let offset = cache?.ropeOffset
        let qPe = applyRotaryPosition(rope, to: qPeRaw, offset: offset)
        let kPeRotated = applyRotaryPosition(rope, to: kPeRaw, offset: offset)
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

    private func projectQuery(_ x: MLXArray) -> MLXArray {
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
