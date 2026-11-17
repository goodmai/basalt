import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import BasaltCore

package final class MuseGlimmerScalelessRMSNorm: Module, UnaryLayer {
    let eps: Float

    package init(eps: Float) {
        self.eps = eps
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

package final class MuseGlimmerCenteredRMSNorm: Module, UnaryLayer {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    package init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let f = x.asType(.float32)
        let variance = mean(f * f, axis: -1, keepDims: true)
        let normalized = f * rsqrt(variance + eps)
        return (normalized * (1.0 + weight.asType(.float32))).asType(dtype)
    }
}

package final class MuseGlimmerTextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    package init(_ config: MuseGlimmerTextConfiguration) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

package final class MuseGlimmerTextAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "qk_norm") var qkNorm: MuseGlimmerScalelessRMSNorm

    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float
    let qkScaleFactor: Float
    let isSliding: Bool
    let rope: RoPE?

    package init(_ config: MuseGlimmerTextConfiguration, layerIndex: Int) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDimensions = config.headDimensions
        self.scale = pow(Float(config.headDimensions), -0.5)
        self.qkScaleFactor = config.qkScaleFactor
        let theta = config.layerRopeTheta[layerIndex]
        let useRope = theta != 0
        self.isSliding = config.layerTypes[layerIndex] == "sliding_attention"

        let dim = config.hiddenSize
        let queryDim = config.attentionHeads * config.headDimensions
        let kvDim = config.kvHeads * config.headDimensions

        self._queryProj.wrappedValue = Linear(dim, queryDim, bias: config.attentionBias)
        self._keyProj.wrappedValue = Linear(dim, kvDim, bias: config.attentionBias)
        self._valueProj.wrappedValue = Linear(dim, kvDim, bias: config.attentionBias)
        self._gateProj.wrappedValue = Linear(dim, queryDim, bias: false)
        self._outputProj.wrappedValue = Linear(queryDim, dim, bias: config.attentionBias)
        self._qkNorm.wrappedValue = MuseGlimmerScalelessRMSNorm(eps: config.rmsNormEps)
        self.rope = useRope ? RoPE(dimensions: config.headDimensions, traditional: false, base: theta) : nil
        super.init()
    }

    package func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = queryProj(x).reshaped(batch, length, heads, headDimensions)
        var keys = keyProj(x).reshaped(batch, length, kvHeads, headDimensions)
        let values = valueProj(x).reshaped(batch, length, kvHeads, headDimensions).transposed(0, 2, 1, 3)

        queries = (qkNorm(queries) * qkScaleFactor).transposed(0, 2, 1, 3)
        keys = qkNorm(keys).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        let merged = attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return outputProj(merged * sigmoid(gateProj(x)))
    }
}
