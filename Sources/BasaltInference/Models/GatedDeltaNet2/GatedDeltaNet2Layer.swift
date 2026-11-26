import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package final class GatedDeltaNet2Attention: Module {
    let layerIdx: Int
    let numHeads: Int
    let kvHeads: Int
    let headDim: Int
    let headVDim: Int
    let allowNegEigval: Bool
    let betaGuard: Bool
    let useShortConv: Bool
    let chunkSize: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_conv1d") var qConv1d: BailingHybridShortConv1d?
    @ModuleInfo(key: "k_conv1d") var kConv1d: BailingHybridShortConv1d?
    @ModuleInfo(key: "v_conv1d") var vConv1d: BailingHybridShortConv1d?
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "w_proj") var wProj: Linear
    @ModuleInfo(key: "a_proj") var aProj: Linear
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    package init(_ config: GatedDeltaNet2Configuration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.numHeads = config.numVHeads
        self.kvHeads = config.numHeads
        self.headDim = config.headDim
        self.headVDim = config.headVDim
        self.allowNegEigval = config.allowNegEigval
        self.betaGuard = config.betaGuard
        self.useShortConv = config.useShortConv
        self.chunkSize = config.chunkSize
        self.scale = 1.0 / Float(config.headDim).squareRoot()

        let hidden = config.hiddenSize
        let qkDim = config.qkProjDim
        let vDim = config.vProjDim
        let decayDim = config.numVHeads * config.headDim

        self._qProj.wrappedValue = Linear(hidden, qkDim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, qkDim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, vDim, bias: false)
        if config.useShortConv {
            self._qConv1d.wrappedValue = BailingHybridShortConv1d(channels: qkDim, kernelSize: config.convKernelSize)
            self._kConv1d.wrappedValue = BailingHybridShortConv1d(channels: qkDim, kernelSize: config.convKernelSize)
            self._vConv1d.wrappedValue = BailingHybridShortConv1d(channels: vDim, kernelSize: config.convKernelSize)
        }
        self._bProj.wrappedValue = Linear(hidden, decayDim, bias: false)
        self._wProj.wrappedValue = Linear(hidden, vDim, bias: false)
        self._aProj.wrappedValue = Linear(hidden, decayDim, bias: false)
        self._aLog.wrappedValue = MLXArray.zeros([config.numVHeads], dtype: .float32)
        self._dtBias.wrappedValue = MLXArray.zeros([config.numVHeads, config.headDim], dtype: .float32)
        self._gProj.wrappedValue = Linear(hidden, vDim, bias: false)
        self._oNorm.wrappedValue = RMSNorm(dimensions: config.headVDim, eps: config.rmsNormEps)
        self._oProj.wrappedValue = Linear(vDim, hidden, bias: false)
    }

    package func callAsFunction(
        _ x: MLXArray,
        cache: ArraysCache?,
        useChunked: Bool = true,
        useMetal: Bool = true
    ) -> MLXArray {
        let (batch, steps, dtype) = (x.dim(0), x.dim(1), x.dtype)
        guard steps > 0 else {
            return MLXArray.zeros([batch, 0, x.dim(2)], dtype: dtype)
        }
        let ssmMask = cache?.makeMask(N: steps)
        let (qRaw, kRaw, vRaw) = projectQKV(x, cache: cache, mask: ssmMask)

        let q = BailingHybridKDAOps.l2Normalize(qRaw.reshaped(batch, steps, kvHeads, headDim)) * scale
        let k = BailingHybridKDAOps.l2Normalize(kRaw.reshaped(batch, steps, kvHeads, headDim))
        let v = vRaw.reshaped(batch, steps, numHeads, headVDim).asType(.float32)

        let aIn = aProj(x).reshaped(batch, steps, numHeads, headDim).asType(.float32)
        let gLog = -MLX.exp(aLog.asType(.float32)).reshaped(1, 1, numHeads, 1)
            * softplus(aIn + dtBias.asType(.float32))
        let eraseScale: Float = allowNegEigval ? 2.0 : 1.0
        let erase = sigmoid(bProj(x).reshaped(batch, steps, numHeads, headDim).asType(.float32)) * eraseScale
        let write = sigmoid(wProj(x).reshaped(batch, steps, numHeads, headVDim).asType(.float32))

        let inputs = GatedDeltaNet2Inputs(
            q: q, k: k, v: v, gLog: gLog, erase: erase, write: write
        ).withBetaGuard(betaGuard)
        let (recurrenceOut, nextState) = GatedDeltaNet2Ops.forward(
            inputs,
            state: cache?[3],
            mask: ssmMask,
            chunkSize: useChunked ? chunkSize : 1,
            useMetal: useMetal && !training
        )
        if let cache {
            cache[3] = contiguous(nextState.asType(.float32))
            cache.offset += steps
            cache.advance(steps)
        }

        let gate = silu(gProj(x).reshaped(batch, steps, numHeads, headVDim).asType(.float32)).asType(dtype)
        let gated = oNorm(recurrenceOut.asType(dtype)) * gate
        return oProj(gated.reshaped(batch, steps, -1))
    }

    private func projectQKV(
        _ x: MLXArray,
        cache: ArraysCache?,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        var qIn = qProj(x)
        var kIn = kProj(x)
        var vIn = vProj(x)
        if let mask {
            let expanded = mask.expandedDimensions(axis: -1)
            qIn = MLX.where(expanded, qIn, MLXArray.zeros(like: qIn))
            kIn = MLX.where(expanded, kIn, MLXArray.zeros(like: kIn))
            vIn = MLX.where(expanded, vIn, MLXArray.zeros(like: vIn))
        }
        if useShortConv, let qConv1d, let kConv1d, let vConv1d {
            let (qOut, qState) = qConv1d(qIn, state: cache?[0])
            let (kOut, kState) = kConv1d(kIn, state: cache?[1])
            let (vOut, vState) = vConv1d(vIn, state: cache?[2])
            if let cache {
                cache[0] = contiguous(qState)
                cache[1] = contiguous(kState)
                cache[2] = contiguous(vState)
            }
            return (qOut, kOut, vOut)
        }
        if let cache {
            let batch = x.dim(0)
            cache[0] = MLXArray.zeros([batch, 0, kvHeads * headDim], dtype: x.dtype)
            cache[1] = MLXArray.zeros([batch, 0, kvHeads * headDim], dtype: x.dtype)
            cache[2] = MLXArray.zeros([batch, 0, numHeads * headVDim], dtype: x.dtype)
        }
        return (silu(qIn), silu(kIn), silu(vIn))
    }
}

package final class GDN2SelfAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    package init(_ config: GatedDeltaNet2Configuration) {
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.headDim = config.attentionHeadDim
        self.scale = 1.0 / Float(config.attentionHeadDim).squareRoot()
        self.rope = RoPE(dimensions: config.attentionHeadDim, traditional: false, base: config.ropeTheta)

        let hidden = config.hiddenSize
        self._qProj.wrappedValue = Linear(hidden, heads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(heads * headDim, hidden, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    package func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var queries = qNorm(qProj(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        return oProj(attended.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

package final class GDN2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    package init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

package final class GDN2DecoderLayer: Module {
    let isAttention: Bool
    let isSliding: Bool

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn") var attn: Module
    @ModuleInfo(key: "mlp") var mlp: GDN2MLP

    package init(_ config: GatedDeltaNet2Configuration, layerIdx: Int) {
        self.isAttention = config.isAttentionLayer(layerIdx)
        self.isSliding = config.isSlidingAttentionLayer(layerIdx)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if isAttention {
            self._attn.wrappedValue = GDN2SelfAttention(config)
        } else {
            self._attn.wrappedValue = GatedDeltaNet2Attention(config, layerIdx: layerIdx)
        }
        self._mlp.wrappedValue = GDN2MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    package func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        useChunked: Bool = true,
        useMetal: Bool = true
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let mixed: MLXArray
        if let full = attn as? GDN2SelfAttention {
            mixed = full(normed, mask: mask, cache: cache)
        } else if let linear = attn as? GatedDeltaNet2Attention {
            mixed = linear(normed, cache: cache as? ArraysCache, useChunked: useChunked, useMetal: useMetal)
        } else {
            preconditionFailure("Unexpected attention module in GDN2DecoderLayer")
        }
        let hidden = x + mixed
        return hidden + mlp(postAttentionLayerNorm(hidden))
    }
}
