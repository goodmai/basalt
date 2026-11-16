import Foundation
import MLX
import MLXLMCommon
import MLXNN

package final class FlintMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    package init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

package final class FlintSparseMoEBlock: Module, UnaryLayer {
    let numExperts: Int
    let topK: Int
    let normTopkProb: Bool
    let scoreFunction: String

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: FlintMLP?
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    package init(_ args: FlintConfiguration) {
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerToken
        self.normTopkProb = args.normTopkProb
        self.scoreFunction = args.routerScoreFunction
        self._gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )
        if args.sharedExpertIntermediateSize > 0 {
            self._sharedExpert.wrappedValue = FlintMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.sharedExpertIntermediateSize
            )
        } else {
            self._sharedExpert.wrappedValue = nil
        }
        self._expertBias.wrappedValue = MLXArray.zeros([args.numExperts], dtype: .float32)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x)
        let routed = FlintRouterKernel.route(
            logits,
            expertBias: expertBias,
            k: topK,
            normTopkProb: normTopkProb,
            scoreFunction: scoreFunction,
            weightDType: .float32
        )
        let expertOut = switchMLP(x, routed.indices)
        let weighted = (expertOut.asType(.float32) * routed.weights.expandedDimensions(axis: -1))
            .sum(axis: -2)
            .asType(x.dtype)
        if let sharedExpert {
            return weighted + sharedExpert(x)
        }
        return weighted
    }
}

package final class FlintAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let skipRoPE: Bool
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ParameterInfo(key: "softmax_offset") var softmaxOffset: MLXArray?

    package init(_ args: FlintConfiguration, layerIdx: Int) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = 1.0 / Float(args.headDim).squareRoot()
        self.skipRoPE = args.skipsRoPE(layer: layerIdx)

        self._wq.wrappedValue = Linear(args.hiddenSize, args.attentionHeads * args.headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(args.hiddenSize, args.kvHeads * args.headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(args.hiddenSize, args.kvHeads * args.headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(args.attentionHeads * args.headDim, args.hiddenSize, bias: args.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        self._softmaxOffset.wrappedValue = args.usesLearnedAttentionSinks
            ? MLXArray.zeros([args.attentionHeads], dtype: .float32)
            : nil
        self.rope = RoPE(
            dimensions: args.headDim,
            traditional: false,
            base: args.ropeTheta,
            scale: args.linearRopeScale
        )
    }

    package func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var queries = qNorm(wq(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3)
        var keys = kNorm(wk(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)

        if !skipRoPE {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        let attended = attend(
            queries: queries,
            keys: keys,
            values: values,
            mask: mask,
            cache: cache
        )
        return wo(attended.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }

    private func attend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let sinks = softmaxOffset
        guard let cache else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask, sinks: sinks)
        }
        if let qcache = cache as? QuantizedKVCacheProtocol {
            let (qKeys, qValues) = qcache.updateQuantized(keys: keys, values: values)
            if sinks == nil {
                return quantizedScaledDotProductAttention(
                    queries: queries,
                    quantizedKeys: qKeys,
                    quantizedValues: qValues,
                    scale: scale,
                    mask: mask,
                    groupSize: qcache.groupSize,
                    bits: qcache.bits,
                    mode: qcache.mode
                )
            }
            let denseK = dequantized(
                qKeys.0, scales: qKeys.1, biases: qKeys.2,
                groupSize: qcache.groupSize, bits: qcache.bits, mode: qcache.mode
            ).asType(queries.dtype)
            let denseV = dequantized(
                qValues.0, scales: qValues.1, biases: qValues.2,
                groupSize: qcache.groupSize, bits: qcache.bits, mode: qcache.mode
            ).asType(queries.dtype)
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: denseK, values: denseV, scale: scale, mask: mask, sinks: sinks)
        }
        if sinks == nil, let attnCache = cache as? KVCacheAttentionProtocol {
            return attnCache.updateAndAttend(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        }
        let (cachedK, cachedV) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedK, values: cachedV, scale: scale, mask: mask, sinks: sinks)
    }

    package static func explicitAttentionWithSinks(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        booleanMask: MLXArray?,
        sinks: MLXArray?
    ) -> MLXArray {
        let (b, h, qLen, d) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        let kLen = keys.dim(2)
        let groups = max(1, h / max(1, keys.dim(1)))
        let expandedK = groups > 1
            ? broadcast(keys.expandedDimensions(axis: 2), to: [b, keys.dim(1), groups, kLen, d]).reshaped(b, h, kLen, d)
            : keys
        let expandedV = groups > 1
            ? broadcast(values.expandedDimensions(axis: 2), to: [b, values.dim(1), groups, kLen, d]).reshaped(b, h, kLen, d)
            : values
        var logits = matmul(queries.asType(.float32), expandedK.asType(.float32).transposed(0, 1, 3, 2)) * scale
        if let booleanMask {
            logits = MLX.where(booleanMask, logits, MLXArray(-Float.infinity))
        }
        if let sinks {
            let sinkCol = broadcast(sinks.asType(.float32).reshaped(1, h, 1, 1), to: [b, h, qLen, 1])
            let augmented = concatenated([logits, sinkCol], axis: -1)
            let probs = softmax(augmented, axis: -1, precise: true)[0..., 0..., 0..., ..<kLen]
            return matmul(probs, expandedV.asType(.float32)).asType(queries.dtype)
        }
        let probs = softmax(logits, axis: -1, precise: true)
        return matmul(probs, expandedV.asType(.float32)).asType(queries.dtype)
    }
}
