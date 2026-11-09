import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum DeepSeekOCRText {
    typealias Config = DeepSeekOCRConfiguration.Text

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let rope: RoPE

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        init(_ config: Config) {
            (heads, kvHeads, headDim) = (config.heads, config.kvHeads, config.headDim)
            scale = pow(Float(headDim), -0.5)
            rope = RoPE(dimensions: headDim, traditional: false, base: config.theta)
            _wq.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
            let (b, l) = (x.dim(0), x.dim(1))
            let queries = wq(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
            let keys = wk(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
            let offset = cache?.ropeOffset
            let output = attentionWithCacheUpdate(
                queries: applyRotaryPosition(rope, to: queries, offset: offset), keys: applyRotaryPosition(rope, to: keys, offset: offset),
                values: values, cache: cache, scale: scale, mask: mask)
            return wo(output.transposed(0, 2, 1, 3).reshaped(b, l, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hidden: Int) {
            _gate.wrappedValue = Linear(dimensions, hidden, bias: false)
            _up.wrappedValue = Linear(dimensions, hidden, bias: false)
            _down.wrappedValue = Linear(hidden, dimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
    }

    final class MoE: Module, UnaryLayer {
        let topK: Int
        let scaling: Float
        @ModuleInfo var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo(key: "shared_experts") var shared: MLP?

        init(_ config: Config) {
            let experts = config.routedExperts ?? 1
            (topK, scaling) = (config.expertsPerToken, config.routedScalingFactor ?? 1)
            _gate.wrappedValue = Linear(config.hiddenSize, experts, bias: false)
            _switchMLP.wrappedValue = SwitchGLU(inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize, numExperts: experts)
            _shared.wrappedValue = config.sharedExperts.map { MLP(dimensions: config.hiddenSize, hidden: config.moeIntermediateSize * $0) }
        }

        func route(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
            let scores = softmax(gate(x), axis: -1, precise: true)
            let indices = stopGradient(argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
            return (indices, takeAlong(scores, indices, axis: -1) * scaling)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (indices, scores) = route(x)
            let routed = weightedExpertSum(switchMLP(x, indices), scores)
            return shared.map { routed + $0(x) } ?? routed
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: any UnaryLayer
        @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

        init(_ config: Config, layer: Int) {
            _attention.wrappedValue = Attention(config)
            mlp = config.usesExperts(layer) ? MoE(config) : MLP(dimensions: config.hiddenSize, hidden: config.intermediateSize)
            _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.eps)
            _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.eps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
            let h = x + attention(inputNorm(x), mask: mask, cache: cache)
            return h + mlp(postAttentionNorm(h))
        }
    }

    final class Backbone: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [DecoderLayer]
        let norm: RMSNorm

        init(_ config: Config) {
            _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            layers = (0 ..< config.layers).map { DecoderLayer(config, layer: $0) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.eps)
        }

        func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
            var h = embeddings
            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Backbone
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        init(_ config: Config) {
            _model.wrappedValue = Backbone(config)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }

        func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
            lmHead(model(embeddings, cache: cache))
        }
    }
}
