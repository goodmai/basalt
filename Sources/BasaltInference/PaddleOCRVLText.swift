import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum PaddleOCRVLText {
    struct Rotary {
        let cos: MLXArray
        let sin: MLXArray
    }

    static func positions(after cache: KVCache?, delta: Int, batch: Int, length: Int) -> MLXArray {
        let steps = MLXArray(Int32(0) ..< Int32(length))
        let base = switch cache?.ropeOffset {
        case .batch(let offsets): offsets.asType(.int32).reshaped(-1, 1) + steps + Int32(delta)
        case .scalar(let offset): broadcast(steps + Int32(offset + delta), to: [batch, length])
        case nil: broadcast(steps + Int32(delta), to: [batch, length])
        }
        return broadcast(base[.newAxis], to: [3, batch, length])
    }

    static func rotary(positions: MLXArray, dims: Int, base: Float, sections: [Int], dtype: DType) -> Rotary {
        let invFreq = 1 / pow(MLXArray(base), MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32) / Float(dims))
        let freqs = positions.asType(.float32)[.ellipsis, .newAxis] * invFreq
        let emb = concatenated([freqs, freqs], axis: -1)
        func select(_ values: MLXArray) -> MLXArray {
            let bounds = (sections + sections).reduce(into: [Int]()) { $0.append(($0.last ?? 0) + $1) }.dropLast()
            let chunks = split(values, indices: Array(bounds), axis: -1)
            return concatenated(chunks.enumerated().map { $1[$0 % 3] }, axis: -1)[0..., .newAxis, 0..., 0...]
        }
        return Rotary(cos: select(cos(emb).asType(dtype)), sin: select(sin(emb).asType(dtype)))
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    static func rotated(_ x: MLXArray, _ rotary: Rotary) -> MLXArray {
        let wide = x.asType(.float32)
        return (wide * rotary.cos.asType(.float32) + rotateHalf(wide) * rotary.sin.asType(.float32)).asType(x.dtype)
    }

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        init(_ config: PaddleOCRVLConfiguration) {
            (heads, kvHeads, headDim) = (config.heads, config.kvHeads, config.headDim)
            scale = pow(Float(headDim), -0.5)
            _wq.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?, rotary: Rotary) -> MLXArray {
            let (b, l) = (x.dim(0), x.dim(1))
            var queries = wq(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
            var keys = wk(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
            queries = PaddleOCRVLText.rotated(queries, rotary)
            keys = PaddleOCRVLText.rotated(keys, rotary)
            let output = attentionWithCacheUpdate(queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask)
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

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

        init(_ config: PaddleOCRVLConfiguration) {
            _attention.wrappedValue = Attention(config)
            mlp = MLP(dimensions: config.hiddenSize, hidden: config.intermediateSize)
            _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?, rotary: Rotary) -> MLXArray {
            let h = x + attention(inputNorm(x), mask: mask, cache: cache, rotary: rotary)
            return h + mlp(postAttentionNorm(h))
        }
    }

    final class Backbone: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [DecoderLayer]
        let norm: RMSNorm

        init(_ config: PaddleOCRVLConfiguration) {
            _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            layers = (0 ..< config.layers).map { _ in DecoderLayer(config) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?, rotary: Rotary) -> MLXArray {
            var h = embeddings
            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i], rotary: rotary)
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Backbone
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        init(_ config: PaddleOCRVLConfiguration) {
            _model.wrappedValue = Backbone(config)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }

        func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?, rotary: Rotary) -> MLXArray {
            lmHead(model(embeddings, cache: cache, rotary: rotary))
        }
    }
}
