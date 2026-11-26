import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct WhisperCrossKV {
    let keys: MLXArray
    let values: MLXArray
}

final class WhisperAttention: Module {
    let heads: Int
    @ModuleInfo(key: "query") var wq: Linear
    @ModuleInfo(key: "key") var wk: Linear
    @ModuleInfo(key: "value") var wv: Linear
    @ModuleInfo(key: "out") var wo: Linear

    init(state: Int, heads: Int) {
        self.heads = heads
        _wq.wrappedValue = Linear(state, state)
        _wk.wrappedValue = Linear(state, state, bias: false)
        _wv.wrappedValue = Linear(state, state)
        _wo.wrappedValue = Linear(state, state)
    }

    func splitHeads(_ x: MLXArray) -> MLXArray {
        x.reshaped(x.dim(0), x.dim(1), heads, -1).transposed(0, 2, 1, 3)
    }

    func crossKeysValues(_ features: MLXArray) -> WhisperCrossKV {
        WhisperCrossKV(keys: splitHeads(wk(features)), values: splitHeads(wv(features)))
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        attend(x, keys: splitHeads(wk(x)), values: splitHeads(wv(x)), cache: cache, mask: mask)
    }

    func callAsFunction(_ x: MLXArray, cross: WhisperCrossKV) -> MLXArray {
        attend(x, keys: cross.keys, values: cross.values, cache: nil, mask: .none)
    }

    private func attend(
        _ x: MLXArray, keys: MLXArray, values: MLXArray, cache: KVCache?, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let queries = splitHeads(wq(x))
        let scale = pow(Float(queries.dim(-1)), -0.5)
        let output = attentionWithCacheUpdate(queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask)
        return wo(output.transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(1), -1))
    }
}

final class WhisperBlock: Module {
    @ModuleInfo(key: "attn") var attention: WhisperAttention
    @ModuleInfo(key: "attn_ln") var attentionNorm: LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttention: WhisperAttention?
    @ModuleInfo(key: "cross_attn_ln") var crossAttentionNorm: LayerNorm?
    @ModuleInfo(key: "mlp1") var mlp1: Linear
    @ModuleInfo(key: "mlp2") var mlp2: Linear
    @ModuleInfo(key: "mlp_ln") var mlpNorm: LayerNorm

    init(state: Int, heads: Int, crossAttention: Bool) {
        _attention.wrappedValue = WhisperAttention(state: state, heads: heads)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: state)
        _crossAttention.wrappedValue = crossAttention ? WhisperAttention(state: state, heads: heads) : nil
        _crossAttentionNorm.wrappedValue = crossAttention ? LayerNorm(dimensions: state) : nil
        _mlp1.wrappedValue = Linear(state, 4 * state)
        _mlp2.wrappedValue = Linear(4 * state, state)
        _mlpNorm.wrappedValue = LayerNorm(dimensions: state)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?, cross: WhisperCrossKV?
    ) -> MLXArray {
        var h = x + attention(attentionNorm(x), mask: mask, cache: cache)
        if let crossAttention, let crossAttentionNorm, let cross {
            h = h + crossAttention(crossAttentionNorm(h), cross: cross)
        }
        return h + mlp2(gelu(mlp1(mlpNorm(h))))
    }
}

final class WhisperAudioEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [WhisperBlock]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

    init(_ configuration: WhisperConfiguration) {
        let state = configuration.audioState
        _conv1.wrappedValue = Conv1d(inputChannels: configuration.melCount, outputChannels: state, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: state, outputChannels: state, kernelSize: 3, stride: 2, padding: 1)
        _blocks.wrappedValue = (0 ..< configuration.audioLayers).map { _ in
            WhisperBlock(state: state, heads: configuration.audioHeads, crossAttention: false)
        }
        _lnPost.wrappedValue = LayerNorm(dimensions: state)
    }

    static func sinusoids(length: Int, channels: Int) -> MLXArray {
        let half = channels / 2
        let step = Float(Foundation.log(10_000.0) / Double(half - 1))
        let inverse = exp(MLXArray(Int32(0) ..< Int32(half)).asType(.float32) * -step)
        let time = MLXArray(Int32(0) ..< Int32(length)).asType(.float32).expandedDimensions(axis: 1) * inverse.expandedDimensions(axis: 0)
        return concatenated([sin(time), cos(time)], axis: 1)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = gelu(conv1(mel.asType(conv1.weight.dtype)))
        x = gelu(conv2(x))
        x = x + Self.sinusoids(length: x.dim(1), channels: x.dim(2)).asType(x.dtype)
        for block in blocks {
            x = block(x, mask: .none, cache: nil, cross: nil)
        }
        return lnPost(x)
    }
}

final class WhisperTextDecoder: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [WhisperBlock]
    @ModuleInfo(key: "ln") var ln: LayerNorm

    init(_ configuration: WhisperConfiguration) {
        let state = configuration.textState
        _tokenEmbedding.wrappedValue = Embedding(embeddingCount: configuration.vocabularySize, dimensions: state)
        _positionalEmbedding.wrappedValue = MLXRandom.normal([configuration.textContext, state], scale: 0.02)
        _blocks.wrappedValue = (0 ..< configuration.textLayers).map { _ in
            WhisperBlock(state: state, heads: configuration.textHeads, crossAttention: true)
        }
        _ln.wrappedValue = LayerNorm(dimensions: state)
    }

    func crossKeysValues(_ features: MLXArray) -> [WhisperCrossKV] {
        blocks.compactMap { $0.crossAttention?.crossKeysValues(features) }
    }

    func callAsFunction(_ tokens: MLXArray, cross: [WhisperCrossKV]?, cache: [KVCache]?) -> MLXArray {
        let offset = cache?.first?.offset ?? 0
        let count = tokens.dim(1)
        var x = tokenEmbedding(tokens) + positionalEmbedding[offset ..< (offset + count)]
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = count > 1 ? .causal : .none
        for (index, block) in blocks.enumerated() {
            x = block(x, mask: mask, cache: cache?[index], cross: cross?[index])
        }
        return tokenEmbedding.asLinear(ln(x))
    }
}
