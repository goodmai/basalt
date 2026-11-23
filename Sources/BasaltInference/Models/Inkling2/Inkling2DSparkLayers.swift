import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct Inkling2DSparkConfiguration: Decodable, Sendable, Equatable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDimensions: Int
    var intermediateSize: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var localLayers: Set<Int>
    var slidingWindow: Int
    var blockSize: Int
    var markovRank: Int
    var markovHeadType: String
    var targetLayerIDs: [Int]

    private enum Root: String, CodingKey { case draft = "mtp_config" }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", hiddenLayers = "num_hidden_layers", attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads", headDimensions = "head_dim", intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps", ropeTheta = "rope_theta", localLayers = "local_layer_ids", slidingWindow = "sliding_window_size"
        case blockSize = "block_size", markovRank = "markov_rank", markovHeadType = "markov_head_type", targetLayerIDs = "target_layer_ids"
    }

    init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: Root.self)
        let c = try root.nestedContainer(keyedBy: CodingKeys.self, forKey: .draft)
        (hiddenSize, hiddenLayers) = (try c.decode(Int.self, forKey: .hiddenSize), try c.decode(Int.self, forKey: .hiddenLayers))
        (attentionHeads, kvHeads) = (try c.decode(Int.self, forKey: .attentionHeads), try c.decode(Int.self, forKey: .kvHeads))
        (headDimensions, intermediateSize) = (try c.decode(Int.self, forKey: .headDimensions), try c.decode(Int.self, forKey: .intermediateSize))
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        localLayers = try c.decodeIfPresent(Set<Int>.self, forKey: .localLayers) ?? []
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        (blockSize, markovRank) = (try c.decode(Int.self, forKey: .blockSize), try c.decode(Int.self, forKey: .markovRank))
        markovHeadType = try c.decode(String.self, forKey: .markovHeadType)
        targetLayerIDs = try c.decode([Int].self, forKey: .targetLayerIDs)
    }

    static let maximumBlockSize = 16

    func validated(for target: Inkling2Configuration) throws(Inkling2Configuration.Invalid) -> Self {
        guard markovHeadType == "rnn_hidden" else { throw .init(description: "DSpark Markov head '\(markovHeadType)' is not rnn_hidden") }
        guard hiddenSize == target.hiddenSize, targetLayerIDs.last == target.visitOrder.count, targetLayerIDs == (target.draftTaps ?? []) else {
            throw .init(description: "DSpark reads \(targetLayerIDs) at width \(hiddenSize); the target has \(target.visitOrder.count) visits at width \(target.hiddenSize)")
        }
        guard hiddenLayers > 0, kvHeads > 0, attentionHeads.isMultiple(of: kvHeads), (2 ... Self.maximumBlockSize).contains(blockSize), markovRank > 0, slidingWindow > 0 else {
            throw .init(description: "DSpark dimensions, or a block_size outside 2...\(Self.maximumBlockSize) (the conv states rewind at most \(Self.maximumBlockSize) positions)")
        }
        return self
    }
}

final class Inkling2DSparkAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "k_proj_ctx") var contextKeyProjection: Linear
    @ModuleInfo(key: "v_proj_ctx") var contextValueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "k_norm_ctx") var contextKeyNorm: RMSNorm

    let heads: Int
    let kvHeads: Int
    let window: Int?
    let scale: Float
    let rope: RoPE

    init(_ configuration: Inkling2DSparkConfiguration, isLocal: Bool) {
        let (d, h, kv) = (configuration.hiddenSize, configuration.headDimensions, configuration.kvHeads)
        heads = configuration.attentionHeads
        kvHeads = kv
        window = isLocal ? configuration.slidingWindow : nil
        scale = 1 / Float(h).squareRoot()
        rope = RoPE(dimensions: h, traditional: false, base: configuration.ropeTheta)
        _queryProjection.wrappedValue = Linear(d, heads * h, bias: false)
        _keyProjection.wrappedValue = Linear(d, kv * h, bias: false)
        _valueProjection.wrappedValue = Linear(d, kv * h, bias: false)
        _contextKeyProjection.wrappedValue = Linear(d, kv * h, bias: false)
        _contextValueProjection.wrappedValue = Linear(d, kv * h, bias: false)
        _outputProjection.wrappedValue = Linear(heads * h, d, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: h, eps: configuration.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: h, eps: configuration.rmsNormEps)
        _contextKeyNorm.wrappedValue = RMSNorm(dimensions: h, eps: configuration.rmsNormEps)
    }

    private func split(_ x: MLXArray, _ count: Int) -> MLXArray {
        x.reshaped(x.dim(0), x.dim(1), count, -1)
    }

    func appendContext(_ context: MLXArray, cache: KVCache) {
        let keys = contextKeyNorm(split(contextKeyProjection(context), kvHeads))
        let values = split(contextValueProjection(context), kvHeads)
        _ = cache.update(keys: rope(keys.transposed(0, 2, 1, 3), offset: cache.offset), values: values.transposed(0, 2, 1, 3))
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let offset = cache.offset
        let queries = queryNorm(split(queryProjection(x), heads)).transposed(0, 2, 1, 3)
        let keys = keyNorm(split(keyProjection(x), kvHeads)).transposed(0, 2, 1, 3)
        let values = split(valueProjection(x), kvHeads).transposed(0, 2, 1, 3)
        var (allKeys, allValues) = cache.update(keys: rope(keys, offset: offset), values: values)
        if let window, allKeys.dim(2) > window + l {
            let start = allKeys.dim(2) - window - l
            (allKeys, allValues) = (allKeys[0..., 0..., start...], allValues[0..., 0..., start...])
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: rope(queries, offset: offset), keys: allKeys, values: allValues, scale: scale, mask: .none)
        return outputProjection(output.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class Inkling2DSparkMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ configuration: Inkling2DSparkConfiguration) {
        let (d, i) = (configuration.hiddenSize, configuration.intermediateSize)
        _gate.wrappedValue = Linear(d, i, bias: false)
        _up.wrappedValue = Linear(d, i, bias: false)
        _down.wrappedValue = Linear(i, d, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class Inkling2DSparkLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Inkling2DSparkAttention
    @ModuleInfo var mlp: Inkling2DSparkMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "dflare_norm") var featureNorm: RMSNorm

    init(_ configuration: Inkling2DSparkConfiguration, isLocal: Bool) {
        let (d, eps) = (configuration.hiddenSize, configuration.rmsNormEps)
        _attention.wrappedValue = Inkling2DSparkAttention(configuration, isLocal: isLocal)
        _mlp.wrappedValue = Inkling2DSparkMLP(configuration)
        _inputNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        _featureNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: cache)
        return h + mlp(postAttentionNorm(h))
    }
}

final class Inkling2DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var previousToken: Embedding
    @ModuleInfo(key: "rnn_proj") var recurrence: Linear
    @ModuleInfo(key: "fold") var fold: Linear
    @ModuleInfo(key: "dflare_norm") var featureNorm: RMSNorm

    let rank: Int

    init(_ configuration: Inkling2DSparkConfiguration, vocabulary: Int) {
        let (d, r) = (configuration.hiddenSize, configuration.markovRank)
        rank = r
        _previousToken.wrappedValue = Embedding(embeddingCount: vocabulary, dimensions: r)
        _recurrence.wrappedValue = Linear(2 * r + 2 * d, 3 * r, bias: true)
        _fold.wrappedValue = Linear(r, vocabulary, bias: true)
        _featureNorm.wrappedValue = RMSNorm(dimensions: d, eps: configuration.rmsNormEps)
    }

    func context(hidden: MLXArray, features: MLXArray) -> MLXArray {
        let inputs = concatenated([hidden, broadcast(features, to: hidden.shape)], axis: -1).asType(.float32)
        let weight = recurrence.weight[0..., (2 * rank)...].asType(.float32)
        return matmul(inputs, weight.T) + recurrence.bias!.asType(.float32)
    }

    func step(state: MLXArray, previous: MLXArray, context: MLXArray) -> (MLXArray, MLXArray) {
        let inputs = concatenated([state, previousToken(previous)], axis: -1).asType(.float32)
        let raw = (context + matmul(inputs, recurrence.weight[0..., ..<(2 * rank)].asType(.float32).T)).asType(state.dtype)
        let gate = sigmoid(raw[.ellipsis, ..<rank])
        let candidate = tanh(raw[.ellipsis, rank ..< (2 * rank)])
        let next = gate * state + (1 - gate) * candidate
        return (next, fold(tanh(raw[.ellipsis, (2 * rank)...])))
    }
}
