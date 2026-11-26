import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public protocol DiffusionLanguageModel: LanguageModel {
    var maskTokenId: Int { get }
    var bidirectionalPrompt: Bool { get }
    func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray
    func projectLogits(_ hidden: MLXArray) -> MLXArray
}

struct LLaDAConfiguration: Decodable, Sendable {
    let hiddenSize: Int
    let layers: Int
    let heads: Int
    let kvHeads: Int
    let mlpHiddenSize: Int
    let vocabularySize: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let maskTokenId: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "d_model", layers = "n_layers", heads = "n_heads", kvHeads = "n_kv_heads"
        case mlpHiddenSize = "mlp_hidden_size", embeddingSize = "embedding_size", vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps", ropeTheta = "rope_theta", maskTokenId = "mask_token_id"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        layers = try c.decode(Int.self, forKey: .layers)
        heads = try c.decode(Int.self, forKey: .heads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        mlpHiddenSize = try c.decode(Int.self, forKey: .mlpHiddenSize)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .embeddingSize) ?? c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        maskTokenId = try c.decode(Int.self, forKey: .maskTokenId)
        guard heads > 0, kvHeads > 0, heads % kvHeads == 0, hiddenSize % heads == 0, (0 ..< vocabularySize).contains(maskTokenId) else {
            throw DecodingError.dataCorruptedError(forKey: .maskTokenId, in: c, debugDescription: "llada needs heads divisible by key-value heads and d_model, and a mask token inside the vocabulary")
        }
    }

    var headDim: Int { hiddenSize / heads }
}

final class LLaDABlock: Module {
    let config: LLaDAConfiguration
    let rope: RoPE
    @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "attn_out") var wo: Linear
    @ModuleInfo(key: "ff_norm") var feedForwardNorm: RMSNorm
    @ModuleInfo(key: "ff_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "ff_out") var down: Linear

    init(_ config: LLaDAConfiguration) {
        self.config = config
        rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        _attentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _wq.wrappedValue = Linear(config.hiddenSize, config.heads * config.headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _wv.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _wo.wrappedValue = Linear(config.heads * config.headDim, config.hiddenSize, bias: false)
        _feedForwardNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _gate.wrappedValue = Linear(config.hiddenSize, config.mlpHiddenSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.mlpHiddenSize, bias: false)
        _down.wrappedValue = Linear(config.mlpHiddenSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let n = attentionNorm(x)
        let queries = applyRotaryPosition(rope, to: wq(n).reshaped(b, l, config.heads, -1).transposed(0, 2, 1, 3), offset: cache?.ropeOffset)
        let keys = applyRotaryPosition(rope, to: wk(n).reshaped(b, l, config.kvHeads, -1).transposed(0, 2, 1, 3), offset: cache?.ropeOffset)
        let values = wv(n).reshaped(b, l, config.kvHeads, -1).transposed(0, 2, 1, 3)
        let attention = attentionWithCacheUpdate(queries: queries, keys: keys, values: values, cache: cache, scale: pow(Float(config.headDim), -0.5), mask: mask)
        let h = x + wo(attention.transposed(0, 2, 1, 3).reshaped(b, l, -1))
        let f = feedForwardNorm(h)
        return h + down(silu(gate(f)) * up(f))
    }
}

final class LLaDAInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LLaDABlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: LLaDAConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.layers).map { _ in LLaDABlock(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }
}

public final class LLaDAModel: Module, LLMModel, DiffusionLanguageModel {
    public static let modelType = "llada"
    public let vocabularySize: Int
    public let maskTokenId: Int
    public var bidirectionalPrompt: Bool { true }
    let config: LLaDAConfiguration
    @ModuleInfo(key: "model") var model: LLaDAInner

    init(_ config: LLaDAConfiguration) {
        self.config = config
        vocabularySize = config.vocabularySize
        maskTokenId = config.maskTokenId
        _model.wrappedValue = LLaDAInner(config)
    }

    public convenience init(configurationData data: Data) throws {
        self.init(try JSONDecoder().decode(LLaDAConfiguration.self, from: data))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectLogits(hiddenStates(inputs, cache: cache))
    }

    public func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = model.embedTokens(inputs)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = cache?.first.map { createAttentionMask(h: h, cache: $0) } ?? .none
        for (index, layer) in model.layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[index])
        }
        return model.norm(h)
    }

    public func projectLogits(_ hidden: MLXArray) -> MLXArray {
        model.lmHead(hidden)
    }

    public var loraLayers: [Module] { model.layers }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        (0 ..< config.layers).map { _ in KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("rotary_emb") }
    }
}
