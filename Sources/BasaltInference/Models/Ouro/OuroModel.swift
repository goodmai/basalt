import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

struct OuroConfiguration: Decodable, Sendable {
    let hiddenSize: Int
    let layers: Int
    let intermediateSize: Int
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let ropeTheta: Float
    let steps: Int
    let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", layers = "num_hidden_layers", intermediateSize = "intermediate_size"
        case heads = "num_attention_heads", kvHeads = "num_key_value_heads", headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps", vocabularySize = "vocab_size", ropeTheta = "rope_theta"
        case steps = "total_ut_steps", tieWordEmbeddings = "tie_word_embeddings"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        layers = try c.decode(Int.self, forKey: .layers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        heads = try c.decode(Int.self, forKey: .heads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / heads
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        steps = try c.decodeIfPresent(Int.self, forKey: .steps) ?? 4
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        guard steps >= 1, layers >= 1, heads > 0, kvHeads > 0, heads % kvHeads == 0 else {
            throw DecodingError.dataCorruptedError(forKey: .steps, in: c, debugDescription: "ouro needs total_ut_steps ≥ 1, layers ≥ 1 and heads divisible by key-value heads")
        }
    }
}

final class OuroAttention: Module {
    let config: OuroConfiguration
    let scale: Float
    let rope: RoPE
    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(_ config: OuroConfiguration) {
        self.config = config
        scale = pow(Float(config.headDim), -0.5)
        rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        _wq.wrappedValue = Linear(config.hiddenSize, config.heads * config.headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _wv.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _wo.wrappedValue = Linear(config.heads * config.headDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let queries = applyRotaryPosition(rope, to: wq(x).reshaped(b, l, config.heads, -1).transposed(0, 2, 1, 3), offset: cache?.ropeOffset)
        let keys = applyRotaryPosition(rope, to: wk(x).reshaped(b, l, config.kvHeads, -1).transposed(0, 2, 1, 3), offset: cache?.ropeOffset)
        let values = wv(x).reshaped(b, l, config.kvHeads, -1).transposed(0, 2, 1, 3)
        let output = attentionWithCacheUpdate(queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask)
        return wo(output.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class OuroMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: OuroConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class OuroDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: OuroAttention
    @ModuleInfo(key: "mlp") var mlp: OuroMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "input_layernorm_2") var attentionOutputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var mlpInputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm_2") var mlpOutputNorm: RMSNorm

    init(_ config: OuroConfiguration) {
        _attention.wrappedValue = OuroAttention(config)
        _mlp.wrappedValue = OuroMLP(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionOutputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlpInputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlpOutputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let h = x + attentionOutputNorm(attention(inputNorm(x), mask: mask, cache: cache))
        return h + mlpOutputNorm(mlp(mlpInputNorm(h)))
    }
}

final class OuroInner: Module {
    let config: OuroConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [OuroDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "early_exit_gate") var exitGate: Linear
    var loopExitThreshold = LoopExit.environmentThreshold
    private(set) var exitSteps = 0
    private(set) var exitPasses = 0

    init(_ config: OuroConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.layers).map { _ in OuroDecoderLayer(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _exitGate.wrappedValue = Linear(config.hiddenSize, 1, bias: true)
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(tokens)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        let exit = exitCaches(tokens, cache)
        var (cumulative, remaining) = (MLXArray(Float(0)), MLXArray(Float(1)))
        for step in 0 ..< config.steps {
            for (index, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[step * layers.count + index])
            }
            h = norm(h)
            guard let exit else { continue }
            exitPasses += 1
            guard step + 1 < config.steps else { break }
            let lambda = sigmoid(exitGate(h).asType(.float32))
            cumulative = cumulative + lambda * remaining
            remaining = remaining * (1 - lambda)
            if let threshold = loopExitThreshold, cumulative.min().item(Float.self) >= threshold {
                fillSkippedSteps(exit, after: step)
                break
            }
        }
        if exit != nil { exitSteps += 1 }
        return h
    }

    private func exitCaches(_ tokens: MLXArray, _ cache: [KVCache]?) -> [KVCache]? {
        guard loopExitThreshold != nil, tokens.dim(-1) == 1, let cache, cache.allSatisfy({ $0 is KVCacheSimple }) else { return nil }
        return cache
    }

    private func fillSkippedSteps(_ cache: [KVCache], after step: Int) {
        for skipped in (step + 1) ..< config.steps {
            for layer in 0 ..< layers.count {
                LoopExit.copyLastPosition(from: cache[step * layers.count + layer], to: cache[skipped * layers.count + layer])
            }
        }
    }
}

public final class OuroModel: Module, LLMModel {
    public static let modelType = "ouro"
    public let vocabularySize: Int
    let config: OuroConfiguration
    @ModuleInfo(key: "model") var model: OuroInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: OuroConfiguration) {
        self.config = config
        vocabularySize = config.vocabularySize
        _model.wrappedValue = OuroInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public convenience init(configurationData data: Data) throws {
        self.init(try JSONDecoder().decode(OuroConfiguration.self, from: data))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
    }

    public var loraLayers: [Module] { model.layers }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        (0 ..< config.steps * config.layers).map { _ in KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("rotary_emb.inv_freq") && !(config.tieWordEmbeddings && $0.key.hasPrefix("lm_head.")) }
    }
}
