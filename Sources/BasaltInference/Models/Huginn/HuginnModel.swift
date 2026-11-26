import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

struct HuginnConfiguration: Decodable, Sendable {
    let hiddenSize: Int
    let heads: Int
    let headDim: Int
    let preludeLayers: Int
    let coreLayers: Int
    let codaLayers: Int
    let intermediateSize: Int
    let vocabularySize: Int
    let normEps: Float
    let ropeBase: Float
    let steps: Int
    let qkBias: Bool
    let bias: Bool
    let tieEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "n_embd", heads = "n_heads", headDim = "head_dim", intermediateSize = "intermediate_size"
        case preludeLayers = "n_layers_in_prelude", coreLayers = "n_layers_in_recurrent_block", codaLayers = "n_layers_in_coda"
        case vocabularySize = "vocab_size", paddedVocabularySize = "padded_vocab_size", normEps = "norm_eps", ropeBase = "rope_base"
        case meanRecurrence = "mean_recurrence", steps = "num_steps", qkBias = "qk_bias", bias, tieEmbeddings = "tie_embeddings"
        case injectionType = "injection_type"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        heads = try c.decode(Int.self, forKey: .heads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / max(heads, 1)
        preludeLayers = try c.decodeIfPresent(Int.self, forKey: .preludeLayers) ?? 2
        coreLayers = try c.decodeIfPresent(Int.self, forKey: .coreLayers) ?? 4
        codaLayers = try c.decodeIfPresent(Int.self, forKey: .codaLayers) ?? 2
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .paddedVocabularySize) ?? c.decode(Int.self, forKey: .vocabularySize)
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-6
        ropeBase = try c.decodeIfPresent(Float.self, forKey: .ropeBase) ?? 50_000
        let mean = try c.decodeIfPresent(Int.self, forKey: .meanRecurrence) ?? 32
        steps = try c.decodeIfPresent(Int.self, forKey: .steps) ?? mean
        qkBias = try c.decodeIfPresent(Bool.self, forKey: .qkBias) ?? true
        bias = try c.decodeIfPresent(Bool.self, forKey: .bias) ?? false
        tieEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieEmbeddings) ?? true
        let injection = try c.decodeIfPresent(String.self, forKey: .injectionType) ?? "linear"
        guard hiddenSize > 0, heads > 0, intermediateSize > 0, vocabularySize > 0, normEps > 0, headDim * heads == hiddenSize, coreLayers >= 1, preludeLayers >= 0, codaLayers >= 0, steps >= 1, injection == "linear" else {
            throw DecodingError.dataCorruptedError(
                forKey: .heads, in: c,
                debugDescription: "huginn_raven needs n_heads > 0 with n_heads × head_dim = n_embd, a recurrent block of ≥ 1 layer, ≥ 1 recurrence step and injection_type linear")
        }
    }
}

final class HuginnAttention: Module {
    let config: HuginnConfiguration
    let rope: RoPE
    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "qk_bias") var qkBias: MLXArray?

    init(_ config: HuginnConfiguration) {
        self.config = config
        rope = RoPE(dimensions: config.headDim, traditional: true, base: config.ropeBase)
        _wqkv.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: false)
        _proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _qkBias.wrappedValue = config.qkBias ? MLXArray.zeros([2, 1, config.heads, config.headDim]) : nil
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let parts = wqkv(x).split(parts: 3, axis: -1).map { $0.reshaped(b, l, config.heads, config.headDim) }
        var (q, k) = (parts[0], parts[1])
        if let qkBias {
            q = q + qkBias[0]
            k = k + qkBias[1]
        }
        let queries = applyRotaryPosition(rope, to: q.transposed(0, 2, 1, 3), offset: cache?.ropeOffset)
        let keys = applyRotaryPosition(rope, to: k.transposed(0, 2, 1, 3), offset: cache?.ropeOffset)
        let values = parts[2].transposed(0, 2, 1, 3)
        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: pow(Float(config.headDim), -0.5), mask: mask)
        return proj(output.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class HuginnMLP: Module {
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ config: HuginnConfiguration) {
        _fc.wrappedValue = Linear(config.hiddenSize, 2 * config.intermediateSize, bias: false)
        _proj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (gate, up) = fc(x).split(axis: -1)
        return proj(silu(gate) * up)
    }
}

final class HuginnBlock: Module {
    @ModuleInfo(key: "attn") var attention: HuginnAttention
    @ModuleInfo(key: "mlp") var mlp: HuginnMLP
    @ModuleInfo(key: "norm_1") var norm1: RMSNorm
    @ModuleInfo(key: "norm_2") var norm2: RMSNorm
    @ModuleInfo(key: "norm_3") var norm3: RMSNorm
    @ModuleInfo(key: "norm_4") var norm4: RMSNorm

    init(_ config: HuginnConfiguration) {
        _attention.wrappedValue = HuginnAttention(config)
        _mlp.wrappedValue = HuginnMLP(config)
        _norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _norm3.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _norm4.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let h = norm2(attention(norm1(x), mask: mask, cache: cache) + x)
        return norm4(mlp(norm3(h)) + h)
    }
}

final class HuginnTransformer: Module {
    let config: HuginnConfiguration
    @ModuleInfo(key: "wte") var wte: Embedding
    @ModuleInfo(key: "prelude") var prelude: [HuginnBlock]
    @ModuleInfo(key: "adapter") var adapter: Linear
    @ModuleInfo(key: "core_block") var core: [HuginnBlock]
    @ModuleInfo(key: "coda") var coda: [HuginnBlock]
    @ModuleInfo(key: "ln_f") var lnF: RMSNorm
    var loopExitThreshold = LoopExit.environmentThreshold
    private(set) var exitSteps = 0
    private(set) var exitPasses = 0

    init(_ config: HuginnConfiguration) {
        self.config = config
        _wte.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _prelude.wrappedValue = (0 ..< config.preludeLayers).map { _ in HuginnBlock(config) }
        _adapter.wrappedValue = Linear(2 * config.hiddenSize, config.hiddenSize, bias: config.bias)
        _core.wrappedValue = (0 ..< config.coreLayers).map { _ in HuginnBlock(config) }
        _coda.wrappedValue = (0 ..< config.codaLayers).map { _ in HuginnBlock(config) }
        _lnF.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    var cacheCount: Int { config.preludeLayers + config.steps * config.coreLayers + config.codaLayers }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var embeds = wte(tokens) * Float(config.hiddenSize).squareRoot()
        let mask = createAttentionMask(h: embeds, cache: cache?.first)
        for (index, block) in prelude.enumerated() {
            embeds = block(embeds, mask: mask, cache: cache?[index])
        }
        var x = lnF(recur(embeds, mask: mask, cache: cache, exit: exitCaches(tokens, cache)))
        let codaStart = config.preludeLayers + config.steps * config.coreLayers
        for (index, block) in coda.enumerated() {
            x = block(x, mask: mask, cache: cache?[codaStart + index])
        }
        return lnF(x)
    }

    private func recur(_ embeds: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: [KVCache]?, exit: [KVCache]?) -> MLXArray {
        var x = MLXArray.zeros(like: embeds)
        for step in 0 ..< config.steps {
            let previous = x
            x = adapter(concatenated([x, embeds], axis: -1))
            for (index, block) in core.enumerated() {
                x = block(x, mask: mask, cache: cache?[coreCacheIndex(step: step, layer: index)])
            }
            guard let exit else { continue }
            exitPasses += 1
            guard step + 1 < config.steps else { break }
            if let threshold = loopExitThreshold, Self.relativeChange(x, from: previous) < threshold {
                fillSkippedSteps(exit, after: step)
                break
            }
        }
        if exit != nil { exitSteps += 1 }
        return x
    }

    static func relativeChange(_ x: MLXArray, from previous: MLXArray) -> Float {
        let (current, delta) = (x.asType(.float32), (x - previous).asType(.float32))
        return sqrt(sum(delta * delta, axis: -1) / maximum(sum(current * current, axis: -1), MLXArray(Float(1e-12)))).mean(axis: -1).max().item(Float.self)
    }

    private func coreCacheIndex(step: Int, layer: Int) -> Int { config.preludeLayers + step * config.coreLayers + layer }

    private func exitCaches(_ tokens: MLXArray, _ cache: [KVCache]?) -> [KVCache]? {
        guard loopExitThreshold != nil, tokens.dim(-1) == 1, let cache, cache.allSatisfy({ $0 is KVCacheSimple }) else { return nil }
        return cache
    }

    private func fillSkippedSteps(_ cache: [KVCache], after step: Int) {
        var written: [MLXArray] = []
        for skipped in (step + 1) ..< config.steps {
            for layer in 0 ..< config.coreLayers {
                written += LoopExit.copyLastPosition(from: cache[coreCacheIndex(step: step, layer: layer)], to: cache[coreCacheIndex(step: skipped, layer: layer)])
            }
        }
        eval(written)
    }
}

public final class HuginnModel: Module, LLMModel {
    public static let modelType = "huginn_raven"
    public let vocabularySize: Int
    let config: HuginnConfiguration
    @ModuleInfo(key: "transformer") var transformer: HuginnTransformer
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: HuginnConfiguration) {
        self.config = config
        vocabularySize = config.vocabularySize
        _transformer.wrappedValue = HuginnTransformer(config)
        if !config.tieEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public convenience init(configurationData data: Data) throws {
        self.init(try JSONDecoder().decode(HuginnConfiguration.self, from: data))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = transformer(inputs, cache: cache)
        return lmHead.map { $0(hidden) } ?? transformer.wte.asLinear(hidden)
    }

    public var loraLayers: [Module] { transformer.prelude + transformer.core + transformer.coda }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        (0 ..< transformer.cacheCount).map { _ in KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { $0.key != "freqs_cis" && !$0.key.contains("rotary_emb") && !(config.tieEmbeddings && $0.key.hasPrefix("lm_head.")) }
    }
}
