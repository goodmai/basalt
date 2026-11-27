import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

final class PLTInner: Module {
    let config: PLTConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [PLTDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: PLTConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.layers).map { _ in PLTDecoderLayer(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    var cacheCount: Int { config.loops * config.layers + config.loops - 1 }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        let embedded = embedTokens(tokens)
        var h = embedded
        var shared: [(keys: MLXArray, values: MLXArray)] = []
        for (index, layer) in layers.enumerated() {
            let out = layer.first(h, cache: cache?[index])
            (h, shared) = (out.hidden, shared + [(out.keys, out.values)])
        }
        for loop in 1 ..< max(config.loops, 1) {
            let previous = config.normalizePerLoop ? norm(h) : h
            h = config.embeddingScale * embedded + config.hiddenScale * shifted(previous, cache: cache?[shiftIndex(loop)])
            for (index, layer) in layers.enumerated() {
                h = layer.looped(h, shared: shared[index], cache: cache?[loop * config.layers + index])
            }
        }
        return norm(h)
    }

    func shiftIndex(_ loop: Int) -> Int { config.loops * config.layers + loop - 1 }

    func shifted(_ hidden: MLXArray, cache: KVCache?) -> MLXArray {
        let (b, l, d) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        var carried = MLXArray.zeros([b, 1, d], dtype: hidden.dtype)
        if let cache {
            if cache.offset > 0, let stored = cache.state.first {
                carried = stored[0..., 0, (cache.offset - 1) ..< cache.offset, 0...].asType(hidden.dtype)
            }
            _ = cache.update(keys: expandedDimensions(hidden, axis: 1), values: MLXArray.zeros([b, 1, l, 1], dtype: hidden.dtype))
        }
        return concatenated([carried, hidden[0..., ..<(l - 1)]], axis: 1)
    }
}

public final class PLTModel: Module, LLMModel {
    public static let modelType = "iquestpltcoder"
    public let vocabularySize: Int
    let config: PLTConfiguration
    @ModuleInfo(key: "model") var model: PLTInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: PLTConfiguration) {
        self.config = config
        vocabularySize = config.vocabularySize
        _model.wrappedValue = PLTInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public convenience init(configurationData data: Data) throws {
        self.init(try JSONDecoder().decode(PLTConfiguration.self, from: data))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
    }

    public var loraLayers: [Module] { model.layers }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        (0 ..< model.cacheCount).map { _ in KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: weights.compactMap { key, value in
            guard !key.contains("rotary_emb.inv_freq"), !(config.tieWordEmbeddings && key.hasPrefix("lm_head.")) else { return nil }
            return (Self.gateKey(key), value)
        })
    }

    static func gateKey(_ key: String) -> String {
        guard let range = key.range(of: ".plt_gate."), !key[range.upperBound...].hasPrefix("gate_norm.") else { return key }
        return key.replacingCharacters(in: range, with: ".plt_gate.proj.")
    }
}
