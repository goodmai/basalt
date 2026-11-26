import BasaltCore
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM

package final class Mamba3Block: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: Mamba3Mixer
    @ModuleInfo(key: "norm2") var norm2: RMSNorm?
    @ModuleInfo(key: "mlp") var mlp: Mamba3GatedMLP?

    package init(_ config: Mamba3Configuration, layerIdx: Int) {
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._mixer.wrappedValue = Mamba3Mixer(config, layerIdx: layerIdx)
        if config.useGatedMLP {
            self._norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._mlp.wrappedValue = Mamba3GatedMLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }

    package func callAsFunction(_ x: MLXArray, cache: ArraysCache?, useChunked: Bool = true) -> MLXArray {
        var hidden = x + mixer(norm(x), cache: cache, useChunked: useChunked)
        if let norm2, let mlp {
            hidden = hidden + mlp(norm2(hidden))
        }
        return hidden
    }
}

public final class Mamba3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Mamba3Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(_ config: Mamba3Configuration) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { Mamba3Block(config, layerIdx: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil, useChunked: Bool = true) -> MLXArray {
        var hidden = embedTokens(inputs)
        for (index, layer) in layers.enumerated() {
            let layerCache = cache.flatMap { index < $0.count ? ($0[index] as? ArraysCache) : nil }
            hidden = layer(hidden, cache: layerCache, useChunked: useChunked)
        }
        return norm(hidden)
    }
}

public final class Mamba3Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public static let modelType = Mamba3Configuration.modelType
    public static let supportedModelTypes = Mamba3Configuration.supportedModelTypes

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: Mamba3Configuration

    @ModuleInfo(key: "model") public var model: Mamba3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Mamba3Configuration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(repeating: config.numHeads, count: config.numHiddenLayers)
        self._model.wrappedValue = Mamba3ModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public convenience init(configurationData data: Data) throws {
        let config = try JSONDecoder.json5().decode(Mamba3Configuration.self, from: data).validated()
        self.init(config)
    }

    public var loraLayers: [Module] { model.layers }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs, cache: cache, useChunked: true)
    }

    package func forward(_ inputs: MLXArray, cache: [KVCache]?, useChunked: Bool) -> MLXArray {
        let hidden = model(inputs, cache: cache, useChunked: useChunked)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        (0 ..< configuration.numHiddenLayers).map { _ in ArraysCache(size: 4) }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let mapped = Mamba3Checkpoint.sanitize(weights, config: configuration)
        FlintModel.quantizeModulesMatchingPackedWeights(in: self, weights: mapped)
        return mapped
    }
}

enum Mamba3Checkpoint {
    static func sanitize(_ weights: [String: MLXArray], config: Mamba3Configuration) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (rawKey, value) in weights {
            let key = canonicalKey(rawKey)
            guard !shouldDrop(key, config: config) else { continue }
            if key.hasSuffix(".mlp.fc1.weight"), value.ndim == 2, value.dim(0) == 2 * config.intermediateSize {
                let prefix = String(key.dropLast("fc1.weight".count))
                out[prefix + "up_proj.weight"] = value[0 ..< config.intermediateSize, 0...]
                out[prefix + "gate_proj.weight"] = value[config.intermediateSize..., 0...]
                continue
            }
            out[key] = normalizeShape(value, key: key, config: config)
        }
        return out
    }

    private static func canonicalKey(_ raw: String) -> String {
        if raw == "backbone.embedding.weight" || raw == "model.embeddings.weight" || raw == "embedding.weight" {
            return "model.embed_tokens.weight"
        }
        if raw == "backbone.norm_f.weight" || raw == "norm_f.weight" {
            return "model.norm.weight"
        }
        var key = raw
        if key.hasPrefix("backbone.") {
            key = "model." + key.dropFirst("backbone.".count)
        }
        for (from, to) in [
            (".attn.", ".mixer."),
            (".input_layernorm.", ".norm."),
            (".mixer_norm.", ".norm."),
            (".post_attention_layernorm.", ".norm2."),
            (".mlp_norm.", ".norm2."),
            (".mlp.fc2.", ".mlp.down_proj."),
        ] {
            key = key.replacingOccurrences(of: from, with: to)
        }
        return key
    }

    private static func shouldDrop(_ key: String, config: Mamba3Configuration) -> Bool {
        if config.tieWordEmbeddings && key.hasPrefix("lm_head.") { return true }
        if !config.useGatedMLP && (key.contains(".mlp.") || key.contains(".norm2.")) { return true }
        if !config.isMIMO && (key.hasSuffix(".mimo_x") || key.hasSuffix(".mimo_z") || key.hasSuffix(".mimo_o")) {
            return true
        }
        let parts = key.split(separator: ".", maxSplits: 3)
        if parts.count >= 3, parts[0] == "model", parts[1] == "layers",
           let idx = Int(parts[2]), idx >= config.numHiddenLayers
        {
            return true
        }
        return false
    }

    private static func normalizeShape(_ value: MLXArray, key: String, config: Mamba3Configuration) -> MLXArray {
        if (key.hasSuffix(".B_bias") || key.hasSuffix(".C_bias")), value.ndim == 2, config.mimoRank == 1 {
            return value.reshaped(config.numHeads, 1, config.dState)
        }
        return value
    }
}

extension MLXInferenceEngine {
    static func registerMamba3Aliases() async {
        for type in Mamba3Model.supportedModelTypes {
            await LLMTypeRegistry.shared.registerModelType(type) { data in
                try Mamba3Model(configurationData: data)
            }
            await VLMTypeRegistry.shared.registerModelType(type) { data in
                try Mamba3Model(configurationData: data)
            }
        }
    }
}
