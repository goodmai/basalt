import BasaltCore
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM

public final class GatedDeltaNet2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [GDN2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let config: GatedDeltaNet2Configuration
    private let firstFullAttentionIndex: Int?
    private let firstSlidingAttentionIndex: Int?

    public init(_ config: GatedDeltaNet2Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        let built = (0 ..< config.numHiddenLayers).map { GDN2DecoderLayer(config, layerIdx: $0) }
        self._layers.wrappedValue = built
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.firstFullAttentionIndex = built.firstIndex { $0.isAttention && !$0.isSliding }
        self.firstSlidingAttentionIndex = built.firstIndex { $0.isSliding }
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        useChunked: Bool = true,
        useMetal: Bool = true
    ) -> MLXArray {
        var hidden = embedTokens(inputs)
        let seqLen = hidden.dim(1)
        let fullMask: MLXFast.ScaledDotProductAttentionMaskMode = firstFullAttentionIndex.map { slot in
            let rep = cache.flatMap { slot < $0.count ? $0[slot] : nil }
            return makeAttentionMask(n: seqLen, cache: rep, windowSize: nil)
        } ?? .none
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode = firstSlidingAttentionIndex.map { slot in
            let rep = cache.flatMap { slot < $0.count ? $0[slot] : nil }
            return makeAttentionMask(n: seqLen, cache: rep, windowSize: config.slidingWindow)
        } ?? .none

        for (index, layer) in layers.enumerated() {
            let layerCache = cache.flatMap { index < $0.count ? $0[index] : nil }
            let layerMask: MLXFast.ScaledDotProductAttentionMaskMode = if layer.isSliding {
                slidingMask
            } else if layer.isAttention {
                fullMask
            } else {
                .none
            }
            hidden = layer(hidden, mask: layerMask, cache: layerCache, useChunked: useChunked, useMetal: useMetal)
        }
        return norm(hidden)
    }
}

public final class GatedDeltaNet2Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public static let modelType = GatedDeltaNet2Configuration.modelType
    public static let supportedModelTypes = GatedDeltaNet2Configuration.supportedModelTypes

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: GatedDeltaNet2Configuration

    @ModuleInfo(key: "model") public var model: GatedDeltaNet2ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: GatedDeltaNet2Configuration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { idx in
            config.isAttentionLayer(idx) ? config.numKeyValueHeads : config.numVHeads
        }
        self._model.wrappedValue = GatedDeltaNet2ModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public convenience init(configurationData data: Data) throws {
        let config = try JSONDecoder.json5().decode(GatedDeltaNet2Configuration.self, from: data).validated()
        self.init(config)
    }

    public var loraLayers: [Module] { model.layers }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs, cache: cache, useChunked: true, useMetal: true)
    }

    package func forward(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        useChunked: Bool,
        useMetal: Bool
    ) -> MLXArray {
        let hidden = model(inputs, cache: cache, useChunked: useChunked, useMetal: useMetal)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try model.layers.map { layer in
            if layer.isSliding {
                return try makeHybridAttentionKVCache(
                    parameters: parameters,
                    slidingWindow: configuration.slidingWindow,
                    usesSlidingWindow: true
                )
            }
            if layer.isAttention {
                return try makeAttentionKVCache(parameters: parameters)
            }
            return ArraysCache(size: 4)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let mapped = GatedDeltaNet2Checkpoint.sanitize(weights, config: configuration)
        FlintModel.quantizeModulesMatchingPackedWeights(in: self, weights: mapped)
        return mapped
    }
}

enum GatedDeltaNet2Checkpoint {
    static func sanitize(_ weights: [String: MLXArray], config: GatedDeltaNet2Configuration) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (rawKey, rawValue) in weights {
            let key = canonicalKey(rawKey)
            guard !shouldDrop(key, config: config) else { continue }
            if key.hasSuffix("_conv1d.weight") {
                let target = String(key.dropLast(".weight".count)) + ".conv.weight"
                out[target] = normalizeConvWeight(rawValue, kernelSize: config.convKernelSize)
            } else if key.hasSuffix("_conv1d.conv.weight") {
                out[key] = normalizeConvWeight(rawValue, kernelSize: config.convKernelSize)
            } else if key.hasSuffix(".attn.dt_bias"), rawValue.ndim == 1, rawValue.size == config.numVHeads * config.headDim {
                out[key] = rawValue.reshaped(config.numVHeads, config.headDim)
            } else {
                out[key] = rawValue
            }
        }
        return out
    }

    private static func canonicalKey(_ raw: String) -> String {
        if raw == "transformer.wte.weight" || raw == "model.embeddings.weight" {
            return "model.embed_tokens.weight"
        }
        if raw == "transformer.ln_f.weight" {
            return "model.norm.weight"
        }
        var key = raw
        for (from, to) in [
            ("transformer.h.", "model.layers."),
            ("model.blocks.", "model.layers."),
        ] where key.hasPrefix(from) {
            key = to + key.dropFirst(from.count)
            break
        }
        for (from, to) in [
            (".self_attn.", ".attn."),
            (".linear_attn.", ".attn."),
            (".mixer.", ".attn."),
            (".norm_1.", ".input_layernorm."),
            (".attn_norm.", ".input_layernorm."),
            (".norm_2.", ".post_attention_layernorm."),
            (".ffn_norm.", ".post_attention_layernorm."),
            (".mlp_norm.", ".post_attention_layernorm."),
            (".attn.erase_proj.", ".attn.b_proj."),
            (".attn.write_proj.", ".attn.w_proj."),
            (".attn.decay_proj.", ".attn.a_proj."),
            (".attn.gate_proj.", ".attn.g_proj."),
        ] {
            key = key.replacingOccurrences(of: from, with: to)
        }
        return key
    }

    private static func shouldDrop(_ key: String, config: GatedDeltaNet2Configuration) -> Bool {
        if config.tieWordEmbeddings && key.hasPrefix("lm_head.") { return true }
        if key.contains("rotary_emb.inv_freq") { return true }
        if !config.useShortConv && key.contains("_conv1d.") { return true }
        let parts = key.split(separator: ".", maxSplits: 3)
        if parts.count >= 3, parts[0] == "model", parts[1] == "layers",
           let idx = Int(parts[2]), idx >= config.numHiddenLayers
        {
            return true
        }
        return false
    }

    private static func normalizeConvWeight(_ weight: MLXArray, kernelSize: Int) -> MLXArray {
        if weight.ndim == 2 {
            return weight.expandedDimensions(axis: -1)
        }
        if weight.ndim == 3 && weight.dim(1) == 1 && weight.dim(2) == kernelSize && kernelSize != 1 {
            return weight.swappedAxes(1, 2)
        }
        return weight
    }
}

extension MLXInferenceEngine {
    static func registerGatedDeltaNet2Aliases() async {
        for type in GatedDeltaNet2Model.supportedModelTypes {
            await LLMTypeRegistry.shared.registerModelType(type) { data in
                try GatedDeltaNet2Model(configurationData: data)
            }
            await VLMTypeRegistry.shared.registerModelType(type) { data in
                try GatedDeltaNet2Model(configurationData: data)
            }
        }
    }
}
