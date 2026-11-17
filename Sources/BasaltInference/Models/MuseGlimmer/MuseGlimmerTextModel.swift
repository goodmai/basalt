import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import BasaltCore

private struct MuseGlimmerTextEnvelope: Decodable {
    let textConfiguration: MuseGlimmerTextConfiguration?

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
    }
}

extension MuseGlimmerTextConfiguration {
    public static func decodeFlexible(from data: Data) throws -> MuseGlimmerTextConfiguration {
        let decoder = JSONDecoder.json5()
        if let nested = try decoder.decode(MuseGlimmerTextEnvelope.self, from: data).textConfiguration {
            return try nested.validated()
        }
        return try decoder.decode(MuseGlimmerTextConfiguration.self, from: data).validated()
    }

    public func validated() throws(BasaltError) -> MuseGlimmerTextConfiguration {
        guard vocabularySize > 0, hiddenSize > 0, intermediateSize > 0, hiddenLayers > 0, headDimensions > 0 else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "muse_glimmer requires positive vocab_size, hidden_size, intermediate_size, head_dim and num_hidden_layers"
            )
        }
        guard attentionHeads > 0, kvHeads > 0, attentionHeads.isMultiple(of: kvHeads) else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "muse_glimmer num_attention_heads (\(attentionHeads)) must be divisible by num_key_value_heads (\(kvHeads))"
            )
        }
        guard layerTypes.count == hiddenLayers, layerRopeTheta.count == hiddenLayers else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "muse_glimmer layer_types (\(layerTypes.count)) and layer_rope_theta (\(layerRopeTheta.count)) must match num_hidden_layers (\(hiddenLayers))"
            )
        }
        if let invalid = layerTypes.first(where: { $0 != "sliding_attention" && $0 != "full_attention" }) {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "unsupported muse_glimmer layer_type '\(invalid)'"
            )
        }
        return self
    }
}

package final class MuseGlimmerTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MuseGlimmerTextAttention
    @ModuleInfo var mlp: MuseGlimmerTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: MuseGlimmerCenteredRMSNorm

    let isSliding: Bool

    package init(_ config: MuseGlimmerTextConfiguration, layerIndex: Int) {
        self._selfAttention.wrappedValue = MuseGlimmerTextAttention(config, layerIndex: layerIndex)
        self._mlp.wrappedValue = MuseGlimmerTextMLP(config)
        self._inputLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(dimensions: config.hiddenSize, eps: config.postNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(dimensions: config.hiddenSize, eps: config.postNormEps)
        self.isSliding = config.layerTypes[layerIndex] == "sliding_attention"
        super.init()
    }

    package func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let afterAttn = x + postAttentionLayerNorm(selfAttention(inputLayerNorm(x), mask: mask, cache: cache))
        return afterAttn + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(afterAttn)))
    }
}

public final class MuseGlimmerTextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: MuseGlimmerScalelessRMSNorm
    @ModuleInfo var layers: [MuseGlimmerTextDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    let config: MuseGlimmerTextConfiguration
    private let fullAttentionIndex: Int?
    private let slidingAttentionIndex: Int?

    public init(_ config: MuseGlimmerTextConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._embedNorm.wrappedValue = MuseGlimmerScalelessRMSNorm(eps: config.rmsNormEps)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { MuseGlimmerTextDecoderLayer(config, layerIndex: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.fullAttentionIndex = config.layerTypes.firstIndex(of: "full_attention")
        self.slidingAttentionIndex = config.layerTypes.firstIndex(of: "sliding_attention")
        super.init()
    }

    public func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputEmbedding: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbedding {
            h = inputEmbedding
        } else if let inputs {
            h = embedNorm(embedTokens(inputs))
        } else {
            preconditionFailure("MuseGlimmerTextModelInner requires inputs or inputEmbedding")
        }
        let seqLen = h.dim(1)
        let fullRep = fullAttentionIndex.flatMap { slot in cache.flatMap { slot < $0.count ? $0[slot] : nil } }
        let slidingRep = slidingAttentionIndex.flatMap { slot in cache.flatMap { slot < $0.count ? $0[slot] : nil } }
        let fullMask = makeAttentionMask(n: seqLen, cache: fullRep, windowSize: nil)
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode = slidingAttentionIndex == nil
            ? .none
            : makeAttentionMask(n: seqLen, cache: slidingRep, windowSize: config.slidingWindow)

        for (index, layer) in layers.enumerated() {
            let layerCache = cache.flatMap { index < $0.count ? $0[index] : nil }
            h = layer(h, mask: layer.isSliding ? slidingMask : fullMask, cache: layerCache)
        }
        return norm(h)
    }
}

public final class MuseGlimmerTextModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public static let modelType = "muse_glimmer_text"
    public static let multimodalModelType = "muse_glimmer"

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: MuseGlimmerTextConfiguration

    @ModuleInfo(key: "model") public var model: MuseGlimmerTextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: MuseGlimmerTextConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self._model.wrappedValue = MuseGlimmerTextModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    public var loraLayers: [Module] { model.layers }

    public var toolCallFormat: ToolCallFormat? { .atem }

    public var reasoningConfig: ReasoningConfig? {
        ReasoningConfig(
            startDelimiter: "to=self<|message|>",
            endDelimiter: "<|eom|>",
            promptStrategy: .none,
            isSpecialToken: true
        )
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs: inputs, cache: cache, inputEmbedding: nil)
    }

    package func forward(inputs: MLXArray?, cache: [KVCache]?, inputEmbedding: MLXArray?) -> MLXArray {
        let hidden = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
        var logits = if let lmHead {
            lmHead(hidden)
        } else {
            model.embedTokens.asLinear(hidden)
        }
        logits = logits * configuration.outputMultiplier
        let cap = configuration.finalLogitSoftcapping
        if cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try configuration.layerTypes.map { layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: layerType == "sliding_attention"
            )
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray](minimumCapacity: weights.count)
        for (rawKey, value) in weights {
            if rawKey.contains("rotary_emb.inv_freq")
                || rawKey.hasPrefix("vision_tower.")
                || rawKey.hasPrefix("vision_adapter.")
                || rawKey.hasPrefix("vision_projection")
                || rawKey.hasPrefix("model.vision_tower.")
                || rawKey.hasPrefix("model.vision_adapter.")
                || rawKey.hasPrefix("model.vision_projection") {
                continue
            }
            var key = rawKey
            if key.hasPrefix("language_model.model.") {
                key = "model." + key.dropFirst("language_model.model.".count)
            } else if key.hasPrefix("model.language_model.") {
                key = "model." + key.dropFirst("model.language_model.".count)
            } else if key.hasPrefix("language_model.lm_head.") {
                key = "lm_head." + key.dropFirst("language_model.lm_head.".count)
            }
            if configuration.tieWordEmbeddings && key.hasPrefix("lm_head.") {
                continue
            }
            out[key] = value
        }
        return out
    }
}

extension MLXInferenceEngine {
    static func registerMuseGlimmerAliases() async {
        for type in [MuseGlimmerTextModel.modelType, MuseGlimmerTextModel.multimodalModelType, "muse-glimmer"] {
            await LLMTypeRegistry.shared.registerModelType(type) { data in
                MuseGlimmerTextModel(try MuseGlimmerTextConfiguration.decodeFlexible(from: data))
            }
        }
        await VLMTypeRegistry.shared.registerModelType(MuseGlimmerTextModel.multimodalModelType) { data in
            let config = try JSONDecoder.json5().decode(MuseGlimmerConfiguration.self, from: data)
            _ = try config.textConfiguration.validated()
            try config.visionConfiguration.validateMuseGlimmer()
            return MuseGlimmerVLM(config)
        }
    }
}

extension MuseGlimmerVisionConfiguration {
    func validateMuseGlimmer() throws(BasaltError) {
        guard hiddenSize > 0, intermediateSize > 0, hiddenLayers > 0, attentionHeads > 0,
              hiddenSize.isMultiple(of: attentionHeads), mergeSize > 0, patchSize > 0, patchTemporal > 0,
              positionEmbeddingHeight > 0, positionEmbeddingWidth > 0, layerTypes.count == hiddenLayers
        else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "invalid muse_glimmer vision_config dimensions or layer_types (\(layerTypes.count) vs \(hiddenLayers))"
            )
        }
    }
}
