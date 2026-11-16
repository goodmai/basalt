import Foundation
import MLXLMCommon
import BasaltCore

public struct FlintConfiguration: Codable, Sendable, Equatable, ModelConfigurationValidating {
    public var modelType: String
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeScaling: [String: StringOrNumber]?
    public var attentionBias: Bool
    public var useSlidingWindow: Bool
    public var slidingWindow: Int?
    public var layerTypes: [String]
    public var noRopeLayerIndices: [Int]
    public var softmaxType: String
    public var attentionSink: Bool?
    public var decoderSparseStep: Int
    public var moeIntermediateSize: Int
    public var sharedExpertIntermediateSize: Int
    public var numExpertsPerToken: Int
    public var numExperts: Int
    public var normTopkProb: Bool
    public var routerScoreFunction: String
    public var routerExpertBias: Bool
    public var mlpOnlyLayers: [Int]
    public var tieWordEmbeddings: Bool
    public var maxPositionEmbeddings: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case useSlidingWindow = "use_sliding_window"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case noRopeLayerIndices = "no_rope_layer_indices"
        case softmaxType = "softmax_type"
        case attentionSink = "attention_sink"
        case decoderSparseStep = "decoder_sparse_step"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case numExpertsPerToken = "num_experts_per_tok"
        case numExperts = "num_experts"
        case numLocalExperts = "num_local_experts"
        case normTopkProb = "norm_topk_prob"
        case routerScoreFunction = "router_score_function"
        case scoringFunc = "scoring_func"
        case routerExpertBias = "router_expert_bias"
        case useExpertBias = "use_expert_bias"
        case mlpOnlyLayers = "mlp_only_layers"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "flint"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 150_144
        let decodedHidden = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        hiddenSize = decodedHidden
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        let layers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 49
        hiddenLayers = layers
        let heads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        attentionHeads = heads
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? max(1, decodedHidden / max(1, heads))
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        let ropeParams = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling) ?? ropeParams
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? ropeParams?["rope_theta"]?.asFloat()
            ?? 10_000.0
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        let slidingEnabled = try c.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? true
        useSlidingWindow = slidingEnabled
        let rawWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 513
        slidingWindow = slidingEnabled ? rawWindow : nil
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: max(0, layers))
        noRopeLayerIndices = try c.decodeIfPresent([Int].self, forKey: .noRopeLayerIndices) ?? []
        softmaxType = try c.decodeIfPresent(String.self, forKey: .softmaxType) ?? "learnable"
        attentionSink = try c.decodeIfPresent(Bool.self, forKey: .attentionSink)
        decoderSparseStep = try c.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        sharedExpertIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 1024
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 6
        let primaryExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts)
        numExperts = try primaryExperts ?? c.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 256
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        let primaryScore = try c.decodeIfPresent(String.self, forKey: .routerScoreFunction)
        routerScoreFunction = try primaryScore ?? c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        let primaryBias = try c.decodeIfPresent(Bool.self, forKey: .routerExpertBias)
        routerExpertBias = try primaryBias ?? c.decodeIfPresent(Bool.self, forKey: .useExpertBias) ?? false
        mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(useSlidingWindow, forKey: .useSlidingWindow)
        try c.encodeIfPresent(slidingWindow, forKey: .slidingWindow)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(noRopeLayerIndices, forKey: .noRopeLayerIndices)
        try c.encode(softmaxType, forKey: .softmaxType)
        try c.encodeIfPresent(attentionSink, forKey: .attentionSink)
        try c.encode(decoderSparseStep, forKey: .decoderSparseStep)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(sharedExpertIntermediateSize, forKey: .sharedExpertIntermediateSize)
        try c.encode(numExpertsPerToken, forKey: .numExpertsPerToken)
        try c.encode(numExperts, forKey: .numExperts)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(routerScoreFunction, forKey: .routerScoreFunction)
        try c.encode(routerExpertBias, forKey: .routerExpertBias)
        try c.encode(mlpOnlyLayers, forKey: .mlpOnlyLayers)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
    }
}

extension FlintConfiguration {
    public var usesLearnedAttentionSinks: Bool {
        softmaxType == "learnable" || attentionSink == true
    }

    public var effectiveSlidingWindow: Int? {
        guard useSlidingWindow, let slidingWindow, slidingWindow > 0 else { return nil }
        return slidingWindow
    }

    public func usesSlidingWindow(layer: Int) -> Bool {
        guard effectiveSlidingWindow != nil, layer >= 0, layer < layerTypes.count else { return false }
        return layerTypes[layer] == "sliding_attention"
    }

    public func skipsRoPE(layer: Int) -> Bool {
        noRopeLayerIndices.contains(layer)
    }

    public func isSparseMoE(layer: Int) -> Bool {
        guard !mlpOnlyLayers.contains(layer), numExperts > 0 else { return false }
        return (layer + 1).isMultiple(of: max(1, decoderSparseStep))
    }

    public var linearRopeScale: Float {
        guard let ropeScaling, ropeScaling["type"] == .string("linear"),
              let factor = ropeScaling["factor"]?.asFloat(), factor > 0 else {
            return 1.0
        }
        return 1.0 / factor
    }

    public func validated() throws -> FlintConfiguration {
        try validateModelConfiguration()
        return self
    }

    public func validateModelConfiguration() throws {
        guard vocabularySize > 0, hiddenSize > 0, hiddenLayers > 0 else {
            throw BasaltError.modelConfigurationInvalid(path: modelType, reason: "flint requires positive vocab, hidden_size and num_hidden_layers")
        }
        guard attentionHeads > 0, kvHeads > 0, attentionHeads.isMultiple(of: kvHeads), headDim > 0 else {
            throw BasaltError.modelConfigurationInvalid(path: modelType, reason: "flint attention heads (\(attentionHeads)) must be divisible by kv heads (\(kvHeads))")
        }
        guard layerTypes.count == hiddenLayers else {
            throw BasaltError.modelConfigurationInvalid(path: modelType, reason: "layer_types has \(layerTypes.count) entries but num_hidden_layers is \(hiddenLayers)")
        }
        let hasSparseLayer = (0 ..< hiddenLayers).contains { isSparseMoE(layer: $0) }
        if hasSparseLayer {
            guard numExpertsPerToken >= 1, numExpertsPerToken <= numExperts else {
                throw BasaltError.modelConfigurationInvalid(path: modelType, reason: "flint num_experts_per_tok (\(numExpertsPerToken)) must be in 1...\(numExperts)")
            }
        }
    }
}
