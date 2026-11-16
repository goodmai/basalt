import Foundation
import MLXLMCommon
import BasaltCore

public struct RicursiveConfiguration: Codable, Sendable, Equatable, ModelConfigurationValidating {
    public var modelType: String
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var slidingWindow: Int?
    public var layerTypes: [String]
    public var useGatedAttention: Bool
    public var gateFullPrecision: Bool
    public var useValres: Bool
    public var ffnClampThreshold: Float?
    public var embedScale: Float?
    public var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case useGatedAttention = "use_gated_attention"
        case gateFullPrecision = "gate_full_precision"
        case useValres = "use_valres"
        case ffnClampThreshold = "ffn_clamp_threshold"
        case embedScale = "embed_scale"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "ricursive"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 100_352
        let decodedHidden = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
        hiddenSize = decodedHidden
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 40_960
        let layers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 80
        hiddenLayers = layers
        let heads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 80
        attentionHeads = heads
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2_097_152
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< max(0, layers)).map { $0 % 5 == 4 ? "full_attention" : "sliding_attention" }
        useGatedAttention = try c.decodeIfPresent(Bool.self, forKey: .useGatedAttention) ?? true
        gateFullPrecision = try c.decodeIfPresent(Bool.self, forKey: .gateFullPrecision) ?? true
        useValres = try c.decodeIfPresent(Bool.self, forKey: .useValres) ?? false
        ffnClampThreshold = try c.decodeIfPresent(Float.self, forKey: .ffnClampThreshold)
        let rawEmbedScale = try c.decodeIfPresent(Double.self, forKey: .embedScale)
        embedScale = rawEmbedScale.map(Float.init)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

extension RicursiveConfiguration {
    public var effectiveSlidingWindow: Int? {
        guard let slidingWindow, slidingWindow > 0 else { return nil }
        return slidingWindow
    }

    public func usesSlidingWindow(layer: Int) -> Bool {
        guard effectiveSlidingWindow != nil, layer >= 0, layer < layerTypes.count else { return false }
        return layerTypes[layer] == "sliding_attention"
    }

    public func validated() throws -> RicursiveConfiguration {
        try validateModelConfiguration()
        return self
    }

    public func validateModelConfiguration() throws {
        guard vocabularySize > 0, hiddenSize > 0, intermediateSize > 0, hiddenLayers > 0 else {
            throw BasaltError.modelConfigurationInvalid(
                path: modelType,
                reason: "ricursive requires positive vocab_size, hidden_size, intermediate_size and num_hidden_layers"
            )
        }
        guard attentionHeads > 0, kvHeads > 0, attentionHeads.isMultiple(of: kvHeads), headDim > 0 else {
            throw BasaltError.modelConfigurationInvalid(
                path: modelType,
                reason: "ricursive num_attention_heads (\(attentionHeads)) must be divisible by num_key_value_heads (\(kvHeads))"
            )
        }
        guard layerTypes.count == hiddenLayers else {
            throw BasaltError.modelConfigurationInvalid(
                path: modelType,
                reason: "ricursive layer_types has \(layerTypes.count) entries, expected \(hiddenLayers)"
            )
        }
        let validKinds: Set<String> = ["sliding_attention", "full_attention"]
        if let invalid = layerTypes.first(where: { !validKinds.contains($0) }) {
            throw BasaltError.modelConfigurationInvalid(
                path: modelType,
                reason: "unsupported ricursive layer_type '\(invalid)'"
            )
        }
    }
}
