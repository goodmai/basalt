import BasaltCore
import Foundation
import MLXLMCommon

public struct HyperConnectionConfiguration: Codable, Sendable, Equatable, ModelConfigurationValidating {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var tieWordEmbeddings: Bool
    public var hyperConnection: HyperConnectionSpec

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case hcVariant = "hc_variant"
        case hcExpansionRate = "hc_expansion_rate"
        case hcSinkhornIterations = "hc_sinkhorn_iterations"
        case hcSinkhornEps = "hc_sinkhorn_eps"
        case hcGateMode = "hc_gate_mode"
        case hcCollapseMode = "hc_collapse_mode"
        case hcDynamic = "hc_dynamic"
        case hcAlphaInit = "hc_alpha_init"
    }

    public init(
        vocabSize: Int,
        hiddenSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        hyperConnection: HyperConnectionSpec = HyperConnectionSpec()
    ) {
        self.modelType = hyperConnection.variant == .ihc ? "ihc" : "mhc"
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = hiddenSize / max(numAttentionHeads, 1)
        self.intermediateSize = hiddenSize * 2
        self.rmsNormEps = 1e-6
        self.ropeTheta = 10_000.0
        self.tieWordEmbeddings = false
        self.hyperConnection = hyperConnection
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let decodedModelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mhc"
        modelType = decodedModelType
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        let decodedHidden = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenSize = decodedHidden
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        let decodedHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numAttentionHeads = decodedHeads
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? decodedHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (decodedHidden / max(decodedHeads, 1))
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        hyperConnection = try Self.decodeSpec(from: c, modelType: decodedModelType)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(hyperConnection.variant, forKey: .hcVariant)
        try c.encode(hyperConnection.expansionRate, forKey: .hcExpansionRate)
        try c.encode(hyperConnection.sinkhornIterations, forKey: .hcSinkhornIterations)
        try c.encode(hyperConnection.sinkhornEps, forKey: .hcSinkhornEps)
        try c.encode(hyperConnection.gateMode, forKey: .hcGateMode)
        try c.encode(hyperConnection.collapseMode, forKey: .hcCollapseMode)
        try c.encode(hyperConnection.dynamic, forKey: .hcDynamic)
        try c.encode(hyperConnection.alphaInit, forKey: .hcAlphaInit)
    }
}

extension HyperConnectionConfiguration {
    private static func decodeSpec(
        from c: KeyedDecodingContainer<CodingKeys>,
        modelType: String
    ) throws -> HyperConnectionSpec {
        let fallbackVariant: HyperConnectionVariant = modelType.lowercased() == "ihc" ? .ihc : .mhc
        let variant = try c.decodeIfPresent(HyperConnectionVariant.self, forKey: .hcVariant) ?? fallbackVariant
        let rate = try c.decodeIfPresent(Int.self, forKey: .hcExpansionRate) ?? 4
        let iters = try c.decodeIfPresent(Int.self, forKey: .hcSinkhornIterations) ?? 20
        let eps = try c.decodeIfPresent(Float.self, forKey: .hcSinkhornEps) ?? 1e-6
        let gateMode = try c.decodeIfPresent(HyperConnectionGateMode.self, forKey: .hcGateMode) ?? .sigmoid
        let collapseMode = try c.decodeIfPresent(HyperConnectionCollapseMode.self, forKey: .hcCollapseMode) ?? .mean
        let dynamic = try c.decodeIfPresent(Bool.self, forKey: .hcDynamic) ?? true
        let alphaInit = try c.decodeIfPresent(Float.self, forKey: .hcAlphaInit) ?? 0.01
        return HyperConnectionSpec(
            variant: variant,
            expansionRate: rate,
            sinkhornIterations: iters,
            sinkhornEps: eps,
            gateMode: gateMode,
            collapseMode: collapseMode
        ).with(dynamic: dynamic, alphaInit: alphaInit)
    }

    public func validated() throws -> HyperConnectionConfiguration {
        try validateModelConfiguration()
        return self
    }

    public func validateModelConfiguration() throws {
        guard vocabSize > 0, hiddenSize > 0, numHiddenLayers > 0, intermediateSize > 0 else {
            throw BasaltError.modelConfigurationInvalid(
                path: modelType,
                reason: "mhc requires positive vocab_size, hidden_size, num_hidden_layers and intermediate_size"
            )
        }
        guard numAttentionHeads > 0, numKeyValueHeads > 0,
              numAttentionHeads.isMultiple(of: numKeyValueHeads), headDim > 0 else {
            throw BasaltError.modelConfigurationInvalid(
                path: modelType,
                reason: "mhc num_attention_heads (\(numAttentionHeads)) must be divisible by num_key_value_heads (\(numKeyValueHeads)) with positive head_dim"
            )
        }
        guard rmsNormEps.isFinite && rmsNormEps > 0 && ropeTheta.isFinite && ropeTheta > 0 else {
            throw BasaltError.modelConfigurationInvalid(
                path: modelType,
                reason: "mhc requires finite positive rms_norm_eps and rope_theta"
            )
        }
        try hyperConnection.validate()
    }
}
