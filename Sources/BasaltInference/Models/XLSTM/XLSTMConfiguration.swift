import Foundation

public enum XLSTMWeightMode: String, Codable, Sendable, Equatable {
    case single
    case fused
}

public enum XLSTMSLSTMGateMode: String, Codable, Sendable, Equatable {
    case logSigmoid = "logsigmoid"
    case exp = "exp"
}

public struct XLSTMConfiguration: Codable, Sendable, Equatable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numHeads: Int
    public let qkDimFactor: Double
    public let vDimFactor: Double
    public let ffnProjFactor: Double
    public let ffnRoundUpToMultipleOf: Int
    public let chunkSize: Int
    public let gateSoftCap: Float?
    public let outputLogitSoftCap: Float?
    public let normEps: Float
    public let eps: Float
    public let addOutNorm: Bool
    public let tieWordEmbeddings: Bool
    public let weightMode: XLSTMWeightMode
    public let useBias: Bool
    public let normReductionForceFloat32: Bool
    public let slstmAt: [Int]
    public let slstmGateMode: XLSTMSLSTMGateMode

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case embeddingDim = "embedding_dim"
        case numHiddenLayers = "num_hidden_layers"
        case numBlocks = "num_blocks"
        case numHeads = "num_heads"
        case numAttentionHeads = "num_attention_heads"
        case qkDimFactor = "qk_dim_factor"
        case vDimFactor = "v_dim_factor"
        case ffnProjFactor = "ffn_proj_factor"
        case ffnRoundUpToMultipleOf = "ffn_round_up_to_multiple_of"
        case chunkSize = "chunk_size"
        case gateSoftCap = "gate_soft_cap"
        case outputLogitSoftCap = "output_logit_soft_cap"
        case normEps = "norm_eps"
        case eps = "eps"
        case addOutNorm = "add_out_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case weightMode = "weight_mode"
        case useBias = "use_bias"
        case normReductionForceFloat32 = "norm_reduction_force_float32"
        case slstmAt = "slstm_at"
        case slstmGateMode = "slstm_gate_mode"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try Self.decodeRequiredInt(c, keys: [.hiddenSize, .embeddingDim])
        numHiddenLayers = try Self.decodeRequiredInt(c, keys: [.numHiddenLayers, .numBlocks])
        numHeads = try Self.decodeRequiredInt(c, keys: [.numHeads, .numAttentionHeads])
        qkDimFactor = try c.decodeIfPresent(Double.self, forKey: .qkDimFactor) ?? 0.5
        vDimFactor = try c.decodeIfPresent(Double.self, forKey: .vDimFactor) ?? 1.0
        ffnProjFactor = try c.decodeIfPresent(Double.self, forKey: .ffnProjFactor) ?? 2.667
        ffnRoundUpToMultipleOf = try c.decodeIfPresent(Int.self, forKey: .ffnRoundUpToMultipleOf) ?? 64
        chunkSize = try c.decodeIfPresent(Int.self, forKey: .chunkSize) ?? 64
        gateSoftCap = c.contains(.gateSoftCap) ? try c.decodeIfPresent(Float.self, forKey: .gateSoftCap) : 15.0
        outputLogitSoftCap = c.contains(.outputLogitSoftCap)
            ? try c.decodeIfPresent(Float.self, forKey: .outputLogitSoftCap) : 30.0
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-6
        eps = try c.decodeIfPresent(Float.self, forKey: .eps) ?? 1e-6
        addOutNorm = try c.decodeIfPresent(Bool.self, forKey: .addOutNorm) ?? true
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        weightMode = try c.decodeIfPresent(XLSTMWeightMode.self, forKey: .weightMode) ?? .single
        useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        normReductionForceFloat32 = try c.decodeIfPresent(Bool.self, forKey: .normReductionForceFloat32) ?? true
        slstmAt = try c.decodeIfPresent([Int].self, forKey: .slstmAt) ?? []
        slstmGateMode = try c.decodeIfPresent(XLSTMSLSTMGateMode.self, forKey: .slstmGateMode) ?? .logSigmoid
        try validateDimensions(in: c)
        try validateHyperparameters(in: c)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numHeads, forKey: .numHeads)
        try c.encode(qkDimFactor, forKey: .qkDimFactor)
        try c.encode(vDimFactor, forKey: .vDimFactor)
        try c.encode(ffnProjFactor, forKey: .ffnProjFactor)
        try c.encode(ffnRoundUpToMultipleOf, forKey: .ffnRoundUpToMultipleOf)
        try c.encode(chunkSize, forKey: .chunkSize)
        try c.encode(gateSoftCap, forKey: .gateSoftCap)
        try c.encode(outputLogitSoftCap, forKey: .outputLogitSoftCap)
        try c.encode(normEps, forKey: .normEps)
        try c.encode(eps, forKey: .eps)
        try c.encode(addOutNorm, forKey: .addOutNorm)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(weightMode, forKey: .weightMode)
        try c.encode(useBias, forKey: .useBias)
        try c.encode(normReductionForceFloat32, forKey: .normReductionForceFloat32)
        try c.encode(slstmAt, forKey: .slstmAt)
        try c.encode(slstmGateMode, forKey: .slstmGateMode)
    }

    public var qkDim: Int { Int(Double(hiddenSize) * qkDimFactor) }
    public var qkHeadDim: Int { qkDim / numHeads }
    public var vDim: Int { Int(Double(hiddenSize) * vDimFactor) }
    public var vHeadDim: Int { vDim / numHeads }
    public var slstmHeadDim: Int { hiddenSize / numHeads }

    public var ffnHiddenDim: Int {
        let raw = Int(Double(hiddenSize) * ffnProjFactor)
        return ((raw + ffnRoundUpToMultipleOf - 1) / ffnRoundUpToMultipleOf) * ffnRoundUpToMultipleOf
    }

    public func isSLSTM(layer index: Int) -> Bool {
        slstmAt.contains(index)
    }

    public var fixedStateBytes: Int64 {
        let slstmCount = Set(slstmAt).count
        let mlstmCount = numHiddenLayers - slstmCount
        let mlstmPerLayer = Int64(numHeads) * (Int64(qkHeadDim) * Int64(vHeadDim) + Int64(qkHeadDim) + 1) * 4
        let slstmPerLayer = Int64(4 * hiddenSize) * 4
        return Int64(mlstmCount) * mlstmPerLayer + Int64(slstmCount) * slstmPerLayer
    }

    private func validateDimensions(in c: KeyedDecodingContainer<CodingKeys>) throws {
        let baseChecks: [(Bool, CodingKeys, String)] = [
            (vocabSize >= 1, .vocabSize, "xlstm requires vocab_size >= 1"),
            (hiddenSize >= 1, .hiddenSize, "xlstm requires hidden_size >= 1"),
            (numHiddenLayers >= 1, .numHiddenLayers, "xlstm requires num_hidden_layers >= 1"),
            (numHeads >= 1, .numHeads, "xlstm requires num_heads >= 1"),
            (qkDimFactor.isFinite && qkDimFactor > 0, .qkDimFactor, "xlstm requires finite positive qk_dim_factor"),
            (vDimFactor.isFinite && vDimFactor > 0, .vDimFactor, "xlstm requires finite positive v_dim_factor"),
        ]
        try Self.enforce(baseChecks, in: c)
        let slstmDivisible = slstmAt.isEmpty || (hiddenSize >= numHeads && hiddenSize.isMultiple(of: numHeads))
        let derivedChecks: [(Bool, CodingKeys, String)] = [
            (qkDim >= numHeads && qkDim.isMultiple(of: numHeads), .qkDimFactor, "xlstm requires qk_dim divisible by num_heads"),
            (vDim >= numHeads && vDim.isMultiple(of: numHeads), .vDimFactor, "xlstm requires v_dim divisible by num_heads"),
            (slstmDivisible, .numHeads, "xlstm slstm layers require hidden_size divisible by num_heads"),
            (slstmAt.allSatisfy { (0 ..< numHiddenLayers).contains($0) }, .slstmAt, "xlstm slstm_at indices must lie in 0 ..< num_hidden_layers"),
        ]
        try Self.enforce(derivedChecks, in: c)
    }

    private func validateHyperparameters(in c: KeyedDecodingContainer<CodingKeys>) throws {
        let factorOK = ffnProjFactor.isFinite && ffnProjFactor > 0 && ffnRoundUpToMultipleOf >= 1
        let gateOK = gateSoftCap.map { $0.isFinite && $0 > 0 } ?? true
        let logitOK = outputLogitSoftCap.map { $0.isFinite && $0 > 0 } ?? true
        let checks: [(Bool, CodingKeys, String)] = [
            (factorOK, .ffnProjFactor, "xlstm requires positive ffn_proj_factor and ffn_round_up_to_multiple_of"),
            (chunkSize >= 1, .chunkSize, "xlstm requires chunk_size >= 1"),
            (normEps.isFinite && normEps > 0, .normEps, "xlstm requires positive norm_eps"),
            (eps.isFinite && eps > 0, .eps, "xlstm requires positive eps"),
            (gateOK, .gateSoftCap, "xlstm requires positive gate_soft_cap when set"),
            (logitOK, .outputLogitSoftCap, "xlstm requires positive output_logit_soft_cap when set"),
        ]
        try Self.enforce(checks, in: c)
        if ffnHiddenDim < 1 {
            throw DecodingError.dataCorruptedError(forKey: .ffnProjFactor, in: c, debugDescription: "xlstm requires ffn_hidden_dim >= 1")
        }
    }

    private static func enforce(_ checks: [(Bool, CodingKeys, String)], in c: KeyedDecodingContainer<CodingKeys>) throws {
        for (valid, key, message) in checks where !valid {
            throw DecodingError.dataCorruptedError(forKey: key, in: c, debugDescription: message)
        }
    }

    private static func decodeRequiredInt(_ c: KeyedDecodingContainer<CodingKeys>, keys: [CodingKeys]) throws -> Int {
        for key in keys {
            if let value = try c.decodeIfPresent(Int.self, forKey: key) { return value }
        }
        return try c.decode(Int.self, forKey: keys[0])
    }
}
