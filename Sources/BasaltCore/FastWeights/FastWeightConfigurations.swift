import Foundation

private enum ConfigDecode {
    static func first<K: CodingKey, T: Decodable>(
        _ c: KeyedDecodingContainer<K>,
        _ k1: K,
        _ k2: K? = nil,
        _ k3: K? = nil
    ) throws -> T? {
        if let v = try c.decodeIfPresent(T.self, forKey: k1) { return v }
        if let k2, let v = try c.decodeIfPresent(T.self, forKey: k2) { return v }
        if let k3, let v = try c.decodeIfPresent(T.self, forKey: k3) { return v }
        return nil
    }

    static func value<K: CodingKey, T: Decodable>(
        _ c: KeyedDecodingContainer<K>,
        _ k1: K,
        fallback: T
    ) throws -> T {
        try c.decodeIfPresent(T.self, forKey: k1) ?? fallback
    }

    static func alias<K: CodingKey, T: Decodable>(
        _ c: KeyedDecodingContainer<K>,
        _ k1: K,
        _ k2: K,
        fallback: T
    ) throws -> T {
        try first(c, k1, k2) ?? fallback
    }

    static func required<K: CodingKey, T: Decodable>(
        _ c: KeyedDecodingContainer<K>,
        _ k1: K,
        _ k2: K? = nil,
        _ k3: K? = nil
    ) throws -> T {
        if let v: T = try first(c, k1, k2, k3) { return v }
        return try c.decode(T.self, forKey: k1)
    }

    static func verify<K: CodingKey>(
        _ checks: [(Bool, K, String)],
        in c: KeyedDecodingContainer<K>
    ) throws {
        for (ok, key, message) in checks where !ok {
            throw DecodingError.dataCorruptedError(forKey: key, in: c, debugDescription: message)
        }
    }

    static func exactProduct(_ a: Int, _ b: Int, equals target: Int) -> Bool {
        let p = a.multipliedReportingOverflow(by: b)
        return a >= 1 && b >= 1 && !p.overflow && p.partialValue == target
    }
}

public struct TTTConfiguration: Decodable, Sendable, Equatable {
    public enum Variant: String, Decodable, Sendable, Equatable, CaseIterable {
        case linear
        case mlp
    }

    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let headDim: Int
    public let variant: Variant
    public let miniBatchSize: Int
    public let tttBaseLR: Float
    public let convKernel: Int
    public let useGate: Bool
    public let shareQK: Bool
    public let mlpExpansion: Int
    public let rmsNormEps: Float
    public let tttNormEps: Float
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size", hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers", numAttentionHeads = "num_attention_heads", numHeads = "num_heads"
        case headDim = "head_dim", variant = "ttt_layer_type", miniBatchSize = "mini_batch_size"
        case tttBaseLR = "ttt_base_lr", convKernel = "conv_kernel", useGate = "use_gate", shareQK = "share_qk"
        case mlpExpansion = "ttt_mlp_expansion", rmsNormEps = "rms_norm_eps", tttNormEps = "ttt_norm_eps"
        case ropeTheta = "rope_theta", tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try ConfigDecode.value(c, .intermediateSize, fallback: hiddenSize * 4)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try ConfigDecode.required(c, .numAttentionHeads, .numHeads)
        let defaultHeadDim = numAttentionHeads > 0 ? hiddenSize / numAttentionHeads : 0
        headDim = try ConfigDecode.value(c, .headDim, fallback: defaultHeadDim)
        variant = try ConfigDecode.value(c, .variant, fallback: .linear)
        miniBatchSize = try ConfigDecode.value(c, .miniBatchSize, fallback: 16)
        tttBaseLR = try ConfigDecode.value(c, .tttBaseLR, fallback: 1.0)
        convKernel = try ConfigDecode.value(c, .convKernel, fallback: 4)
        useGate = try ConfigDecode.value(c, .useGate, fallback: true)
        shareQK = try ConfigDecode.value(c, .shareQK, fallback: false)
        mlpExpansion = try ConfigDecode.value(c, .mlpExpansion, fallback: 4)
        rmsNormEps = try ConfigDecode.value(c, .rmsNormEps, fallback: 1e-6)
        tttNormEps = try ConfigDecode.value(c, .tttNormEps, fallback: 1e-6)
        ropeTheta = try ConfigDecode.value(c, .ropeTheta, fallback: 10_000.0)
        tieWordEmbeddings = try ConfigDecode.value(c, .tieWordEmbeddings, fallback: false)
        try validate(in: c)
    }

    private func validate(in c: KeyedDecodingContainer<CodingKeys>) throws {
        try ConfigDecode.verify([
            (vocabSize >= 1, .vocabSize, "ttt requires vocab_size >= 1"),
            (hiddenSize >= 1, .hiddenSize, "ttt requires hidden_size >= 1"),
            (intermediateSize >= 1, .intermediateSize, "ttt requires intermediate_size >= 1"),
            (numHiddenLayers >= 1, .numHiddenLayers, "ttt requires num_hidden_layers >= 1"),
            (ConfigDecode.exactProduct(numAttentionHeads, headDim, equals: hiddenSize), .headDim, "ttt requires num_attention_heads * head_dim == hidden_size"),
            (miniBatchSize >= 1, .miniBatchSize, "ttt requires mini_batch_size >= 1"),
            (convKernel >= 0, .convKernel, "ttt requires conv_kernel >= 0"),
            (mlpExpansion >= 1, .mlpExpansion, "ttt requires ttt_mlp_expansion >= 1"),
            (tttBaseLR >= 0 && tttBaseLR.isFinite, .tttBaseLR, "ttt requires finite non-negative ttt_base_lr"),
            (rmsNormEps > 0, .rmsNormEps, "ttt requires rms_norm_eps > 0"),
            (tttNormEps > 0, .tttNormEps, "ttt requires ttt_norm_eps > 0"),
            (ropeTheta > 0, .ropeTheta, "ttt requires rope_theta > 0"),
        ], in: c)
    }
}

public struct TitansConfiguration: Decodable, Sendable, Equatable {
    public enum Variant: String, Decodable, Sendable, Equatable, CaseIterable {
        case mac
        case mag
        case mal
    }

    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let variant: Variant
    public let memoryDepth: Int
    public let memoryHiddenMultiplier: Int
    public let numPersistentTokens: Int
    public let chunkSize: Int
    public let windowSize: Int
    public let convKernel: Int
    public let baseLR: Float
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool

    public var memoryHiddenDim: Int { headDim * memoryHiddenMultiplier }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", vocabSize = "vocab_size", hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size", numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads", numHeads = "num_heads"
        case numKeyValueHeads = "num_key_value_heads", headDim = "head_dim"
        case titansVariant = "titans_variant", variant, memoryDepth = "memory_depth"
        case memoryHiddenMultiplier = "memory_hidden_multiplier", numPersistentTokens = "num_persistent_tokens"
        case chunkSize = "chunk_size", windowSize = "window_size", slidingWindow = "sliding_window"
        case convKernel = "conv_kernel", baseLR = "base_lr", memoryLR = "memory_lr"
        case rmsNormEps = "rms_norm_eps", ropeTheta = "rope_theta", tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try ConfigDecode.value(c, .intermediateSize, fallback: hiddenSize * 4)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try ConfigDecode.required(c, .numAttentionHeads, .numHeads)
        numKeyValueHeads = try ConfigDecode.value(c, .numKeyValueHeads, fallback: numAttentionHeads)
        let defaultHeadDim = numAttentionHeads > 0 ? hiddenSize / numAttentionHeads : 0
        headDim = try ConfigDecode.value(c, .headDim, fallback: defaultHeadDim)
        variant = try Self.resolveVariant(c)
        memoryDepth = try ConfigDecode.value(c, .memoryDepth, fallback: 2)
        memoryHiddenMultiplier = try ConfigDecode.value(c, .memoryHiddenMultiplier, fallback: 2)
        numPersistentTokens = try ConfigDecode.value(c, .numPersistentTokens, fallback: 4)
        chunkSize = try ConfigDecode.value(c, .chunkSize, fallback: 16)
        windowSize = try ConfigDecode.alias(c, .windowSize, .slidingWindow, fallback: 64)
        convKernel = try ConfigDecode.value(c, .convKernel, fallback: 4)
        baseLR = try ConfigDecode.alias(c, .baseLR, .memoryLR, fallback: 0.1)
        rmsNormEps = try ConfigDecode.value(c, .rmsNormEps, fallback: 1e-6)
        ropeTheta = try ConfigDecode.value(c, .ropeTheta, fallback: 10_000.0)
        tieWordEmbeddings = try ConfigDecode.value(c, .tieWordEmbeddings, fallback: false)
        try validate(in: c)
    }

    private static func resolveVariant(_ c: KeyedDecodingContainer<CodingKeys>) throws -> Variant {
        if let explicit: Variant = try ConfigDecode.first(c, .titansVariant, .variant) { return explicit }
        switch (try c.decodeIfPresent(String.self, forKey: .modelType))?.lowercased() {
        case "titans_mac": return .mac
        case "titans_mal": return .mal
        default: return .mag
        }
    }

    private func validate(in c: KeyedDecodingContainer<CodingKeys>) throws {
        let gqaOK = numKeyValueHeads >= 1 && numAttentionHeads.isMultiple(of: numKeyValueHeads)
        try ConfigDecode.verify([
            (vocabSize >= 1, .vocabSize, "titans requires vocab_size >= 1"),
            (hiddenSize >= 1, .hiddenSize, "titans requires hidden_size >= 1"),
            (intermediateSize >= 1, .intermediateSize, "titans requires intermediate_size >= 1"),
            (numHiddenLayers >= 1, .numHiddenLayers, "titans requires num_hidden_layers >= 1"),
            (ConfigDecode.exactProduct(numAttentionHeads, headDim, equals: hiddenSize), .headDim, "titans requires num_attention_heads * head_dim == hidden_size"),
            (gqaOK, .numKeyValueHeads, "titans requires num_attention_heads divisible by num_key_value_heads"),
            (memoryDepth == 1 || memoryDepth == 2, .memoryDepth, "titans requires memory_depth in {1, 2}"),
            (memoryHiddenMultiplier >= 1, .memoryHiddenMultiplier, "titans requires memory_hidden_multiplier >= 1"),
            (numPersistentTokens >= 0, .numPersistentTokens, "titans requires num_persistent_tokens >= 0"),
            (chunkSize >= 1, .chunkSize, "titans requires chunk_size >= 1"),
            (windowSize >= 1, .windowSize, "titans requires window_size >= 1"),
            (convKernel >= 0, .convKernel, "titans requires conv_kernel >= 0"),
            (baseLR >= 0 && baseLR.isFinite, .baseLR, "titans requires finite non-negative base_lr"),
            (rmsNormEps > 0, .rmsNormEps, "titans requires rms_norm_eps > 0"),
            (ropeTheta > 0, .ropeTheta, "titans requires rope_theta > 0"),
        ], in: c)
    }
}

public struct LaCTConfiguration: Decodable, Sendable, Equatable {
    public enum Objective: String, Decodable, Sendable, Equatable, CaseIterable {
        case dotProduct = "dot_product"
        case l2 = "l2"
    }

    public enum UpdateRule: String, Decodable, Sendable, Equatable, CaseIterable {
        case sgd
        case momentum
        case muon
    }

    public enum MemoryMode: String, Decodable, Sendable, Equatable, CaseIterable {
        case swiglu
        case linear
    }

    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let numFastWeightHeads: Int
    public let fastWeightHeadDim: Int
    public let fastWeightInterMulti: Float
    public let chunkSize: Int
    public let windowSize: Int
    public let objective: Objective
    public let updateRule: UpdateRule
    public let memoryMode: MemoryMode
    public let useWeightNorm: Bool
    public let muonSteps: Int
    public let momentumBeta: Float
    public let baseLR: Float
    public let convKernel: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool

    public var fastWeightHiddenDim: Int {
        max(1, Int((Float(fastWeightHeadDim) * fastWeightInterMulti).rounded()))
    }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size", hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
        case hiddenRatio = "hidden_ratio", numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads", numAttnHeads = "num_attn_heads", numHeads = "num_heads"
        case numKeyValueHeads = "num_key_value_heads", headDim = "head_dim"
        case numFastWeightHeads = "num_fw_heads", numLactHeads = "num_lact_heads", fastWeightHeadDim = "fw_head_dim"
        case fastWeightInterMulti = "inter_multi", lactChunkSize = "lact_chunk_size", chunkSize = "chunk_size"
        case windowSize = "window_size", slidingWindow = "sliding_window", objective
        case updateRule = "update_rule", useMuon = "use_muon", useMomentum = "use_momentum"
        case memoryMode = "memory_mode", useWeightNorm = "use_weight_norm", fwWeightNorm = "fw_weight_norm"
        case muonSteps = "muon_steps", momentumBeta = "momentum_beta", baseLR = "base_lr", lrScale = "lr_scale"
        case convKernel = "conv_kernel", rmsNormEps = "rms_norm_eps", normEps = "norm_eps"
        case ropeTheta = "rope_theta", tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        let ratio = try ConfigDecode.value(c, .hiddenRatio, fallback: Float(4.0))
        intermediateSize = try ConfigDecode.value(c, .intermediateSize, fallback: Int((Float(hiddenSize) * ratio).rounded()))
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try ConfigDecode.required(c, .numAttentionHeads, .numAttnHeads, .numHeads)
        numKeyValueHeads = try ConfigDecode.value(c, .numKeyValueHeads, fallback: numAttentionHeads)
        let defaultHeadDim = numAttentionHeads > 0 ? hiddenSize / numAttentionHeads : 0
        headDim = try ConfigDecode.value(c, .headDim, fallback: defaultHeadDim)
        numFastWeightHeads = try ConfigDecode.alias(c, .numFastWeightHeads, .numLactHeads, fallback: numAttentionHeads)
        let defaultFWDim = numFastWeightHeads > 0 ? hiddenSize / numFastWeightHeads : 0
        fastWeightHeadDim = try ConfigDecode.value(c, .fastWeightHeadDim, fallback: defaultFWDim)
        fastWeightInterMulti = try ConfigDecode.value(c, .fastWeightInterMulti, fallback: 1.0)
        chunkSize = try ConfigDecode.alias(c, .lactChunkSize, .chunkSize, fallback: 2048)
        windowSize = try ConfigDecode.alias(c, .windowSize, .slidingWindow, fallback: 2048)
        objective = try ConfigDecode.value(c, .objective, fallback: .dotProduct)
        updateRule = try Self.resolveUpdateRule(c)
        memoryMode = try ConfigDecode.value(c, .memoryMode, fallback: .swiglu)
        useWeightNorm = try ConfigDecode.alias(c, .useWeightNorm, .fwWeightNorm, fallback: true)
        muonSteps = try ConfigDecode.value(c, .muonSteps, fallback: 5)
        momentumBeta = try ConfigDecode.value(c, .momentumBeta, fallback: 0.9)
        baseLR = try ConfigDecode.alias(c, .baseLR, .lrScale, fallback: 0.1)
        convKernel = try ConfigDecode.value(c, .convKernel, fallback: 0)
        rmsNormEps = try ConfigDecode.alias(c, .rmsNormEps, .normEps, fallback: 1e-6)
        ropeTheta = try ConfigDecode.value(c, .ropeTheta, fallback: 10_000.0)
        tieWordEmbeddings = try ConfigDecode.value(c, .tieWordEmbeddings, fallback: false)
        try validate(in: c)
    }

    private static func resolveUpdateRule(_ c: KeyedDecodingContainer<CodingKeys>) throws -> UpdateRule {
        if let explicit = try c.decodeIfPresent(UpdateRule.self, forKey: .updateRule) { return explicit }
        let useMuon = try c.decodeIfPresent(Bool.self, forKey: .useMuon)
        let useMomentum = try c.decodeIfPresent(Bool.self, forKey: .useMomentum)
        if useMuon == true { return .muon }
        if useMuon == false { return useMomentum == true ? .momentum : .sgd }
        return useMomentum == false ? .sgd : .muon
    }

    private func validate(in c: KeyedDecodingContainer<CodingKeys>) throws {
        let gqaOK = numKeyValueHeads >= 1 && numAttentionHeads.isMultiple(of: numKeyValueHeads)
        let fwProd = numFastWeightHeads.multipliedReportingOverflow(by: fastWeightHeadDim)
        let fwOK = numFastWeightHeads >= 1 && fastWeightHeadDim >= 1 && !fwProd.overflow && fwProd.partialValue >= 1
        try ConfigDecode.verify([
            (vocabSize >= 1, .vocabSize, "lact requires vocab_size >= 1"),
            (hiddenSize >= 1, .hiddenSize, "lact requires hidden_size >= 1"),
            (intermediateSize >= 1, .intermediateSize, "lact requires intermediate_size >= 1"),
            (numHiddenLayers >= 1, .numHiddenLayers, "lact requires num_hidden_layers >= 1"),
            (ConfigDecode.exactProduct(numAttentionHeads, headDim, equals: hiddenSize), .headDim, "lact requires num_attention_heads * head_dim == hidden_size"),
            (gqaOK, .numKeyValueHeads, "lact requires num_attention_heads divisible by num_key_value_heads"),
            (fwOK, .fastWeightHeadDim, "lact requires positive fast-weight head dimensions"),
            (fastWeightInterMulti > 0 && fastWeightInterMulti.isFinite, .fastWeightInterMulti, "lact requires positive finite inter_multi"),
            (chunkSize >= 1, .chunkSize, "lact requires chunk_size >= 1"),
            (windowSize >= 1, .windowSize, "lact requires window_size >= 1"),
            (convKernel >= 0, .convKernel, "lact requires conv_kernel >= 0"),
            (muonSteps >= 1, .muonSteps, "lact requires muon_steps >= 1"),
            (momentumBeta >= 0 && momentumBeta <= 1, .momentumBeta, "lact requires momentum_beta in [0, 1]"),
            (baseLR >= 0 && baseLR.isFinite, .baseLR, "lact requires finite non-negative base_lr"),
            (rmsNormEps > 0, .rmsNormEps, "lact requires rms_norm_eps > 0"),
            (ropeTheta > 0, .ropeTheta, "lact requires rope_theta > 0"),
        ], in: c)
    }
}
