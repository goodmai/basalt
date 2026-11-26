import Foundation

public struct GatedDeltaNet2Configuration: Codable, Sendable, Equatable {
    public static let modelType = "gdn2"
    public static let supportedModelTypes = ["gdn2", "gated_deltanet_2", "gated-deltanet-2"]

    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numHeads: Int
    public var numVHeads: Int
    public var headDim: Int
    public var headVDim: Int
    public var allowNegEigval: Bool
    public var betaGuard: Bool
    public var useShortConv: Bool
    public var convKernelSize: Int
    public var chunkSize: Int
    public var rmsNormEps: Float
    public var tieWordEmbeddings: Bool
    public var attentionLayerIndices: [Int]
    public var slidingAttentionLayerIndices: [Int]
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var attentionHeadDim: Int
    public var slidingWindow: Int?
    public var ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_heads"
        case numVHeads = "num_v_heads"
        case headDim = "head_dim"
        case headVDim = "head_v_dim"
        case allowNegEigval = "allow_neg_eigval"
        case betaGuard = "beta_guard"
        case useShortConv = "use_short_conv"
        case convKernelSize = "conv_kernel_size"
        case chunkSize = "chunk_size"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionLayerIndices = "attention_layer_indices"
        case slidingAttentionLayerIndices = "sliding_attention_layer_indices"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case attentionHeadDim = "attention_head_dim"
        case slidingWindow = "sliding_window"
        case ropeTheta = "rope_theta"
    }

    private enum AliasKeys: String, CodingKey {
        case nEmbd = "n_embd"
        case nLayer = "n_layer"
        case nHead = "n_head"
        case numKeyHeads = "num_key_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case numValueHeads = "num_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case valueHeadDim = "value_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case expandV = "expand_v"
        case convSize = "conv_size"
        case shortConvKernelSize = "short_conv_kernel_size"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case normEps = "norm_eps"
        case attnLayers = "attn_layers"
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case slidingWindowSize = "sliding_window_size"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let a = try decoder.container(keyedBy: AliasKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? Self.modelType
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try Self.decodeRequiredInt(c, .hiddenSize, a, .nEmbd)
        numHiddenLayers = try Self.decodeRequiredInt(c, .numHiddenLayers, a, .nLayer)
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? (hiddenSize * 4)

        let primaryHeads = try Self.firstInt(c, [.numHeads], a, [.numKeyHeads, .linearNumKeyHeads, .nHead])
        let decodedHeads = try primaryHeads ?? c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 4
        numHeads = decodedHeads
        numVHeads = try Self.firstInt(c, [.numVHeads], a, [.numValueHeads, .linearNumValueHeads]) ?? decodedHeads
        let decodedHeadDim = try Self.firstInt(c, [.headDim], a, [.linearKeyHeadDim])
            ?? (decodedHeads > 0 ? hiddenSize / decodedHeads : 0)
        headDim = decodedHeadDim
        let expandV = try a.decodeIfPresent(Float.self, forKey: .expandV) ?? 1.0
        let defaultVDim = max(1, Int((Float(decodedHeadDim) * expandV).rounded()))
        headVDim = try Self.firstInt(c, [.headVDim], a, [.valueHeadDim, .linearValueHeadDim]) ?? defaultVDim

        allowNegEigval = try c.decodeIfPresent(Bool.self, forKey: .allowNegEigval) ?? false
        betaGuard = try c.decodeIfPresent(Bool.self, forKey: .betaGuard) ?? false
        useShortConv = try c.decodeIfPresent(Bool.self, forKey: .useShortConv) ?? true
        convKernelSize = try Self.firstInt(c, [.convKernelSize], a, [.convSize, .shortConvKernelSize, .linearConvKernelDim]) ?? 4
        chunkSize = try c.decodeIfPresent(Int.self, forKey: .chunkSize) ?? 64
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps)
            ?? a.decodeIfPresent(Float.self, forKey: .normEps)
            ?? 1e-5
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        let resolvedLayers = try Self.resolveAttentionLayers(c, a, layers: numHiddenLayers)
        attentionLayerIndices = resolvedLayers.attention
        slidingAttentionLayerIndices = resolvedLayers.sliding
        let attnHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? decodedHeads
        numAttentionHeads = attnHeads
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? attnHeads
        attentionHeadDim = try c.decodeIfPresent(Int.self, forKey: .attentionHeadDim)
            ?? (attnHeads > 0 ? hiddenSize / attnHeads : decodedHeadDim)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
            ?? a.decodeIfPresent(Int.self, forKey: .slidingWindowSize)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
    }

    private static func firstInt(
        _ primary: KeyedDecodingContainer<CodingKeys>,
        _ primaryKeys: [CodingKeys],
        _ alias: KeyedDecodingContainer<AliasKeys>,
        _ aliasKeys: [AliasKeys]
    ) throws -> Int? {
        for key in primaryKeys {
            if let value = try primary.decodeIfPresent(Int.self, forKey: key) { return value }
        }
        for key in aliasKeys {
            if let value = try alias.decodeIfPresent(Int.self, forKey: key) { return value }
        }
        return nil
    }

    private static func decodeRequiredInt(
        _ primary: KeyedDecodingContainer<CodingKeys>,
        _ primaryKey: CodingKeys,
        _ alias: KeyedDecodingContainer<AliasKeys>,
        _ aliasKey: AliasKeys
    ) throws -> Int {
        if let value = try primary.decodeIfPresent(Int.self, forKey: primaryKey) { return value }
        if let fallback = try alias.decodeIfPresent(Int.self, forKey: aliasKey) { return fallback }
        return try primary.decode(Int.self, forKey: primaryKey)
    }

    private static func resolveAttentionLayers(
        _ c: KeyedDecodingContainer<CodingKeys>,
        _ a: KeyedDecodingContainer<AliasKeys>,
        layers: Int
    ) throws -> (attention: [Int], sliding: [Int]) {
        if let types = try a.decodeIfPresent([String].self, forKey: .layerTypes), !types.isEmpty {
            let sliding = types.enumerated().compactMap { $0.element.contains("sliding") ? $0.offset : nil }
            let attention = types.enumerated().compactMap { idx, kind -> Int? in
                kind.contains("sliding") || kind == "full_attention" || kind == "attention" ? idx : nil
            }
            return (attention, sliding)
        }
        let explicitSliding = try c.decodeIfPresent([Int].self, forKey: .slidingAttentionLayerIndices) ?? []
        if let explicit = try c.decodeIfPresent([Int].self, forKey: .attentionLayerIndices)
            ?? a.decodeIfPresent([Int].self, forKey: .attnLayers)
        {
            return (Array(Set(explicit + explicitSliding)).sorted(), explicitSliding.sorted())
        }
        if let interval = try a.decodeIfPresent(Int.self, forKey: .fullAttentionInterval), interval > 0 {
            let full = (0 ..< layers).filter { ($0 + 1).isMultiple(of: interval) }
            return (full, explicitSliding.sorted())
        }
        return (explicitSliding.sorted(), explicitSliding.sorted())
    }
}

extension GatedDeltaNet2Configuration {
    public var qkProjDim: Int { numHeads * headDim }
    public var vProjDim: Int { numVHeads * headVDim }

    public func isAttentionLayer(_ index: Int) -> Bool {
        attentionLayerIndices.contains(index)
    }

    public func isSlidingAttentionLayer(_ index: Int) -> Bool {
        slidingAttentionLayerIndices.contains(index)
    }

    public var attentionLayerCount: Int {
        (0 ..< max(0, numHiddenLayers)).filter { isAttentionLayer($0) }.count
    }

    public var linearLayerCount: Int {
        max(0, numHiddenLayers - attentionLayerCount)
    }

    public var recurrentStateBytes: ByteCount {
        recurrentStateFootprint().bytes
    }

    public func recurrentStateFootprint(batchSize: Int = 1, activationBytes: Int = 4) -> RecurrentStateFootprint {
        let b = max(0, batchSize)
        let convHistory = useShortConv ? max(convKernelSize - 1, 0) : 0
        let convBytes = Int64(convHistory * (2 * qkProjDim + vProjDim) * max(0, activationBytes))
        let stateBytes = Int64(numVHeads * headVDim * headDim * 4)
        let bytesPerLayer = Int64(b) * (convBytes + stateBytes)
        return RecurrentStateFootprint(recurrentLayers: linearLayerCount, bytesPerLayer: ByteCount(bytesPerLayer))
    }

    public func validated() throws(BasaltError) -> GatedDeltaNet2Configuration {
        try validateCoreDims()
        try validateHybridAttention()
        return self
    }

    private func validateCoreDims() throws(BasaltError) {
        guard vocabSize > 0, hiddenSize > 0, numHiddenLayers > 0, intermediateSize > 0 else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "gdn2 requires positive vocab_size, hidden_size, num_hidden_layers and intermediate_size"
            )
        }
        guard numHeads > 0, numVHeads >= numHeads, numVHeads.isMultiple(of: numHeads), headDim > 0, headVDim > 0 else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "gdn2 requires num_v_heads (\(numVHeads)) divisible by num_heads (\(numHeads)) and positive head_dim/head_v_dim"
            )
        }
        guard chunkSize >= 1, rmsNormEps > 0, !useShortConv || convKernelSize >= 1 else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "gdn2 requires chunk_size >= 1, positive rms_norm_eps and conv_kernel_size >= 1 when use_short_conv is true"
            )
        }
    }

    private func validateHybridAttention() throws(BasaltError) {
        let validRange = 0 ..< numHiddenLayers
        guard attentionLayerIndices.allSatisfy({ validRange.contains($0) }),
              slidingAttentionLayerIndices.allSatisfy({ validRange.contains($0) })
        else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "gdn2 attention layer indices must fall within 0..<\(numHiddenLayers)"
            )
        }
        guard !attentionLayerIndices.isEmpty else { return }
        guard numAttentionHeads > 0, numKeyValueHeads > 0,
              numAttentionHeads.isMultiple(of: numKeyValueHeads),
              attentionHeadDim > 0, attentionHeadDim.isMultiple(of: 2)
        else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "gdn2 hybrid attention requires num_attention_heads divisible by num_key_value_heads and even attention_head_dim"
            )
        }
        if !slidingAttentionLayerIndices.isEmpty {
            guard let window = slidingWindow, window > 0 else {
                throw .modelConfigurationInvalid(
                    path: modelType,
                    reason: "gdn2 sliding attention layers require positive sliding_window"
                )
            }
        }
    }
}
