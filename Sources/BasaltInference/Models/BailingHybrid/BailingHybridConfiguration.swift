import BasaltCore
import Foundation
import MLXLMCommon

public struct BailingHybridConfiguration: Codable, Sendable, Equatable {
    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var tieWordEmbeddings: Bool
    public var layerGroupSize: Int
    public var firstKDenseReplace: Int
    public var convKernelSize: Int
    public var noKDALoRA: Bool
    public var kdaSafeGate: Bool
    public var kdaLowerBound: Float
    public var qLoraRank: Int?
    public var kvLoraRank: Int
    public var qkNopeHeadDim: Int
    public var qkRopeHeadDim: Int
    public var vHeadDim: Int
    public var ropeTheta: Float
    public var ropeInterleave: Bool
    public var ropeScaling: [String: StringOrNumber]?
    public var useQKVBias: Bool
    public var useBias: Bool
    public var useQKNorm: Bool
    public var gatedAttentionGranularity: String?
    public var numExperts: Int
    public var numExpertsPerTok: Int
    public var numSharedExperts: Int
    public var moeIntermediateSize: Int
    public var moeSharedExpertIntermediateSize: Int?
    public var nGroup: Int
    public var topkGroup: Int
    public var normTopkProb: Bool
    public var routedScalingFactor: Float
    public var moeRouterEnableExpertBias: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerGroupSize = "layer_group_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case convKernelSize = "short_conv_kernel_size"
        case noKDALoRA = "no_kda_lora"
        case kdaSafeGate = "kda_safe_gate"
        case kdaLowerBound = "kda_lower_bound"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case ropeTheta = "rope_theta"
        case ropeInterleave = "rope_interleave"
        case ropeScaling = "rope_scaling"
        case useQKVBias = "use_qkv_bias"
        case useBias = "use_bias"
        case useQKNorm = "use_qk_norm"
        case gatedAttentionGranularity = "gated_attention_proj_granularity_type"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
    }

    private enum AliasKeys: String, CodingKey {
        case convKernelSize = "conv_kernel_size"
        case useKDALoRA = "use_kda_lora"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let alias = try decoder.container(keyedBy: AliasKeys.self)
        modelType = try Self.value(c, .modelType, default: BailingHybridModel.modelType)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try Self.value(c, .numKeyValueHeads, default: numAttentionHeads)
        headDim = try Self.value(c, .headDim, default: hiddenSize / max(numAttentionHeads, 1))
        rmsNormEps = try Self.value(c, .rmsNormEps, default: 1e-6)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        maxPositionEmbeddings = try Self.value(c, .maxPositionEmbeddings, default: 131_072)
        tieWordEmbeddings = try Self.value(c, .tieWordEmbeddings, default: false)
        layerGroupSize = try Self.value(c, .layerGroupSize, default: 4)
        firstKDenseReplace = try Self.value(c, .firstKDenseReplace, default: 1)
        convKernelSize = try Self.decodeConvKernel(c, alias)
        noKDALoRA = try Self.decodeNoKDALoRA(c, alias)
        kdaSafeGate = try Self.value(c, .kdaSafeGate, default: true)
        kdaLowerBound = try Self.value(c, .kdaLowerBound, default: -5.0)
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        kvLoraRank = try Self.value(c, .kvLoraRank, default: 512)
        qkNopeHeadDim = try Self.value(c, .qkNopeHeadDim, default: 128)
        qkRopeHeadDim = try Self.value(c, .qkRopeHeadDim, default: 64)
        vHeadDim = try Self.value(c, .vHeadDim, default: 128)
        ropeTheta = try Self.value(c, .ropeTheta, default: 6_000_000.0)
        ropeInterleave = try Self.value(c, .ropeInterleave, default: true)
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        useQKVBias = try Self.value(c, .useQKVBias, default: false)
        useBias = try Self.value(c, .useBias, default: false)
        useQKNorm = try Self.value(c, .useQKNorm, default: false)
        gatedAttentionGranularity = try Self.decodeGranularity(c)
        numExperts = try Self.value(c, .numExperts, default: 0)
        numExpertsPerTok = try Self.value(c, .numExpertsPerTok, default: 8)
        numSharedExperts = try Self.value(c, .numSharedExperts, default: 0)
        moeIntermediateSize = try Self.value(c, .moeIntermediateSize, default: 512)
        moeSharedExpertIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeSharedExpertIntermediateSize)
        nGroup = try Self.value(c, .nGroup, default: 1)
        topkGroup = try Self.value(c, .topkGroup, default: 1)
        normTopkProb = try Self.value(c, .normTopkProb, default: true)
        routedScalingFactor = try Self.value(c, .routedScalingFactor, default: 1.0)
        moeRouterEnableExpertBias = try Self.value(c, .moeRouterEnableExpertBias, default: true)
    }

    private static func value<T: Decodable>(
        _ container: KeyedDecodingContainer<CodingKeys>,
        _ key: CodingKeys,
        default fallback: T
    ) throws -> T {
        try container.decodeIfPresent(T.self, forKey: key) ?? fallback
    }

    private static func decodeConvKernel(
        _ container: KeyedDecodingContainer<CodingKeys>,
        _ alias: KeyedDecodingContainer<AliasKeys>
    ) throws -> Int {
        if let primary = try container.decodeIfPresent(Int.self, forKey: .convKernelSize) {
            return primary
        }
        return try alias.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 4
    }

    private static func decodeNoKDALoRA(
        _ container: KeyedDecodingContainer<CodingKeys>,
        _ alias: KeyedDecodingContainer<AliasKeys>
    ) throws -> Bool {
        if let explicit = try container.decodeIfPresent(Bool.self, forKey: .noKDALoRA) {
            return explicit
        }
        if let useLoRA = try alias.decodeIfPresent(Bool.self, forKey: .useKDALoRA) {
            return !useLoRA
        }
        return true
    }

    private static func decodeGranularity(_ container: KeyedDecodingContainer<CodingKeys>) throws -> String? {
        guard container.contains(.gatedAttentionGranularity) else {
            return "head_wise"
        }
        return try container.decodeIfPresent(String.self, forKey: .gatedAttentionGranularity)
    }
}

extension BailingHybridConfiguration {
    public var qkHeadDim: Int {
        qkNopeHeadDim + qkRopeHeadDim
    }

    public var sharedExpertIntermediateSize: Int {
        (moeSharedExpertIntermediateSize ?? moeIntermediateSize) * numSharedExperts
    }

    public func isMLALayer(_ index: Int) -> Bool {
        let group = max(layerGroupSize, 1)
        let full = (numHiddenLayers / group) * group
        return (index + 1).isMultiple(of: group) || index >= full
    }

    public func isMoELayer(_ index: Int) -> Bool {
        numExperts > 0 && index >= firstKDenseReplace
    }

    public func validated() throws(BasaltError) -> BailingHybridConfiguration {
        try validateCoreGeometry()
        try validateMLAGeometry()
        try validateMoEGeometry()
        return self
    }

    private func validateCoreGeometry() throws(BasaltError) {
        guard vocabSize > 0, hiddenSize > 0, numHiddenLayers > 0, intermediateSize > 0 else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "bailing_hybrid requires positive vocab_size, hidden_size, num_hidden_layers and intermediate_size"
            )
        }
        guard numAttentionHeads > 0, numKeyValueHeads > 0, headDim > 0, layerGroupSize > 0, convKernelSize > 0 else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "bailing_hybrid requires positive head counts, head_dim, layer_group_size and short_conv_kernel_size"
            )
        }
        guard firstKDenseReplace >= 0 else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "bailing_hybrid first_k_dense_replace (\(firstKDenseReplace)) must be non-negative"
            )
        }
    }

    private func validateMLAGeometry() throws(BasaltError) {
        guard kvLoraRank > 0, qkNopeHeadDim >= 0, qkRopeHeadDim > 0, qkRopeHeadDim.isMultiple(of: 2),
              qkHeadDim > 0, vHeadDim > 0
        else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "bailing_hybrid MLA requires positive kv_lora_rank, even positive qk_rope_head_dim and positive v_head_dim"
            )
        }
        if let qLoraRank, qLoraRank <= 0 {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "bailing_hybrid q_lora_rank (\(qLoraRank)) must be positive when set"
            )
        }
        if let granularity = gatedAttentionGranularity, granularity != "head_wise" && granularity != "element_wise" {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "bailing_hybrid unsupported gated_attention_proj_granularity_type '\(granularity)'"
            )
        }
    }

    private func validateMoEGeometry() throws(BasaltError) {
        guard numExperts >= 0, numSharedExperts >= 0 else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "bailing_hybrid num_experts and num_shared_experts must be non-negative"
            )
        }
        guard numExperts > 0 else { return }
        guard moeIntermediateSize > 0, (1 ... numExperts).contains(numExpertsPerTok) else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "bailing_hybrid num_experts_per_tok (\(numExpertsPerTok)) must be in 1...\(numExperts)"
            )
        }
        guard nGroup > 0, numExperts.isMultiple(of: nGroup), (1 ... nGroup).contains(topkGroup) else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "bailing_hybrid n_group (\(nGroup)) must divide num_experts (\(numExperts)) with topk_group (\(topkGroup)) in 1...\(nGroup)"
            )
        }
        let maxSelectable = topkGroup * (numExperts / nGroup)
        guard numExpertsPerTok <= maxSelectable else {
            throw .modelConfigurationInvalid(
                path: modelType,
                reason: "bailing_hybrid num_experts_per_tok (\(numExpertsPerTok)) exceeds topk_group capacity (\(maxSelectable))"
            )
        }
    }
}
