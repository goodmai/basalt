import BasaltCore
import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM

public enum ModelFamilyAliases {
    public static let llmModelTypes: [String] = [
        "lfm3", "lfm-3", "lfm3_moe", "lfm-3-moe",
        "deepseek_v2", "deepseek-v2", "deepseek_v3", "deepseek-v3",
        "mimo", "mimo_v2_flash", "mimo-v2-flash",
        "minimax", "minimax_m1", "minimax-m1", "minimax_m2", "minimax-m2",
        "bitnet", "bitnet_b1_58", "bitnet-b1.58",
        "falcon_h1", "falcon-h1",
        "hunyuan", "hunyuan_mt", "hunyuan-mt", "hunyuan_v1_dense",
    ]

    public static let vlmModelTypes: [String] = [
        "lfm2_vl", "lfm2-vl", "lfm3_vl", "lfm3-vl",
    ]

    public static func registerAll() async {
        let llm = LLMTypeRegistry.shared
        let decoder = JSONDecoder.json5()
        for type in ["lfm3", "lfm-3"] {
            await llm.registerModelType(type) { LFM2Model(try decoder.decode(LFM2Configuration.self, from: $0)) }
        }
        for type in ["lfm3_moe", "lfm-3-moe"] {
            await llm.registerModelType(type) { LFM2MoEModel(try decoder.decode(LFM2MoEConfiguration.self, from: $0)) }
        }
        await llm.registerModelType("deepseek-v2") { DeepseekV2Model(try decoder.decode(DeepseekV2Configuration.self, from: $0)) }
        await llm.registerModelType("deepseek-v3") { DeepseekV3Model(try decoder.decode(DeepseekV3Configuration.self, from: $0)) }
        await llm.registerModelType("mimo-v2-flash") { MiMoV2FlashModel(try decoder.decode(MiMoV2FlashConfiguration.self, from: $0)) }
        for type in ["minimax_m1", "minimax-m1", "minimax_m2", "minimax-m2"] {
            await llm.registerModelType(type) { MiniMaxModel(try decoder.decode(MiniMaxConfiguration.self, from: $0)) }
        }
        for type in ["bitnet_b1_58", "bitnet-b1.58"] {
            await llm.registerModelType(type) { BitnetModel(try decoder.decode(BitnetConfiguration.self, from: $0)) }
        }
        await llm.registerModelType("falcon-h1") { FalconH1Model(try decoder.decode(FalconH1Configuration.self, from: $0)) }
        for type in ["hunyuan", "hunyuan_mt", "hunyuan-mt"] {
            await llm.registerModelType(type) { HunyuanModel(try decoder.decode(HunyuanConfiguration.self, from: $0)) }
        }
        let vlm = VLMTypeRegistry.shared
        for type in ["lfm3_vl", "lfm3-vl"] {
            await vlm.registerModelType(type) { LFM2VL(try decoder.decode(LFM2VLConfiguration.self, from: $0)) }
        }
    }
}

struct ModelCapabilities: Equatable, Sendable {
    enum Feature: String, CaseIterable, Sendable {
        case speculativeLookup = "speculative lookup"
        case mtp = "MTP"
        case batchedDecode = "batched decode"
        case promptCacheReuse = "prompt-cache reuse"
        case cacheBlend = "CacheBlend"
        case kvZones = "KV zones"
        case kvQuantization = "KV quantization"
        case kvHeadSplit = "KV head split"
        case quest = "Quest"
        case grammarJumpForward = "grammar jump-forward"
    }

    let verified: Set<Feature>

    static let unrestricted = ModelCapabilities(verified: Set(Feature.allCases))
    static let conservative = ModelCapabilities(verified: [])

    func allows(_ feature: Feature) -> Bool { verified.contains(feature) }

    func kvBits(_ precision: KVPrecision?) -> Int? {
        allows(.kvQuantization) && precision?.turbo != true && precision?.isKVarN != true ? precision?.uniform.keyBits : nil
    }

    func kvScheme(_ precision: KVPrecision?) -> String? { allows(.kvQuantization) ? precision?.scheme : nil }

    static func of(modelType: String?) -> ModelCapabilities? {
        modelType.flatMap { families[$0.lowercased()] }
    }

    static func refuseUnverifiedKVBits(_ precision: KVPrecision?, modelType: String?, path: String) throws(BasaltError) {
        guard precision != nil, of(modelType: modelType)?.allows(.kvQuantization) != true else { return }
        if modelType?.lowercased() == BailingHybridModel.modelType {
            throw .modelConfigurationInvalid(
                path: path,
                reason: "model_type 'bailing_hybrid' stores an absorbed MLA latent KV cache (kv_lora_rank + qk_rope_head_dim), so --kv-bits cannot apply; serve it without --kv-bits or with --kv-bits auto")
        }
        throw .modelConfigurationInvalid(
            path: path,
            reason: "model_type '\(modelType ?? "none")' has no verified quantized KV cache, so --kv-bits cannot apply; serve it without --kv-bits or with --kv-bits auto")
    }

    static func unverifiedWarning(modelType: String?) -> String {
        "unverified model family '\(modelType ?? "no model_type")': features \(Feature.allCases.map(\.rawValue).joined(separator: ", ")) stay off"
    }

    static let families: [String: ModelCapabilities] = {
        let all = Set(Feature.allCases)
        let attention = ModelCapabilities(verified: all.subtracting([.mtp]))
        let qwen35 = ModelCapabilities(verified: all.subtracting([.cacheBlend]))
        let recurrent = ModelCapabilities(verified: all.subtracting([.mtp, .cacheBlend]))
        let nemotron = ModelCapabilities(verified: all.subtracting([.cacheBlend]))
        let gemma3 = ModelCapabilities(verified: all.subtracting([.mtp, .cacheBlend, .batchedDecode]))
        let gemma4 = ModelCapabilities(verified: all.subtracting([.mtp, .quest]))
        let flint = ModelCapabilities(verified: [.speculativeLookup, .batchedDecode, .promptCacheReuse, .kvQuantization, .grammarJumpForward])
        let falconH1 = ModelCapabilities(verified: [.speculativeLookup, .promptCacheReuse, .kvQuantization, .grammarJumpForward])
        let bailingHybrid = ModelCapabilities(verified: [.batchedDecode, .promptCacheReuse, .grammarJumpForward])
        let table: [([String], ModelCapabilities)] = [
            (["llama", "qwen3", "qwen3_moe", "mimo", "minimax", "minimax_m1", "minimax-m1", "minimax_m2", "minimax-m2", "hunyuan", "hunyuan_mt", "hunyuan-mt", "hunyuan_v1_dense"], attention),
            (["qwen3_5", "qwen3_5_text", "qwen3_6", "qwen3.6", "qwen3_8", "qwen3.8", PrismHadamard.modelType], qwen35),
            (["qwen3_5_moe", "qwen3_5_moe_text", "qwen3_6_moe", "qwen3_8_moe", "qwen3.8_moe"], qwen35),
            ([NemotronLightningModel.modelType, NemotronLightningModel.aliasModelType, "nemotron-lightning"], nemotron),
            ([NemotronTranscribeModel.modelType, "lfm2", "lfm2_moe", "lfm3", "lfm-3", "lfm3_moe", "lfm-3-moe", "mimo_v2_flash", "mimo-v2-flash", RicursiveModel.modelType, RicursiveModel.aliasModelType, MuseGlimmerTextModel.multimodalModelType, MuseGlimmerTextModel.modelType, "muse-glimmer"], recurrent),
            (["gemma3", "gemma3_text"], gemma3),
            (["gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"], gemma4),
            ([FlintModel.modelType], flint),
            ([OuroModel.modelType], flint),
            (["falcon_h1", "falcon-h1"], falconH1),
            ([BailingHybridModel.modelType], bailingHybrid),
            (Inkling2Model.modelTypes, ModelCapabilities(verified: [.speculativeLookup, .mtp, .promptCacheReuse, .grammarJumpForward])),
            ([PaddleOCRVL.modelType], ModelCapabilities(verified: all.subtracting([.mtp, .cacheBlend, .kvQuantization, .kvZones, .kvHeadSplit, .quest]))),
            ([DeepSeekOCR.modelType, DeepSeekOCR2.modelType], ModelCapabilities(verified: all.subtracting([.mtp, .cacheBlend, .kvQuantization, .kvZones, .kvHeadSplit, .quest]))),
        ]
        return Dictionary(uniqueKeysWithValues: table.flatMap { types, capabilities in types.map { ($0, capabilities) } })
    }()
}
