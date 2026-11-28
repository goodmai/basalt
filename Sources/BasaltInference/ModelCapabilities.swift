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
        await registerFastWeightModels()
    }

    private static func registerFastWeightModels() async {
        let registry = LLMTypeRegistry.shared
        await registry.registerModelType(TTTModel.modelType) { data in try TTTModel(configurationData: data) }
        for type in TitansModel.modelTypes {
            await registry.registerModelType(type) { data in try TitansModel(configurationData: data) }
        }
        for type in LaCTModel.modelTypes {
            await registry.registerModelType(type) { data in try LaCTModel(configurationData: data) }
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
        allows(.kvQuantization) && precision?.turbo != true && precision?.isKVarN != true && precision?.isGaugeRoPE != true
            ? precision?.uniform.keyBits : nil
    }

    func kvScheme(_ precision: KVPrecision?) -> String? { allows(.kvQuantization) ? precision?.scheme : nil }

    static func of(modelType: String?) -> ModelCapabilities? {
        modelType.flatMap { families[$0.lowercased()] }
    }

    static let diffusionModelTypes: Set<String> = [LLaDAModel.modelType]

    static func refuseDiffusionWithoutFlag(dllm: Bool, modelType: String?, path: String) throws(BasaltError) {
        guard !dllm, let modelType, diffusionModelTypes.contains(modelType.lowercased()) else { return }
        throw .modelConfigurationInvalid(
            path: path,
            reason: "model_type '\(modelType)' is a masked-diffusion language model and cannot decode token by token; serve it with --experimental-dllm (#819)")
    }

    static func refuseIncompleteCheckpoint(at directory: URL, modelType: String?, path: String) throws(BasaltError) {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        if !files.contains(where: { $0.hasSuffix(".safetensors") }), files.contains(where: { $0.hasSuffix(".npz") }) {
            throw .modelConfigurationInvalid(path: path, reason: "the weights are an .npz archive (the old mlx-examples layout) and basalt reads only safetensors; use a safetensors build of this model or convert the archive")
        }
        guard modelType == WhisperModel.modelType, !files.contains("tokenizer.json") else { return }
        let source = WhisperTokenizerSource.repository(config: ModelCache.config(at: directory)) ?? "the matching openai/whisper-* repository"
        throw .modelConfigurationInvalid(path: path, reason: "this Whisper checkpoint ships no tokenizer.json; download the model again with basalt models download, which fetches it from \(source), or copy it into the folder")
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

    static let gaugeRoPEModelTypes: Set<String> = [
        "qwen3_5", "qwen3_5_text", "qwen3_6", "qwen3.6", "qwen3_8", "qwen3.8",
        "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_6_moe", "qwen3_8_moe", "qwen3.8_moe",
    ]

    static func gaugeRoPELayout(
        _ precision: KVPrecision?, config: [String: Any]?, ropeYarn: Double?, path: String
    ) throws(BasaltError) -> GaugeRoPELayout? {
        guard precision?.isGaugeRoPE == true else { return nil }
        let modelType = (config?["model_type"] as? String)?.lowercased() ?? "none"
        guard gaugeRoPEModelTypes.contains(modelType) else {
            let why = modelType.hasPrefix("gemma4")
                ? "Gemma 4 rotates only part of each global-layer head with proportional RoPE (p-RoPE) and attends over the whole cache itself, so the per-frequency latent basis does not apply"
                : "the per-frequency latent basis is verified only on Qwen 3.5, 3.6 and 3.8 (qwen3_5)"
            throw .modelConfigurationInvalid(
                path: path, reason: "model_type '\(modelType)' cannot serve --kv-bits gaugerope (experimental, #647): \(why); serve it without --kv-bits gaugerope")
        }
        guard ropeYarn == nil, let layout = config.flatMap(GaugeRoPELayout.init(config:)) else {
            throw .modelConfigurationInvalid(
                path: path,
                reason: "--kv-bits gaugerope inverts plain RoPE (rope_type default with rope_theta) to build its position-independent code, and this checkpoint's RoPE is \(ropeYarn == nil ? "not plain" : "staged with --rope-yarn"); serve it without --kv-bits gaugerope (#647)")
        }
        return layout
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
        let rwkv7 = ModelCapabilities(verified: [.promptCacheReuse, .grammarJumpForward])
        let hyperConnection = ModelCapabilities(verified: [.speculativeLookup, .batchedDecode, .promptCacheReuse, .kvQuantization, .quest, .grammarJumpForward])
        let mamba3 = ModelCapabilities(verified: [.promptCacheReuse, .grammarJumpForward])
        let gdn2 = ModelCapabilities(verified: [.batchedDecode, .promptCacheReuse, .grammarJumpForward])
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
            ([HuginnModel.modelType], flint),
            ([PLTModel.modelType], ModelCapabilities(verified: [.speculativeLookup, .promptCacheReuse, .grammarJumpForward])),
            (HyperConnectionModel.modelTypes, hyperConnection),
            (["falcon_h1", "falcon-h1"], falconH1),
            ([BailingHybridModel.modelType], bailingHybrid),
            (["deepseek_v3", "deepseek-v3"], ModelCapabilities(verified: [.speculativeLookup, .batchedDecode, .promptCacheReuse, .quest, .grammarJumpForward])),
            ([RWKV7Model.modelType, XLSTMModel.modelType], rwkv7),
            (Mamba3Model.supportedModelTypes, mamba3),
            (GatedDeltaNet2Model.supportedModelTypes, gdn2),
            (["qwen3_next"], mamba3),
            ([TTTModel.modelType] + TitansModel.modelTypes + LaCTModel.modelTypes, rwkv7),
            (Inkling2Model.modelTypes, ModelCapabilities(verified: [.speculativeLookup, .mtp, .promptCacheReuse, .grammarJumpForward])),
            ([PaddleOCRVL.modelType], ModelCapabilities(verified: all.subtracting([.mtp, .cacheBlend, .kvQuantization, .kvZones, .kvHeadSplit, .quest]))),
            ([DeepSeekOCR.modelType, DeepSeekOCR2.modelType], ModelCapabilities(verified: all.subtracting([.mtp, .cacheBlend, .kvQuantization, .kvZones, .kvHeadSplit, .quest]))),
        ]
        return Dictionary(uniqueKeysWithValues: table.flatMap { types, capabilities in types.map { ($0, capabilities) } })
    }()
}
