import BasaltCore

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

    func kvBits(_ precision: KVPrecision?) -> Int? { allows(.kvQuantization) ? precision?.uniform.keyBits : nil }

    static func of(modelType: String?) -> ModelCapabilities? {
        modelType.flatMap { families[$0.lowercased()] }
    }

    static func refuseUnverifiedKVBits(_ precision: KVPrecision?, modelType: String?, path: String) throws(BasaltError) {
        guard precision != nil, of(modelType: modelType)?.allows(.kvQuantization) != true else { return }
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
        let gemma3 = ModelCapabilities(verified: all.subtracting([.mtp, .cacheBlend, .batchedDecode]))
        let gemma4 = ModelCapabilities(verified: all.subtracting([.mtp, .quest]))
        let table: [([String], ModelCapabilities)] = [
            (["llama", "qwen3", "qwen3_moe", "minimax_m2", "minimax-m2"], attention),
            (["qwen3_5", "qwen3_6", "qwen3.6", "qwen3_8", "qwen3.8", PrismHadamard.modelType], qwen35),
            (["qwen3_5_moe", "qwen3_6_moe", "qwen3_8_moe", "qwen3.8_moe"], qwen35),
            (["nemotron_h", NemotronTranscribeModel.modelType, "lfm2", "lfm2_moe"], recurrent),
            (["gemma3", "gemma3_text"], gemma3),
            (["gemma4", "gemma4_text"], gemma4),
            ([PaddleOCRVL.modelType], ModelCapabilities(verified: all.subtracting([.mtp, .cacheBlend, .kvQuantization, .kvZones, .kvHeadSplit, .quest]))),
            ([DeepSeekOCR.modelType], ModelCapabilities(verified: all.subtracting([.mtp, .cacheBlend, .kvQuantization, .kvZones, .kvHeadSplit, .quest]))),
        ]
        return Dictionary(uniqueKeysWithValues: table.flatMap { types, capabilities in types.map { ($0, capabilities) } })
    }()
}
