import Foundation

public enum TTSModelFamily: String, Sendable, Codable, Equatable, CaseIterable {
    case kokoro82M = "kokoro_82m"
    case qwen3TTS = "qwen3_tts"
    case chatterbox = "chatterbox"
    case cosyVoice3 = "cosyvoice3"
    case zonos3Mini = "zonos3_mini"
    case appleSpeech = "apple_speech"

    public var defaultSampleRate: Int {
        switch self {
        case .zonos3Mini: Zonos3Catalog.sampleRate
        case .kokoro82M, .qwen3TTS, .chatterbox, .cosyVoice3, .appleSpeech: SpeechRequest.defaultSampleRate
        }
    }

    public var estimatedTTFAMs: Int {
        switch self {
        case .kokoro82M: 45
        case .qwen3TTS: 95
        case .chatterbox: 130
        case .cosyVoice3: 110
        case .zonos3Mini: 85
        case .appleSpeech: 20
        }
    }

    public var licenseSPDX: String {
        switch self {
        case .kokoro82M, .qwen3TTS, .cosyVoice3, .zonos3Mini: "Apache-2.0"
        case .chatterbox: "MIT"
        case .appleSpeech: "Apple-System"
        }
    }

    public static func detect(
        identifier: String,
        config: [String: Any]? = nil,
        directory: URL? = nil
    ) -> TTSModelFamily? {
        try? validateSupported(identifier: identifier, config: config, directory: directory)
    }

    public static func validateSupported(
        identifier: String,
        config: [String: Any]? = nil,
        directory: URL? = nil
    ) throws(BasaltError) -> TTSModelFamily {
        let resolvedConfig = config ?? directory.flatMap(loadConfigJSON)
        try rejectLegacyCosyVoice(identifier: identifier, config: resolvedConfig, directory: directory)
        let lower = identifier.lowercased()
        if lower == ComposeSpec.appleSpeech || lower == SpeechComputeBackend.appleSpeech.rawValue {
            return .appleSpeech
        }
        if let dirFamily = detectFromDirectory(directory) {
            return dirFamily
        }
        try rejectLegacyZonos(identifier: identifier, config: resolvedConfig)
        if let cfgFamily = detectFromConfig(resolvedConfig) {
            return cfgFamily
        }
        if let idFamily = detectFromIdentifier(lower) {
            return idFamily
        }
        throw .modelConfigurationInvalid(
            path: directory?.path ?? identifier,
            reason: "unsupported TTS model '\(identifier)'; supported families are Kokoro-82M, Qwen3-TTS, Chatterbox, Fun-CosyVoice3, ZONOS3-Mini and apple-speech"
        )
    }
}

extension TTSModelFamily {
    private static func loadConfigJSON(_ directory: URL) -> [String: Any]? {
        let url = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static func matchesCosyVoice3ID(_ lower: String) -> Bool {
        lower.firstMatch(of: /cosyvoice[-_]?3(?![0-9])/) != nil
    }

    private static func rejectLegacyCosyVoice(
        identifier: String,
        config: [String: Any]?,
        directory: URL?
    ) throws(BasaltError) {
        let path = directory?.path ?? identifier
        let lower = identifier.lowercased()
        if lower.contains("cosyvoice"), !matchesCosyVoice3ID(lower) {
            throw legacyCosyVoiceError(path: path)
        }
        if let config, isLegacyCosyVoiceConfig(config) {
            throw legacyCosyVoiceError(path: path)
        }
        if let directory, isLegacyCosyVoiceDirectory(directory) {
            throw legacyCosyVoiceError(path: path)
        }
    }

    private static func isLegacyCosyVoiceConfig(_ config: [String: Any]) -> Bool {
        let modelType = (config["model_type"] as? String ?? "").lowercased()
        let legacyTypes: Set<String> = ["cosyvoice", "cosyvoice1", "cosyvoice_v1", "cosyvoice2", "cosyvoice_v2"]
        if legacyTypes.contains(modelType) { return true }
        let tokenizer = (config["speech_tokenizer"] as? String ?? "").lowercased()
        if tokenizer.contains("speech_tokenizer_v1") || tokenizer.contains("speech_tokenizer_v2") { return true }
        let estimator = (config["flow_estimator"] as? String ?? config["estimator_type"] as? String ?? "").lowercased()
        return estimator == "unet"
    }

    private static func isLegacyCosyVoiceDirectory(_ directory: URL) -> Bool {
        let fm = FileManager.default
        let hasV3 = fm.fileExists(atPath: directory.appending(component: "cosyvoice3.yaml").path)
        if !hasV3 {
            for legacy in ["cosyvoice.yaml", "cosyvoice2.yaml", "speech_tokenizer_v1.onnx", "speech_tokenizer_v2.onnx"]
            where fm.fileExists(atPath: directory.appending(component: legacy).path) {
                return true
            }
        }
        return false
    }

    private static func legacyCosyVoiceError(path: String) -> BasaltError {
        .modelConfigurationInvalid(
            path: path,
            reason: "legacy CosyVoice v1/v2 checkpoints are not supported; only native Fun-CosyVoice3-0.5B (cosyvoice3) is supported"
        )
    }

    private static func matchesZonos3ID(_ lower: String) -> Bool {
        lower.firstMatch(of: /zonos[-_]?3(?![0-9])/) != nil
    }

    private static func rejectLegacyZonos(identifier: String, config: [String: Any]?) throws(BasaltError) {
        let lower = identifier.lowercased()
        if lower.firstMatch(of: /(?:zyphra\/zonos(?![-_]?3)|zonos[-_]?v?0|zonos[-_]?v?1|zonos[-_]?v?2)/) != nil {
            throw .modelConfigurationInvalid(
                path: identifier,
                reason: "legacy Zonos v0.1/v2 checkpoints are not supported; use Zyphra/ZONOS3-Mini-hf (zonos3_mini)"
            )
        }
        let modelType = (config?["model_type"] as? String ?? "").lowercased()
        if ["zonos", "zonos_v0_1", "zonos2", "zonos_v2"].contains(modelType) {
            throw .modelConfigurationInvalid(
                path: identifier,
                reason: "legacy Zonos v0.1/v2 checkpoints are not supported; use Zyphra/ZONOS3-Mini-hf (zonos3_mini)"
            )
        }
    }

    private static func detectFromDirectory(_ directory: URL?) -> TTSModelFamily? {
        guard let directory else { return nil }
        let fm = FileManager.default
        if fm.fileExists(atPath: directory.appending(component: "cosyvoice3.yaml").path) {
            return .cosyVoice3
        }
        if fm.fileExists(atPath: directory.appending(component: "configuration_zonos3.py").path)
            || fm.fileExists(atPath: directory.appending(component: "voices.safetensors").path) {
            return .zonos3Mini
        }
        if fm.fileExists(atPath: directory.appending(component: CoreAISpeechBundle.kokoroANEManifest).path) {
            return .kokoro82M
        }
        if fm.fileExists(atPath: directory.appending(component: CoreAISpeechBundle.manifest).path) {
            if let data = try? Data(contentsOf: directory.appending(component: CoreAISpeechBundle.manifest)),
               let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
               let rawID = json["model_id"] as? String ?? json["modelID"] as? String,
               let byID = detectFromIdentifier(rawID.lowercased()) {
                return byID
            }
            return .kokoro82M
        }
        return nil
    }

    private static func detectFromConfig(_ config: [String: Any]?) -> TTSModelFamily? {
        guard let config else { return nil }
        switch (config["model_type"] as? String ?? "").lowercased() {
        case "cosyvoice3", "fun_cosyvoice3", "cosyvoice_3": return .cosyVoice3
        case "kokoro", "kokoro_82m", "istftnet", "styletts2": return .kokoro82M
        case "qwen3_tts", "qwen3-tts", "qwen3_tts_talker": return .qwen3TTS
        case "chatterbox", "chatterbox_tts", "chatterbox_turbo", "chatterbox_multilingual": return .chatterbox
        case "zonos3_mini", "zonos3", "zonos-3", "zonos_3": return .zonos3Mini
        default: return detectFromConfigStructure(config)
        }
    }

    private static func detectFromConfigStructure(_ config: [String: Any]) -> TTSModelFamily? {
        if (config["architectures"] as? [String])?.contains("Zonos3MiniForTextToSpeech") == true { return .zonos3Mini }
        if (config["lm"] as? [String: Any])?["gdn_period"] != nil, config["codec"] != nil { return .zonos3Mini }
        if config["s3gen"] != nil { return .chatterbox }
        if config["istftnet"] != nil || config["plbert"] != nil { return .kokoro82M }
        if config["talker_config"] != nil, config["codec_rate_hz"] as? Int == 12 { return .qwen3TTS }
        if config["fsq_speech_tokens"] as? Int == 6_561 { return .cosyVoice3 }
        return nil
    }

    private static func detectFromIdentifier(_ lower: String) -> TTSModelFamily? {
        if matchesCosyVoice3ID(lower) {
            return .cosyVoice3
        }
        if matchesZonos3ID(lower) {
            return .zonos3Mini
        }
        if lower.contains("kokoro") {
            return .kokoro82M
        }
        if lower.contains("qwen3-tts") || lower.contains("qwen3_tts") {
            return .qwen3TTS
        }
        if lower.contains("chatterbox") {
            return .chatterbox
        }
        return nil
    }
}

public enum ChatterboxTagParser {
    public enum ParalinguisticTag: String, Sendable, Codable, Equatable, CaseIterable {
        case laugh
        case chuckle
        case sigh
        case cough
        case gasp
        case breath
    }

    public enum Segment: Sendable, Codable, Equatable {
        case text(String)
        case tag(ParalinguisticTag)
    }

    public static func parse(_ input: String) -> [Segment] {
        var segments: [Segment] = []
        var buffer = ""
        var cursor = input.startIndex
        while cursor < input.endIndex {
            if input[cursor] == "[",
               let close = input[cursor...].firstIndex(of: "]") {
                let inside = input[input.index(after: cursor) ..< close]
                    .trimmingCharacters(in: .whitespaces)
                    .lowercased()
                if let tag = ParalinguisticTag(rawValue: inside) {
                    if !buffer.isEmpty {
                        segments.append(.text(buffer))
                        buffer.removeAll(keepingCapacity: true)
                    }
                    segments.append(.tag(tag))
                    cursor = input.index(after: close)
                    continue
                }
            }
            buffer.append(input[cursor])
            cursor = input.index(after: cursor)
        }
        if !buffer.isEmpty {
            segments.append(.text(buffer))
        }
        return segments
    }
}

public enum CosyVoice3PromptBuilder {
    public static let systemPrefix = "You are a helpful assistant."
    public static let endOfPromptToken = "<|endofprompt|>"
    public static let inlineTags: [String] = [
        "[breath]", "[laughter]", "[sigh]", "[cough]", "[mn]", "<strong>", "</strong>",
    ]

    public static func build(
        input: String,
        instruct: String? = nil,
        promptTranscript: String? = nil
    ) -> String {
        if let instruct {
            let trimmed = instruct.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                return "\(systemPrefix) \(trimmed)\(endOfPromptToken)\(input)"
            }
        }
        let prefixTranscript = promptTranscript ?? ""
        return "\(systemPrefix)\(endOfPromptToken)\(prefixTranscript)\(input)"
    }

    public static func extractInlineControlSpans(_ text: String) -> [String] {
        var found: [String] = []
        var cursor = text.startIndex
        while cursor < text.endIndex {
            if let matched = inlineTags.first(where: { text[cursor...].hasPrefix($0) }) {
                found.append(matched)
                cursor = text.index(cursor, offsetBy: matched.count)
                continue
            }
            if text[cursor] == "[", let close = text[cursor...].firstIndex(of: "]") {
                let span = String(text[cursor ... close])
                if span.count >= 3, span.count <= 32 {
                    found.append(span)
                    cursor = text.index(after: close)
                    continue
                }
            }
            cursor = text.index(after: cursor)
        }
        return found
    }
}

public enum TTSVoiceCatalog {
    public static let kokoroVoiceIDs: [String] = [
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
        "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
        "am_onyx", "am_puck", "am_santa",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        "ef_dora", "em_alex", "em_santa",
        "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        "if_sara", "im_nicola",
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        "pf_dora", "pm_alex", "pm_santa",
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ]

    public static let qwen3Languages: [String] = [
        "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it",
    ]

    public static let chatterboxLanguages: [String] = [
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
        "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
    ]

    public static let cosyVoice3Languages: [String] = [
        "zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru",
    ]

    public static func kokoroLanguage(for voiceID: String) -> String {
        switch voiceID.prefix(2) {
        case "bf", "bm": "en-GB"
        case "ef", "em": "es-ES"
        case "ff": "fr-FR"
        case "hf", "hm": "hi-IN"
        case "if", "im": "it-IT"
        case "jf", "jm": "ja-JP"
        case "pf", "pm": "pt-BR"
        case "zf", "zm": "zh-CN"
        default: "en-US"
        }
    }

    public static func kokoroDisplayName(for voiceID: String) -> String {
        guard let underscore = voiceID.firstIndex(of: "_") else { return voiceID.capitalized }
        let stem = voiceID[voiceID.index(after: underscore)...]
        return stem.isEmpty ? voiceID : stem.prefix(1).uppercased() + stem.dropFirst()
    }

    public static var kokoroVoiceEntries: [CoreAISpeechBundle.VoiceEntry] {
        kokoroVoiceIDs.enumerated().map { index, id in
            CoreAISpeechBundle.VoiceEntry(
                id: id,
                name: kokoroDisplayName(for: id),
                language: kokoroLanguage(for: id),
                index: index
            )
        }
    }

    public static func kokoroVoices(model: String, compute: SpeechComputeBackend) -> [VoiceSpec] {
        kokoroVoiceEntries.map {
            VoiceSpec(id: $0.id, name: $0.name, language: $0.language, model: model, compute: compute)
        }
    }

    public static func kokoroVoiceIndex(for requested: String, language: String?) -> Int? {
        if let idx = kokoroVoiceIDs.firstIndex(where: { $0 == requested }) {
            return idx
        }
        if requested == "alloy" || requested == "default" {
            if let language {
                let prefix = language.lowercased()
                if let matched = kokoroVoiceIDs.firstIndex(where: { kokoroLanguage(for: $0).lowercased().hasPrefix(prefix) }) {
                    return matched
                }
            }
            return kokoroVoiceIDs.firstIndex(of: "af_heart") ?? 0
        }
        return kokoroVoiceIDs.firstIndex { kokoroDisplayName(for: $0).caseInsensitiveCompare(requested) == .orderedSame }
    }

    public static func qwen3Voices(model: String, compute: SpeechComputeBackend = .mlx) -> [VoiceSpec] {
        let presets: [(String, String, String)] = [
            ("chelsie", "Chelsie", "en"), ("ethan", "Ethan", "en"),
            ("serena", "Serena", "zh"), ("vivian", "Vivian", "zh"),
            ("ryan", "Ryan", "en"), ("aiden", "Aiden", "ru"),
            ("ono_anna", "Ono Anna", "ja"), ("sohee", "Sohee", "ko"),
        ]
        return presets.map { VoiceSpec(id: $0.0, name: $0.1, language: $0.2, model: model, compute: compute) }
    }

    public static func chatterboxVoices(model: String, compute: SpeechComputeBackend = .mlx) -> [VoiceSpec] {
        [
            VoiceSpec(id: "default", name: "Default", language: "en", model: model, compute: compute),
            VoiceSpec(id: "expressive", name: "Expressive", language: "en", model: model, compute: compute),
            VoiceSpec(id: "narrator_ru", name: "Narrator RU", language: "ru", model: model, compute: compute),
        ]
    }

    public static func cosyVoice3Voices(model: String, compute: SpeechComputeBackend = .mlx) -> [VoiceSpec] {
        [
            VoiceSpec(id: "cosy_default", name: "CosyVoice3 Default", language: "en", model: model, compute: compute),
            VoiceSpec(id: "cosy_zh", name: "CosyVoice3 Mandarin", language: "zh", model: model, compute: compute),
            VoiceSpec(id: "cosy_ru", name: "CosyVoice3 Russian", language: "ru", model: model, compute: compute),
        ]
    }

    public static func zonos3Voices(model: String, compute: SpeechComputeBackend = .mlx) -> [VoiceSpec] {
        Zonos3Catalog.voices(model: model, compute: compute)
    }
}
