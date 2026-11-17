import Foundation

public enum Zonos3Catalog {
    public static let sampleRate = 44_100
    public static let frameHopSamples = 2_940

    public static let markupTags: [String] = [
        "laugh", "breath", "sigh", "cough", "panting", "shouting", "crying",
    ]

    public static let supportedLanguages: Set<String> = [
        "en_us", "en_gb", "fr_fr", "de", "es", "it", "pt_br", "ja", "cmn", "ko",
        "ar", "bn", "bg", "cs", "da", "nl", "fi", "ka", "el", "gu", "he",
        "hi", "hu", "id", "kn", "ms", "ml", "mr", "no", "or", "pl", "pa",
        "ro", "ru", "sk", "sv", "tl", "ta", "te", "th", "tr", "uk", "ur", "vi",
    ]

    public static let languageAliases: [String: String] = [
        "en": "en_us", "en_au": "en_gb", "en_ca": "en_us", "en_ie": "en_gb",
        "fr": "fr_fr", "pt": "pt_br", "zh": "cmn", "zh_cn": "cmn", "zh_tw": "cmn",
        "nb": "no", "nn": "no", "fil": "tl", "iw": "he", "in": "id",
    ]

    public static let defaultVoices: [(id: String, language: String, style: String?)] = [
        ("Savannah", "en-US", "American"),
        ("Wyatt", "en-US", "American"),
        ("Cillian", "en-IE", "Irish"),
        ("Imogen", "en-GB", "British"),
        ("Rupert", "en-GB", "British"),
        ("Jett", "en-US", "energetic"),
        ("Morgana", "en-US", "witch"),
        ("Sakura", "en-US", "anime"),
        ("Thaddeus", "en-US", "announcer"),
        ("Willow", "en-US", "ASMR"),
        ("Gaspard", "fr-FR", nil),
        ("Liesel", "de-DE", nil),
        ("Wolfgang", "de-DE", nil),
        ("Ananya", "hi-IN", nil),
        ("Arjun", "hi-IN", nil),
        ("Ginevra", "it-IT", nil),
        ("Lorenzo", "it-IT", nil),
        ("Minjun", "ko-KR", nil),
        ("Seoyeon", "ko-KR", nil),
        ("Haoran", "zh-CN", "energetic"),
        ("Meilin", "zh-CN", "energetic"),
        ("Leonor", "pt-BR", nil),
        ("Thiago", "pt-BR", nil),
        ("Rodrigo", "es-ES", nil),
        ("Ximena", "es-ES", nil),
        ("Chaitanya", "te-IN", nil),
        ("Sravya", "te-IN", nil),
    ]

    public static func normalizeLanguage(_ raw: String?) -> String? {
        guard let raw, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "en_us"
        }
        let clean = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        if supportedLanguages.contains(clean) { return clean }
        if let mapped = languageAliases[clean] { return mapped }
        if let base = clean.split(separator: "_").first.map(String.init) {
            if supportedLanguages.contains(base) { return base }
            if let mapped = languageAliases[base] { return mapped }
        }
        return nil
    }

    public static func collapseLineBreaks(_ text: String) -> String {
        var out = ""
        out.reserveCapacity(text.count)
        var sawBreak = false
        var trailingSpace = false
        for scalar in text.unicodeScalars {
            let isSpace = scalar == " "
            let isOtherWhitespace = scalar.properties.isWhitespace && !isSpace
            if isOtherWhitespace {
                sawBreak = true
                continue
            }
            if isSpace {
                if sawBreak { continue }
                trailingSpace = true
                out.unicodeScalars.append(scalar)
                continue
            }
            if sawBreak {
                if !trailingSpace && !out.isEmpty { out.append(" ") }
                sawBreak = false
            }
            trailingSpace = false
            out.unicodeScalars.append(scalar)
        }
        if sawBreak && !trailingSpace && !out.isEmpty { out.append(" ") }
        return out
    }

    public static func voices(model: String, compute: SpeechComputeBackend = .mlx) -> [VoiceSpec] {
        defaultVoices.map { entry in
            let title = if let style = entry.style { "\(entry.id) (\(style))" } else { entry.id }
            return VoiceSpec(id: entry.id, name: title, language: entry.language, model: model, compute: compute)
        }
    }
}
