import CryptoKit
import Foundation

public enum DocLocale: String, CaseIterable, Sendable, Codable, Hashable {
    case ru
    case zh
    case es
    case fr
    case ja
    case de
    case ko
    case id

    public var nativeName: String {
        switch self {
        case .ru: "Русский"
        case .zh: "中文"
        case .es: "Español"
        case .fr: "Français"
        case .ja: "日本語"
        case .de: "Deutsch"
        case .ko: "한국어"
        case .id: "Bahasa Indonesia"
        }
    }

    public var englishName: String {
        switch self {
        case .ru: "Russian"
        case .zh: "Simplified Chinese"
        case .es: "Spanish"
        case .fr: "French"
        case .ja: "Japanese"
        case .de: "German"
        case .ko: "Korean"
        case .id: "Indonesian"
        }
    }

    public var readmeFilename: String { "README.\(rawValue).md" }

    public var docsDirectory: String { "docs/i18n/\(rawValue)" }

    public static let canonicalDocs: [String] = [
        "architecture.md",
        "benchmarks.md",
        "cloud-decisions.md",
        "commands.md",
        "compose.md",
        "install.md",
        "integrations.md",
        "mac-app.md",
        "models.md",
        "pipelines.md",
        "research-foundations.md",
        "research-tools.md",
        "rest-api.md",
    ]

    public func targetPath(forEnglishPath path: String) -> String {
        if path == "README.md" { return readmeFilename }
        if path.hasPrefix("docs/") {
            return "\(docsDirectory)/\(path.dropFirst(5))"
        }
        return "\(docsDirectory)/\(path)"
    }
}

public enum DocTranslationError: Error, Sendable, Equatable {
    case missingPlaceholder(String)
    case emptyTranslation
    case backendFailure(String)
}

public enum MarkdownSegment: Sendable, Equatable {
    case protected(String)
    case translatable(String)
}

public struct ProtectedText: Sendable, Equatable {
    public let masked: String
    public let tokens: [String: String]

    public init(masked: String, tokens: [String: String]) {
        self.masked = masked
        self.tokens = tokens
    }

    public func restore(in translated: String) throws(DocTranslationError) -> String {
        var result = translated
        for key in tokens.keys.sorted() {
            guard result.contains(key) else {
                throw .missingPlaceholder(key)
            }
            if let value = tokens[key] {
                result = result.replacingOccurrences(of: key, with: value)
            }
        }
        return result
    }
}

public enum MarkdownSegmenter {
    public static func split(_ markdown: String) -> [MarkdownSegment] {
        let lines = markdown.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var segments: [MarkdownSegment] = []
        var buffer: [String] = []
        var fence: String?

        func flush(protected: Bool) {
            guard !buffer.isEmpty else { return }
            let joined = buffer.joined(separator: "\n")
            segments.append(protected ? .protected(joined) : .translatable(joined))
            buffer.removeAll(keepingCapacity: true)
        }

        for line in lines {
            if let activeFence = fence {
                buffer.append(line)
                if isClosingFence(line, marker: activeFence) {
                    flush(protected: true)
                    fence = nil
                }
                continue
            }
            if let opener = openingFence(line) {
                flush(protected: false)
                fence = opener
                buffer.append(line)
            } else if isStandaloneProtectedLine(line) {
                flush(protected: false)
                segments.append(.protected(line))
            } else {
                buffer.append(line)
            }
        }
        flush(protected: fence != nil)
        return segments
    }

    public static func join(_ segments: [MarkdownSegment]) -> String {
        segments.map { segment in
            switch segment {
            case .protected(let text), .translatable(let text): text
            }
        }.joined(separator: "\n")
    }

    public static func maskInline(_ text: String) -> ProtectedText {
        var tokens: [String: String] = [:]
        var nextIndex = 0

        func replaceMatches(_ pattern: String, in input: String) -> String {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { return input }
            let ns = input as NSString
            let matches = regex.matches(in: input, range: NSRange(location: 0, length: ns.length))
            guard !matches.isEmpty else { return input }
            var output = ""
            var cursor = 0
            for match in matches {
                let range = match.range
                if range.location > cursor {
                    output += ns.substring(with: NSRange(location: cursor, length: range.location - cursor))
                }
                let raw = ns.substring(with: range)
                let key = "__BASALT_PH_\(nextIndex)__"
                nextIndex += 1
                tokens[key] = raw
                output += key
                cursor = range.location + range.length
            }
            if cursor < ns.length {
                output += ns.substring(from: cursor)
            }
            return output
        }

        let afterCode = replaceMatches("`[^`\\n]+`", in: text)
        let afterURLs = replaceMatches("https?://[^\\s)\\]>]+", in: afterCode)
        return ProtectedText(masked: afterURLs, tokens: tokens)
    }

    private static func openingFence(_ line: String) -> String? {
        let trimmed = line.drop(while: { $0 == " " })
        if trimmed.hasPrefix("```") { return "```" }
        if trimmed.hasPrefix("~~~") { return "~~~" }
        return nil
    }

    private static func isClosingFence(_ line: String, marker: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        return trimmed.hasPrefix(marker) && trimmed.allSatisfy { $0 == marker.first }
    }

    private static func isStandaloneProtectedLine(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("<p align=\"center\">") && trimmed.contains("<img ") { return true }
        if trimmed.hasPrefix("[![") && trimmed.contains("img.shields.io") { return true }
        return false
    }
}

public struct DocTranslationEntry: Sendable, Codable, Equatable {
    public var sourceSHA256: String
    public var translatedLocales: [String: String]

    public init(sourceSHA256: String, translatedLocales: [String: String] = [:]) {
        self.sourceSHA256 = sourceSHA256
        self.translatedLocales = translatedLocales
    }
}

public struct DocTranslationManifest: Sendable, Codable, Equatable {
    public var version: Int
    public var files: [String: DocTranslationEntry]

    public init(version: Int = 1, files: [String: DocTranslationEntry] = [:]) {
        self.version = version
        self.files = files
    }

    public static func sha256Hex(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    public func staleTargets(
        englishSources: [String: String],
        locales: [DocLocale] = DocLocale.allCases,
        existingTargets: Set<String>
    ) -> [(source: String, locale: DocLocale, target: String)] {
        var result: [(String, DocLocale, String)] = []
        for path in englishSources.keys.sorted() {
            guard let content = englishSources[path] else { continue }
            let digest = Self.sha256Hex(content)
            let entry = files[path]
            for locale in locales {
                let target = locale.targetPath(forEnglishPath: path)
                let recorded = entry?.translatedLocales[locale.rawValue]
                if !existingTargets.contains(target) || recorded != digest {
                    result.append((path, locale, target))
                }
            }
        }
        return result
    }

    public mutating func record(sourcePath: String, sourceText: String, locale: DocLocale) {
        let digest = Self.sha256Hex(sourceText)
        var entry = files[sourcePath] ?? DocTranslationEntry(sourceSHA256: digest)
        entry.sourceSHA256 = digest
        entry.translatedLocales[locale.rawValue] = digest
        files[sourcePath] = entry
    }
}

public enum DocLinkRewriter {
    public static func readmeLanguageBar(active: DocLocale?) -> String {
        let english = active == nil ? "**English**" : "[English](README.md)"
        let others = DocLocale.allCases.map { locale in
            locale == active ? "**\(locale.nativeName)**" : "[\(locale.nativeName)](\(locale.readmeFilename))"
        }
        return ([english] + others + ["**[Website — bas-alt.vercel.app](https://bas-alt.vercel.app)**"])
            .joined(separator: " · ")
    }

    public static func docLanguageBar(docFilename: String, active: DocLocale?) -> String {
        let english = active == nil
            ? "**English**"
            : "[English](../../\(docFilename))"
        let others = DocLocale.allCases.map { locale in
            if locale == active { return "**\(locale.nativeName)**" }
            let href = active == nil ? "i18n/\(locale.rawValue)/\(docFilename)" : "../\(locale.rawValue)/\(docFilename)"
            return "[\(locale.nativeName)](\(href))"
        }
        return "> Languages: " + ([english] + others).joined(separator: " · ")
    }

    public static func rewriteReadmeLinks(_ markdown: String, locale: DocLocale) -> String {
        var result = markdown
        for doc in DocLocale.canonicalDocs {
            result = result.replacingOccurrences(
                of: "(docs/\(doc)",
                with: "(\(locale.docsDirectory)/\(doc)"
            )
        }
        return result
    }
}

public enum TranslationPromptBuilder {
    public static func systemPrompt(target: DocLocale) -> String {
        """
        You are a technical documentation translator for the Basalt Apple Silicon MLX inference project.
        Translate the user's English Markdown into natural, idiomatic \(target.englishName) (\(target.nativeName)).
        Rules:
        - Preserve every placeholder token matching __BASALT_PH_N__ verbatim without modifying or translating it.
        - Preserve Markdown headings, tables, lists, links, emphasis, and line structure.
        - Keep product names (Basalt, MLX, Apple Silicon, Swift, OpenAI, Anthropic, MCP, Liquid Glass), CLI flags, API paths, and model IDs in English.
        - Output only the translated Markdown with no preamble or commentary.
        """
    }

    public static func userPrompt(maskedMarkdown: String, target: DocLocale) -> String {
        "Translate the following Markdown section from English into \(target.englishName) (\(target.nativeName)):\n\n\(maskedMarkdown)"
    }
}

public enum DocTranslationPipeline {
    public static func translateDocument(
        _ markdown: String,
        sourcePath: String,
        to locale: DocLocale,
        translateChunk: @Sendable (String, DocLocale) async throws(DocTranslationError) -> String
    ) async throws(DocTranslationError) -> String {
        let segments = MarkdownSegmenter.split(markdown)
        var output: [MarkdownSegment] = []
        output.reserveCapacity(segments.count)
        for segment in segments {
            switch segment {
            case .protected(let raw):
                output.append(.protected(raw))
            case .translatable(let raw):
                if raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    output.append(.translatable(raw))
                    continue
                }
                let protected = MarkdownSegmenter.maskInline(raw)
                let translated = try await translateChunk(protected.masked, locale)
                guard !translated.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    throw .emptyTranslation
                }
                let restored = try protected.restore(in: translated)
                output.append(.translatable(restored))
            }
        }
        var joined = MarkdownSegmenter.join(output)
        if sourcePath == "README.md" {
            joined = DocLinkRewriter.rewriteReadmeLinks(joined, locale: locale)
        }
        return joined
    }
}
