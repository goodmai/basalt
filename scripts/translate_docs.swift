#!/usr/bin/env swift
import CryptoKit
import Foundation

struct CLIOptions {
    var root: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    var endpoint: String = "http://127.0.0.1:8080/v1"
    var model: String = "default"
    var locales: [String] = ["ru", "zh", "es", "fr", "ja", "de", "ko", "id"]
    var checkOnly: Bool = false
    var dryRun: Bool = false
    var syncManifest: Bool = false
    var force: Bool = false
}

struct ManifestEntry: Codable {
    var sourceSHA256: String
    var translatedLocales: [String: String]
}

struct Manifest: Codable {
    var version: Int = 1
    var files: [String: ManifestEntry] = [:]
}

let localeMeta: [String: (native: String, english: String)] = [
    "ru": ("Русский", "Russian"),
    "zh": ("中文", "Simplified Chinese"),
    "es": ("Español", "Spanish"),
    "fr": ("Français", "French"),
    "ja": ("日本語", "Japanese"),
    "de": ("Deutsch", "German"),
    "ko": ("한국어", "Korean"),
    "id": ("Bahasa Indonesia", "Indonesian"),
]

let canonicalSources: [String] = [
    "README.md",
    "docs/architecture.md",
    "docs/benchmarks.md",
    "docs/cloud-decisions.md",
    "docs/commands.md",
    "docs/compose.md",
    "docs/install.md",
    "docs/integrations.md",
    "docs/mac-app.md",
    "docs/models.md",
    "docs/pipelines.md",
    "docs/research-foundations.md",
    "docs/research-tools.md",
    "docs/rest-api.md",
]

func parseArgs(_ args: [String]) -> CLIOptions {
    var opts = CLIOptions()
    var i = 1
    while i < args.count {
        switch args[i] {
        case "--check": opts.checkOnly = true
        case "--dry-run": opts.dryRun = true
        case "--sync-manifest": opts.syncManifest = true
        case "--force": opts.force = true
        case "--root":
            i += 1
            if i < args.count { opts.root = URL(fileURLWithPath: args[i]) }
        case "--endpoint":
            i += 1
            if i < args.count { opts.endpoint = args[i] }
        case "--model":
            i += 1
            if i < args.count { opts.model = args[i] }
        case "--locales":
            i += 1
            if i < args.count {
                opts.locales = args[i].split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            }
        default: break
        }
        i += 1
    }
    return opts
}

func sha256Hex(_ text: String) -> String {
    SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
}

func targetPath(for source: String, locale: String) -> String {
    if source == "README.md" { return "README.\(locale).md" }
    if source.hasPrefix("docs/") { return "docs/i18n/\(locale)/\(source.dropFirst(5))" }
    return "docs/i18n/\(locale)/\(source)"
}

func loadManifest(at url: URL) -> Manifest {
    guard let data = try? Data(contentsOf: url),
          let decoded = try? JSONDecoder().decode(Manifest.self, from: data) else {
        return Manifest()
    }
    return decoded
}

func saveManifest(_ manifest: Manifest, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(manifest)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: url)
}

func maskInline(_ text: String) -> (masked: String, tokens: [String: String]) {
    var tokens: [String: String] = [:]
    var idx = 0
    func sub(_ pattern: String, in input: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return input }
        let ns = input as NSString
        let matches = regex.matches(in: input, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return input }
        var out = ""
        var pos = 0
        for m in matches {
            if m.range.location > pos { out += ns.substring(with: NSRange(location: pos, length: m.range.location - pos)) }
            let key = "__BASALT_PH_\(idx)__"
            idx += 1
            tokens[key] = ns.substring(with: m.range)
            out += key
            pos = m.range.location + m.range.length
        }
        if pos < ns.length { out += ns.substring(from: pos) }
        return out
    }
    let step1 = sub("`[^`\\n]+`", in: text)
    let step2 = sub("https?://[^\\s)\\]>]+", in: step1)
    return (step2, tokens)
}

func splitMarkdown(_ markdown: String) -> [(protected: Bool, text: String)] {
    let lines = markdown.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    var out: [(Bool, String)] = []
    var buf: [String] = []
    var fence: String?
    func flush(_ isProt: Bool) {
        guard !buf.isEmpty else { return }
        out.append((isProt, buf.joined(separator: "\n")))
        buf.removeAll(keepingCapacity: true)
    }
    for line in lines {
        if let f = fence {
            buf.append(line)
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix(f) && t.allSatisfy({ $0 == f.first }) {
                flush(true)
                fence = nil
            }
            continue
        }
        let stripped = line.drop(while: { $0 == " " })
        if stripped.hasPrefix("```") {
            flush(false)
            fence = "```"
            buf.append(line)
        } else if stripped.hasPrefix("~~~") {
            flush(false)
            fence = "~~~"
            buf.append(line)
        } else if line.trimmingCharacters(in: .whitespaces).hasPrefix("[![") && line.contains("img.shields.io") {
            flush(false)
            out.append((true, line))
        } else {
            buf.append(line)
        }
    }
    flush(fence != nil)
    return out
}

func callEndpoint(endpoint: String, model: String, locale: String, text: String) async throws -> String {
    guard let meta = localeMeta[locale], let url = URL(string: "\(endpoint)/chat/completions") else {
        throw NSError(domain: "TranslateDocs", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid endpoint or locale \(locale)"])
    }
    let system = "You are a technical documentation translator for Basalt (Apple Silicon MLX). Translate English Markdown into natural \(meta.english) (\(meta.native)). Preserve every __BASALT_PH_N__ placeholder, Markdown syntax, product names, CLI flags, and model IDs verbatim. Output only translated Markdown."
    let body: [String: Any] = [
        "model": model,
        "temperature": 0.1,
        "messages": [
            ["role": "system", "content": system],
            ["role": "user", "content": text],
        ],
    ]
    var req = URLRequest(url: url)
    req.httpMethod = "POST"
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    req.httpBody = try JSONSerialization.data(withJSONObject: body)
    let (data, response) = try await URLSession.shared.data(for: req)
    let code = (response as? HTTPURLResponse)?.statusCode ?? 0
    guard (200 ..< 300).contains(code),
          let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let choices = json["choices"] as? [[String: Any]],
          let message = choices.first?["message"] as? [String: Any],
          let content = message["content"] as? String else {
        throw NSError(domain: "TranslateDocs", code: code, userInfo: [NSLocalizedDescriptionKey: "Translation request failed (HTTP \(code))"])
    }
    return content
}

struct StaleItem {
    let source: String
    let locale: String
    let target: String
    let text: String
    let digest: String
}

func syncManifestEntries(opts: CLIOptions, manifest: inout Manifest, manifestURL: URL) throws {
    let fm = FileManager.default
    for source in canonicalSources {
        guard let text = try? String(contentsOf: opts.root.appending(path: source), encoding: .utf8) else { continue }
        let digest = sha256Hex(text)
        var entry = manifest.files[source] ?? ManifestEntry(sourceSHA256: digest, translatedLocales: [:])
        entry.sourceSHA256 = digest
        for locale in opts.locales {
            if fm.fileExists(atPath: opts.root.appending(path: targetPath(for: source, locale: locale)).path) {
                entry.translatedLocales[locale] = digest
            }
        }
        manifest.files[source] = entry
    }
    try saveManifest(manifest, to: manifestURL)
    print("Updated \(manifestURL.path)")
}

func collectStale(opts: CLIOptions, manifest: Manifest) -> [StaleItem] {
    let fm = FileManager.default
    var stale: [StaleItem] = []
    for source in canonicalSources {
        guard let text = try? String(contentsOf: opts.root.appending(path: source), encoding: .utf8) else { continue }
        let digest = sha256Hex(text)
        for locale in opts.locales where localeMeta[locale] != nil {
            let dstRel = targetPath(for: source, locale: locale)
            let dstExists = fm.fileExists(atPath: opts.root.appending(path: dstRel).path)
            if opts.force || !dstExists || manifest.files[source]?.translatedLocales[locale] != digest {
                stale.append(StaleItem(source: source, locale: locale, target: dstRel, text: text, digest: digest))
            }
        }
    }
    return stale
}

func translateItem(_ item: StaleItem, opts: CLIOptions, manifest: inout Manifest, manifestURL: URL) async throws {
    let segments = splitMarkdown(item.text)
    let count = segments.filter { !$0.protected && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }.count
    if opts.dryRun {
        print("PLAN [\(item.locale)] \(item.source) -> \(item.target) (\(count) chunks)")
        return
    }
    print("Translating [\(item.locale)] \(item.source) -> \(item.target) (\(count) chunks)...")
    var translatedParts: [String] = []
    for seg in segments {
        if seg.protected || seg.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            translatedParts.append(seg.text)
            continue
        }
        let (masked, tokens) = maskInline(seg.text)
        var reply = try await callEndpoint(endpoint: opts.endpoint, model: opts.model, locale: item.locale, text: masked)
        for key in tokens.keys.sorted() {
            if let orig = tokens[key] { reply = reply.replacingOccurrences(of: key, with: orig) }
        }
        translatedParts.append(reply)
    }
    let dstURL = opts.root.appending(path: item.target)
    try FileManager.default.createDirectory(at: dstURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try translatedParts.joined(separator: "\n").write(to: dstURL, atomically: true, encoding: .utf8)
    var entry = manifest.files[item.source] ?? ManifestEntry(sourceSHA256: item.digest, translatedLocales: [:])
    entry.sourceSHA256 = item.digest
    entry.translatedLocales[item.locale] = item.digest
    manifest.files[item.source] = entry
    try saveManifest(manifest, to: manifestURL)
}

func run() async throws {
    let opts = parseArgs(CommandLine.arguments)
    let manifestURL = opts.root.appending(path: "docs/i18n/.manifest.json")
    var manifest = loadManifest(at: manifestURL)
    if opts.syncManifest {
        try syncManifestEntries(opts: opts, manifest: &manifest, manifestURL: manifestURL)
        return
    }
    let stale = collectStale(opts: opts, manifest: manifest)
    if opts.checkOnly {
        if stale.isEmpty {
            print("All documentation translations are up to date.")
            return
        }
        for item in stale { print("STALE [\(item.locale)] \(item.source) -> \(item.target)") }
        exit(1)
    }
    for item in stale {
        try await translateItem(item, opts: opts, manifest: &manifest, manifestURL: manifestURL)
    }
}

try await run()
