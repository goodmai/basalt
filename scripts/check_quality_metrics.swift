#!/usr/bin/env swift
import Foundation

struct Options {
    var root = FileManager.default.currentDirectoryPath
    var badgesDir: String?
    var osvOK = false
    var maxCrashTraps = 18
    var maxFileLines = 600
}

struct SourceStats {
    let files: Int
    let loc: Int
    let overLimit: [String]
    let fatalErrorCount: Int
    let tryBangCount: Int
    let asBangCount: Int
}

struct TestStats {
    let files: Int
    let loc: Int
    let suites: Int
    let tests: Int
    let xctestSites: [String]
    let globalSeedSites: [String]
}

struct QualityMetrics {
    let source: SourceStats
    let test: TestStats
    let concurrencyHatches: Int
    let enKeys: Int
    let ruKeys: Int
    let l10nIssues: [String]
    let pinnedActions: Int
    let totalActions: Int
    let unpinnedActions: [String]
    let resolvedDeps: Int
    let osvOK: Bool
    let goldenModels: Int
    let referenceFamilies: Int

    var crashTraps: Int { source.fatalErrorCount + source.tryBangCount + source.asBangCount }
    var filesWithinLimit: Int { source.files - source.overLimit.count }
    var testRatioPercent: Double { source.loc > 0 ? (Double(test.loc) * 100.0) / Double(source.loc) : 0 }
    var crashTrapsPerKLOC: Double { source.loc > 0 ? (Double(crashTraps) * 1000.0) / Double(source.loc) : 0 }
}

struct BadgePayload: Encodable {
    let schemaVersion = 1
    let label: String
    let message: String
    let color: String
}

struct MetricsSnapshot: Encodable {
    let sourceFiles: Int
    let sourceLOC: Int
    let filesOverLimit: Int
    let testFiles: Int
    let testLOC: Int
    let testRatioPercent: Double
    let suiteCount: Int
    let testCount: Int
    let xctestCount: Int
    let globalSeedCount: Int
    let concurrencyHatches: Int
    let crashTraps: Int
    let fatalError: Int
    let tryBang: Int
    let asBang: Int
    let enKeys: Int
    let ruKeys: Int
    let l10nIssues: Int
    let pinnedActions: Int
    let totalActions: Int
    let resolvedDeps: Int
    let osvOK: Bool
    let goldenModels: Int
    let referenceFamilies: Int
}

func writeStderr(_ message: String) {
    FileHandle.standardError.write(Data(message.utf8))
}

func parseOptions(_ args: [String]) -> Options {
    var opts = Options()
    var i = 1
    while i < args.count {
        switch args[i] {
        case "--write-badges":
            i += 1
            if i < args.count { opts.badgesDir = args[i] }
        case "--osv-ok":
            opts.osvOK = true
        case "--max-crash-traps":
            i += 1
            if i < args.count, let v = Int(args[i]) { opts.maxCrashTraps = v }
        case "--max-file-lines":
            i += 1
            if i < args.count, let v = Int(args[i]) { opts.maxFileLines = v }
        default:
            if !args[i].hasPrefix("-") { opts.root = args[i] }
        }
        i += 1
    }
    return opts
}

func matchHashes(_ chars: [Unicode.Scalar], from index: Int, count: Int) -> Bool {
    guard index + count <= chars.count else { return false }
    for k in 0..<count where chars[index + k] != "#" { return false }
    return true
}

func skipStringLiteral(_ chars: [Unicode.Scalar], start: Int, hashes: Int, out: inout String) -> Int {
    var i = start
    let multi = i + 2 < chars.count && chars[i + 1] == "\"" && chars[i + 2] == "\""
    i += multi ? 3 : 1
    while i < chars.count {
        if multi {
            if chars[i] == "\"", i + 2 < chars.count, chars[i + 1] == "\"", chars[i + 2] == "\"",
               matchHashes(chars, from: i + 3, count: hashes) {
                return i + 3 + hashes
            }
        } else {
            if chars[i] == "\\", hashes == 0 {
                i += 2
                continue
            }
            if chars[i] == "\\", hashes > 0, matchHashes(chars, from: i + 1, count: hashes) {
                i += 2 + hashes
                continue
            }
            if chars[i] == "\"", matchHashes(chars, from: i + 1, count: hashes) {
                return i + 1 + hashes
            }
            if chars[i] == "\n" {
                out.append("\n")
                return i + 1
            }
        }
        if chars[i] == "\n" { out.append("\n") }
        i += 1
    }
    return i
}

func stripSwiftCommentsAndStrings(_ text: String) -> String {
    let chars = Array(text.unicodeScalars)
    var out = ""
    out.reserveCapacity(chars.count)
    var i = 0
    var blockDepth = 0
    while i < chars.count {
        let c = chars[i]
        let next: Unicode.Scalar? = i + 1 < chars.count ? chars[i + 1] : nil
        if blockDepth > 0 {
            if c == "/", next == "*" { blockDepth += 1; i += 2; continue }
            if c == "*", next == "/" { blockDepth -= 1; i += 2; continue }
            if c == "\n" { out.append("\n") }
            i += 1
            continue
        }
        if c == "/", next == "/" {
            while i < chars.count, chars[i] != "\n" { i += 1 }
            continue
        }
        if c == "/", next == "*" {
            blockDepth = 1
            i += 2
            continue
        }
        if c == "#" {
            var hashes = 0
            while i + hashes < chars.count, chars[i + hashes] == "#" { hashes += 1 }
            if i + hashes < chars.count, chars[i + hashes] == "\"" {
                i = skipStringLiteral(chars, start: i + hashes, hashes: hashes, out: &out)
                out.append(" ")
                continue
            }
        }
        if c == "\"" {
            i = skipStringLiteral(chars, start: i, hashes: 0, out: &out)
            out.append(" ")
            continue
        }
        out.unicodeScalars.append(c)
        i += 1
    }
    return out
}

func swiftFiles(under directory: URL) -> [URL] {
    let fm = FileManager.default
    guard let en = fm.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey]) else { return [] }
    var result: [URL] = []
    for case let url as URL in en where url.pathExtension == "swift" {
        result.append(url)
    }
    return result.sorted { $0.path < $1.path }
}

func regexCount(_ pattern: String, in text: String) -> Int {
    guard let rx = try? NSRegularExpression(pattern: pattern) else { return 0 }
    return rx.numberOfMatches(in: text, range: NSRange(text.startIndex..., in: text))
}

func scanSources(root: URL, maxFileLines: Int) -> SourceStats {
    let urls = swiftFiles(under: root.appendingPathComponent("Sources"))
    var loc = 0
    var overLimit: [String] = []
    var fatalCount = 0
    var tryBang = 0
    var asBang = 0
    for url in urls {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
        let lines = text.components(separatedBy: .newlines)
        let lineCount = text.filter { $0 == "\n" }.count
        let rel = url.path.replacingOccurrences(of: root.path + "/", with: "")
        if lineCount > maxFileLines { overLimit.append("\(rel) (\(lineCount))") }
        loc += lines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.count
        let code = stripSwiftCommentsAndStrings(text)
        fatalCount += regexCount(#"\bfatalError\s*\("#, in: code)
        tryBang += regexCount(#"\btry!"#, in: code)
        asBang += regexCount(#"\bas!"#, in: code)
    }
    return SourceStats(
        files: urls.count, loc: loc, overLimit: overLimit,
        fatalErrorCount: fatalCount, tryBangCount: tryBang, asBangCount: asBang
    )
}

func scanTests(root: URL) -> TestStats {
    let urls = swiftFiles(under: root.appendingPathComponent("Tests"))
    var loc = 0
    var suites = 0
    var tests = 0
    var xctests: [String] = []
    var globalSeeds: [String] = []
    for url in urls {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
        let rel = url.path.replacingOccurrences(of: root.path + "/", with: "")
        loc += text.components(separatedBy: .newlines).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.count
        let code = stripSwiftCommentsAndStrings(text)
        suites += regexCount(#"@Suite\b"#, in: code)
        tests += regexCount(#"@Test\b"#, in: code)
        if regexCount(#":\s*XCTestCase\b"#, in: code) > 0 { xctests.append(rel) }
        if regexCount(#"\bMLXRandom\s*\.\s*seed\s*\("#, in: code) > 0 { globalSeeds.append(rel) }
    }
    return TestStats(
        files: urls.count, loc: loc, suites: suites, tests: tests,
        xctestSites: xctests, globalSeedSites: globalSeeds
    )
}

func parseL10nKeys(at url: URL) -> (keys: [String], duplicates: [String]) {
    guard let content = try? String(contentsOf: url, encoding: .utf8),
          let rx = try? NSRegularExpression(pattern: "^\\s*\"((?:[^\"\\\\]|\\\\.)*)\"\\s*=") else {
        return ([], [])
    }
    var seen = Set<String>()
    var keys: [String] = []
    var dups: [String] = []
    for line in content.components(separatedBy: .newlines) {
        let range = NSRange(line.startIndex..., in: line)
        guard let m = rx.firstMatch(in: line, range: range),
              let keyRange = Range(m.range(at: 1), in: line) else { continue }
        let key = String(line[keyRange])
        keys.append(key)
        if !seen.insert(key).inserted { dups.append(key) }
    }
    return (keys, dups)
}

func scanL10n(root: URL) -> (en: Int, ru: Int, issues: [String]) {
    let enURL = root.appendingPathComponent("Localization/en.lproj/Localizable.strings")
    let ruURL = root.appendingPathComponent("Localization/ru.lproj/Localizable.strings")
    guard FileManager.default.fileExists(atPath: enURL.path),
          FileManager.default.fileExists(atPath: ruURL.path) else {
        return (0, 0, [])
    }
    let en = parseL10nKeys(at: enURL)
    let ru = parseL10nKeys(at: ruURL)
    var issues: [String] = []
    if !en.duplicates.isEmpty { issues.append("en duplicate keys: \(en.duplicates.count)") }
    if !ru.duplicates.isEmpty { issues.append("ru duplicate keys: \(ru.duplicates.count)") }
    let enSet = Set(en.keys)
    let ruSet = Set(ru.keys)
    let missing = enSet.subtracting(ruSet).sorted()
    let extra = ruSet.subtracting(enSet).sorted()
    if !missing.isEmpty { issues.append("ru missing \(missing.count) key(s): \(missing.prefix(3).joined(separator: ", "))") }
    if !extra.isEmpty { issues.append("ru extra \(extra.count) key(s): \(extra.prefix(3).joined(separator: ", "))") }
    return (enSet.count, ruSet.count, issues)
}

func scanWorkflows(root: URL) -> (pinned: Int, total: Int, unpinned: [String]) {
    let dir = root.appendingPathComponent(".github/workflows")
    let files = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "yml" || $0.pathExtension == "yaml" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard let usesRx = try? NSRegularExpression(pattern: #"(?m)^\s*-?\s*uses:\s*([^\s#]+)"#),
          let shaRx = try? NSRegularExpression(pattern: #"@[0-9a-f]{40}$"#) else {
        return (0, 0, [])
    }
    var pinned = 0
    var total = 0
    var unpinned: [String] = []
    for file in files {
        guard let text = try? String(contentsOf: file, encoding: .utf8) else { continue }
        let range = NSRange(text.startIndex..., in: text)
        for m in usesRx.matches(in: text, range: range) {
            guard let r = Range(m.range(at: 1), in: text) else { continue }
            let ref = String(text[r])
            if ref.hasPrefix("./") { continue }
            total += 1
            let rr = NSRange(ref.startIndex..., in: ref)
            if shaRx.firstMatch(in: ref, range: rr) != nil {
                pinned += 1
            } else {
                unpinned.append("\(file.lastPathComponent): \(ref)")
            }
        }
    }
    return (pinned, total, unpinned)
}

func collectMetrics(_ opts: Options) -> QualityMetrics {
    let root = URL(fileURLWithPath: opts.root)
    let source = scanSources(root: root, maxFileLines: opts.maxFileLines)
    let test = scanTests(root: root)
    let allowText = (try? String(contentsOf: root.appendingPathComponent("scripts/concurrency-hatches.allow"), encoding: .utf8)) ?? ""
    let hatchCount = allowText.components(separatedBy: .newlines)
        .map { $0.trimmingCharacters(in: .whitespaces) }
        .filter { !$0.isEmpty && !$0.hasPrefix("#") }.count
    let l10n = scanL10n(root: root)
    let wf = scanWorkflows(root: root)
    let resolvedText = (try? String(contentsOf: root.appendingPathComponent("Package.resolved"), encoding: .utf8)) ?? ""
    let deps = regexCount("\"identity\"\\s*:", in: resolvedText)
    let goldenDir = root.appendingPathComponent("Tests/BasaltTests/Fixtures/golden")
    let golden = ((try? FileManager.default.contentsOfDirectory(at: goldenDir, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "json" }.count
    let refDir = root.appendingPathComponent("Tests/BasaltTests/Fixtures/ReferenceLogits")
    let refFamilies = ((try? FileManager.default.contentsOfDirectory(at: refDir, includingPropertiesForKeys: [.isDirectoryKey])) ?? [])
        .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }.count

    return QualityMetrics(
        source: source, test: test, concurrencyHatches: hatchCount,
        enKeys: l10n.en, ruKeys: l10n.ru, l10nIssues: l10n.issues,
        pinnedActions: wf.pinned, totalActions: wf.total, unpinnedActions: wf.unpinned,
        resolvedDeps: deps, osvOK: opts.osvOK, goldenModels: golden, referenceFamilies: refFamilies
    )
}

func formatGrouped(_ value: Int) -> String {
    let digits = Array(String(value))
    var out = ""
    for (idx, ch) in digits.enumerated() {
        if idx > 0 && (digits.count - idx) % 3 == 0 { out.append(",") }
        out.append(ch)
    }
    return out
}

func formatFixed(_ value: Double, decimals: Int) -> String {
    var scale = 1
    for _ in 0..<decimals { scale *= 10 }
    let scaled = Int((value * Double(scale)).rounded())
    let whole = scaled / scale
    let frac = abs(scaled % scale)
    if decimals == 0 { return "\(whole)" }
    let fracStr = String(frac)
    let pad = String(repeating: "0", count: max(0, decimals - fracStr.count))
    return "\(whole).\(pad)\(fracStr)"
}

func writeEncoded<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(value)
    try data.write(to: url)
}

func writeBadges(_ m: QualityMetrics, to dirPath: String) throws {
    let dir = URL(fileURLWithPath: dirPath)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let kloc = "\(formatFixed(Double(m.source.loc) / 1000.0, decimals: 0))K"
    let hatchColor = m.concurrencyHatches == 0 ? "brightgreen" : "yellow"
    try writeEncoded(BadgePayload(label: "concurrency hatches", message: "\(m.concurrencyHatches) in \(m.source.files) files", color: hatchColor), to: dir.appendingPathComponent("concurrency-hatches.json"))
    let testColor = (m.test.xctestSites.isEmpty && m.test.globalSeedSites.isEmpty) ? "brightgreen" : "orange"
    let testMsg = "\(formatGrouped(m.test.tests)) · \(m.test.xctestSites.isEmpty ? "100% Swift Testing" : "\(m.test.xctestSites.count) XCTest")"
    try writeEncoded(BadgePayload(label: "tests", message: testMsg, color: testColor), to: dir.appendingPathComponent("swift-testing.json"))
    let pct = m.source.files > 0 ? Int((Double(m.filesWithinLimit) * 100.0 / Double(m.source.files)).rounded()) : 100
    let sizeColor = m.source.overLimit.isEmpty ? "brightgreen" : "yellow"
    try writeEncoded(BadgePayload(label: "files ≤600 LOC", message: "\(m.filesWithinLimit)/\(m.source.files) (\(pct)%)", color: sizeColor), to: dir.appendingPathComponent("file-size.json"))
    let trapColor = m.crashTrapsPerKLOC <= 0.2 ? "brightgreen" : (m.crashTrapsPerKLOC <= 0.5 ? "green" : "yellow")
    let trapMsg = "\(m.crashTraps) in \(kloc) LOC (\(formatFixed(m.crashTrapsPerKLOC, decimals: 2))/k)"
    try writeEncoded(BadgePayload(label: "crash traps", message: trapMsg, color: trapColor), to: dir.appendingPathComponent("crash-traps.json"))
    let l10nColor = m.l10nIssues.isEmpty ? "brightgreen" : "red"
    let l10nMsg = m.l10nIssues.isEmpty ? "\(m.enKeys)/\(m.ruKeys) (100%)" : "\(m.l10nIssues.count) issue(s)"
    try writeEncoded(BadgePayload(label: "l10n EN/RU", message: l10nMsg, color: l10nColor), to: dir.appendingPathComponent("l10n-parity.json"))
    let scOK = m.unpinnedActions.isEmpty && m.osvOK
    let scMsg = "\(m.pinnedActions)/\(m.totalActions) SHA · \(m.osvOK ? "0 CVEs (\(m.resolvedDeps) deps)" : "OSV unchecked")"
    try writeEncoded(BadgePayload(label: "supply chain", message: scMsg, color: scOK ? "brightgreen" : "yellow"), to: dir.appendingPathComponent("supply-chain.json"))
    try writeEncoded(BadgePayload(label: "golden logits", message: "\(m.goldenModels) models · \(m.referenceFamilies) families", color: "blue"), to: dir.appendingPathComponent("golden-logits.json"))
    let snapshot = MetricsSnapshot(
        sourceFiles: m.source.files, sourceLOC: m.source.loc, filesOverLimit: m.source.overLimit.count,
        testFiles: m.test.files, testLOC: m.test.loc, testRatioPercent: (m.testRatioPercent * 10).rounded() / 10,
        suiteCount: m.test.suites, testCount: m.test.tests, xctestCount: m.test.xctestSites.count,
        globalSeedCount: m.test.globalSeedSites.count, concurrencyHatches: m.concurrencyHatches,
        crashTraps: m.crashTraps, fatalError: m.source.fatalErrorCount, tryBang: m.source.tryBangCount, asBang: m.source.asBangCount,
        enKeys: m.enKeys, ruKeys: m.ruKeys, l10nIssues: m.l10nIssues.count,
        pinnedActions: m.pinnedActions, totalActions: m.totalActions, resolvedDeps: m.resolvedDeps,
        osvOK: m.osvOK, goldenModels: m.goldenModels, referenceFamilies: m.referenceFamilies
    )
    try writeEncoded(snapshot, to: dir.appendingPathComponent("metrics.json"))
}

func validate(_ m: QualityMetrics, _ opts: Options) -> [String] {
    var failures: [String] = []
    if !m.source.overLimit.isEmpty { failures.append("Sources files > \(opts.maxFileLines) lines: \(m.source.overLimit.joined(separator: ", "))") }
    if !m.test.xctestSites.isEmpty { failures.append("Legacy XCTestCase found: \(m.test.xctestSites.joined(separator: ", "))") }
    if !m.test.globalSeedSites.isEmpty { failures.append("Global MLXRandom.seed found in Tests: \(m.test.globalSeedSites.joined(separator: ", "))") }
    if m.crashTraps > opts.maxCrashTraps { failures.append("Crash traps in Sources grew to \(m.crashTraps) (ceiling \(opts.maxCrashTraps): fatalError=\(m.source.fatalErrorCount), try!=\(m.source.tryBangCount), as!=\(m.source.asBangCount); see CONTRIBUTING.md for --max-crash-traps)") }
    for issue in m.l10nIssues { failures.append("L10n parity: \(issue)") }
    if !m.unpinnedActions.isEmpty { failures.append("Unpinned external GitHub Actions: \(m.unpinnedActions.joined(separator: ", "))") }
    return failures
}

do {
    let opts = parseOptions(CommandLine.arguments)
    let m = collectMetrics(opts)
    print("Quality Metrics:")
    print("  Concurrency hatches : \(m.concurrencyHatches) across \(m.source.files) files (\(formatGrouped(m.source.loc)) LOC)")
    print("  File size ≤\(opts.maxFileLines) LOC  : \(m.filesWithinLimit)/\(m.source.files)")
    print("  Swift Testing       : \(formatGrouped(m.test.tests)) tests in \(m.test.suites) suites (\(m.test.xctestSites.count) XCTest, \(m.test.globalSeedSites.count) MLXRandom.seed, \(formatFixed(m.testRatioPercent, decimals: 1))% test/source LOC)")
    print("  Crash traps         : \(m.crashTraps) (fatalError=\(m.source.fatalErrorCount), try!=\(m.source.tryBangCount), as!=\(m.source.asBangCount); \(formatFixed(m.crashTrapsPerKLOC, decimals: 2))/KLOC, ceiling \(opts.maxCrashTraps))")
    print("  L10n EN/RU parity   : \(m.enKeys)/\(m.ruKeys) (\(m.l10nIssues.isEmpty ? "OK" : m.l10nIssues.joined(separator: "; ")))")
    print("  Supply chain        : \(m.pinnedActions)/\(m.totalActions) SHA-pinned actions, \(m.resolvedDeps) deps, OSV=\(m.osvOK ? "clean" : "unchecked")")
    print("  Golden logits       : \(m.goldenModels) real models, \(m.referenceFamilies) reference families")

    if let dir = opts.badgesDir {
        try writeBadges(m, to: dir)
        print("  Badges written to   : \(dir)")
    }

    let failures = validate(m, opts)
    if !failures.isEmpty {
        for f in failures { writeStderr("FAIL: \(f)\n") }
        exit(1)
    }
    print("All fast-check quality metrics passed.")
} catch {
    writeStderr("ERROR: \(error)\n")
    exit(1)
}
