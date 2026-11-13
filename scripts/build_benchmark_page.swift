#!/usr/bin/env swift
import Foundation

struct CLIOptions {
    var root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    var out: URL?
    var logs: URL?
    var checkOnly = false

    static func parse(_ args: [String]) -> CLIOptions {
        var options = CLIOptions()
        var index = 0
        while index < args.count {
            switch args[index] {
            case "--root" where index + 1 < args.count:
                options.root = URL(fileURLWithPath: args[index + 1])
                index += 2
            case "--out" where index + 1 < args.count:
                options.out = URL(fileURLWithPath: args[index + 1])
                index += 2
            case "--logs" where index + 1 < args.count:
                options.logs = URL(fileURLWithPath: args[index + 1])
                index += 2
            case "--check":
                options.checkOnly = true
                index += 1
            default:
                index += 1
            }
        }
        return options
    }
}

struct GoldenPromptTrace: Decodable {
    let tokens: [Int]
}

struct GoldenFixtureDoc: Decodable {
    let revision: String
    let tolerance: Double
    let prompts: [String: GoldenPromptTrace]
}

struct PublishedMetrics: Codable, Equatable {
    let activeMemory: String?
    let decodeTokPerSec: String?
    let perplexity: String?
}

struct CollectionItem: Codable, Equatable {
    let repoId: String
    let hfURL: String
    let architectureClass: String
    let fixturePath: String
    let revision: String
    let klTolerance: Double
    let verifiedPrompts: [String]
    let verifiedGreedySteps: Int
    let verifiedPaths: [String]
    let publishedMeasurements: PublishedMetrics
}

struct HFCollectionManifest: Codable, Equatable {
    let schemaVersion: Int
    let publish: Bool
    let requiresHumanReview: Bool
    let title: String
    let description: String
    let hardwareBaseline: String
    let items: [CollectionItem]
}

struct HarnessSummary {
    let name: String
    let path: String
    let filesCount: Int
    let totalBytes: Int
    let detail: String
}

struct RunnerLogSummary {
    let file: String
    let header: String
    let summaryRows: [[String]]
}

func regexCaptures(_ pattern: String, in text: String) -> [String] {
    guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
    let range = NSRange(text.startIndex..., in: text)
    return regex.matches(in: text, range: range).compactMap { match in
        Range(match.range(at: 1), in: text).map { String(text[$0]) }
    }
}

func parsePipeTable(in markdown: String, afterHeading heading: String) -> (headers: [String], rows: [[String]]) {
    let lines = markdown.components(separatedBy: "\n")
    guard let headingIndex = lines.firstIndex(where: { $0.trimmingCharacters(in: .whitespaces).hasPrefix(heading) }) else {
        return ([], [])
    }
    var tableLines: [String] = []
    for line in lines[(headingIndex + 1)...] {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("#") && !tableLines.isEmpty { break }
        if trimmed.hasPrefix("|") {
            tableLines.append(trimmed)
        } else if !tableLines.isEmpty {
            break
        }
    }
    guard tableLines.count >= 2 else { return ([], []) }
    let splitRow: (String) -> [String] = { row in
        var inner = row.replacingOccurrences(of: #"\|"#, with: "\u{001F}")
        if inner.hasPrefix("|") { inner.removeFirst() }
        if inner.hasSuffix("|") { inner.removeLast() }
        return inner.components(separatedBy: "|").map {
            $0.replacingOccurrences(of: "\u{001F}", with: "|").trimmingCharacters(in: .whitespaces)
        }
    }
    let headers = splitRow(tableLines[0])
    let rows = tableLines.dropFirst(2).map(splitRow)
    return (headers, rows)
}

func verifiedPaths(for prompts: [String]) -> [String] {
    var paths: [String] = []
    if prompts.contains(where: { $0.hasPrefix("chat-") }) { paths.append("single-turn chat") }
    if prompts.contains("chat-russian") { paths.append("multilingual") }
    if prompts.contains(where: { $0.hasPrefix("code-") }) { paths.append("code generation") }
    if prompts.contains(where: { $0.contains("tool-call") }) { paths.append("JSON tool calling") }
    if prompts.contains(where: { $0.contains("agent-trace") }) { paths.append("8k multi-turn agent trace") }
    if prompts.contains(where: { $0.contains("resume") }) { paths.append("prefix-cache anchor resume") }
    return paths.isEmpty ? ["greedy logit replay"] : paths
}

func loadPublished08Metrics(modelsMarkdown: String) -> [String: PublishedMetrics] {
    let table = parsePipeTable(in: modelsMarkdown, afterHeading: "### Measured for 0.8")
    var result: [String: PublishedMetrics] = [:]
    for row in table.rows where row.count >= 4 {
        guard let repo = regexCaptures(#"`([^`]+)`"#, in: row[0]).first else { continue }
        let mem = row[2] == "not measured" || row[2] == "—" ? nil : row[2]
        let tok = row[3] == "not measured" || row[3] == "—" ? nil : row[3]
        result[repo] = PublishedMetrics(activeMemory: mem, decodeTokPerSec: tok, perplexity: nil)
    }
    return result
}

func hasEvidenceLink(_ text: String) -> Bool {
    ["/pull/", "/issues/", "#", "Fixtures/", "Tests/", "logs/"].contains { text.contains($0) }
}

func validateProvenanceTable(_ benchmarks: String) -> [String] {
    let provenance = parsePipeTable(in: benchmarks, afterHeading: "## Where the README's numbers come from")
    if provenance.rows.isEmpty {
        return ["docs/benchmarks.md: missing provenance table under '## Where the README's numbers come from'"]
    }
    return provenance.rows.enumerated().compactMap { idx, row in
        guard row.count >= 2, !hasEvidenceLink(row[1]) else { return nil }
        return "docs/benchmarks.md: provenance row \(idx + 1) lacks a PR, issue, test, fixture or log link"
    }
}

func validateClaims(readme: String, benchmarks: String) -> [String] {
    var errors = validateProvenanceTable(benchmarks)
    let lines = readme.components(separatedBy: "\n")
    guard let infStart = lines.firstIndex(where: { $0.hasPrefix("## Inference:") }) else { return errors }
    let metrics = ["tok/s", "×", " GB"]
    for (offset, line) in lines[(infStart + 1)...].enumerated() {
        if line.hasPrefix("## ") { break }
        let isMetricBullet = line.hasPrefix("- ") && metrics.contains(where: { line.contains($0) })
        if isMetricBullet && !hasEvidenceLink(line) {
            errors.append("README.md:\(infStart + offset + 2): quantitative inference bullet lacks a PR, test or fixture link")
        }
    }
    return errors
}

func loadGoldenItems(root: URL, readme: String, published: [String: PublishedMetrics]) -> (items: [CollectionItem], errors: [String]) {
    var items: [CollectionItem] = []
    var errors: [String] = []
    for (idx, line) in readme.components(separatedBy: "\n").enumerated() where line.contains("Fixtures/golden/") {
        let cols = line.trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "|")).components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
        guard cols.count >= 3,
              let repo = regexCaptures(#"`([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)`"#, in: cols[0]).first,
              let fixtureRel = regexCaptures(#"\((Tests/BasaltTests/Fixtures/golden/[A-Za-z0-9._-]+\.json)\)"#, in: cols[2]).first else {
            errors.append("README.md:\(idx + 1): malformed golden-tested model row")
            continue
        }
        let fixtureURL = root.appendingPathComponent(fixtureRel)
        guard let data = try? Data(contentsOf: fixtureURL),
              let doc = try? JSONDecoder().decode(GoldenFixtureDoc.self, from: data),
              !doc.prompts.isEmpty else {
            errors.append("README.md:\(idx + 1): unreadable or empty golden fixture \(fixtureRel)")
            continue
        }
        let promptNames = doc.prompts.keys.sorted()
        let totalSteps = doc.prompts.values.reduce(0) { $0 + $1.tokens.count }
        let metrics = published[repo] ?? PublishedMetrics(activeMemory: nil, decodeTokPerSec: nil, perplexity: nil)
        items.append(CollectionItem(
            repoId: repo,
            hfURL: "https://huggingface.co/\(repo)",
            architectureClass: cols[1],
            fixturePath: fixtureRel,
            revision: doc.revision,
            klTolerance: doc.tolerance,
            verifiedPrompts: promptNames,
            verifiedGreedySteps: totalSteps,
            verifiedPaths: verifiedPaths(for: promptNames),
            publishedMeasurements: metrics
        ))
    }
    if items.isEmpty { errors.append("README.md: found zero golden-tested models") }
    return (items, errors)
}

func inspectHarnesses(root: URL) -> [HarnessSummary] {
    let specs: [(String, String, [String], String)] = [
        (
            "Ladder to the Moon",
            "benchmarks/ladder",
            ["run.py", "selfcheck.py", "lunar_prompt.md", "lunar_task.md"],
            "5 executable rungs ending in 4-body Earth-to-Moon propagation + selfcheck.py (9 offline assertions)"
        ),
        (
            "ARC-AGI (pass@2)",
            "benchmarks/arc-agi",
            [
                "arc_agi_benchmark.py", "arc_sandbox.py", "arc_helpers.py",
                "compare-tasks/task_1.json", "compare-tasks/task_2.json", "compare-tasks/task_3.json",
                "compare-tasks/task_4.json", "compare-tasks/task_5.json"
            ],
            "Official pass@2 harness, sandboxed solver verification and 5 local compare tasks"
        ),
        (
            "Speculative Lookup",
            "benchmarks/speculative-lookup",
            ["code.txt", "prose.txt", "reasoning.txt"],
            "Code, prose and reasoning draft/verify corpora for exact n-gram lookup"
        ),
        (
            "CacheBlend",
            "benchmarks/cacheblend",
            ["file0.txt", "file1.txt", "file2.txt", "file3.txt", "file4.txt", "file5.txt"],
            "6 multi-chunk RAG context files for prefix reuse and selective KV recompute"
        )
    ]
    return specs.compactMap { name, rel, files, detail in
        let dir = root.appendingPathComponent(rel)
        var count = 0
        var bytes = 0
        for item in files {
            if let data = try? Data(contentsOf: dir.appendingPathComponent(item)) {
                count += 1
                bytes += data.count
            }
        }
        guard count > 0 else { return nil }
        return HarnessSummary(name: name, path: rel, filesCount: count, totalBytes: bytes, detail: detail)
    }
}

func referenceFamilies(root: URL) -> [String] {
    let dir = root.appendingPathComponent("Tests/BasaltTests/Fixtures/ReferenceLogits")
    let entries = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
    return entries.filter { !$0.hasPrefix(".") && $0 != "README.md" }.sorted()
}

func parseRunnerLogs(in dir: URL?) -> [RunnerLogSummary] {
    guard let dir, let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return [] }
    var results: [RunnerLogSummary] = []
    for name in names.sorted() where name.hasSuffix(".log") {
        guard let text = try? String(contentsOf: dir.appendingPathComponent(name), encoding: .utf8) else { continue }
        var header = ""
        var rows: [[String]] = []
        for line in text.components(separatedBy: "\n") {
            if header.isEmpty, line.contains("start ") && line.contains("commit=") {
                header = line.trimmingCharacters(in: .whitespaces)
            }
            if let range = line.range(of: "summary |") {
                let tail = String(line[range.upperBound...]).trimmingCharacters(in: .whitespaces)
                if tail.hasPrefix("---") { continue }
                let cells = tail.trimmingCharacters(in: CharacterSet(charactersIn: "|")).components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
                if !cells.isEmpty { rows.append(cells) }
            }
        }
        if !rows.isEmpty {
            results.append(RunnerLogSummary(file: name, header: header, summaryRows: rows))
        }
    }
    return results
}

func escapeHTML(_ text: String) -> String {
    var out = text
        .replacingOccurrences(of: "&", with: "&amp;")
        .replacingOccurrences(of: "<", with: "&lt;")
        .replacingOccurrences(of: ">", with: "&gt;")
        .replacingOccurrences(of: "\"", with: "&quot;")
    if let linkRegex = try? NSRegularExpression(pattern: #"\[([^\]]+)\]\((https?://[^)]+|[A-Za-z0-9._/#-]+)\)"#) {
        let range = NSRange(out.startIndex..., in: out)
        out = linkRegex.stringByReplacingMatches(in: out, range: range, withTemplate: #"<a href="$2">$1</a>"#)
    }
    if let codeRegex = try? NSRegularExpression(pattern: #"`([^`]+)`"#) {
        let range = NSRange(out.startIndex..., in: out)
        out = codeRegex.stringByReplacingMatches(in: out, range: range, withTemplate: #"<code>$1</code>"#)
    }
    return out
}

func renderTableHTML(headers: [String], rows: [[String]]) -> String {
    guard !headers.isEmpty, !rows.isEmpty else { return "" }
    let head = "<tr>" + headers.map { "<th>\(escapeHTML($0))</th>" }.joined() + "</tr>"
    let body = rows.map { row in
        "<tr>" + row.map { "<td>\(escapeHTML($0))</td>" }.joined() + "</tr>"
    }.joined(separator: "\n")
    return "<table><thead>\(head)</thead><tbody>\n\(body)\n</tbody></table>"
}

func renderRunnerLogsHTML(_ logs: [RunnerLogSummary]) -> String {
    guard !logs.isEmpty else { return "" }
    let blocks = logs.map { log -> String in
        let headers = log.summaryRows.first ?? []
        let rows = Array(log.summaryRows.dropFirst())
        let meta = log.header.isEmpty ? "" : "<p class=\"meta\"><code>\(escapeHTML(log.header))</code></p>"
        return "<section class=\"log-card\"><h3><code>\(escapeHTML(log.file))</code></h3>\(meta)\(renderTableHTML(headers: headers, rows: rows))</section>"
    }.joined(separator: "\n")
    return "<h2>Logged Runner Tables</h2>\n\(blocks)"
}

func wrapHTMLDocument(body: String) -> String {
    """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>basalt — Golden Checkpoints &amp; Measured Benchmarks</title>
      <style>
        :root { color-scheme: light dark; --bg: #0f1115; --fg: #e8ecf1; --muted: #9aa4b2; --card: #171a21; --border: #2a303c; --accent: #7cc4fa; }
        @media (prefers-color-scheme: light) { :root { --bg: #ffffff; --fg: #14171f; --muted: #525d6d; --card: #f6f8fa; --border: #d8dee4; --accent: #0969da; } }
        body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--fg); max-width: 1120px; margin: 0 auto; padding: 32px 20px 64px; }
        .banner { background: var(--card); border: 1px solid var(--border); border-left: 4px solid var(--accent); padding: 12px 16px; border-radius: 6px; margin-bottom: 24px; font-size: 14px; }
        h1 { font-size: 28px; margin: 0 0 8px; }
        h2 { font-size: 20px; margin: 34px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
        p, .meta { color: var(--muted); margin: 8px 0 14px; }
        a { color: var(--accent); text-decoration: none; }
        a:hover { text-decoration: underline; }
        code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; background: var(--card); padding: 2px 5px; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; margin: 12px 0 20px; font-size: 14px; }
        th, td { border: 1px solid var(--border); padding: 8px 10px; text-align: left; vertical-align: top; }
        th { background: var(--card); font-weight: 600; }
      </style>
    </head>
    <body>
    \(body)
    </body>
    </html>
    """ + "\n"
}

func renderHTML(
    items: [CollectionItem],
    families: [String],
    harnesses: [HarnessSummary],
    benchmarksMarkdown: String,
    logs: [RunnerLogSummary]
) -> String {
    let features = parsePipeTable(in: benchmarksMarkdown, afterHeading: "## Features at a glance")
    let provenance = parsePipeTable(in: benchmarksMarkdown, afterHeading: "## Where the README's numbers come from")
    let results = parsePipeTable(in: benchmarksMarkdown, afterHeading: "### Results")
    let goldenRows = items.map { item in
        [
            "[\(item.repoId)](\(item.hfURL))",
            item.architectureClass,
            "`\(item.revision)` (KL ≤ \(item.klTolerance))",
            "\(item.verifiedPrompts.count) prompts · \(item.verifiedGreedySteps) steps",
            item.verifiedPaths.joined(separator: ", "),
            item.publishedMeasurements.decodeTokPerSec ?? "fixture-verified (tok/s not in 0.8 table)",
            "[fixture](../../\(item.fixturePath))"
        ]
    }
    let harnessRows = harnesses.map { [$0.name, "`\($0.path)`", "\($0.filesCount) files (\($0.totalBytes) B)", $0.detail] }
    let goldenTable = renderTableHTML(
        headers: ["Checkpoint", "Class", "Golden Gate", "Replay Coverage", "Verified Paths", "0.8 Decode tok/s", "Fixture"],
        rows: goldenRows
    )
    let harnessTable = renderTableHTML(headers: ["Harness", "Directory", "Tracked Assets", "Verification Scope"], rows: harnessRows)
    let famList = families.map { "<code>\(escapeHTML($0))</code>" }.joined(separator: ", ")
    return wrapHTMLDocument(body: """
      <div class="banner"><strong>Review status:</strong> local static artifact generated by <code>scripts/build_benchmark_page.swift</code>. Outward-facing publishing of this page and <a href="hf-collection.json"><code>hf-collection.json</code></a> stays a human decision.</div>
      <h1>basalt — Measured Benchmarks &amp; Golden Checkpoints</h1>
      <p>Every number below is backed by a golden logit fixture, a reproducible harness in <code>benchmarks/</code>, or a logged run linked to its PR. Hardware baseline: Apple M4 Pro 24 GB, release build.</p>
      <h2>Golden-Verified Hugging Face Checkpoints</h2>
      \(goldenTable)
      <p>Per-family reference logits in <code>Tests/BasaltTests/Fixtures/ReferenceLogits</code> (\(families.count) families): \(famList).</p>
      <h2>Measured Engine Features</h2>
      \(renderTableHTML(headers: features.headers, rows: features.rows))
      <h2>README First-Screen Claim Provenance</h2>
      \(renderTableHTML(headers: provenance.headers, rows: provenance.rows))
      <h2>Benchmark Harnesses</h2>
      \(harnessTable)
      <h2>Ladder &amp; ARC-AGI Logged Results</h2>
      \(renderTableHTML(headers: results.headers, rows: results.rows))
      \(renderRunnerLogsHTML(logs))
    """)
}

func run() throws {
    let options = CLIOptions.parse(Array(CommandLine.arguments.dropFirst()))
    let root = options.root
    let outDir = options.out ?? root.appendingPathComponent("docs/benchmark-page")
    let readme = try String(contentsOf: root.appendingPathComponent("README.md"), encoding: .utf8)
    let benchmarks = try String(contentsOf: root.appendingPathComponent("docs/benchmarks.md"), encoding: .utf8)
    let models = (try? String(contentsOf: root.appendingPathComponent("docs/models.md"), encoding: .utf8)) ?? ""

    var errors = validateClaims(readme: readme, benchmarks: benchmarks)
    let published = loadPublished08Metrics(modelsMarkdown: models)
    let loaded = loadGoldenItems(root: root, readme: readme, published: published)
    errors.append(contentsOf: loaded.errors)
    if !errors.isEmpty {
        errors.forEach { print($0) }
        exit(1)
    }

    let manifest = HFCollectionManifest(
        schemaVersion: 1,
        publish: false,
        requiresHumanReview: true,
        title: "basalt — Golden-Verified Apple Silicon Checkpoints",
        description: "Open-weight MLX checkpoints verified against reference top-64 logprobs and greedy token traces on every basalt inference change. Measurements include only logged 0.8 runs.",
        hardwareBaseline: "Apple M4 Pro 24 GB, macOS 27, Swift 6.4 release build",
        items: loaded.items
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let jsonText = String(decoding: try encoder.encode(manifest), as: UTF8.self) + "\n"
    let htmlText = renderHTML(
        items: loaded.items,
        families: referenceFamilies(root: root),
        harnesses: inspectHarnesses(root: root),
        benchmarksMarkdown: benchmarks,
        logs: parseRunnerLogs(in: options.logs)
    )

    let jsonURL = outDir.appendingPathComponent("hf-collection.json")
    let htmlURL = outDir.appendingPathComponent("index.html")
    if options.checkOnly {
        let existingJSON = try? String(contentsOf: jsonURL, encoding: .utf8)
        let existingHTML = try? String(contentsOf: htmlURL, encoding: .utf8)
        if existingJSON != jsonText || existingHTML != htmlText {
            print("docs/benchmark-page is out of date; run: swift scripts/build_benchmark_page.swift")
            exit(1)
        }
        print("benchmark page: up to date (\(loaded.items.count) golden checkpoints, all claims linked).")
        return
    }
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    try Data(jsonText.utf8).write(to: jsonURL, options: .atomic)
    try Data(htmlText.utf8).write(to: htmlURL, options: .atomic)
    print("Wrote \(jsonURL.path) and \(htmlURL.path)")
}

do {
    try run()
} catch {
    print("build_benchmark_page failed: \(error)")
    exit(1)
}
