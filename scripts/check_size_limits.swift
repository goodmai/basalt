#!/usr/bin/env swift
import Foundation

struct Metrics: Equatable {
    var ccn: Int
    var nloc: Int
    var params: Int
    var line: Int
}

let maxCCN = 15
let maxNLOC = 60
let maxParams = 6
let maxFileLines = 600
let lizard = (ProcessInfo.processInfo.environment["LIZARD"] ?? "uvx --quiet lizard==1.24.0").split(separator: " ").map(String.init)

@discardableResult
func run(_ arguments: [String], input: String? = nil) throws -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = arguments
    let output = Pipe()
    process.standardOutput = output
    process.standardError = FileHandle.nullDevice
    try process.run()
    let data = output.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else { throw CocoaError(.executableLoad) }
    return String(decoding: data, as: UTF8.self)
}

func git(_ arguments: String...) throws -> String {
    try run(["git", "-c", "core.quotePath=false"] + arguments)
}

func csvFields(_ line: Substring) -> [String] {
    var fields: [String] = [], field = "", quoted = false
    var characters = line.makeIterator()
    while let character = characters.next() {
        switch character {
        case "\"" where quoted: quoted = false
        case "\"": quoted = true
        case "," where !quoted:
            fields.append(field)
            field = ""
        default: field.append(character)
        }
    }
    fields.append(field)
    return fields
}

func functions(in source: String?) throws -> [String: Metrics] {
    guard let source else { return [:] }
    let file = FileManager.default.temporaryDirectory.appending(path: "size-\(UUID().uuidString).swift")
    try source.write(to: file, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: file) }
    var found: [String: Metrics] = [:]
    for row in try run(lizard + ["-l", "swift", "--csv", file.path]).split(separator: "\n") {
        let fields = csvFields(row)
        guard fields.count >= 10, let nloc = Int(fields[0]), let ccn = Int(fields[1]), let params = Int(fields[3]),
              let line = Int(fields[9]) else { continue }
        let metrics = Metrics(ccn: ccn, nloc: nloc, params: params, line: line)
        if let known = found[fields[8]], (known.ccn, known.nloc, known.params) >= (ccn, nloc, params) { continue }
        found[fields[8]] = metrics
    }
    return found
}

func violations(path: String, before: [String: Metrics], after: [String: Metrics], base: String?, head: String) -> [String] {
    var hits: [String] = []
    for (name, metrics) in after.sorted(by: { $0.key < $1.key }) {
        let was = before[name]
        for (metric, value, previous, limit) in [
            ("CCN", metrics.ccn, was?.ccn ?? 0, maxCCN), ("lines of code", metrics.nloc, was?.nloc ?? 0, maxNLOC),
            ("parameters", metrics.params, was?.params ?? 0, maxParams),
        ] where value > limit && value > previous {
            hits.append("\(path):\(metrics.line): \(name): \(metric) \(value) > \(limit)\(previous > limit ? ", was \(previous)" : "")")
        }
    }
    let lines = head.filter { $0 == "\n" }.count, wasLines = (base ?? "").filter { $0 == "\n" }.count
    if !path.hasPrefix("Tests/"), lines > maxFileLines, lines > wasLines {
        hits.append("\(path): \(lines) lines > \(maxFileLines)\(wasLines > maxFileLines ? ", was \(wasLines)" : "")")
    }
    return hits
}

let arguments = Array(CommandLine.arguments.dropFirst())
let base = arguments.first { !$0.hasPrefix("--") } ?? "origin/main"
let paths = try git("diff", "--name-only", "--diff-filter=d", "-z", "\(base)...HEAD", "--", "*.swift").split(separator: "\0").map(String.init)
var hits: [String] = []
var touched = 0, sums = (before: 0, after: 0), peaks = (before: 0, after: 0)
for path in paths {
    let base = try? git("show", "\(base):\(path)")
    let head = (try? git("show", "HEAD:\(path)")) ?? ""
    let (before, after) = (try functions(in: base), try functions(in: head))
    hits += violations(path: path, before: before, after: after, base: base, head: head)
    touched += Set(before.keys).union(after.keys).filter { name in
        before[name].map { [$0.ccn, $0.nloc, $0.params] } != after[name].map { [$0.ccn, $0.nloc, $0.params] }
    }.count
    sums.before += before.values.map(\.ccn).reduce(0, +)
    sums.after += after.values.map(\.ccn).reduce(0, +)
    peaks = (max(peaks.before, before.values.map(\.ccn).max() ?? 0), max(peaks.after, after.values.map(\.ccn).max() ?? 0))
}
if arguments.contains("--impact") {
    let total = try run(lizard + ["-l", "swift", "--csv", "Sources"]).split(separator: "\n").count
    let share = total == 0 ? 0 : Double(touched) / Double(total) * 100
    print("Impact: \(paths.count) Swift file(s), \(touched) of \(total) functions in Sources touched (\(String(format: "%.1f", share))%), "
        + "CCN sum \(sums.before) → \(sums.after), largest CCN \(peaks.before) → \(peaks.after)")
}
for hit in hits { print(hit) }
if hits.isEmpty {
    print("No size or complexity limit exceeded or grown.")
} else {
    print("\n\(hits.count) limit(s) exceeded or grown. Split along an existing seam (see \"Size and complexity\" in CONTRIBUTING.md).")
    exit(1)
}
