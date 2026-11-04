#!/usr/bin/env swift
import Foundation

let hunk = try! Regex(#"^@@ -\S+ \+(\d+)(?:,(\d+))? @@"#)

struct GitFailure: Error {
    let arguments: [String]
}

func git(_ arguments: String...) throws -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["git", "-c", "core.quotePath=false"] + arguments
    let output = Pipe(), discard = FileHandle(forWritingAtPath: "/dev/null")
    process.standardOutput = output
    process.standardError = discard
    try process.run()
    let data = output.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    try output.fileHandleForReading.close()
    try discard?.close()
    guard process.terminationStatus == 0 else { throw GitFailure(arguments: arguments) }
    return String(decoding: data, as: UTF8.self)
}

func changedSources(base: String) throws -> [String] {
    try git("diff", "--name-only", "--diff-filter=d", "-z", "\(base)...HEAD", "--", "Sources/*.swift")
        .split(separator: "\0").map(String.init)
}

func addedLineNumbers(base: String, path: String) throws -> Set<Int> {
    var lines: Set<Int> = []
    let diff = try git("diff", "--no-color", "--no-ext-diff", "--unified=0", "\(base)...HEAD", "--", path)
    for row in diff.split(whereSeparator: \.isNewline) {
        guard let match = row.prefixMatch(of: hunk), let start = match.output[1].substring.flatMap({ Int($0) }) else { continue }
        let count = match.output[2].substring.flatMap { Int($0) } ?? 1
        lines.formUnion(start ..< start + count)
    }
    return lines
}

struct Segment: Decodable {
    let line: Int
    let count: Double
    let hasCount: Bool
    let isRegionEntry: Bool
    let isGap: Bool

    init(from decoder: any Decoder) throws {
        var fields = try decoder.unkeyedContainer()
        line = try fields.decode(Int.self)
        _ = try fields.decode(Int.self)
        count = try fields.decode(Double.self)
        hasCount = try fields.decode(Bool.self)
        isRegionEntry = try fields.decode(Bool.self)
        isGap = try fields.decode(Bool.self)
    }

    var startsRegion: Bool { hasCount && isRegionEntry && !isGap }
}

struct Export: Decodable {
    struct Unit: Decodable {
        let files: [File]
    }

    struct File: Decodable {
        let filename: String
        let segments: [Segment]
    }

    let data: [Unit]
}

func coverage(of segments: [Segment], at lines: Set<Int>) -> [Int: Bool] {
    var result: [Int: Bool] = [:]
    var wrapped: Segment?
    var next = 0
    for line in 1 ..< (lines.max() ?? 0) + 1 {
        let start = next
        while next < segments.count, segments[next].line <= line { next += 1 }
        let onLine = segments[start ..< next]
        defer { if let last = onLine.last { wrapped = last } }
        guard lines.contains(line) else { continue }
        let starts = onLine.filter(\.startsRegion)
        let skipped = onLine.first.map { !$0.hasCount && $0.isRegionEntry } ?? false
        guard !skipped, wrapped?.hasCount == true || !starts.isEmpty else { continue }
        result[line] = max(wrapped?.count ?? 0, starts.map(\.count).max() ?? 0) > 0
    }
    return result
}

func ranges(_ lines: [Int]) -> String {
    var parts: [String] = []
    var index = 0
    while index < lines.count {
        var end = index
        while end + 1 < lines.count, lines[end + 1] == lines[end] + 1 { end += 1 }
        parts.append(end == index ? "\(lines[index])" : "\(lines[index])-\(lines[end])")
        index = end + 1
    }
    return parts.joined(separator: ", ")
}

struct Usage: Error, CustomStringConvertible {
    var description: String { "usage: check_changed_coverage.swift <codecov.json> [base, default origin/main] [--min percent, default 60]" }
}

var arguments = Array(CommandLine.arguments.dropFirst())
var minimum = 60.0
do {
    if let flag = arguments.firstIndex(of: "--min") {
        guard flag + 1 < arguments.count, let value = Double(arguments[flag + 1]) else { throw Usage() }
        minimum = value
        arguments.removeSubrange(flag ... flag + 1)
    }
    guard (1 ... 2).contains(arguments.count) else { throw Usage() }
    let base = arguments.count == 2 ? arguments[1] : "origin/main"
    let files = try JSONDecoder().decode(Export.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[0])))
        .data.flatMap(\.files).filter { !$0.filename.contains("/.build/") }
    var covered = 0, executable = 0
    for path in try changedSources(base: base) {
        guard let file = files.first(where: { $0.filename.hasSuffix("/" + path) }) else {
            print("\(path): not in the coverage report (no test target builds it), not counted")
            continue
        }
        let lines = coverage(of: file.segments, at: try addedLineNumbers(base: base, path: path))
        guard !lines.isEmpty else { continue }
        let missed = lines.filter { !$0.value }.keys.sorted()
        covered += lines.count - missed.count
        executable += lines.count
        print("\(path): \(lines.count - missed.count)/\(lines.count) changed executable lines covered" + (missed.isEmpty ? "" : "; uncovered: \(ranges(missed))"))
    }
    guard executable > 0 else {
        print("No changed executable lines in Sources.")
        exit(0)
    }
    let percent = Double(covered) * 100 / Double(executable)
    print("\nChanged-line coverage: \(String(format: "%.1f", percent))% (\(covered)/\(executable)), minimum \(String(format: "%g", minimum))%.")
    if percent < minimum {
        print("Below the minimum: add a test that runs the uncovered lines (see CONTRIBUTING.md).")
        exit(1)
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
