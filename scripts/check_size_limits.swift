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
let maxTypeLines = 200
let maxExtensionLines = 150
let legacyTypeTarget = 450
let lizard = (ProcessInfo.processInfo.environment["LIZARD"] ?? "uvx --quiet lizard==1.24.0").split(separator: " ").map(String.init)

@discardableResult
func run(_ arguments: [String]) throws -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = arguments
    let output = Pipe()
    process.standardOutput = output
    process.standardError = FileHandle.nullDevice
    try process.run()
    let data = output.fileHandleForReading.readDataToEndOfFile()
    try output.fileHandleForReading.close()
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


enum Lexeme {
    case code(parens: Int)
    case string(multiline: Bool, hashes: Int)
    case blockComment
}

struct Lexer {
    let text: [Character]
    var out: [Character] = []
    var stack: [Lexeme] = [.code(parens: 0)]
    var i = 0

    func at(_ token: String) -> Bool {
        Array(token).enumerated().allSatisfy { i + $0.offset < text.count && text[i + $0.offset] == $0.element }
    }

    mutating func blank() {
        out.append(text[i] == "\n" ? "\n" : " ")
        i += 1
    }

    mutating func comment() {
        if at("*/") {
            stack.removeLast()
            i += 2
        } else if at("/*") {
            stack.append(.blockComment)
            i += 2
        } else {
            blank()
        }
    }

    mutating func string(multiline: Bool, hashes: Int) {
        let pounds = String(repeating: "#", count: hashes)
        let closing = (multiline ? "\"\"\"" : "\"") + pounds
        if at("\\" + pounds + "(") {
            stack.append(.code(parens: 1))
            i += 2 + hashes
        } else if hashes == 0, at("\\") {
            i += 2
        } else if at(closing) {
            stack.removeLast()
            i += closing.count
        } else {
            blank()
        }
    }

    mutating func openString() -> Bool {
        var hashes = 0
        while i + hashes < text.count, text[i + hashes] == "#" { hashes += 1 }
        let quote = i + hashes
        guard quote < text.count, text[quote] == "\"" else { return false }
        let multiline = quote + 2 < text.count && text[quote + 1] == "\"" && text[quote + 2] == "\""
        stack.append(.string(multiline: multiline, hashes: hashes))
        i = quote + (multiline ? 3 : 1)
        return true
    }

    mutating func code(parens: Int) {
        let c = text[i]
        if at("//") {
            while i < text.count, text[i] != "\n" { i += 1 }
        } else if at("/*") {
            stack.append(.blockComment)
            i += 2
        } else if (c == "#" || c == "\"") && openString() {
            return
        } else if stack.count > 1, c == "(" || c == ")" {
            let depth = parens + (c == "(" ? 1 : -1)
            if depth == 0 { stack.removeLast() } else { stack[stack.count - 1] = .code(parens: depth) }
            i += 1
        } else {
            out.append(c)
            i += 1
        }
    }

    mutating func run() -> [Character] {
        while i < text.count {
            switch stack[stack.count - 1] {
            case .blockComment: comment()
            case .string(let multiline, let hashes): string(multiline: multiline, hashes: hashes)
            case .code(let parens): code(parens: parens)
            }
        }
        return out
    }
}

func codeOnly(_ source: String) -> [Character] {
    var lexer = Lexer(text: Array(source))
    return lexer.run()
}

let declaration = try! Regex(#"^\s*(?:@[\w.]+(?:\([^)]*\))?\s+)*(?:(?:public|private|fileprivate|internal|package|open|final|indirect|nonisolated)\s+)*(struct|class|enum|actor|extension|protocol)\s+([A-Za-z_][\w.]*)"#)

func typeBodies(in source: String?) -> [String: Metrics] {
    guard let source else { return [:] }
    let code = codeOnly(source)
    let lines = String(code).split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    var starts: [Int] = [0]
    for line in lines { starts.append(starts[starts.count - 1] + line.count + 1) }
    var found: [String: Metrics] = [:], seen: [String: Int] = [:]
    for (number, line) in lines.enumerated() {
        guard let match = line.firstMatch(of: declaration), let kind = match.output[1].substring.map(String.init),
              let name = match.output[2].substring.map(String.init) else { continue }
        if kind == "class", ["func", "var", "let", "subscript", "init"].contains(name) { continue }
        var index = starts[number] + line.distance(from: line.startIndex, to: match.range.upperBound)
        while index < code.count, code[index] != "{", code[index] != ";" { index += 1 }
        guard index < code.count, code[index] == "{" else { continue }
        var depth = 0, end = index
        repeat {
            if code[end] == "{" { depth += 1 } else if code[end] == "}" { depth -= 1 }
            end += 1
        } while depth > 0 && end < code.count
        let body = String(code[index ..< end]).split(separator: "\n").dropFirst().dropLast()
        let nloc = body.filter { !$0.allSatisfy(\.isWhitespace) }.count
        let key = "\(kind) \(name)"
        seen[key, default: 0] += 1
        found[seen[key]! > 1 ? "\(key) #\(seen[key]!)" : key] = Metrics(ccn: 0, nloc: nloc, params: 0, line: number + 1)
    }
    return found
}

func typeViolations(path: String, before: [String: Metrics], after: [String: Metrics]) -> [String] {
    after.sorted(by: { $0.key < $1.key }).compactMap { name, metrics in
        let limit = name.hasPrefix("extension ") ? maxExtensionLines : maxTypeLines
        let previous = before[name]?.nloc ?? 0
        guard metrics.nloc > limit, metrics.nloc > previous else { return nil }
        return "\(path):\(metrics.line): \(name): \(metrics.nloc) lines of code > \(limit)\(previous > limit ? ", was \(previous)" : "")"
    }
}

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments.contains("--inventory") {
    let files = FileManager.default.enumerator(atPath: "Sources")?.compactMap { $0 as? String }.filter { $0.hasSuffix(".swift") } ?? []
    for file in files.sorted() {
        let source = try String(contentsOfFile: "Sources/\(file)", encoding: .utf8)
        for (name, metrics) in typeBodies(in: source).sorted(by: { $0.key < $1.key }) where metrics.nloc > legacyTypeTarget {
            print("Sources/\(file):\(metrics.line): \(name): \(metrics.nloc) lines of code > \(legacyTypeTarget)")
        }
    }
    exit(0)
}
let base = arguments.first { !$0.hasPrefix("--") } ?? "origin/main"
let paths = try git("diff", "--name-only", "--diff-filter=d", "-z", "\(base)...HEAD", "--", "*.swift").split(separator: "\0").map(String.init)
var hits: [String] = []
var touched = 0, sums = (before: 0, after: 0), peaks = (before: 0, after: 0)
for path in paths {
    let base = try? git("show", "\(base):\(path)")
    let head = (try? git("show", "HEAD:\(path)")) ?? ""
    let (before, after) = (try functions(in: base), try functions(in: head))
    hits += violations(path: path, before: before, after: after, base: base, head: head)
    hits += typeViolations(path: path, before: typeBodies(in: base), after: typeBodies(in: head))
    guard path.hasPrefix("Sources/") else { continue }
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
    print("Impact: \(paths.filter { $0.hasPrefix("Sources/") }.count) Swift file(s) in Sources, \(touched) of \(total) functions in Sources touched (\(String(format: "%.1f", share))%), "
        + "CCN sum \(sums.before) → \(sums.after), largest CCN \(peaks.before) → \(peaks.after)")
}
for hit in hits { print(hit) }
if hits.isEmpty {
    print("No size or complexity limit exceeded or grown.")
} else {
    print("\n\(hits.count) limit(s) exceeded or grown. Split along an existing seam (see \"Size and complexity\" in CONTRIBUTING.md).")
    exit(1)
}
