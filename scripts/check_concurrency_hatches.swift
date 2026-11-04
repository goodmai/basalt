#!/usr/bin/env swift
import Foundation

let allowlistPath = "scripts/concurrency-hatches.allow"
let regexStartAfter = Set("=([{,:;!&|?^~+-*%<>".unicodeScalars.map(Optional.some) + [nil])

let hatches: [(kind: String, pattern: Regex<AnyRegexOutput>)] = [
    ("@unchecked Sendable", #"@unchecked\s+Sendable\b"#),
    ("nonisolated(unsafe)", #"\bnonisolated\s*\(\s*unsafe\s*\)"#),
    ("NSLock", #"\bNSLock\b"#),
    ("NSRecursiveLock", #"\bNSRecursiveLock\b"#),
    ("NSCondition", #"\bNSCondition(?:Lock)?\b"#),
    ("DispatchQueue", #"\bDispatchQueue\b"#),
    ("DispatchSemaphore", #"\bDispatchSemaphore\b"#),
    ("os_unfair_lock", #"\bos_unfair_lock\w*"#),
    ("Task.detached", #"\bTask\s*(?:<[^<>{}()]*>)?\s*\.\s*(?:immediateDetached|detached)\b"#),
    ("Thread", #"\bThread\s*(?:\.\s*detachNewThread\b|[({])"#),
].map { ($0.0, try! Regex($0.1)) }

enum State {
    case code
    case lineComment
    case blockComment
    case string(hashes: Int, multiline: Bool)
    case regex(hashes: Int)
    case regexClass
    case interpolation(depth: Int)

    var endsAtNewline: Bool {
        switch self {
        case .lineComment, .regex, .string(_, multiline: false): true
        default: false
        }
    }

    var isCode: Bool {
        switch self {
        case .code, .interpolation: true
        default: false
        }
    }
}

struct Lexer {
    let text: [Unicode.Scalar]
    var stack: [State] = [.code]
    var i = 0

    func at(_ token: String) -> Bool {
        text[i...].starts(with: token.unicodeScalars)
    }

    func previousSignificant() -> Unicode.Scalar? {
        var j = i - 1
        while j >= 0, text[j] == " " || text[j] == "\t" { j -= 1 }
        return j < 0 || text[j] == "\n" ? nil : text[j]
    }

    mutating func skipEscape(_ length: Int) {
        i += length
        if i < text.count, text[i] != "\n" { i += 1 }
    }

    mutating func blockComment() {
        if at("/*") {
            stack.append(.blockComment)
            i += 2
        } else if at("*/") {
            stack.removeLast()
            i += 2
        } else {
            i += 1
        }
    }

    mutating func string(hashes: Int, multiline: Bool) {
        let pounds = String(repeating: "#", count: hashes)
        let escape = "\\" + pounds
        let closer = (multiline ? "\"\"\"" : "\"") + pounds
        if at(escape + "(") {
            stack.append(.interpolation(depth: 0))
            i += escape.count + 1
        } else if at(escape) {
            skipEscape(escape.count)
        } else if at(closer) {
            stack.removeLast()
            i += closer.count
        } else {
            i += 1
        }
    }

    mutating func regex(hashes: Int) {
        if text[i] == "[" {
            stack.append(.regexClass)
            i += 1
        } else if text[i] == "\\" {
            skipEscape(1)
        } else if at("/" + String(repeating: "#", count: hashes)) {
            stack.removeLast()
            i += 1 + hashes
        } else {
            i += 1
        }
    }

    mutating func regexClass() {
        if text[i] == "\\" {
            skipEscape(1)
            return
        }
        if text[i] == "]" { stack.removeLast() }
        i += 1
    }

    mutating func interpolation(depth: Int) {
        switch text[i] {
        case ")":
            if depth == 0 { stack.removeLast() } else { stack[stack.count - 1] = .interpolation(depth: depth - 1) }
            i += 1
        case "(":
            stack[stack.count - 1] = .interpolation(depth: depth + 1)
            i += 1
        default:
            code()
        }
    }

    mutating func code() {
        if at("//") || at("/*") {
            stack.append(at("//") ? .lineComment : .blockComment)
            i += 2
        } else if text[i] == "/", regexStartAfter.contains(previousSignificant()), !at("/ ") {
            stack.append(.regex(hashes: 0))
            i += 1
        } else {
            literal()
        }
    }

    mutating func literal() {
        var hashes = 0
        while i + hashes < text.count, text[i + hashes] == "#" { hashes += 1 }
        let rest = text[(i + hashes)...]
        if rest.starts(with: "\"\"\"".unicodeScalars) {
            stack.append(.string(hashes: hashes, multiline: true))
            i += hashes + 3
        } else if rest.first == "\"" {
            stack.append(.string(hashes: hashes, multiline: false))
            i += hashes + 1
        } else if hashes > 0, rest.first == "/" {
            stack.append(.regex(hashes: hashes))
            i += hashes + 1
        } else {
            i += max(hashes, 1)
        }
    }

    mutating func step() {
        switch stack[stack.count - 1] {
        case .code: code()
        case .lineComment: i += 1
        case .blockComment: blockComment()
        case .string(let hashes, let multiline): string(hashes: hashes, multiline: multiline)
        case .regex(let hashes): regex(hashes: hashes)
        case .regexClass: regexClass()
        case .interpolation(let depth): interpolation(depth: depth)
        }
    }

    mutating func codeOnly() -> String {
        var kept = text
        while i < text.count {
            if text[i] == "\n" {
                while stack.last?.endsAtNewline == true { stack.removeLast() }
                i += 1
                continue
            }
            let start = i, isCode = stack[stack.count - 1].isCode
            step()
            if !isCode {
                for j in start ..< min(i, text.count) where text[j] != "\n" { kept[j] = " " }
            }
        }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: kept)
        return String(view)
    }
}

struct Hit {
    let path: String
    let line: Int
    let kind: String
}

func hits(in path: String, source: String) -> [Hit] {
    var lexer = Lexer(text: Array(source.unicodeScalars))
    let code = lexer.codeOnly()
    return hatches.flatMap { hatch in
        code.matches(of: hatch.pattern).map { match in
            Hit(path: path, line: 1 + code[..<match.range.lowerBound].count(where: \.isNewline), kind: hatch.kind)
        }
    }
}

func sourceHits() throws -> [Hit] {
    let files = FileManager.default.enumerator(atPath: "Sources")?.compactMap { $0 as? String }.filter { $0.hasSuffix(".swift") } ?? []
    return try files.sorted().flatMap { file in
        let path = "Sources/\(file)"
        return hits(in: path, source: try String(contentsOfFile: path, encoding: .utf8))
    }
}

struct Malformed: Error, CustomStringConvertible {
    let line: Int
    let text: String
    var description: String { "\(allowlistPath):\(line): expected `path:symbol: reason`, got `\(text)`" }
}

func allowed() throws -> [String: Int] {
    guard FileManager.default.fileExists(atPath: allowlistPath) else { return [:] }
    var counts: [String: Int] = [:]
    let rows = try String(contentsOfFile: allowlistPath, encoding: .utf8).split(separator: "\n", omittingEmptySubsequences: false)
    for (index, row) in rows.enumerated() where !row.trimmingCharacters(in: .whitespaces).isEmpty {
        guard let pathEnd = row.firstIndex(of: ":"), let symbolEnd = row[row.index(after: pathEnd)...].firstRange(of: ": "),
              !row[row.index(after: pathEnd) ..< symbolEnd.lowerBound].isEmpty,
              !row[symbolEnd.upperBound...].trimmingCharacters(in: .whitespaces).isEmpty
        else { throw Malformed(line: index + 1, text: String(row)) }
        counts[String(row[..<symbolEnd.lowerBound]), default: 0] += 1
    }
    return counts
}

do {
    let found = Dictionary(grouping: try sourceHits()) { "\($0.path):\($0.kind)" }
    let allowance = try allowed()
    var problems: [String] = []
    for key in found.keys.sorted() where found[key]!.count > allowance[key, default: 0] {
        problems += found[key]!.map { "\($0.path):\($0.line): \($0.kind)" }
    }
    for key in allowance.keys.sorted() where allowance[key]! > found[key]?.count ?? 0 {
        problems.append("\(allowlistPath): \(key) is listed \(allowance[key]!) time(s) but found \(found[key]?.count ?? 0): remove the stale line")
    }
    for problem in problems { print(problem) }
    if problems.isEmpty {
        print("No concurrency hatch outside \(allowlistPath).")
    } else {
        print("\n\(problems.count) problem(s). Use a value type, an actor, `sending`, `Mutex` or an atomic; a hatch that must stay gets a line `path:symbol: why it is needed and what serializes it` in \(allowlistPath), which only shrinks (see CONTRIBUTING.md).")
        exit(1)
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
