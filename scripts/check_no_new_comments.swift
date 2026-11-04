#!/usr/bin/env swift
import Foundation

let directive = try! Regex(#"^//\s*swift-tools-version:"#)
let hunk = try! Regex(#"^@@ -\S+ \+(\d+)(?:,(\d+))? @@"#)
let regexStartAfter = Set("=([{,:;!&|?^~+-*%<>".unicodeScalars.map(Optional.some) + [nil])

struct GitFailure: Error {
    let arguments: [String]
}

func git(_ arguments: String...) throws -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["git", "-c", "core.quotePath=false"] + arguments
    let output = Pipe()
    process.standardOutput = output
    process.standardError = FileHandle(forWritingAtPath: "/dev/null")
    try process.run()
    let data = output.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else { throw GitFailure(arguments: arguments) }
    return String(decoding: data, as: UTF8.self)
}

func changedSwiftFiles(base: String) throws -> [String] {
    try git("diff", "--name-only", "--diff-filter=d", "-z", "\(base)...HEAD", "--", "*.swift")
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
}

struct Lexer {
    let text: [Unicode.Scalar]
    var stack: [State] = [.code]
    var found: Set<Int> = []
    var line = 1
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

    mutating func newline() {
        while stack.last?.endsAtNewline == true { stack.removeLast() }
        line += 1
        i += 1
    }

    mutating func blockComment() {
        found.insert(line)
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
            found.insert(line)
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

    mutating func commentLines() -> Set<Int> {
        while i < text.count {
            if text[i] == "\n" {
                newline()
                continue
            }
            switch stack[stack.count - 1] {
            case .code: code()
            case .lineComment:
                found.insert(line)
                i += 1
            case .blockComment: blockComment()
            case .string(let hashes, let multiline): string(hashes: hashes, multiline: multiline)
            case .regex(let hashes): regex(hashes: hashes)
            case .regexClass: regexClass()
            case .interpolation(let depth): interpolation(depth: depth)
            }
        }
        return found
    }
}

func commentHits(base: String) throws -> [String] {
    var hits: [String] = []
    for path in try changedSwiftFiles(base: base) {
        let source = try git("show", "HEAD:\(path)")
        let rows = source.unicodeScalars.split(separator: "\n", omittingEmptySubsequences: false).map { String(String.UnicodeScalarView($0)) }
        var lexer = Lexer(text: Array(source.unicodeScalars))
        for number in lexer.commentLines().intersection(try addedLineNumbers(base: base, path: path)).sorted() {
            let text = number <= rows.count ? rows[number - 1] : ""
            if path == "Package.swift", number == 1, text.prefixMatch(of: directive) != nil { continue }
            hits.append("\(path):\(number): \(text.trimmingCharacters(in: .whitespacesAndNewlines))")
        }
    }
    return hits
}

do {
    let hits = try commentHits(base: CommandLine.arguments.dropFirst().first ?? "origin/main")
    for hit in hits { print(hit) }
    if hits.isEmpty {
        print("No comments added.")
    } else {
        print("\n\(hits.count) added comment line(s). Move the explanation to the commit message, the PR or a test name (see CONTRIBUTING.md).")
        exit(1)
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
