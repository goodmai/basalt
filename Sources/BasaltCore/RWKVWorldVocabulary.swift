import Foundation

public struct RWKVWorldVocabulary: Sendable {
    public static let fileName = "rwkv_vocab_v20230424.txt"
    let tokens: [Int: [UInt8]]
    let ids: [[UInt8]: Int]
    let longest: [Int]

    public enum ParseError: Error, Equatable { case malformed(line: Int) }

    public init(contents: String) throws(ParseError) {
        var tokens: [Int: [UInt8]] = [:]
        for (number, line) in contents.split(separator: "\n", omittingEmptySubsequences: true).enumerated() {
            guard let first = line.firstIndex(of: " "), let last = line.lastIndex(of: " "), first < last,
                  let id = Int(line[..<first]), let length = Int(line[line.index(after: last)...]),
                  let bytes = Self.literal(line[line.index(after: first) ..< last]), bytes.count == length else {
                throw .malformed(line: number + 1)
            }
            tokens[id] = bytes
        }
        self.tokens = tokens
        ids = Dictionary(tokens.map { ($0.value, $0.key) }, uniquingKeysWith: min)
        var longest = Array(repeating: 0, count: 256)
        for bytes in tokens.values where !bytes.isEmpty { longest[Int(bytes[0])] = max(longest[Int(bytes[0])], bytes.count) }
        self.longest = longest
    }

    public func encode(_ text: String) -> [Int] {
        let bytes = Array(text.utf8)
        var (result, index) = ([Int](), 0)
        while index < bytes.count {
            var length = min(longest[Int(bytes[index])], bytes.count - index)
            while length > 0, ids[Array(bytes[index ..< index + length])] == nil { length -= 1 }
            guard length > 0, let id = ids[Array(bytes[index ..< index + length])] else { index += 1; continue }
            result.append(id)
            index += length
        }
        return result
    }

    public func decode(_ ids: [Int]) -> String {
        String(decoding: ids.flatMap { tokens[$0] ?? [] }, as: UTF8.self)
    }

    public func bytes(of id: Int) -> [UInt8]? { tokens[id] }
    public func id(of bytes: [UInt8]) -> Int? { ids[bytes] }

    static func literal(_ raw: Substring) -> [UInt8]? {
        var scalars = Array(raw.unicodeScalars)
        while scalars.first == " " { scalars.removeFirst() }
        while scalars.last == " " { scalars.removeLast() }
        let isBytes = scalars.first == "b"
        if isBytes { scalars.removeFirst() }
        guard let quote = scalars.first, quote == "'" || quote == "\"", scalars.count >= 2, scalars.last == quote else { return nil }
        let body = Array(scalars.dropFirst().dropLast())
        var out: [UInt8] = []
        var i = 0
        while i < body.count {
            guard body[i] == "\\", i + 1 < body.count else {
                out += isBytes ? [UInt8(truncatingIfNeeded: body[i].value)] : Array(String(body[i]).utf8)
                i += 1
                continue
            }
            guard let (scalar, width) = escape(body, at: i + 1) else { return nil }
            out += isBytes ? [UInt8(truncatingIfNeeded: scalar)] : Array(String(Unicode.Scalar(scalar) ?? "\u{FFFD}").utf8)
            i += 1 + width
        }
        return out
    }

    private static func escape(_ body: [Unicode.Scalar], at index: Int) -> (UInt32, Int)? {
        let simple: [Unicode.Scalar: UInt32] = ["n": 10, "t": 9, "r": 13, "\\": 92, "'": 39, "\"": 34, "0": 0, "a": 7, "b": 8, "f": 12, "v": 11]
        if let value = simple[body[index]] { return (value, 1) }
        let digits: Int? = switch body[index] { case "x": 2; case "u": 4; case "U": 8; default: nil }
        guard let digits, index + digits < body.count,
              let value = UInt32(String(String.UnicodeScalarView(body[(index + 1) ... (index + digits)])), radix: 16) else { return nil }
        return (value, digits + 1)
    }
}
