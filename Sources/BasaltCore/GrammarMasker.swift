import Foundation
import Synchronization

public struct TokenMask: Sendable, Equatable {
    public private(set) var words: [UInt32]

    public init(size: Int) {
        words = [UInt32](repeating: 0, count: (size + 31) / 32)
    }

    public mutating func insert(_ id: Int) {
        guard id >= 0, id >> 5 < words.count else { return }
        words[id >> 5] |= 1 << UInt32(id & 31)
    }

    public func contains(_ id: Int) -> Bool {
        id >= 0 && id >> 5 < words.count && words[id >> 5] & (1 << UInt32(id & 31)) != 0
    }

    public var isEmpty: Bool { words.allSatisfy { $0 == 0 } }
}

public final class TokenVocabulary: Sendable {
    public let size: Int
    let bytes: [[UInt8]]
    let sorted: [Int32]

    public init(tokenBytes: [[UInt8]]) {
        size = tokenBytes.count
        bytes = tokenBytes
        sorted = tokenBytes.indices.filter { !tokenBytes[$0].isEmpty }
            .sorted { tokenBytes[$0].lexicographicallyPrecedes(tokenBytes[$1]) }
            .map(Int32.init)
    }

    public func bytes(of id: Int) -> [UInt8] {
        bytes.indices.contains(id) ? bytes[id] : []
    }

    public struct Unreadable: Error, Sendable {
        public let reason: String
    }

    public static func load(tokenizerJSON data: Data) throws(Unreadable) -> TokenVocabulary {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any] else {
            throw Unreadable(reason: "tokenizer.json has no model section")
        }
        var pieces: [(String, Int)] = []
        if let vocab = model["vocab"] as? [String: Int] {
            pieces = vocab.map { ($0.key, $0.value) }
        } else if let vocab = model["vocab"] as? [[Any]] {
            pieces = vocab.enumerated().compactMap { index, entry in (entry.first as? String).map { ($0, index) } }
        } else {
            throw Unreadable(reason: "tokenizer.json has no readable vocab")
        }
        let added = ((root["added_tokens"] as? [[String: Any]]) ?? []).compactMap { $0["id"] as? Int }
        let decoder = root["decoder"].flatMap { try? JSONSerialization.data(withJSONObject: $0) }
            .flatMap { String(data: $0, encoding: .utf8) } ?? ""
        let byteLevel = decoder.contains("ByteLevel")
        let size = max(pieces.map(\.1).max() ?? -1, added.max() ?? -1) + 1
        var tokenBytes = [[UInt8]](repeating: [], count: size)
        for (piece, id) in pieces where id >= 0 {
            tokenBytes[id] = (byteLevel ? byteLevelBytes(piece) : sentencePieceBytes(piece)) ?? []
        }
        for id in added where id >= 0 { tokenBytes[id] = [] }
        return TokenVocabulary(tokenBytes: tokenBytes)
    }

    private static let byteLevelTable: [Unicode.Scalar: UInt8] = {
        let printable = Array(33...126) + Array(161...172) + Array(174...255)
        var table: [Unicode.Scalar: UInt8] = [:]
        var shifted = 0
        for byte in 0...255 {
            if printable.contains(byte) {
                table[Unicode.Scalar(UInt8(byte))] = UInt8(byte)
            } else {
                table[Unicode.Scalar(256 + shifted)!] = UInt8(byte)
                shifted += 1
            }
        }
        return table
    }()

    static func byteLevelBytes(_ piece: String) -> [UInt8]? {
        var out: [UInt8] = []
        for scalar in piece.unicodeScalars {
            guard let byte = byteLevelTable[scalar] else { return nil }
            out.append(byte)
        }
        return out
    }

    static func sentencePieceBytes(_ piece: String) -> [UInt8]? {
        if piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
           let byte = UInt8(piece.dropFirst(3).prefix(2), radix: 16) {
            return [byte]
        }
        return Array(piece.replacingOccurrences(of: "\u{2581}", with: " ").utf8)
    }
}

public final class GrammarMasker: Sendable {
    public let grammar: JSONGrammar
    public let vocabulary: TokenVocabulary
    public static let defaultCacheBytes = 8 << 20
    let cacheBytes: Int

    private struct Partition {
        var accepted: TokenMask
        var dependent: [Int32]
    }

    private struct Caches {
        var contextIndependent: [GrammarFrame: Partition] = [:]
        var byState: [[GrammarFrame]: TokenMask] = [:]
        var bytes = 0
    }

    private let caches = Mutex(Caches())

    public init(grammar: JSONGrammar, vocabulary: TokenVocabulary, cacheBytes: Int = defaultCacheBytes) {
        self.grammar = grammar
        self.vocabulary = vocabulary
        self.cacheBytes = cacheBytes
    }

    var cachedBytes: Int { caches.withLock { $0.bytes } }

    public func mask(for matcher: JSONGrammarMatcher) -> TokenMask {
        caches.withLock { caches in
            if let known = caches.byState[matcher.stack] { return known }
            let top = matcher.stack[matcher.stack.count - 1]
            let maskBytes = MemoryLayout<UInt32>.stride * (vocabulary.size + 31) / 32
            let frameBytes = MemoryLayout<GrammarFrame>.stride
            if caches.contextIndependent[top] == nil {
                var accepted = TokenMask(size: vocabulary.size)
                let local = JSONGrammarMatcher(grammar: grammar, stack: [top])
                let dependent = scan(vocabulary.sorted, from: local, into: &accepted)
                Self.make(room: maskBytes + frameBytes + 4 * dependent.count, in: &caches, limit: cacheBytes)
                caches.contextIndependent[top] = Partition(accepted: accepted, dependent: dependent)
            }
            let partition = caches.contextIndependent[top]!
            var mask = partition.accepted
            _ = scan(partition.dependent, from: matcher, into: &mask)
            Self.make(room: maskBytes + frameBytes * matcher.stack.count, in: &caches, limit: cacheBytes)
            caches.byState[matcher.stack] = mask
            return mask
        }
    }

    private static func make(room bytes: Int, in caches: inout Caches, limit: Int) {
        if caches.bytes + bytes > limit {
            caches.contextIndependent.removeAll()
            caches.byState.removeAll()
            caches.bytes = 0
        }
        caches.bytes += bytes
    }

    private func scan(_ ids: [Int32], from start: JSONGrammarMatcher, into accepted: inout TokenMask) -> [Int32] {
        var dependent: [Int32] = []
        var states = [start]
        var validDepth = 0
        var previous: [UInt8] = []
        var index = 0
        while index < ids.count {
            let token = vocabulary.bytes[Int(ids[index])]
            var depth = min(validDepth, Self.sharedPrefixLength(previous, token))
            var outcome = JSONGrammarMatcher.Step.consumed
            while depth < token.count {
                var next = states[depth]
                outcome = next.advance(token[depth])
                guard outcome == .consumed else { break }
                depth += 1
                if depth < states.count { states[depth] = next } else { states.append(next) }
            }
            validDepth = depth
            guard outcome != .consumed else {
                accepted.insert(Int(ids[index]))
                previous = token
                index += 1
                continue
            }
            repeat {
                if outcome == .needsContext { dependent.append(ids[index]) }
                previous = vocabulary.bytes[Int(ids[index])]
                index += 1
            } while index < ids.count && Self.sharedPrefixLength(vocabulary.bytes[Int(ids[index])], token) > depth
        }
        return dependent
    }

    private static func sharedPrefixLength(_ a: [UInt8], _ b: [UInt8]) -> Int {
        var length = 0
        let limit = min(a.count, b.count)
        while length < limit && a[length] == b[length] { length += 1 }
        return length
    }
}
