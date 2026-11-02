import Foundation
import MLX
import MLXLMCommon

struct DRYPenalty: LogitProcessor {
    static let defaultBase: Float = 1.75
    static let defaultAllowedLength = 2
    static let defaultSequenceBreakers = ["\n", ":", "\"", "*"]
    static let window = 2048
    static let maximumPenalty: Float = 10_000

    private final class History {
        var tokens: [Int] = []
        var unread: MLXArray?
        var reversed: [Int] = []
        var matches: [Int] = []
        var penalties: [Int: Float] = [:]

        init() {}

        init(copying other: History) {
            tokens = other.tokens
            unread = other.unread
        }
    }

    let multiplier: Float
    let base: Float
    let allowedLength: Int
    let breakers: Set<Int>
    private var history = History()

    init(multiplier: Float, base: Float = defaultBase, allowedLength: Int = defaultAllowedLength, breakers: Set<Int>) {
        self.multiplier = multiplier
        self.base = base
        self.allowedLength = allowedLength
        self.breakers = breakers
    }

    static func breakerIds(_ breakers: [String], tokenizer: any MLXLMCommon.Tokenizer) -> Set<Int> {
        var ids: Set<Int> = []
        var id = 0
        while tokenizer.convertIdToToken(id) != nil {
            let piece = tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
            if breakers.contains(where: piece.contains) { ids.insert(id) }
            id += 1
        }
        return ids
    }

    mutating func prompt(_ prompt: MLXArray) {
        history.tokens = Array(prompt.asArray(Int.self).suffix(Self.window))
    }

    func process(logits: MLXArray) -> MLXArray {
        readSampledToken()
        let penalties = continuationPenalties()
        guard !penalties.isEmpty else { return logits }
        let indices = MLXArray(penalties.keys.map(Int32.init))[.newAxis, 0...]
        let amounts = MLXArray(Array(penalties.values))[.newAxis, 0...].asType(logits.dtype)
        return putAlong(logits, indices, values: takeAlong(logits, indices, axis: -1) - amounts, axis: -1)
    }

    mutating func didSample(token: MLXArray) {
        readSampledToken()
        history.unread = token
    }

    func copy() -> DRYPenalty {
        var copy = self
        copy.history = History(copying: history)
        return copy
    }

    private func readSampledToken() {
        guard let token = history.unread else { return }
        history.unread = nil
        history.tokens.append(token.item(Int.self))
        if history.tokens.count > 2 * Self.window {
            history.tokens.removeFirst(history.tokens.count - Self.window)
        }
    }

    private func continuationPenalties() -> [Int: Float] {
        history.penalties.removeAll(keepingCapacity: true)
        let tail = history.tokens.suffix(Self.window)
        let longest = tail.reversed().prefix { !breakers.contains($0) }.count
        guard tail.count > allowedLength, longest >= allowedLength else { return history.penalties }
        history.reversed.removeAll(keepingCapacity: true)
        history.reversed.append(contentsOf: tail.reversed())
        Self.prefixMatches(of: history.reversed, into: &history.matches)
        for start in 1 ..< history.reversed.count {
            let length = min(history.matches[start], longest)
            guard length >= allowedLength else { continue }
            let penalty = min(multiplier * pow(base, Float(length - allowedLength)), Self.maximumPenalty)
            let token = history.reversed[start - 1]
            history.penalties[token] = max(history.penalties[token] ?? 0, penalty)
        }
        return history.penalties
    }

    static func prefixMatches(of tokens: [Int], into matches: inout [Int]) {
        let count = tokens.count
        matches.removeAll(keepingCapacity: true)
        matches.append(contentsOf: repeatElement(0, count: count))
        var left = 0
        var right = 0
        for index in 1 ..< max(count, 1) {
            var length = index < right ? min(right - index, matches[index - left]) : 0
            while index + length < count, tokens[length] == tokens[index + length] { length += 1 }
            matches[index] = length
            if index + length > right { (left, right) = (index, index + length) }
        }
    }
}
