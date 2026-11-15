import Foundation

public struct SentenceStreamChunker: Sendable, Equatable {
    public static let defaultClauseMinChars = 80

    private static let immediateTerminators: Set<Character> = ["\n", "!", "?", "…", "。", "！", "？"]
    private static let clauseDelimiters: Set<Character> = [";", "—", ":"]
    private static let abbreviations: Set<String> = [
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
        "e.g.", "i.e.", "fig.", "no.", "vol.", "inc.", "ltd.", "co.",
        "т.д.", "т.п.", "т.е.", "т.к.", "т.н.", "т.", "д.", "п.",
        "г.", "гг.", "см.", "стр.", "рис.", "табл.", "ул.", "пр.", "им.", "др.", "обл.", "руб.", "коп.",
    ]

    public let clauseMinChars: Int
    private var buffer = ""

    public init(clauseMinChars: Int = Self.defaultClauseMinChars) {
        self.clauseMinChars = max(1, clauseMinChars)
    }

    public mutating func push(_ delta: String) -> [String] {
        guard !delta.isEmpty else { return [] }
        buffer.append(delta)
        var emitted: [String] = []
        while let cut = nextBoundary() {
            let chunk = buffer[..<cut].trimmingCharacters(in: .whitespacesAndNewlines)
            buffer = String(buffer[cut...].drop { $0 == " " || $0 == "\t" || $0 == "\r" })
            if !chunk.isEmpty {
                emitted.append(chunk)
            }
        }
        return emitted
    }

    public mutating func flush() -> String? {
        let remaining = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        buffer.removeAll(keepingCapacity: false)
        return remaining.isEmpty ? nil : remaining
    }

    private func nextBoundary() -> String.Index? {
        var index = buffer.startIndex
        var count = 0
        while index < buffer.endIndex {
            let char = buffer[index]
            count += 1
            let after = buffer.index(after: index)
            if Self.immediateTerminators.contains(char) {
                return consumeTrailingRepeated(from: after, matching: char)
            }
            if char == ".", isPeriodBoundary(at: index, after: after) {
                return after
            }
            if Self.clauseDelimiters.contains(char), isClauseBoundary(after: after, prefixCount: count) {
                return after
            }
            index = after
        }
        return nil
    }

    private func consumeTrailingRepeated(from start: String.Index, matching char: Character) -> String.Index {
        guard char == "\n" else { return start }
        var cursor = start
        while cursor < buffer.endIndex, buffer[cursor] == "\n" || buffer[cursor] == "\r" {
            cursor = buffer.index(after: cursor)
        }
        return cursor
    }

    private func isPeriodBoundary(at period: String.Index, after: String.Index) -> Bool {
        guard after < buffer.endIndex else { return false }
        let next = buffer[after]
        guard next.isWhitespace else { return false }
        let prefix = buffer[...period]
        return !Self.endsWithAbbreviation(prefix)
    }

    private func isClauseBoundary(after: String.Index, prefixCount: Int) -> Bool {
        guard prefixCount >= clauseMinChars, after < buffer.endIndex else { return false }
        return buffer[after].isWhitespace
    }

    private static func endsWithAbbreviation(_ prefix: Substring) -> Bool {
        let trimmed = prefix.dropLast()
        let tokenStart = trimmed.lastIndex { $0.isWhitespace || $0 == "(" || $0 == "\"" || $0 == "«" }
            .map { trimmed.index(after: $0) } ?? trimmed.startIndex
        let word = prefix[tokenStart...].lowercased()
        if abbreviations.contains(word) {
            return true
        }
        let lettersOnly = word.dropLast()
        if lettersOnly.count == 1, let single = lettersOnly.first, single.isLetter {
            return true
        }
        return false
    }
}
