import Foundation

public struct RepetitionGuard: Sendable {
    private let probeSize: Int
    private let threshold: Int
    private let minLength: Int
    private let retain: Int

    private var buffer: String = ""

    public private(set) var score: Double = 0

    public init(probeSize: Int = 64, threshold: Int = 4, minLength: Int = 400, retain: Int = 4096) {
        self.probeSize = probeSize
        self.threshold = threshold
        self.minLength = minLength
        self.retain = retain
    }

    public mutating func shouldStop(after chunk: String) -> Bool {
        buffer += chunk
        guard buffer.count >= minLength else { return false }
        if buffer.count > retain {
            buffer.removeFirst(buffer.count - retain)
        }
        guard buffer.count > probeSize else { return false }

        let probe = String(buffer.suffix(probeSize))
        var occurrences = 0
        var searchRange = buffer.startIndex..<buffer.endIndex
        while occurrences < threshold, let found = buffer.range(of: probe, range: searchRange) {
            occurrences += 1
            guard found.upperBound < buffer.endIndex else { break }
            searchRange = found.upperBound..<buffer.endIndex
        }
        score = Self.score(occurrences: occurrences, threshold: threshold)
        guard occurrences >= threshold else { return false }
        var utf8 = buffer
        return utf8.withUTF8 { ptr in
            Self.endsInVerbatimCycle(ptr.span, probeSize: probeSize, repeats: threshold)
        }
    }

    static func endsInVerbatimCycle(_ bytes: [UInt8], probeSize: Int, repeats: Int) -> Bool {
        endsInVerbatimCycle(bytes.span, probeSize: probeSize, repeats: repeats)
    }

    static func endsInVerbatimCycle(_ bytes: Span<UInt8>, probeSize: Int, repeats: Int) -> Bool {
        let probeStart = bytes.count - probeSize
        guard probeStart > 0 else { return false }
        let firstByte = bytes[probeStart]
        var start = probeStart - 1
        while start >= 0 {
            if bytes[start] == firstByte, matches(bytes, lhs: start, rhs: probeStart, count: probeSize) {
                let period = probeStart - start
                let span = max((repeats - 1) * period + probeSize, repeats * probeSize)
                guard span <= bytes.count else { return false }
                if isPeriodic(bytes, from: bytes.count - span + period, period: period) {
                    return true
                }
            }
            start -= 1
        }
        return false
    }

    private static func matches(_ bytes: Span<UInt8>, lhs: Int, rhs: Int, count: Int) -> Bool {
        for offset in 0 ..< count where bytes[lhs + offset] != bytes[rhs + offset] {
            return false
        }
        return true
    }

    private static func isPeriodic(_ bytes: Span<UInt8>, from start: Int, period: Int) -> Bool {
        for index in start ..< bytes.count where bytes[index] != bytes[index - period] {
            return false
        }
        return true
    }

    public static func score<E: Equatable>(tokens: [E], probeSize: Int, threshold: Int) -> Double {
        score(tokens: tokens.span, probeSize: probeSize, threshold: threshold)
    }

    public static func score<E: Equatable>(tokens: Span<E>, probeSize: Int, threshold: Int) -> Double {
        guard probeSize > 0, tokens.count > probeSize else { return 0 }
        let probeStart = tokens.count - probeSize
        var occurrences = 0
        var i = 0
        while occurrences < threshold, i + probeSize <= tokens.count {
            var matched = true
            for offset in 0 ..< probeSize where tokens[i + offset] != tokens[probeStart + offset] {
                matched = false
                break
            }
            if matched {
                occurrences += 1
                i += probeSize
            } else {
                i += 1
            }
        }
        return score(occurrences: occurrences, threshold: threshold)
    }

    static func score(occurrences: Int, threshold: Int) -> Double {
        guard threshold > 1 else { return occurrences >= threshold ? 1 : 0 }
        return min(1, max(0, Double(occurrences - 1) / Double(threshold - 1)))
    }
}

public struct TokenRepetitionWindow: ~Copyable, Sendable {
    private let probeSize: Int
    private let threshold: Int
    private let minTokens: Int
    private let retain: Int
    private var head: Int = 0
    private var _storage: UniqueArray<Int>

    public private(set) var score: Double = 0

    public init(probeSize: Int = 16, threshold: Int = 4, minTokens: Int = 64, retain: Int = 512) {
        self.probeSize = max(1, probeSize)
        self.threshold = max(1, threshold)
        self.minTokens = max(1, minTokens)
        self.retain = max(1, retain)
        var initial = UniqueArray<Int>()
        initial.reserveCapacity(max(2, retain * 2))
        self._storage = initial
    }

    package var storage: UniqueArray<Int> {
        borrow { _storage }
        mutate { &_storage }
    }

    public var count: Int { _storage.count - head }

    public borrowing func withTokens<R>(_ body: (Span<Int>) -> R) -> R {
        body(_storage.span.extracting(head ..< _storage.count))
    }

    @discardableResult
    public mutating func append(_ token: Int) -> Double {
        _storage.append(token)
        if _storage.count - head > retain {
            head = _storage.count - retain
        }
        if head >= retain {
            let kept = _storage.count - head
            do {
                var span = _storage.mutableSpan
                for index in 0 ..< kept {
                    span[index] = span[head + index]
                }
            }
            _storage.removeLast(head)
            head = 0
        }
        let active = _storage.count - head
        guard active >= minTokens else {
            score = 0
            return 0
        }
        score = RepetitionGuard.score(
            tokens: _storage.span.extracting(head ..< _storage.count),
            probeSize: probeSize,
            threshold: threshold)
        return score
    }
}
