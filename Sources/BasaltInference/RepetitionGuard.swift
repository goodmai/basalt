import Foundation

// MARK: — Degenerate-loop detector
//
// Quantised models fall into verbatim repetition and then never stop: Qwen3.8 at
// 2-bit answered "127 + 458" by repeating the question 70 times until it hit the
// ceiling, and Gemma4-12B produced 459 lines of the same comment block on the
// Fourier task. Both burned the full budget — minutes of GPU time for output that
// was worthless after the first cycle.
//
// The generation loop feeds every chunk through this and stops early once the
// tail is provably cyclic, reporting `.repetitionLoop` so the caller can tell a
// degenerate run from a real answer. Cheap by construction: a bounded suffix and
// one hash lookup per chunk, no scan of the whole transcript.

public struct RepetitionGuard: Sendable {
    /// Length of the suffix used as the probe, in characters.
    private let probeSize: Int
    /// How many times the probe must recur before the tail counts as cyclic.
    private let threshold: Int
    /// Don't judge before this much text exists; short answers legitimately repeat.
    private let minLength: Int
    /// Bound on retained text, so memory stays flat on long generations.
    private let retain: Int

    private var buffer: String = ""

    public private(set) var score: Double = 0

    public init(probeSize: Int = 64, threshold: Int = 4, minLength: Int = 400, retain: Int = 4096) {
        self.probeSize = probeSize
        self.threshold = threshold
        self.minLength = minLength
        self.retain = retain
    }

    /// Feeds a freshly generated chunk. Returns true when generation should stop.
    ///
    /// Counts recurrences of the trailing probe rather than comparing adjacent
    /// fixed-size windows: window comparison only fires when the cycle length
    /// happens to divide the window, so a 118-character cycle slips past a
    /// 120-character window. Occurrence counting is period-agnostic.
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
        return occurrences >= threshold && Self.endsInVerbatimCycle(Array(buffer.utf8), probeSize: probeSize, repeats: threshold)
    }

    static func endsInVerbatimCycle(_ bytes: [UInt8], probeSize: Int, repeats: Int) -> Bool {
        let probeStart = bytes.count - probeSize
        guard probeStart > 0 else { return false }
        let probe = bytes[probeStart...]
        var start = probeStart - 1
        while start >= 0 {
            if bytes[start] == probe.first, bytes[start..<(start + probeSize)].elementsEqual(probe) {
                let period = probeStart - start
                let span = max(repeats * period, repeats * probeSize)
                guard span <= bytes.count else { return false }
                if (bytes.count - span + period..<bytes.count).allSatisfy({ bytes[$0] == bytes[$0 - period] }) {
                    return true
                }
            }
            start -= 1
        }
        return false
    }

    public static func score<E: Equatable>(tokens: [E], probeSize: Int, threshold: Int) -> Double {
        guard probeSize > 0, tokens.count > probeSize else { return 0 }
        let probe = tokens[(tokens.count - probeSize)...]
        var occurrences = 0
        var i = 0
        while occurrences < threshold, i + probeSize <= tokens.count {
            if tokens[i..<(i + probeSize)].elementsEqual(probe) {
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
