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
        while let found = buffer.range(of: probe, range: searchRange) {
            occurrences += 1
            if occurrences >= threshold { return true }
            guard found.upperBound < buffer.endIndex else { break }
            searchRange = found.upperBound..<buffer.endIndex
        }
        return false
    }
}
