import Foundation
import MLX
import MLXLMCommon
import BasaltCore

struct ReasoningMarkers: Sendable {
    var openers: Set<Int> = []
    var closers: [[Int]] = []
    var closerTexts: [[UInt8]] = MLXInferenceEngine.thinkingClosers.map { Array($0.utf8) }

    init(openers: Set<Int> = [], closers: [[Int]] = []) {
        self.openers = openers
        self.closers = closers
    }

    init(tokenizer: any MLXLMCommon.Tokenizer) {
        for opener in MLXInferenceEngine.thinkingOpeners {
            let ids = tokenizer.encode(text: opener, addSpecialTokens: false)
            if ids.count == 1 { openers.insert(ids[0]) }
        }
        closers = Set(MLXInferenceEngine.thinkingClosers.map { tokenizer.encode(text: $0, addSpecialTokens: false) })
            .filter { !$0.isEmpty }.sorted { $0.count > $1.count }
    }
}

struct JSONConstraintProcessor: LogitProcessor {
    enum Phase: Sendable {
        case awaitingAnswer
        case reasoning
        case constrained
    }

    private final class Progress {
        var phase: Phase
        var matcher: JSONGrammarMatcher
        var recent: [Int] = []
        var recentText: [UInt8] = []
        var unread: MLXArray?

        init(phase: Phase, matcher: JSONGrammarMatcher) {
            self.phase = phase
            self.matcher = matcher
        }

        func copy() -> Progress {
            let copy = Progress(phase: phase, matcher: matcher)
            copy.recent = recent
            copy.recentText = recentText
            copy.unread = unread
            return copy
        }
    }

    let masker: GrammarMasker
    let stopTokens: Set<Int>
    let markers: ReasoningMarkers
    private var progress: Progress
    private let wordIndex: MLXArray
    private let bitOffset: MLXArray

    init(masker: GrammarMasker, stopTokens: Set<Int>, markers: ReasoningMarkers, phase: Phase) {
        self.masker = masker
        self.stopTokens = stopTokens
        self.markers = markers
        progress = Progress(
            phase: markers.openers.isEmpty && phase == .awaitingAnswer ? .constrained : phase,
            matcher: JSONGrammarMatcher(grammar: masker.grammar))
        let ids = MLXArray(0..<Int32(masker.vocabulary.size))
        wordIndex = ids >> 5
        bitOffset = (ids & 31).asType(.uint32)
    }

    func copy() -> JSONConstraintProcessor {
        var copy = self
        copy.progress = progress.copy()
        return copy
    }

    var phase: Phase {
        readSampledToken()
        return progress.phase
    }

    var matcher: JSONGrammarMatcher {
        readSampledToken()
        return progress.matcher
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func allowedTokens() -> TokenMask {
        let phase = phase
        var allowed = masker.mask(for: progress.matcher)
        if phase == .awaitingAnswer {
            for opener in markers.openers { allowed.insert(opener) }
        }
        if progress.matcher.isAccepting || allowed.isEmpty {
            for stop in stopTokens { allowed.insert(stop) }
        }
        return allowed
    }

    func process(logits: MLXArray) -> MLXArray {
        guard phase != .reasoning else { return logits }
        let allowedTokens = allowedTokens()
        guard !allowedTokens.isEmpty else { return logits }
        let words = MLXArray(allowedTokens.words)
        var allowed = (words.take(wordIndex) >> bitOffset) & 1
        let width = logits.dim(-1)
        if width > masker.vocabulary.size {
            allowed = concatenated([allowed, MLXArray.zeros([width - masker.vocabulary.size], dtype: .uint32)])
        } else if width < masker.vocabulary.size {
            allowed = allowed[..<width]
        }
        return MLX.where(allowed .== 1, logits, MLXArray(-Float.infinity).asType(logits.dtype))
    }

    mutating func didSample(token: MLXArray) {
        readSampledToken()
        progress.unread = token
    }

    private func readSampledToken() {
        guard let token = progress.unread else { return }
        progress.unread = nil
        let id = token.item(Int.self)
        switch progress.phase {
        case .reasoning:
            progress.recent.append(id)
            if progress.recent.count > 16 { progress.recent.removeFirst(progress.recent.count - 16) }
            progress.recentText += masker.vocabulary.bytes(of: id)
            if progress.recentText.count > 64 { progress.recentText.removeFirst(progress.recentText.count - 64) }
            if markers.closers.contains(where: { progress.recent.ends(with: $0) }) {
                progress.phase = .constrained
            } else if let answer = markers.closerTexts.compactMap({ progress.recentText.text(after: $0) }).first {
                progress.phase = .constrained
                _ = progress.matcher.accept(answer)
            }
        case .awaitingAnswer where markers.openers.contains(id):
            progress.phase = .reasoning
        case .awaitingAnswer, .constrained:
            progress.phase = .constrained
            guard !stopTokens.contains(id) else { return }
            _ = progress.matcher.accept(masker.vocabulary.bytes(of: id))
        }
    }
}

private extension Array where Element: Equatable {
    func ends(with suffix: [Element]) -> Bool {
        count >= suffix.count && Array(self[(count - suffix.count)...]) == suffix
    }

    func text(after marker: [Element]) -> [Element]? {
        guard !marker.isEmpty, count >= marker.count else { return nil }
        for start in stride(from: count - marker.count, through: 0, by: -1) where Array(self[start..<(start + marker.count)]) == marker {
            return Array(self[(start + marker.count)...])
        }
        return nil
    }
}
