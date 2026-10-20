import Foundation
import MLX
import MLXLMCommon
import BasaltCore

struct ReasoningMarkers: Sendable {
    var openers: Set<Int> = []
    var closers: [[Int]] = []

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
        case released
    }

    let masker: GrammarMasker
    let stopTokens: Set<Int>
    let markers: ReasoningMarkers
    private(set) var phase: Phase
    private(set) var matcher: JSONGrammarMatcher
    private var recent: [Int] = []
    private let wordIndex: MLXArray
    private let bitOffset: MLXArray

    init(masker: GrammarMasker, stopTokens: Set<Int>, markers: ReasoningMarkers, phase: Phase) {
        self.masker = masker
        self.stopTokens = stopTokens
        self.markers = markers
        self.phase = markers.openers.isEmpty && phase == .awaitingAnswer ? .constrained : phase
        matcher = JSONGrammarMatcher(grammar: masker.grammar)
        let ids = MLXArray(0..<Int32(masker.vocabulary.size))
        wordIndex = ids >> 5
        bitOffset = (ids & 31).asType(.uint32)
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func allowedTokens() -> TokenMask {
        var allowed = masker.mask(for: matcher)
        if phase == .awaitingAnswer {
            for opener in markers.openers { allowed.insert(opener) }
        }
        if matcher.isAccepting || allowed.isEmpty {
            for stop in stopTokens { allowed.insert(stop) }
        }
        return allowed
    }

    func process(logits: MLXArray) -> MLXArray {
        guard phase == .constrained || phase == .awaitingAnswer else { return logits }
        let words = MLXArray(allowedTokens().words)
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
        let id = token.item(Int.self)
        switch phase {
        case .released:
            return
        case .reasoning:
            recent.append(id)
            if recent.count > 16 { recent.removeFirst(recent.count - 16) }
            if markers.closers.contains(where: { recent.ends(with: $0) }) { phase = .constrained }
        case .awaitingAnswer where markers.openers.contains(id):
            phase = .reasoning
        case .awaitingAnswer, .constrained:
            phase = .constrained
            guard !stopTokens.contains(id) else { return }
            if !matcher.accept(masker.vocabulary.bytes(of: id)) { phase = .released }
        }
    }
}

private extension Array where Element: Equatable {
    func ends(with suffix: [Element]) -> Bool {
        count >= suffix.count && Array(self[(count - suffix.count)...]) == suffix
    }
}
