import Foundation
import MLX
import MLXLMCommon
import MLXGuidedGeneration
import Synchronization
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
            guard let first = ids.first else { continue }
            let piece = tokenizer.decode(tokenIds: [first], skipSpecialTokens: false)
            let opensAsSpecialToken = piece.count > 1 && opener.hasPrefix(piece) && tokenizer.decode(tokenIds: [first], skipSpecialTokens: true).isEmpty
            if ids.count == 1 || opensAsSpecialToken { openers.insert(first) }
        }
        closers = Set(MLXInferenceEngine.thinkingClosers.map { tokenizer.encode(text: $0, addSpecialTokens: false) })
            .filter { !$0.isEmpty }.sorted { $0.count > $1.count }
    }
}

final class GrammarVocabulary: Sendable {
    let tokenizer: GrammarTokenizer
    let pieces: [[UInt8]]
    let stopTokens: Set<Int>
    let markers: ReasoningMarkers

    init(pieces: [String], vocabType: VocabType, stopTokens: Set<Int>, markers: ReasoningMarkers) throws {
        let pieces = pieces + Array(repeating: "", count: max(0, (stopTokens.max() ?? -1) + 1 - pieces.count))
        var grammarPieces = pieces
        let reserved = stopTokens.union(markers.openers).union(markers.closers.filter { $0.count == 1 }.joined())
        for id in reserved where grammarPieces.indices.contains(id) { grammarPieces[id] = "" }
        let eos = stopTokens.min().map(Int32.init) ?? -1
        tokenizer = try GrammarTokenizer(vocab: grammarPieces, vocabType: vocabType, eosTokenId: eos)
        self.pieces = pieces.map { Array($0.utf8) }
        self.stopTokens = stopTokens
        self.markers = markers
    }

    convenience init(context: ModelContext) throws {
        let extracted = TokenizerVocabExtractor.extractForGrammar(from: context.tokenizer)
        let stopTokens = MLXInferenceEngine.stopTokenIds(context)
        let pastGap = (extracted.vocab.count..<max(extracted.vocab.count, (stopTokens.max() ?? -1) + 1))
            .map { context.tokenizer.convertIdToToken($0) ?? "" }
        try self.init(pieces: extracted.vocab + pastGap, vocabType: extracted.vocabType,
                      stopTokens: stopTokens, markers: ReasoningMarkers(tokenizer: context.tokenizer))
    }

    var size: Int { pieces.count }
}

final class GrammarSession: Sendable {
    struct Progress {
        var committed: Int32 = 0
        var failed = false
        var retired = false
    }

    let schema: String
    private let grammar: GrammarConstraint
    private let progress = Mutex(Progress())

    init(schema: String, grammar: GrammarConstraint) {
        self.schema = schema
        self.grammar = grammar
    }

    func commit(_ id: Int) {
        progress.withLock { progress in
            guard !progress.retired else { return }
            do {
                _ = try grammar.commitToken(Int32(id))
                progress.committed += 1
            } catch {
                progress.failed = true
                GemLogger(module: "JSONConstraint").warn("grammar refused sampled token \(id), the session will not be pooled: \(error)")
            }
        }
    }

    func mask() -> MaskResult? {
        progress.withLock { $0.retired ? nil : try? grammar.computeMask() }
    }

    func rewound() -> GrammarConstraint? {
        progress.withLock { progress in
            guard !progress.retired else { return nil }
            progress.retired = true
            guard !progress.failed, (try? grammar.rollback(progress.committed)) != nil else { return nil }
            return grammar
        }
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
        let session: GrammarSession
        var recent: [Int] = []
        var recentText: [UInt8] = []
        var unread: MLXArray?

        init(phase: Phase, session: GrammarSession) {
            self.phase = phase
            self.session = session
        }
    }

    let vocabulary: GrammarVocabulary
    private var progress: Progress
    private let wordIndex: MLXArray
    private let bitOffset: MLXArray

    init(session: GrammarSession, vocabulary: GrammarVocabulary, phase: Phase) {
        self.vocabulary = vocabulary
        progress = Progress(
            phase: vocabulary.markers.openers.isEmpty && phase == .awaitingAnswer ? .constrained : phase,
            session: session)
        let ids = MLXArray(0..<Int32(vocabulary.size))
        wordIndex = ids >> 5
        bitOffset = (ids & 31).asType(.uint32)
    }

    var phase: Phase {
        readSampledToken()
        return progress.phase
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func allowedTokens() -> [Int32]? {
        let phase = phase
        let words = (vocabulary.size + 31) / 32
        guard let result = progress.session.mask() else {
            return vocabulary.stopTokens.reduce(into: [Int32](repeating: 0, count: words)) { $0.allow($1) }
        }
        guard result.needsApply else { return nil }
        var allowed = result.isTerminated ? [Int32](repeating: 0, count: words) : result.mask
        if phase == .awaitingAnswer {
            for opener in vocabulary.markers.openers { allowed.allow(opener) }
        }
        if result.isTerminated || vocabulary.stopTokens.contains(where: allowed.allows) || allowed.allSatisfy({ $0 == 0 }) {
            for stop in vocabulary.stopTokens { allowed.allow(stop) }
        }
        return allowed
    }

    func process(logits: MLXArray) -> MLXArray {
        guard phase != .reasoning, let words = allowedTokens() else { return logits }
        var allowed = (MLXArray(words.map { UInt32(bitPattern: $0) }).take(wordIndex) >> bitOffset) & 1
        let width = logits.dim(-1)
        if width > vocabulary.size {
            allowed = concatenated([allowed, MLXArray.zeros([width - vocabulary.size], dtype: .uint32)])
        } else if width < vocabulary.size {
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
        let markers = vocabulary.markers
        switch progress.phase {
        case .reasoning:
            progress.recent.append(id)
            if progress.recent.count > 16 { progress.recent.removeFirst(progress.recent.count - 16) }
            progress.recentText += vocabulary.pieces.indices.contains(id) ? vocabulary.pieces[id] : []
            if progress.recentText.count > 64 { progress.recentText.removeFirst(progress.recentText.count - 64) }
            if markers.closers.contains(where: { progress.recent.ends(with: $0) })
                || markers.closerTexts.contains(where: { progress.recentText.contains($0) }) {
                progress.phase = .constrained
            }
        case .awaitingAnswer where markers.openers.contains(id):
            progress.phase = .reasoning
        case .awaitingAnswer, .constrained:
            progress.phase = .constrained
            guard !vocabulary.stopTokens.contains(id) else { return }
            progress.session.commit(id)
        }
    }
}

extension [Int32] {
    func allows(_ id: Int) -> Bool {
        id >= 0 && id >> 5 < count && UInt32(bitPattern: self[id >> 5]) & (1 << UInt32(id & 31)) != 0
    }

    mutating func allow(_ id: Int) {
        guard id >= 0, id >> 5 < count else { return }
        self[id >> 5] = Int32(bitPattern: UInt32(bitPattern: self[id >> 5]) | 1 << UInt32(id & 31))
    }
}

private extension Array where Element: Equatable {
    func ends(with suffix: [Element]) -> Bool {
        count >= suffix.count && Array(self[(count - suffix.count)...]) == suffix
    }

    func contains(_ marker: [Element]) -> Bool {
        !marker.isEmpty && count >= marker.count && (0...(count - marker.count)).contains { Array(self[$0..<($0 + marker.count)]) == marker }
    }
}
