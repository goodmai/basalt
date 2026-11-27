import Foundation

public struct LayerStateHealth: Sendable, Codable, Equatable {
    public var layerIndex: Int
    public var cacheKind: String
    public var stateTensorCount: Int
    public var frobeniusNorm: Double
    public var maxAbsEntry: Double
    public var relativeGrowthVsBaseline: Double?
    public var nonFiniteCount: Int
    public var saturated: Bool
    public var betaGuardTriggerRate: Double?

    public init(
        layerIndex: Int = 0,
        cacheKind: String = "",
        stateTensorCount: Int = 0,
        frobeniusNorm: Double = 0,
        maxAbsEntry: Double = 0,
        nonFiniteCount: Int = 0
    ) {
        self.layerIndex = max(0, layerIndex)
        self.cacheKind = cacheKind
        self.stateTensorCount = max(0, stateTensorCount)
        self.frobeniusNorm = frobeniusNorm.isFinite ? max(0, frobeniusNorm) : 0
        self.maxAbsEntry = maxAbsEntry.isFinite ? max(0, maxAbsEntry) : 0
        self.relativeGrowthVsBaseline = nil
        self.nonFiniteCount = max(0, nonFiniteCount)
        self.saturated = nonFiniteCount > 0
        self.betaGuardTriggerRate = nil
    }

    public func withDiagnostics(
        relativeGrowthVsBaseline: Double? = nil,
        saturated: Bool = false,
        betaGuardTriggerRate: Double? = nil
    ) -> LayerStateHealth {
        var copy = self
        copy.relativeGrowthVsBaseline = relativeGrowthVsBaseline.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil }
        copy.saturated = saturated || copy.nonFiniteCount > 0
        copy.betaGuardTriggerRate = betaGuardTriggerRate.flatMap { $0.isFinite ? min(max($0, 0), 1) : nil }
        return copy
    }

    enum CodingKeys: String, CodingKey {
        case layerIndex = "layer_index"
        case cacheKind = "cache_kind"
        case stateTensorCount = "state_tensor_count"
        case frobeniusNorm = "frobenius_norm"
        case maxAbsEntry = "max_abs_entry"
        case relativeGrowthVsBaseline = "relative_growth_vs_baseline"
        case nonFiniteCount = "non_finite_count"
        case saturated
        case betaGuardTriggerRate = "beta_guard_trigger_rate"
    }
}

public struct RecurrentStateHealthSnapshot: Sendable, Codable, Equatable {
    public var layerCount: Int
    public var recurrentLayerCount: Int
    public var maxFrobeniusNorm: Double
    public var maxAbsEntry: Double
    public var totalNonFiniteCount: Int
    public var saturatedLayerCount: Int
    public var meanBetaGuardRate: Double?
    public var layers: [LayerStateHealth]

    public init(layerCount: Int = 0, layers: [LayerStateHealth] = []) {
        self.layerCount = max(layerCount, layers.count)
        self.recurrentLayerCount = layers.count
        self.maxFrobeniusNorm = layers.map(\.frobeniusNorm).filter(\.isFinite).max() ?? 0
        self.maxAbsEntry = layers.map(\.maxAbsEntry).filter(\.isFinite).max() ?? 0
        self.totalNonFiniteCount = layers.reduce(0) { $0 + max(0, $1.nonFiniteCount) }
        self.saturatedLayerCount = layers.reduce(0) { $0 + ($1.saturated ? 1 : 0) }
        let rates = layers.compactMap(\.betaGuardTriggerRate).filter(\.isFinite)
        self.meanBetaGuardRate = rates.isEmpty ? nil : rates.reduce(0, +) / Double(rates.count)
        self.layers = layers
    }

    enum CodingKeys: String, CodingKey {
        case layerCount = "layer_count"
        case recurrentLayerCount = "recurrent_layer_count"
        case maxFrobeniusNorm = "max_frobenius_norm"
        case maxAbsEntry = "max_abs_entry"
        case totalNonFiniteCount = "total_non_finite_count"
        case saturatedLayerCount = "saturated_layer_count"
        case meanBetaGuardRate = "mean_beta_guard_rate"
        case layers
    }
}

public struct ReasoningTurnRecord: Sendable, Equatable {
    public var thinkingTokens: Int
    public var answerTokens: Int
    public var hitTokenBudgetBeforeClose: Bool
    public var thinkingTokenIDs: [Int]
    public var meanThinkingEntropy: Double?
    public var meanAnswerEntropy: Double?

    public init(
        thinkingTokens: Int = 0,
        answerTokens: Int = 0,
        hitTokenBudgetBeforeClose: Bool = false,
        thinkingTokenIDs: [Int] = [],
        meanThinkingEntropy: Double? = nil,
        meanAnswerEntropy: Double? = nil
    ) {
        self.thinkingTokens = max(0, thinkingTokens)
        self.answerTokens = max(0, answerTokens)
        self.hitTokenBudgetBeforeClose = hitTokenBudgetBeforeClose
        self.thinkingTokenIDs = thinkingTokenIDs
        self.meanThinkingEntropy = meanThinkingEntropy.flatMap { $0.isFinite ? $0 : nil }
        self.meanAnswerEntropy = meanAnswerEntropy.flatMap { $0.isFinite ? $0 : nil }
    }
}

public struct ReasoningTelemetrySnapshot: Sendable, Codable, Equatable {
    public var sampleCount: Int
    public var thinkingToAnswerRatio: Double
    public var meanNGramLoopRate: Double
    public var budgetTruncationRate: Double
    public var meanPostThinkEntropyDrop: Double?
    public var totalThinkingTokens: Int
    public var totalAnswerTokens: Int

    public init(
        sampleCount: Int = 0,
        totalThinkingTokens: Int = 0,
        totalAnswerTokens: Int = 0,
        meanNGramLoopRate: Double = 0,
        budgetTruncationRate: Double = 0,
        meanPostThinkEntropyDrop: Double? = nil
    ) {
        self.sampleCount = max(0, sampleCount)
        self.totalThinkingTokens = max(0, totalThinkingTokens)
        self.totalAnswerTokens = max(0, totalAnswerTokens)
        self.thinkingToAnswerRatio = Self.thinkingRatio(
            thinkingTokens: self.totalThinkingTokens,
            answerTokens: self.totalAnswerTokens
        )
        self.meanNGramLoopRate = meanNGramLoopRate.isFinite ? min(max(meanNGramLoopRate, 0), 1) : 0
        self.budgetTruncationRate = budgetTruncationRate.isFinite ? min(max(budgetTruncationRate, 0), 1) : 0
        self.meanPostThinkEntropyDrop = meanPostThinkEntropyDrop.flatMap { $0.isFinite ? $0 : nil }
    }

    public static func thinkingRatio(thinkingTokens: Int, answerTokens: Int) -> Double {
        let thinking = max(0, thinkingTokens)
        guard thinking > 0 else { return 0 }
        return Double(thinking) / Double(max(1, answerTokens))
    }

    enum CodingKeys: String, CodingKey {
        case sampleCount = "sample_count"
        case thinkingToAnswerRatio = "thinking_to_answer_ratio"
        case meanNGramLoopRate = "mean_ngram_loop_rate"
        case budgetTruncationRate = "budget_truncation_rate"
        case meanPostThinkEntropyDrop = "mean_post_think_entropy_drop"
        case totalThinkingTokens = "total_thinking_tokens"
        case totalAnswerTokens = "total_answer_tokens"
    }
}

public enum ReasoningTelemetryMath: Sendable {
    public static func nGramRepetitionRate(tokens: [Int], n: Int = 4) -> Double {
        guard n > 0, tokens.count > n else { return 0.0 }
        let totalWindows = tokens.count - n + 1
        var seen = Set<ArraySlice<Int>>(minimumCapacity: totalWindows)
        seen.insert(tokens[0 ..< n])
        var repeats = 0
        for start in 1 ..< totalWindows {
            if !seen.insert(tokens[start ..< (start + n)]).inserted {
                repeats += 1
            }
        }
        return Double(repeats) / Double(totalWindows - 1)
    }

    public static func postThinkEntropyDrop(thinkingEntropy: Double?, answerEntropy: Double?) -> Double? {
        guard let thinking = thinkingEntropy, let answer = answerEntropy,
              thinking.isFinite, answer.isFinite
        else { return nil }
        let drop = thinking - answer
        return drop.isFinite ? drop : nil
    }

    public static func splitThinkingAndAnswer(
        text: String,
        openTag: String = "<think>",
        closeTag: String = "</think>"
    ) -> (thinking: String, answer: String, unclosed: Bool) {
        guard !openTag.isEmpty, !closeTag.isEmpty else { return ("", text, false) }
        var thinking = ""
        var answer = ""
        var rest = Substring(text)
        var sawOpen = false
        while !rest.isEmpty {
            let openRange = rest.range(of: openTag)
            if !sawOpen, let closeRange = promptOpenedCloseRange(in: rest, openRange: openRange, closeTag: closeTag) {
                thinking.append(contentsOf: rest[..<closeRange.lowerBound])
                rest = rest[closeRange.upperBound...]
                sawOpen = true
                continue
            }
            guard let openRange else {
                answer.append(contentsOf: rest)
                return (thinking, answer, false)
            }
            sawOpen = true
            answer.append(contentsOf: rest[..<openRange.lowerBound])
            let afterOpen = rest[openRange.upperBound...]
            guard let closeRange = afterOpen.range(of: closeTag) else {
                thinking.append(contentsOf: afterOpen)
                return (thinking, answer, true)
            }
            thinking.append(contentsOf: afterOpen[..<closeRange.lowerBound])
            rest = afterOpen[closeRange.upperBound...]
        }
        return (thinking, answer, false)
    }

    private static func promptOpenedCloseRange(
        in rest: Substring,
        openRange: Range<Substring.Index>?,
        closeTag: String
    ) -> Range<Substring.Index>? {
        guard let closeRange = rest.range(of: closeTag) else { return nil }
        guard let openRange else { return closeRange }
        return closeRange.lowerBound < openRange.lowerBound ? closeRange : nil
    }
}

public struct ThinkingTelemetryReducer: Sendable, Equatable {
    private struct TurnSummary: Sendable, Equatable {
        let thinkingTokens: Int
        let answerTokens: Int
        let truncated: Bool
        let nGramLoopRate: Double
        let postThinkEntropyDrop: Double?
    }

    public static let defaultWindowCapacity = 256
    public static let defaultNGramOrder = 4

    public let windowCapacity: Int
    public let nGramOrder: Int
    private var entries: [TurnSummary] = []
    private var nextWriteIndex = 0

    public init(windowCapacity: Int = defaultWindowCapacity, nGramOrder: Int = defaultNGramOrder) {
        self.windowCapacity = max(1, windowCapacity)
        self.nGramOrder = max(1, nGramOrder)
    }

    public var count: Int { entries.count }
    public var isEmpty: Bool { entries.isEmpty }

    public mutating func record(_ turn: ReasoningTurnRecord) {
        let summary = TurnSummary(
            thinkingTokens: max(0, turn.thinkingTokens),
            answerTokens: max(0, turn.answerTokens),
            truncated: turn.hitTokenBudgetBeforeClose,
            nGramLoopRate: ReasoningTelemetryMath.nGramRepetitionRate(tokens: turn.thinkingTokenIDs, n: nGramOrder),
            postThinkEntropyDrop: ReasoningTelemetryMath.postThinkEntropyDrop(
                thinkingEntropy: turn.meanThinkingEntropy,
                answerEntropy: turn.meanAnswerEntropy
            )
        )
        if entries.count < windowCapacity {
            entries.append(summary)
        } else {
            entries[nextWriteIndex] = summary
        }
        nextWriteIndex = (nextWriteIndex + 1) % windowCapacity
    }

    public func snapshot() -> ReasoningTelemetrySnapshot {
        guard !entries.isEmpty else { return ReasoningTelemetrySnapshot() }
        var totalThinking = 0
        var totalAnswer = 0
        var truncatedCount = 0
        var loopSum = 0.0
        var dropSum = 0.0
        var dropCount = 0
        for entry in entries {
            totalThinking += entry.thinkingTokens
            totalAnswer += entry.answerTokens
            if entry.truncated { truncatedCount += 1 }
            loopSum += entry.nGramLoopRate
            if let drop = entry.postThinkEntropyDrop {
                dropSum += drop
                dropCount += 1
            }
        }
        let sampleCount = entries.count
        return ReasoningTelemetrySnapshot(
            sampleCount: sampleCount,
            totalThinkingTokens: totalThinking,
            totalAnswerTokens: totalAnswer,
            meanNGramLoopRate: loopSum / Double(sampleCount),
            budgetTruncationRate: Double(truncatedCount) / Double(sampleCount),
            meanPostThinkEntropyDrop: dropCount > 0 ? dropSum / Double(dropCount) : nil
        )
    }

    public mutating func reset() {
        entries.removeAll(keepingCapacity: false)
        nextWriteIndex = 0
    }
}
