import BasaltCore
import Foundation

public struct TranslationPairedEvaluation: Sendable, Equatable {
    public let score: TranslationScore
    public let repetitionRatio: Double
    public let belnapState: BelnapState
    public let passed: Bool

    public init(
        score: TranslationScore,
        repetitionRatio: Double,
        belnapState: BelnapState,
        passed: Bool
    ) {
        self.score = score
        self.repetitionRatio = repetitionRatio
        self.belnapState = belnapState
        self.passed = passed
    }
}

public struct TranslationPairedGrader: PairedAnswerGrader, Sendable {
    public let chrfThreshold: Double
    public let strongChrfThreshold: Double
    public let techFidelityThreshold: Double
    public let maxRepetitionRatio: Double

    public init(
        chrfThreshold: Double = 35.0,
        strongChrfThreshold: Double = 55.0,
        techFidelityThreshold: Double = 90.0,
        maxRepetitionRatio: Double = 0.35
    ) {
        self.chrfThreshold = chrfThreshold
        self.strongChrfThreshold = strongChrfThreshold
        self.techFidelityThreshold = techFidelityThreshold
        self.maxRepetitionRatio = maxRepetitionRatio
    }

    public func evaluate(response: String, problem: PairedProblem) -> TranslationPairedEvaluation {
        let source = problem.sourceText ?? problem.prompt
        let score = TranslationMetrics.scoreProxy(
            source: source,
            hypothesis: response,
            reference: problem.expectedAnswer
        )
        let rep = Self.repetitionRatio(response)
        let passed = score.chrfPP >= chrfThreshold
            && score.techFidelity >= techFidelityThreshold
            && rep <= maxRepetitionRatio
        let supported = score.chrfPP >= strongChrfThreshold
        let refuted = score.techFidelity < techFidelityThreshold
            || rep > maxRepetitionRatio
            || score.chrfPP < chrfThreshold
        let belnap = BelnapState.from(supported: supported, refuted: refuted)
        return TranslationPairedEvaluation(
            score: score,
            repetitionRatio: rep,
            belnapState: belnap,
            passed: passed
        )
    }

    public func grade(response: String, problem: PairedProblem) -> PairedGradeResult {
        let eval = evaluate(response: response, problem: problem)
        let penalizedScore = eval.repetitionRatio > maxRepetitionRatio
            ? eval.score.comet * max(0.0, 1.0 - eval.repetitionRatio)
            : eval.score.comet
        let detail = String(
            format: "chrf=%.1f fidelity=%.1f rep=%.2f belnap=%@",
            eval.score.chrfPP, eval.score.techFidelity, eval.repetitionRatio, eval.belnapState.rawValue
        )
        return PairedGradeResult(
            passed: eval.passed,
            score: penalizedScore,
            extractedAnswer: nil,
            detail: detail
        )
    }

    public static func repetitionRatio(_ text: String, ngramSize: Int = 4) -> Double {
        let tokens = text.lowercased()
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
        let n = max(1, ngramSize)
        if tokens.count >= n * 2 {
            return duplicateRatio(tokens, n: n)
        }
        let chars = Array(text.trimmingCharacters(in: .whitespacesAndNewlines))
        let charN = max(3, n)
        guard chars.count >= charN * 3 else { return 0.0 }
        return duplicateRatio(chars, n: charN)
    }

    private static func duplicateRatio<T: Hashable>(_ elements: [T], n: Int) -> Double {
        let total = elements.count - n + 1
        guard total > 1 else { return 0.0 }
        var seen = Set<[T]>()
        var duplicates = 0
        for i in 0 ..< total {
            let gram = Array(elements[i ..< (i + n)])
            if !seen.insert(gram).inserted {
                duplicates += 1
            }
        }
        return Double(duplicates) / Double(total)
    }
}
