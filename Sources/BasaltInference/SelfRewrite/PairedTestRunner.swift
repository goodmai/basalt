import BasaltCore
import Foundation
import MLX
import MLXLMCommon

public struct PairedTestRunConfig: Sendable, Equatable {
    public let seeds: [UInt64]
    public let maxTokens: Int
    public let temperature: Float
    public let stopTokenIds: Set<Int>
    public let budgetCost: Int

    public init(
        seeds: [UInt64] = [42],
        maxTokens: Int = 256,
        temperature: Float = 0.0,
        stopTokenIds: Set<Int> = [],
        budgetCost: Int = 1
    ) {
        self.seeds = seeds.isEmpty ? [42] : seeds
        self.maxTokens = max(1, maxTokens)
        self.temperature = max(0, temperature)
        self.stopTokenIds = stopTokenIds
        self.budgetCost = max(1, budgetCost)
    }
}

public struct PairedEvaluationSpec: Sendable {
    public let encode: @Sendable (String) -> [Int]
    public let decode: @Sendable ([Int]) -> String
    public let grader: any PairedAnswerGrader
    public let config: PairedTestRunConfig

    public init(
        encode: @escaping @Sendable (String) -> [Int],
        decode: @escaping @Sendable ([Int]) -> String,
        grader: any PairedAnswerGrader,
        config: PairedTestRunConfig = PairedTestRunConfig()
    ) {
        self.encode = encode
        self.decode = decode
        self.grader = grader
        self.config = config
    }
}

public struct PairedPracticeOutcome: Sendable, Equatable {
    public let summary: PairedTestSummary
    public let samples: [PairedSamplePair]

    public init(summary: PairedTestSummary, samples: [PairedSamplePair]) {
        self.summary = summary
        self.samples = samples
    }
}

public enum PairedTestRunner {
    public static func evaluatePractice(
        problems: [PairedProblem],
        original: any LanguageModel,
        copy: borrowing WeightCopy,
        spec: PairedEvaluationSpec,
        budget: inout SelfRewriteBudget
    ) throws -> PairedPracticeOutcome {
        guard !problems.isEmpty else { throw SelfRewriteBudgetError.emptyProblemSet }
        var nextBudget = budget
        try nextBudget.consumeForwardPasses(spec.config.budgetCost)
        let samples = try runPaired(problems: problems, original: original, copy: copy, spec: spec)
        try copy.verifyOriginal(original)
        budget = nextBudget
        return PairedPracticeOutcome(summary: PairedTestStatistics.summarize(samples), samples: samples)
    }

    public static func evaluateReserved(
        split: PairedDatasetSplit,
        original: any LanguageModel,
        copy: borrowing WeightCopy,
        spec: PairedEvaluationSpec,
        budget: inout SelfRewriteBudget
    ) throws -> PairedTestSummary {
        var nextBudget = budget
        try nextBudget.consumeForwardPasses(spec.config.budgetCost)
        let summary = try split.evaluateReserved(budget: &nextBudget) { reservedProblems in
            try runPaired(problems: reservedProblems, original: original, copy: copy, spec: spec)
        }
        try copy.verifyOriginal(original)
        budget = nextBudget
        return summary
    }

    private static func runPaired(
        problems: [PairedProblem],
        original: any LanguageModel,
        copy: borrowing WeightCopy,
        spec: PairedEvaluationSpec
    ) throws -> [PairedSamplePair] {
        let copyModel = try copy.model
        var samples: [PairedSamplePair] = []
        samples.reserveCapacity(problems.count * spec.config.seeds.count)
        for problem in problems {
            let promptIds = spec.encode(problem.prompt)
            for seed in spec.config.seeds {
                let baseText = try generate(promptIds: promptIds, model: original, seed: seed, spec: spec)
                let editedText = try generate(promptIds: promptIds, model: copyModel, seed: seed, spec: spec)
                let baseGrade = spec.grader.grade(response: baseText, problem: problem)
                let editedGrade = spec.grader.grade(response: editedText, problem: problem)
                samples.append(PairedSamplePair(
                    problemId: problem.id,
                    seed: seed,
                    baseGrade: baseGrade,
                    editedGrade: editedGrade
                ))
            }
        }
        return samples
    }

    static func generate(
        promptIds: [Int],
        model: any LanguageModel,
        seed: UInt64,
        spec: PairedEvaluationSpec,
        maxTokensOverride: Int? = nil
    ) throws -> String {
        let tokenLimit = max(1, maxTokensOverride ?? spec.config.maxTokens)
        let parameters = GenerateParameters(
            maxTokens: tokenLimit,
            temperature: spec.config.temperature,
            seed: seed
        )
        let input = LMInput(tokens: MLXArray(promptIds.map(Int32.init)))
        defer { Memory.clearCache() }
        let generated = try withRandomState(MLXRandom.RandomState(seed: seed)) {
            var iterator = try TokenIterator(input: input, model: model, parameters: parameters)
            var out: [Int] = []
            out.reserveCapacity(tokenLimit)
            while let token = iterator.next() {
                if spec.config.stopTokenIds.contains(token) { break }
                out.append(token)
            }
            return out
        }
        return spec.decode(generated)
    }
}
