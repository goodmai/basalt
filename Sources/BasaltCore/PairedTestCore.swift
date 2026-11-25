import Foundation

public enum SelfRewriteBudgetError: Error, Sendable, Equatable, CustomStringConvertible {
    case reservedSplitAccessDenied(problemId: String)
    case reservedEvaluationLimitReached(used: Int, limit: Int)
    case forwardPassBudgetExceeded(requested: Int, remaining: Int)
    case toolCallBudgetExceeded(limit: Int)
    case editBudgetExceeded(limit: Int)
    case cloudCostBudgetExceeded(requested: Double, remaining: Double)
    case emptyProblemSet
    case invalidBudget(String)

    public var description: String {
        switch self {
        case .reservedSplitAccessDenied(let problemId):
            return "access to reserved split problem '\(problemId)' is denied"
        case .reservedEvaluationLimitReached(let used, let limit):
            return "reserved split evaluation limit reached (\(used)/\(limit))"
        case .forwardPassBudgetExceeded(let requested, let remaining):
            return "forward pass budget exceeded: requested \(requested), remaining \(remaining)"
        case .toolCallBudgetExceeded(let limit):
            return "tool call budget exceeded (limit \(limit))"
        case .editBudgetExceeded(let limit):
            return "edit application budget exceeded (limit \(limit))"
        case .cloudCostBudgetExceeded(let requested, let remaining):
            return String(format: "cloud cost budget exceeded: requested $%.4f, remaining $%.4f", requested, remaining)
        case .emptyProblemSet:
            return "paired test requires at least one problem"
        case .invalidBudget(let reason):
            return "invalid self-rewrite budget: \(reason)"
        }
    }
}

public struct PairedProblem: Sendable, Codable, Equatable {
    public let id: String
    public let prompt: String
    public let expectedAnswer: String
    public let sourceText: String?
    public let metadata: [String: String]

    public init(
        id: String,
        prompt: String,
        expectedAnswer: String,
        sourceText: String? = nil,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.prompt = prompt
        self.expectedAnswer = expectedAnswer
        self.sourceText = sourceText
        self.metadata = metadata
    }
}

public struct PairedGradeResult: Sendable, Codable, Equatable {
    public let passed: Bool
    public let score: Double
    public let extractedAnswer: String?
    public let detail: String?

    public init(
        passed: Bool,
        score: Double? = nil,
        extractedAnswer: String? = nil,
        detail: String? = nil
    ) {
        self.passed = passed
        self.score = score ?? (passed ? 1.0 : 0.0)
        self.extractedAnswer = extractedAnswer
        self.detail = detail
    }
}

public protocol PairedAnswerGrader: Sendable {
    func grade(response: String, problem: PairedProblem) -> PairedGradeResult
}

public struct PairedSamplePair: Sendable, Codable, Equatable {
    public let problemId: String
    public let seed: UInt64
    public let baseGrade: PairedGradeResult
    public let editedGrade: PairedGradeResult

    public init(
        problemId: String,
        seed: UInt64,
        baseGrade: PairedGradeResult,
        editedGrade: PairedGradeResult
    ) {
        self.problemId = problemId
        self.seed = seed
        self.baseGrade = baseGrade
        self.editedGrade = editedGrade
    }
}

public enum PairedTestStatistics {
    public static func summarize(_ samples: [PairedSamplePair]) -> PairedTestSummary {
        guard !samples.isEmpty else {
            return PairedTestSummary(
                problemCount: 0, baseAccuracy: 0, editedAccuracy: 0,
                standardError: 0, wrongToRight: 0, rightToWrong: 0
            )
        }
        var grouped: [String: (baseSum: Double, editedSum: Double, count: Int)] = [:]
        var order: [String] = []
        var wrongToRight = 0
        var rightToWrong = 0
        for sample in samples {
            if grouped[sample.problemId] == nil { order.append(sample.problemId) }
            var entry = grouped[sample.problemId] ?? (0, 0, 0)
            entry.baseSum += sample.baseGrade.score
            entry.editedSum += sample.editedGrade.score
            entry.count += 1
            grouped[sample.problemId] = entry
            if !sample.baseGrade.passed && sample.editedGrade.passed { wrongToRight += 1 }
            if sample.baseGrade.passed && !sample.editedGrade.passed { rightToWrong += 1 }
        }
        let n = order.count
        var baseMeanSum = 0.0
        var editedMeanSum = 0.0
        var deltas: [Double] = []
        deltas.reserveCapacity(n)
        for id in order {
            guard let entry = grouped[id], entry.count > 0 else { continue }
            let b = entry.baseSum / Double(entry.count)
            let e = entry.editedSum / Double(entry.count)
            baseMeanSum += b
            editedMeanSum += e
            deltas.append(e - b)
        }
        let count = Double(max(1, deltas.count))
        let baseAcc = baseMeanSum / count
        let editedAcc = editedMeanSum / count
        let meanDelta = editedAcc - baseAcc
        let se: Double
        if deltas.count > 1 {
            let sq = deltas.reduce(0.0) { acc, d in
                let diff = d - meanDelta
                return acc + diff * diff
            }
            let variance = sq / Double(deltas.count - 1)
            se = sqrt(max(0.0, variance) / Double(deltas.count))
        } else {
            se = 0.0
        }
        return PairedTestSummary(
            problemCount: deltas.count,
            baseAccuracy: baseAcc,
            editedAccuracy: editedAcc,
            standardError: se,
            wrongToRight: wrongToRight,
            rightToWrong: rightToWrong
        )
    }

    public static func confidenceInterval95(_ summary: PairedTestSummary) -> (low: Double, high: Double) {
        let half = 1.96 * summary.standardError
        return (summary.delta - half, summary.delta + half)
    }

    public static func seExcludesZero(_ summary: PairedTestSummary) -> Bool {
        summary.problemCount > 0 && summary.delta > summary.standardError
    }

    public static func ci95ExcludesZero(_ summary: PairedTestSummary) -> Bool {
        summary.problemCount > 1 && summary.delta > 1.96 * summary.standardError
    }
}

public struct SelfRewriteBudget: Sendable, Codable, Equatable {
    public let maxForwardPasses: Int
    public let maxToolCalls: Int
    public let maxReservedTests: Int
    public let maxEditApplies: Int
    public let maxCloudUSD: Double?
    public private(set) var forwardPassesUsed: Int = 0
    public private(set) var toolCallsUsed: Int = 0
    public private(set) var reservedTestsUsed: Int = 0
    public private(set) var editAppliesUsed: Int = 0
    public private(set) var cloudUSDUsed: Double = 0

    public init(
        maxForwardPasses: Int = 20,
        maxToolCalls: Int = 200,
        maxReservedTests: Int = 1,
        maxEditApplies: Int = 32,
        maxCloudUSD: Double? = nil
    ) throws(SelfRewriteBudgetError) {
        guard maxForwardPasses > 0, maxToolCalls > 0, maxReservedTests >= 0, maxEditApplies > 0 else {
            throw .invalidBudget("limits must be positive (maxReservedTests >= 0)")
        }
        if let maxCloudUSD, (!maxCloudUSD.isFinite || maxCloudUSD < 0) {
            throw .invalidBudget("maxCloudUSD must be finite and non-negative")
        }
        self.maxForwardPasses = maxForwardPasses
        self.maxToolCalls = maxToolCalls
        self.maxReservedTests = maxReservedTests
        self.maxEditApplies = maxEditApplies
        self.maxCloudUSD = maxCloudUSD
    }

    public var remainingForwardPasses: Int { max(0, maxForwardPasses - forwardPassesUsed) }
    public var remainingToolCalls: Int { max(0, maxToolCalls - toolCallsUsed) }
    public var remainingReservedTests: Int { max(0, maxReservedTests - reservedTestsUsed) }
    public var remainingCloudUSD: Double? { maxCloudUSD.map { max(0.0, $0 - cloudUSDUsed) } }
    public var isExhausted: Bool { remainingForwardPasses == 0 || remainingToolCalls == 0 }

    public mutating func consumeForwardPasses(_ count: Int) throws(SelfRewriteBudgetError) {
        guard count >= 0 else { throw .invalidBudget("forward pass count cannot be negative") }
        guard count <= remainingForwardPasses else {
            throw .forwardPassBudgetExceeded(requested: count, remaining: remainingForwardPasses)
        }
        forwardPassesUsed += count
    }

    public mutating func consumeToolCall() throws(SelfRewriteBudgetError) {
        guard toolCallsUsed < maxToolCalls else { throw .toolCallBudgetExceeded(limit: maxToolCalls) }
        toolCallsUsed += 1
    }

    public mutating func consumeEditApply() throws(SelfRewriteBudgetError) {
        guard editAppliesUsed < maxEditApplies else { throw .editBudgetExceeded(limit: maxEditApplies) }
        editAppliesUsed += 1
    }

    public mutating func consumeReservedTest() throws(SelfRewriteBudgetError) {
        guard reservedTestsUsed < maxReservedTests else {
            throw .reservedEvaluationLimitReached(used: reservedTestsUsed, limit: maxReservedTests)
        }
        reservedTestsUsed += 1
    }

    public mutating func consumeCloudCost(usd: Double) throws(SelfRewriteBudgetError) {
        guard usd.isFinite, usd >= 0 else { throw .invalidBudget("cloud cost must be finite and non-negative") }
        if let remaining = remainingCloudUSD, usd > remaining + 1e-12 {
            throw .cloudCostBudgetExceeded(requested: usd, remaining: remaining)
        }
        cloudUSDUsed += usd
    }
}

public struct PairedDatasetSplit: Sendable {
    public let practice: [PairedProblem]
    private let reserved: [PairedProblem]
    private let practiceById: [String: PairedProblem]
    private let reservedIds: Set<String>

    public var reservedCount: Int { reserved.count }

    public func reservedFenceStrings() -> Set<String> {
        Set(reserved.map(\.prompt))
    }

    public init(practice: [PairedProblem], reserved: [PairedProblem]) {
        self.practice = practice
        self.reserved = reserved
        self.practiceById = Dictionary(practice.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        self.reservedIds = Set(reserved.map(\.id))
    }

    public func practiceProblem(id: String) throws(SelfRewriteBudgetError) -> PairedProblem? {
        if reservedIds.contains(id) {
            throw .reservedSplitAccessDenied(problemId: id)
        }
        return practiceById[id]
    }

    public func practiceBatch(ids: [String]) throws(SelfRewriteBudgetError) -> [PairedProblem] {
        var resolved: [PairedProblem] = []
        resolved.reserveCapacity(ids.count)
        for id in ids {
            if let item = try practiceProblem(id: id) {
                resolved.append(item)
            }
        }
        return resolved
    }

    public func evaluateReserved(
        budget: inout SelfRewriteBudget,
        run: ([PairedProblem]) throws -> [PairedSamplePair]
    ) throws -> PairedTestSummary {
        guard !reserved.isEmpty else { throw SelfRewriteBudgetError.emptyProblemSet }
        var nextBudget = budget
        try nextBudget.consumeReservedTest()
        let samples = try run(reserved)
        budget = nextBudget
        return PairedTestStatistics.summarize(samples)
    }

    public static func decontaminated(
        candidates: [PairedProblem],
        benchmarks: [String],
        practiceCount: Int,
        ngramSize: Int = 8,
        seed: UInt64 = 676
    ) -> (split: PairedDatasetSplit, removedCount: Int) {
        let normalizedBench = Set(benchmarks.map(Self.normalizeText).filter { !$0.isEmpty })
        let benchNgrams = Self.collectNgrams(texts: Array(normalizedBench), n: max(1, ngramSize))
        var clean: [PairedProblem] = []
        var removed = 0
        for problem in candidates {
            let norm = Self.normalizeText(problem.prompt)
            if normalizedBench.contains(norm) || Self.overlaps(norm, benchNgrams: benchNgrams, n: max(1, ngramSize)) {
                removed += 1
            } else {
                clean.append(problem)
            }
        }
        let shuffled = Self.deterministicShuffle(clean, seed: seed)
        let cut = min(max(0, practiceCount), shuffled.count)
        let split = PairedDatasetSplit(
            practice: Array(shuffled[..<cut]),
            reserved: Array(shuffled[cut...])
        )
        return (split, removed)
    }

    private static func normalizeText(_ text: String) -> String {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private static func collectNgrams(texts: [String], n: Int) -> Set<String> {
        var out = Set<String>()
        for text in texts {
            let words = text.split(separator: " ")
            guard words.count >= n else { continue }
            for i in 0 ... (words.count - n) {
                out.insert(words[i ..< (i + n)].joined(separator: " "))
            }
        }
        return out
    }

    private static func overlaps(_ normalized: String, benchNgrams: Set<String>, n: Int) -> Bool {
        guard !benchNgrams.isEmpty else { return false }
        let words = normalized.split(separator: " ")
        guard words.count >= n else { return false }
        for i in 0 ... (words.count - n) {
            if benchNgrams.contains(words[i ..< (i + n)].joined(separator: " ")) {
                return true
            }
        }
        return false
    }

    private static func deterministicShuffle(_ items: [PairedProblem], seed: UInt64) -> [PairedProblem] {
        var copy = items
        guard copy.count > 1 else { return copy }
        var state = seed == 0 ? 0x9e37_79b9_7f4a_7c15 : seed
        for i in stride(from: copy.count - 1, through: 1, by: -1) {
            state &+= 0x9e37_79b9_7f4a_7c15
            var z = state
            z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
            z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
            z = z ^ (z >> 31)
            let j = Int(z % UInt64(i + 1))
            if i != j { copy.swapAt(i, j) }
        }
        return copy
    }
}
