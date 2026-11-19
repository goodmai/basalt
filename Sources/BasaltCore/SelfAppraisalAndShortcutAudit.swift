import Foundation

public struct AppraisalEpisode: Sendable, Codable, Equatable {
    public var id: String
    public var reward: Double
    public var selfScore: Double
    public var noveltyScore: Double
    public var tokenCount: Int
    public var verifiedSuccess: Bool

    public init(
        id: String,
        reward: Double,
        selfScore: Double,
        noveltyScore: Double,
        tokenCount: Int,
        verifiedSuccess: Bool
    ) {
        self.id = id
        self.reward = reward
        self.selfScore = selfScore
        self.noveltyScore = noveltyScore
        self.tokenCount = tokenCount
        self.verifiedSuccess = verifiedSuccess
    }

    enum CodingKeys: String, CodingKey {
        case id, reward
        case selfScore = "self_score"
        case noveltyScore = "novelty_score"
        case tokenCount = "token_count"
        case verifiedSuccess = "verified_success"
    }
}

public struct SelfAppraisalConfig: Sendable, Codable, Equatable {
    public var minCombinedScore: Double
    public var minNovelty: Double
    public var maxSelectedEpisodes: Int
    public var requireVerifiedSuccess: Bool

    public init(
        minCombinedScore: Double = 0.60,
        minNovelty: Double = 0.20,
        maxSelectedEpisodes: Int = 64,
        requireVerifiedSuccess: Bool = true
    ) {
        self.minCombinedScore = minCombinedScore
        self.minNovelty = minNovelty
        self.maxSelectedEpisodes = maxSelectedEpisodes
        self.requireVerifiedSuccess = requireVerifiedSuccess
    }

    public func validate() throws(BasaltError) {
        guard minCombinedScore.isFinite, (0.0 ... 1.0).contains(minCombinedScore) else {
            throw .invalidRequestStructure(details: "self-appraisal: minCombinedScore must lie in 0...1")
        }
        guard minNovelty.isFinite, (0.0 ... 1.0).contains(minNovelty) else {
            throw .invalidRequestStructure(details: "self-appraisal: minNovelty must lie in 0...1")
        }
        guard maxSelectedEpisodes >= 1 else {
            throw .invalidRequestStructure(details: "self-appraisal: maxSelectedEpisodes must be >= 1")
        }
    }
}

public enum SelfAppraisalCore {
    public static func combinedScore(for episode: AppraisalEpisode) -> Double {
        let r = min(1.0, max(0.0, episode.reward.isFinite ? episode.reward : 0.0))
        let s = min(1.0, max(0.0, episode.selfScore.isFinite ? episode.selfScore : 0.0))
        let n = min(1.0, max(0.0, episode.noveltyScore.isFinite ? episode.noveltyScore : 0.0))
        return 0.45 * r + 0.35 * s + 0.20 * n
    }

    public static func selectEpisodes(
        _ episodes: [AppraisalEpisode],
        config: SelfAppraisalConfig = .init()
    ) throws(BasaltError) -> [AppraisalEpisode] {
        try config.validate()
        let eligible = episodes.filter { ep in
            if config.requireVerifiedSuccess && !ep.verifiedSuccess { return false }
            let nov = ep.noveltyScore.isFinite ? ep.noveltyScore : 0.0
            guard nov >= config.minNovelty else { return false }
            return combinedScore(for: ep) >= config.minCombinedScore
        }
        let ranked = eligible.sorted { lhs, rhs in
            let ls = combinedScore(for: lhs)
            let rs = combinedScore(for: rhs)
            return ls == rs ? lhs.tokenCount < rhs.tokenCount : ls > rs
        }
        return Array(ranked.prefix(config.maxSelectedEpisodes))
    }
}

public struct RLVRVerificationSample: Sendable, Codable, Equatable {
    public var taskId: String
    public var verifierPassed: Bool
    public var adversarialPerturbationPassed: Bool
    public var intermediateStepsValid: Bool
    public var reasoningTokenCount: Int

    public init(
        taskId: String,
        verifierPassed: Bool,
        adversarialPerturbationPassed: Bool,
        intermediateStepsValid: Bool,
        reasoningTokenCount: Int = 0
    ) {
        self.taskId = taskId
        self.verifierPassed = verifierPassed
        self.adversarialPerturbationPassed = adversarialPerturbationPassed
        self.intermediateStepsValid = intermediateStepsValid
        self.reasoningTokenCount = reasoningTokenCount
    }

    enum CodingKeys: String, CodingKey {
        case taskId = "task_id"
        case verifierPassed = "verifier_passed"
        case adversarialPerturbationPassed = "adversarial_perturbation_passed"
        case intermediateStepsValid = "intermediate_steps_valid"
        case reasoningTokenCount = "reasoning_token_count"
    }
}

public struct ShortcutAuditReport: Sendable, Codable, Equatable {
    public var sampleCount: Int
    public var verifierPassRate: Double
    public var adversarialPassRate: Double
    public var shortcutRate: Double
    public var shortcutFlagged: Bool
    public var flaggedTaskIds: [String]

    public init(
        sampleCount: Int,
        verifierPassRate: Double,
        adversarialPassRate: Double,
        shortcutRate: Double,
        shortcutFlagged: Bool,
        flaggedTaskIds: [String]
    ) {
        self.sampleCount = sampleCount
        self.verifierPassRate = verifierPassRate
        self.adversarialPassRate = adversarialPassRate
        self.shortcutRate = shortcutRate
        self.shortcutFlagged = shortcutFlagged
        self.flaggedTaskIds = flaggedTaskIds
    }

    enum CodingKeys: String, CodingKey {
        case sampleCount = "sample_count"
        case verifierPassRate = "verifier_pass_rate"
        case adversarialPassRate = "adversarial_pass_rate"
        case shortcutRate = "shortcut_rate"
        case shortcutFlagged = "shortcut_flagged"
        case flaggedTaskIds = "flagged_task_ids"
    }
}

public enum ShortcutAuditCore {
    public static func evaluate(
        _ samples: [RLVRVerificationSample],
        maxAllowedShortcutRate: Double = 0.10
    ) throws(BasaltError) -> ShortcutAuditReport {
        guard maxAllowedShortcutRate.isFinite, (0.0 ... 1.0).contains(maxAllowedShortcutRate) else {
            throw .invalidRequestStructure(details: "shortcut audit: maxAllowedShortcutRate must lie in 0...1")
        }
        guard !samples.isEmpty else {
            return ShortcutAuditReport(
                sampleCount: 0, verifierPassRate: 0, adversarialPassRate: 0,
                shortcutRate: 0, shortcutFlagged: false, flaggedTaskIds: []
            )
        }
        let passed = samples.filter(\.verifierPassed)
        let advPassed = samples.filter(\.adversarialPerturbationPassed)
        let shortcuts = passed.filter { !$0.adversarialPerturbationPassed || !$0.intermediateStepsValid }
        let total = Double(samples.count)
        let vRate = Double(passed.count) / total
        let aRate = Double(advPassed.count) / total
        let sRate = passed.isEmpty ? 0.0 : Double(shortcuts.count) / Double(passed.count)
        return ShortcutAuditReport(
            sampleCount: samples.count,
            verifierPassRate: vRate,
            adversarialPassRate: aRate,
            shortcutRate: sRate,
            shortcutFlagged: sRate > maxAllowedShortcutRate,
            flaggedTaskIds: shortcuts.map(\.taskId)
        )
    }
}
