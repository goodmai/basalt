import Foundation

public struct QuantizationAuditThresholds: Sendable, Codable, Equatable {
    public var maxPerplexityDeltaRatio: Double
    public var maxAccuracyDrop: Double
    public var maxGroupDisparity: Double
    public var maxLeakageSignal: Double

    public init(
        maxPerplexityDeltaRatio: Double = 0.15,
        maxAccuracyDrop: Double = 0.03,
        maxGroupDisparity: Double = 0.08,
        maxLeakageSignal: Double = 0.10
    ) {
        self.maxPerplexityDeltaRatio = maxPerplexityDeltaRatio
        self.maxAccuracyDrop = maxAccuracyDrop
        self.maxGroupDisparity = maxGroupDisparity
        self.maxLeakageSignal = maxLeakageSignal
    }

    public func validate() throws(BasaltError) {
        guard maxPerplexityDeltaRatio.isFinite, maxPerplexityDeltaRatio >= 0 else {
            throw .invalidRequestStructure(details: "quantization audit: maxPerplexityDeltaRatio must be non-negative")
        }
        guard maxAccuracyDrop.isFinite, (0.0 ... 1.0).contains(maxAccuracyDrop) else {
            throw .invalidRequestStructure(details: "quantization audit: maxAccuracyDrop must lie in 0...1")
        }
        guard maxGroupDisparity.isFinite, (0.0 ... 1.0).contains(maxGroupDisparity) else {
            throw .invalidRequestStructure(details: "quantization audit: maxGroupDisparity must lie in 0...1")
        }
        guard maxLeakageSignal.isFinite, maxLeakageSignal >= 0 else {
            throw .invalidRequestStructure(details: "quantization audit: maxLeakageSignal must be non-negative")
        }
    }

    enum CodingKeys: String, CodingKey {
        case maxPerplexityDeltaRatio = "max_perplexity_delta_ratio"
        case maxAccuracyDrop = "max_accuracy_drop"
        case maxGroupDisparity = "max_group_disparity"
        case maxLeakageSignal = "max_leakage_signal"
    }
}

public struct FairnessGroupObservation: Sendable, Codable, Equatable {
    public var group: String
    public var baselineScore: Double
    public var quantizedScore: Double

    public init(group: String, baselineScore: Double, quantizedScore: Double) {
        self.group = group
        self.baselineScore = baselineScore
        self.quantizedScore = quantizedScore
    }

    enum CodingKeys: String, CodingKey {
        case group
        case baselineScore = "baseline_score"
        case quantizedScore = "quantized_score"
    }
}

public struct LeakageProbeObservation: Sendable, Codable, Equatable {
    public var promptId: String
    public var memberLogprob: Double
    public var nonMemberLogprob: Double

    public init(promptId: String, memberLogprob: Double, nonMemberLogprob: Double) {
        self.promptId = promptId
        self.memberLogprob = memberLogprob
        self.nonMemberLogprob = nonMemberLogprob
    }

    enum CodingKeys: String, CodingKey {
        case promptId = "prompt_id"
        case memberLogprob = "member_logprob"
        case nonMemberLogprob = "non_member_logprob"
    }
}

public struct QuantizationAuditInput: Sendable, Codable, Equatable {
    public var baselinePerplexity: Double
    public var quantizedPerplexity: Double
    public var baselineAccuracy: Double
    public var quantizedAccuracy: Double
    public var fairnessGroups: [FairnessGroupObservation]
    public var leakageProbes: [LeakageProbeObservation]

    public init(
        baselinePerplexity: Double,
        quantizedPerplexity: Double,
        baselineAccuracy: Double,
        quantizedAccuracy: Double,
        fairnessGroups: [FairnessGroupObservation] = [],
        leakageProbes: [LeakageProbeObservation] = []
    ) {
        self.baselinePerplexity = baselinePerplexity
        self.quantizedPerplexity = quantizedPerplexity
        self.baselineAccuracy = baselineAccuracy
        self.quantizedAccuracy = quantizedAccuracy
        self.fairnessGroups = fairnessGroups
        self.leakageProbes = leakageProbes
    }

    enum CodingKeys: String, CodingKey {
        case baselinePerplexity = "baseline_perplexity"
        case quantizedPerplexity = "quantized_perplexity"
        case baselineAccuracy = "baseline_accuracy"
        case quantizedAccuracy = "quantized_accuracy"
        case fairnessGroups = "fairness_groups"
        case leakageProbes = "leakage_probes"
    }
}

public struct QuantizationAuditVerdict: Sendable, Codable, Equatable {
    public var passed: Bool
    public var perplexityDeltaRatio: Double
    public var accuracyDrop: Double
    public var maxGroupDisparity: Double
    public var dispersionDisparity: Double
    public var leakageSignal: Double
    public var failures: [String]

    public init(
        perplexityDeltaRatio: Double,
        accuracyDrop: Double,
        maxGroupDisparity: Double,
        dispersionDisparity: Double,
        leakageSignal: Double,
        failures: [String] = []
    ) {
        self.passed = failures.isEmpty
        self.perplexityDeltaRatio = perplexityDeltaRatio
        self.accuracyDrop = accuracyDrop
        self.maxGroupDisparity = maxGroupDisparity
        self.dispersionDisparity = dispersionDisparity
        self.leakageSignal = leakageSignal
        self.failures = failures
    }

    enum CodingKeys: String, CodingKey {
        case passed, failures
        case perplexityDeltaRatio = "perplexity_delta_ratio"
        case accuracyDrop = "accuracy_drop"
        case maxGroupDisparity = "max_group_disparity"
        case dispersionDisparity = "dispersion_disparity"
        case leakageSignal = "leakage_signal"
    }
}

public struct QATSchedule: Sendable, Codable, Equatable {
    public var domainAdaptationSteps: Int
    public var qatSteps: Int
    public var fakeQuantBits: Int
    public var groupSize: Int

    public init(domainAdaptationSteps: Int, qatSteps: Int, fakeQuantBits: Int, groupSize: Int) {
        self.domainAdaptationSteps = domainAdaptationSteps
        self.qatSteps = qatSteps
        self.fakeQuantBits = fakeQuantBits
        self.groupSize = groupSize
    }
}

public struct QATBudgetConfig: Sendable, Codable, Equatable {
    public var bits: Int
    public var groupSize: Int
    public var totalStepBudget: Int
    public var isDomainSpecializedCheckpoint: Bool

    public init(
        bits: Int = 4,
        groupSize: Int = 64,
        totalStepBudget: Int = 100,
        isDomainSpecializedCheckpoint: Bool = true
    ) {
        self.bits = bits
        self.groupSize = groupSize
        self.totalStepBudget = totalStepBudget
        self.isDomainSpecializedCheckpoint = isDomainSpecializedCheckpoint
    }

    public func validate() throws(BasaltError) {
        guard [2, 3, 4, 8].contains(bits) else {
            throw .invalidRequestStructure(details: "QAT: bits must be one of 2, 3, 4, 8")
        }
        guard [32, 64, 128].contains(groupSize) else {
            throw .invalidRequestStructure(details: "QAT: groupSize must be one of 32, 64, 128")
        }
        guard totalStepBudget >= 2 else {
            throw .invalidRequestStructure(details: "QAT: totalStepBudget must be at least 2")
        }
    }

    public func recommendedSchedule() throws(BasaltError) -> QATSchedule {
        try validate()
        let domainShare: Double = isDomainSpecializedCheckpoint ? 0.25 : 0.60
        let domainSteps = max(1, Int((Double(totalStepBudget) * domainShare).rounded()))
        let qatSteps = max(1, totalStepBudget - domainSteps)
        return QATSchedule(
            domainAdaptationSteps: domainSteps,
            qatSteps: qatSteps,
            fakeQuantBits: bits,
            groupSize: groupSize
        )
    }
}

public enum QuantizationAuditCore {
    public static func evaluate(
        _ input: QuantizationAuditInput,
        thresholds: QuantizationAuditThresholds = .init()
    ) throws(BasaltError) -> QuantizationAuditVerdict {
        try thresholds.validate()
        guard input.baselinePerplexity.isFinite, input.baselinePerplexity > 0,
              input.quantizedPerplexity.isFinite, input.quantizedPerplexity > 0 else {
            throw .invalidRequestStructure(details: "quantization audit: perplexities must be finite and positive")
        }
        guard input.baselineAccuracy.isFinite, (0.0 ... 1.0).contains(input.baselineAccuracy),
              input.quantizedAccuracy.isFinite, (0.0 ... 1.0).contains(input.quantizedAccuracy) else {
            throw .invalidRequestStructure(details: "quantization audit: accuracies must lie in 0...1")
        }
        let pplRatio = max(0.0, (input.quantizedPerplexity - input.baselinePerplexity) / input.baselinePerplexity)
        let accDrop = max(0.0, input.baselineAccuracy - input.quantizedAccuracy)
        let (maxDisp, dispersion) = fairnessDisparity(input.fairnessGroups)
        let leakage = leakageSignal(input.leakageProbes)
        var failures: [String] = []
        if pplRatio > thresholds.maxPerplexityDeltaRatio {
            failures.append(String(format: "perplexity delta %.3f exceeds %.3f", pplRatio, thresholds.maxPerplexityDeltaRatio))
        }
        if accDrop > thresholds.maxAccuracyDrop {
            failures.append(String(format: "accuracy drop %.3f exceeds %.3f", accDrop, thresholds.maxAccuracyDrop))
        }
        if maxDisp > thresholds.maxGroupDisparity {
            failures.append(String(format: "group fairness disparity %.3f exceeds %.3f", maxDisp, thresholds.maxGroupDisparity))
        }
        if leakage > thresholds.maxLeakageSignal {
            failures.append(String(format: "membership leakage signal %.3f exceeds %.3f", leakage, thresholds.maxLeakageSignal))
        }
        return QuantizationAuditVerdict(
            perplexityDeltaRatio: pplRatio,
            accuracyDrop: accDrop,
            maxGroupDisparity: maxDisp,
            dispersionDisparity: dispersion,
            leakageSignal: leakage,
            failures: failures
        )
    }

    private static func fairnessDisparity(_ groups: [FairnessGroupObservation]) -> (maxDelta: Double, dispersion: Double) {
        guard !groups.isEmpty else { return (0.0, 0.0) }
        let deltas = groups.map { abs($0.baselineScore - $0.quantizedScore) }
        let maxDelta = deltas.max() ?? 0.0
        let quantScores = groups.map(\.quantizedScore)
        let dispersion = (quantScores.max() ?? 0.0) - (quantScores.min() ?? 0.0)
        return (maxDelta, max(0.0, dispersion))
    }

    private static func leakageSignal(_ probes: [LeakageProbeObservation]) -> Double {
        guard !probes.isEmpty else { return 0.0 }
        var sum = 0.0
        for probe in probes {
            let gap = probe.memberLogprob - probe.nonMemberLogprob
            let prob = 1.0 / (1.0 + Foundation::exp(-gap))
            sum += max(0.0, prob - 0.5) * 2.0
        }
        return sum / Double(probes.count)
    }
}
