import Foundation

public struct LayerUpdateRatio: Sendable, Codable, Equatable {
    public var layerName: String
    public var updateNorm: Double
    public var weightNorm: Double
    public var ratio: Double
    public var stepRMS: Double

    public var isFinite: Bool {
        updateNorm.isFinite && weightNorm.isFinite && ratio.isFinite && stepRMS.isFinite
    }

    public init(layerName: String, updateNorm: Double, weightNorm: Double, ratio: Double, stepRMS: Double) {
        (self.layerName, self.updateNorm, self.weightNorm, self.ratio, self.stepRMS) = (
            layerName, updateNorm, weightNorm, ratio, stepRMS
        )
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        layerName = try c.decodeOrDefault(.layerName, default: "")
        updateNorm = try c.decodeOrDefault(.updateNorm, default: 0.0)
        weightNorm = try c.decodeOrDefault(.weightNorm, default: 0.0)
        ratio = try c.decodeOrDefault(.ratio, default: 0.0)
        stepRMS = try c.decodeOrDefault(.stepRMS, default: 0.0)
    }

    func sanitized() -> LayerUpdateRatio {
        LayerUpdateRatio(
            layerName: layerName,
            updateNorm: TrainingStabilityMath.finiteOrZero(updateNorm),
            weightNorm: TrainingStabilityMath.finiteOrZero(weightNorm),
            ratio: TrainingStabilityMath.finiteOrZero(ratio),
            stepRMS: TrainingStabilityMath.finiteOrZero(stepRMS)
        )
    }
}

public struct LogitZLossStats: Sendable, Codable, Equatable {
    public var zLoss: Double
    public var maxLogit: Double

    public var isFinite: Bool { zLoss.isFinite && maxLogit.isFinite }

    public init(zLoss: Double, maxLogit: Double) {
        (self.zLoss, self.maxLogit) = (zLoss, maxLogit)
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        zLoss = try c.decodeOrDefault(.zLoss, default: 0.0)
        maxLogit = try c.decodeOrDefault(.maxLogit, default: 0.0)
    }

    func sanitized() -> LogitZLossStats {
        LogitZLossStats(
            zLoss: TrainingStabilityMath.finiteOrZero(zLoss),
            maxLogit: TrainingStabilityMath.finiteOrZero(maxLogit)
        )
    }
}

public struct GRPOStabilityMetrics: Sendable, Codable, Equatable {
    public var totalGroups: Int
    public var skippedZeroAdvantageGroups: Int
    public var zeroAdvantageGroupRate: Double
    public var clipFraction: Double
    public var meanKL: Double
    public var rewardVariance: Double

    public var isFinite: Bool {
        zeroAdvantageGroupRate.isFinite && clipFraction.isFinite && meanKL.isFinite && rewardVariance.isFinite
    }

    public init(
        totalGroups: Int,
        skippedZeroAdvantageGroups: Int,
        zeroAdvantageGroupRate: Double,
        clipFraction: Double,
        meanKL: Double,
        rewardVariance: Double
    ) {
        self.totalGroups = totalGroups
        self.skippedZeroAdvantageGroups = skippedZeroAdvantageGroups
        self.zeroAdvantageGroupRate = zeroAdvantageGroupRate
        self.clipFraction = clipFraction
        self.meanKL = meanKL
        self.rewardVariance = rewardVariance
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        totalGroups = try c.decodeOrDefault(.totalGroups, default: 0)
        skippedZeroAdvantageGroups = try c.decodeOrDefault(.skippedZeroAdvantageGroups, default: 0)
        zeroAdvantageGroupRate = try c.decodeOrDefault(.zeroAdvantageGroupRate, default: 0.0)
        clipFraction = try c.decodeOrDefault(.clipFraction, default: 0.0)
        meanKL = try c.decodeOrDefault(.meanKL, default: 0.0)
        rewardVariance = try c.decodeOrDefault(.rewardVariance, default: 0.0)
    }

    func sanitized() -> GRPOStabilityMetrics {
        GRPOStabilityMetrics(
            totalGroups: max(0, totalGroups),
            skippedZeroAdvantageGroups: max(0, skippedZeroAdvantageGroups),
            zeroAdvantageGroupRate: TrainingStabilityMath.finiteOrZero(zeroAdvantageGroupRate),
            clipFraction: TrainingStabilityMath.finiteOrZero(clipFraction),
            meanKL: TrainingStabilityMath.finiteOrZero(meanKL),
            rewardVariance: TrainingStabilityMath.finiteOrZero(rewardVariance)
        )
    }
}

public struct TrainingStepTelemetry: Sendable, Codable, Equatable {
    public var step: Int = 0
    public var gradNorm: Double = 0
    public var meanUpdateToWeightRatio: Double = 0
    public var maxUpdateToWeightRatio: Double = 0
    public var meanStepRMS: Double = 0
    public var logitStats: LogitZLossStats?
    public var topLayerRatios: [LayerUpdateRatio] = []

    public var isFinite: Bool {
        gradNorm.isFinite
            && meanUpdateToWeightRatio.isFinite
            && maxUpdateToWeightRatio.isFinite
            && meanStepRMS.isFinite
            && (logitStats?.isFinite ?? true)
            && topLayerRatios.allSatisfy(\.isFinite)
    }

    public init(
        step: Int = 0,
        gradNorm: Double = 0,
        meanUpdateToWeightRatio: Double = 0,
        maxUpdateToWeightRatio: Double = 0,
        meanStepRMS: Double = 0,
        logitStats: LogitZLossStats? = nil
    ) {
        (self.step, self.gradNorm, self.meanUpdateToWeightRatio, self.maxUpdateToWeightRatio, self.meanStepRMS) = (
            step, gradNorm, meanUpdateToWeightRatio, maxUpdateToWeightRatio, meanStepRMS
        )
        self.logitStats = logitStats
        self.topLayerRatios = []
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        step = try c.decodeOrDefault(.step, default: 0)
        gradNorm = try c.decodeOrDefault(.gradNorm, default: 0.0)
        meanUpdateToWeightRatio = try c.decodeOrDefault(.meanUpdateToWeightRatio, default: 0.0)
        maxUpdateToWeightRatio = try c.decodeOrDefault(.maxUpdateToWeightRatio, default: 0.0)
        meanStepRMS = try c.decodeOrDefault(.meanStepRMS, default: 0.0)
        logitStats = try c.decodeIfPresent(LogitZLossStats.self, forKey: .logitStats)
        topLayerRatios = try c.decodeOrDefault(.topLayerRatios, default: [LayerUpdateRatio]())
    }

    public static func from(
        step: Int,
        gradNorm: Double,
        layerRatios: [LayerUpdateRatio],
        logitStats: LogitZLossStats? = nil,
        topK: Int = 5
    ) -> TrainingStepTelemetry {
        var telemetry = TrainingStepTelemetry(step: step, gradNorm: gradNorm, logitStats: logitStats)
        guard let first = layerRatios.first else { return telemetry }
        var (ratioSum, maxRatio, rmsSum) = (0.0, first.ratio, 0.0)
        for item in layerRatios {
            ratioSum += item.ratio
            maxRatio = TrainingStabilityMath.propagateMax(maxRatio, item.ratio)
            rmsSum += item.stepRMS
        }
        let count = Double(layerRatios.count)
        (telemetry.meanUpdateToWeightRatio, telemetry.maxUpdateToWeightRatio, telemetry.meanStepRMS) = (
            ratioSum / count, maxRatio, rmsSum / count
        )
        telemetry.topLayerRatios = Array(layerRatios.sorted(by: Self.higherRatioFirst).prefix(max(0, topK)))
        return telemetry
    }

    func sanitized() -> TrainingStepTelemetry {
        var copy = TrainingStepTelemetry(
            step: step,
            gradNorm: TrainingStabilityMath.finiteOrZero(gradNorm),
            meanUpdateToWeightRatio: TrainingStabilityMath.finiteOrZero(meanUpdateToWeightRatio),
            maxUpdateToWeightRatio: TrainingStabilityMath.finiteOrZero(maxUpdateToWeightRatio),
            meanStepRMS: TrainingStabilityMath.finiteOrZero(meanStepRMS),
            logitStats: logitStats?.sanitized()
        )
        copy.topLayerRatios = topLayerRatios.map { $0.sanitized() }
        return copy
    }

    private static func higherRatioFirst(_ lhs: LayerUpdateRatio, _ rhs: LayerUpdateRatio) -> Bool {
        if lhs.ratio.isNaN || rhs.ratio.isNaN {
            return lhs.ratio.isNaN != rhs.ratio.isNaN ? lhs.ratio.isNaN : lhs.layerName < rhs.layerName
        }
        return lhs.ratio != rhs.ratio ? lhs.ratio > rhs.ratio : lhs.layerName < rhs.layerName
    }
}

public struct TrainingStabilitySummary: Sendable, Codable, Equatable {
    public var sampledSteps: Int = 0
    public var maxGradNorm: Double = 0
    public var meanGradNorm: Double = 0
    public var maxUpdateToWeightRatio: Double = 0
    public var meanUpdateToWeightRatio: Double = 0
    public var meanStepRMS: Double = 0
    public var maxZLoss: Double?
    public var maxLogit: Double?
    public var nonFiniteStepCount: Int = 0
    public var grpo: GRPOStabilityMetrics?
    public var recentSteps: [TrainingStepTelemetry] = []

    public init() {}

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sampledSteps = try c.decodeOrDefault(.sampledSteps, default: 0)
        maxGradNorm = try c.decodeOrDefault(.maxGradNorm, default: 0.0)
        meanGradNorm = try c.decodeOrDefault(.meanGradNorm, default: 0.0)
        maxUpdateToWeightRatio = try c.decodeOrDefault(.maxUpdateToWeightRatio, default: 0.0)
        meanUpdateToWeightRatio = try c.decodeOrDefault(.meanUpdateToWeightRatio, default: 0.0)
        meanStepRMS = try c.decodeOrDefault(.meanStepRMS, default: 0.0)
        maxZLoss = try c.decodeIfPresent(Double.self, forKey: .maxZLoss)
        maxLogit = try c.decodeIfPresent(Double.self, forKey: .maxLogit)
        nonFiniteStepCount = try c.decodeOrDefault(.nonFiniteStepCount, default: 0)
        grpo = try c.decodeIfPresent(GRPOStabilityMetrics.self, forKey: .grpo)
        recentSteps = try c.decodeOrDefault(.recentSteps, default: [TrainingStepTelemetry]())
    }

    public func formattedSummaryLine() -> String {
        var parts = [
            "stability: steps=\(sampledSteps)",
            String(format: "max ||ΔW||/||W||=%.2e", maxUpdateToWeightRatio),
            String(format: "gradNorm=%.4f", maxGradNorm),
            String(format: "stepRMS=%.2e", meanStepRMS),
        ]
        if let maxZLoss { parts.append(String(format: "zLoss=%.2f", maxZLoss)) }
        if let maxLogit { parts.append(String(format: "maxLogit=%.2f", maxLogit)) }
        if let grpo {
            parts.append(String(
                format: "grpo zeroAdv=%.1f%% clip=%.1f%% KL=%.4f",
                grpo.zeroAdvantageGroupRate * 100.0, grpo.clipFraction * 100.0, grpo.meanKL
            ))
        }
        if nonFiniteStepCount > 0 { parts.append("nonFinite=\(nonFiniteStepCount)") }
        return parts.joined(separator: ", ")
    }
}

public enum TrainingStabilityMath: Sendable {
    public static let defaultEps: Double = 1e-12

    public static func updateToWeightRatio(
        before: [Double],
        after: [Double],
        eps: Double = defaultEps
    ) -> (updateNorm: Double, weightNorm: Double, ratio: Double, stepRMS: Double) {
        guard !before.isEmpty, before.count == after.count else { return (0.0, 0.0, 0.0, 0.0) }
        var (sumDeltaSq, sumWeightSq) = (0.0, 0.0)
        for index in before.indices {
            let b = before[index]
            let d = after[index] - b
            sumDeltaSq += d * d
            sumWeightSq += b * b
        }
        let updateNorm = sumDeltaSq.squareRoot()
        let weightNorm = sumWeightSq.squareRoot()
        let safeEps = eps.isFinite && eps > 0 ? eps : defaultEps
        return (updateNorm, weightNorm, updateNorm / (weightNorm + safeEps), (sumDeltaSq / Double(before.count)).squareRoot())
    }

    public static func gradNorm(tensors: [[Double]]) -> Double {
        var sumSq = 0.0
        for tensor in tensors {
            for value in tensor { sumSq += value * value }
        }
        return sumSq.squareRoot()
    }

    public static func logitZLossStats(logitsRows: [[Double]]) -> LogitZLossStats {
        var (zLossSum, maxLogit, count) = (0.0, -Double.infinity, 0)
        for row in logitsRows where !row.isEmpty {
            let stats = rowLogSumExpAndMax(row)
            zLossSum += stats.lse * stats.lse
            maxLogit = propagateMax(maxLogit, stats.maxVal)
            count += 1
        }
        return count > 0 ? LogitZLossStats(zLoss: zLossSum / Double(count), maxLogit: maxLogit) : LogitZLossStats(zLoss: 0.0, maxLogit: 0.0)
    }

    public static func grpoClippedTokenCount(
        logPolicy: [Double],
        logOld: [Double],
        advantage: Double,
        clipEpsilon: Double
    ) -> Int {
        guard !logPolicy.isEmpty, logPolicy.count == logOld.count,
              advantage.isFinite, advantage != 0.0,
              clipEpsilon.isFinite, clipEpsilon > 0.0 else {
            return 0
        }
        let (upper, lower) = (1.0 + clipEpsilon, 1.0 - clipEpsilon)
        var clipped = 0
        for index in logPolicy.indices {
            let ratio = Foundation.exp(min(max(logPolicy[index] - logOld[index], -20.0), 20.0))
            if (advantage > 0.0 && ratio > upper) || (advantage < 0.0 && ratio < lower) { clipped += 1 }
        }
        return clipped
    }

    public static func grpoMetrics(
        updatesApplied: Int,
        skippedGroups: Int,
        clippedTokenCount: Int,
        totalTokenCount: Int,
        meanKL: Double,
        rewards: [Double]
    ) -> GRPOStabilityMetrics {
        let (applied, skipped) = (max(0, updatesApplied), max(0, skippedGroups))
        let totalGroups = applied + skipped
        let totalTokens = max(0, totalTokenCount)
        let clippedTokens = min(max(0, clippedTokenCount), totalTokens)
        return GRPOStabilityMetrics(
            totalGroups: totalGroups,
            skippedZeroAdvantageGroups: skipped,
            zeroAdvantageGroupRate: totalGroups > 0 ? Double(skipped) / Double(totalGroups) : 0.0,
            clipFraction: totalTokens > 0 ? Double(clippedTokens) / Double(totalTokens) : 0.0,
            meanKL: meanKL.isFinite ? max(0.0, meanKL) : 0.0,
            rewardVariance: populationVariance(rewards)
        )
    }

    static func propagateMax(_ lhs: Double, _ rhs: Double) -> Double {
        (lhs.isNaN || rhs.isNaN) ? .nan : Swift.max(lhs, rhs)
    }

    static func finiteOrZero(_ value: Double) -> Double {
        value.isFinite ? value : 0.0
    }

    private static func rowLogSumExpAndMax(_ row: [Double]) -> (lse: Double, maxVal: Double) {
        var rowMax = -Double.infinity
        for value in row {
            if value.isNaN { return (.nan, .nan) }
            if value > rowMax { rowMax = value }
        }
        if rowMax == .infinity { return (.infinity, .infinity) }
        if rowMax == -.infinity { return (-.infinity, -.infinity) }
        var sumExp = 0.0
        for value in row { sumExp += Foundation.exp(value - rowMax) }
        return (rowMax + Foundation.log(sumExp), rowMax)
    }

    private static func populationVariance(_ values: [Double]) -> Double {
        let finite = values.filter(\.isFinite)
        guard finite.count > 1 else { return 0.0 }
        let count = Double(finite.count)
        let mean = finite.reduce(0.0, +) / count
        return finite.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / count
    }
}

public struct TrainingStabilityAccumulator: Sendable, Codable, Equatable {
    public static let defaultMaxRecentSteps = 32

    public let maxRecentSteps: Int
    public private(set) var sampledSteps: Int = 0
    public private(set) var nonFiniteStepCount: Int = 0
    private var gradStats = RunningStat()
    private var meanRatioStats = RunningStat()
    private var maxRatioStats = RunningStat()
    private var rmsStats = RunningStat()
    private var maxZLoss: Double?
    private var maxLogit: Double?
    public private(set) var grpo: GRPOStabilityMetrics?
    public private(set) var recentSteps: [TrainingStepTelemetry] = []

    public init(maxRecentSteps: Int = defaultMaxRecentSteps) {
        self.maxRecentSteps = max(0, maxRecentSteps)
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        maxRecentSteps = max(0, try c.decodeOrDefault(.maxRecentSteps, default: Self.defaultMaxRecentSteps))
        sampledSteps = max(0, try c.decodeOrDefault(.sampledSteps, default: 0))
        nonFiniteStepCount = max(0, try c.decodeOrDefault(.nonFiniteStepCount, default: 0))
        gradStats = try c.decodeOrDefault(.gradStats, default: RunningStat())
        meanRatioStats = try c.decodeOrDefault(.meanRatioStats, default: RunningStat())
        maxRatioStats = try c.decodeOrDefault(.maxRatioStats, default: RunningStat())
        rmsStats = try c.decodeOrDefault(.rmsStats, default: RunningStat())
        maxZLoss = try c.decodeIfPresent(Double.self, forKey: .maxZLoss)
        maxLogit = try c.decodeIfPresent(Double.self, forKey: .maxLogit)
        grpo = try c.decodeIfPresent(GRPOStabilityMetrics.self, forKey: .grpo)
        recentSteps = try c.decodeOrDefault(.recentSteps, default: [TrainingStepTelemetry]())
    }

    public mutating func record(_ step: TrainingStepTelemetry) {
        sampledSteps += 1
        if !step.isFinite { nonFiniteStepCount += 1 }
        gradStats.record(step.gradNorm)
        meanRatioStats.record(step.meanUpdateToWeightRatio)
        maxRatioStats.record(step.maxUpdateToWeightRatio)
        rmsStats.record(step.meanStepRMS)
        recordLogitStats(step.logitStats)
        appendRecent(step)
    }

    public mutating func setGRPO(_ metrics: GRPOStabilityMetrics) {
        grpo = metrics.isFinite ? metrics : metrics.sanitized()
    }

    public func finalize() -> TrainingStabilitySummary {
        var summary = TrainingStabilitySummary()
        (summary.sampledSteps, summary.nonFiniteStepCount) = (sampledSteps, nonFiniteStepCount)
        (summary.maxGradNorm, summary.meanGradNorm) = (gradStats.peak, gradStats.mean)
        (summary.maxUpdateToWeightRatio, summary.meanUpdateToWeightRatio) = (maxRatioStats.peak, meanRatioStats.mean)
        (summary.meanStepRMS, summary.maxZLoss, summary.maxLogit) = (rmsStats.mean, maxZLoss, maxLogit)
        (summary.grpo, summary.recentSteps) = (grpo, recentSteps)
        return summary
    }

    private mutating func recordLogitStats(_ stats: LogitZLossStats?) {
        guard let stats else { return }
        if stats.zLoss.isFinite { maxZLoss = maxZLoss.map { Swift.max($0, stats.zLoss) } ?? stats.zLoss }
        if stats.maxLogit.isFinite { maxLogit = maxLogit.map { Swift.max($0, stats.maxLogit) } ?? stats.maxLogit }
    }

    private mutating func appendRecent(_ step: TrainingStepTelemetry) {
        guard maxRecentSteps > 0 else { return }
        if recentSteps.count >= maxRecentSteps { recentSteps.removeFirst(recentSteps.count - maxRecentSteps + 1) }
        recentSteps.append(step.isFinite ? step : step.sanitized())
    }
}

private struct RunningStat: Sendable, Codable, Equatable {
    var count: Int = 0
    var sum: Double = 0
    var peak: Double = 0

    var mean: Double { count > 0 ? sum / Double(count) : 0.0 }

    init() {}

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        count = max(0, try c.decodeOrDefault(.count, default: 0))
        sum = try c.decodeOrDefault(.sum, default: 0.0)
        peak = try c.decodeOrDefault(.peak, default: 0.0)
    }

    mutating func record(_ value: Double) {
        guard value.isFinite else { return }
        count += 1
        sum += value
        peak = count == 1 ? value : Swift.max(peak, value)
    }
}

private extension KeyedDecodingContainer {
    func decodeOrDefault<T: Decodable>(_ key: Key, default value: T) throws -> T {
        try decodeIfPresent(T.self, forKey: key) ?? value
    }
}
