import Foundation

public struct MoELoadBalanceMetrics: Sendable, Codable, Equatable {
    public var expertCount: Int = 0
    public var totalDispatches: Int = 0
    public var routerEntropy: Double = 0.0
    public var normalizedRouterEntropy: Double = 0.0
    public var cvSquared: Double = 0.0
    public var gini: Double = 0.0
    public var maxToMeanRatio: Double = 0.0
    public var deadExpertCount: Int = 0
    public var deadExpertFraction: Double = 0.0
    public var collapseAlert: Bool = false

    public init() {}

    enum CodingKeys: String, CodingKey {
        case expertCount = "expert_count"
        case totalDispatches = "total_dispatches"
        case routerEntropy = "router_entropy"
        case normalizedRouterEntropy = "normalized_router_entropy"
        case cvSquared = "cv_squared"
        case gini
        case maxToMeanRatio = "max_to_mean_ratio"
        case deadExpertCount = "dead_expert_count"
        case deadExpertFraction = "dead_expert_fraction"
        case collapseAlert = "collapse_alert"
    }
}

public enum MoELoadBalanceMath: Sendable {
    public static func evaluate(
        counts: [Int],
        collapseEntropyThreshold: Double = 0.5,
        collapseDeadFractionThreshold: Double = 0.25
    ) -> MoELoadBalanceMetrics {
        let clamped = counts.map { max(0, $0) }
        let expertCount = clamped.count
        var metrics = MoELoadBalanceMetrics()
        metrics.expertCount = expertCount
        guard expertCount > 0 else { return metrics }

        let totalDispatches = clamped.reduce(0, +)
        let deadExpertCount = clamped.reduce(0) { $0 + ($1 == 0 ? 1 : 0) }
        let expertCountDouble = Double(expertCount)
        metrics.totalDispatches = totalDispatches
        metrics.deadExpertCount = deadExpertCount
        metrics.deadExpertFraction = Double(deadExpertCount) / expertCountDouble
        guard totalDispatches > 0 else { return metrics }

        let totalDouble = Double(totalDispatches)
        let mean = totalDouble / expertCountDouble
        let maxCount = clamped.max() ?? 0
        let rawEntropy = shannonEntropy(clamped, total: totalDouble)
        let normalizedEntropy = expertCount > 1
            ? clampUnit(rawEntropy / Foundation.log(expertCountDouble))
            : 1.0

        metrics.routerEntropy = rawEntropy
        metrics.normalizedRouterEntropy = normalizedEntropy
        metrics.cvSquared = squaredCoefficientOfVariation(clamped, mean: mean)
        metrics.gini = giniCoefficient(clamped, total: totalDouble)
        metrics.maxToMeanRatio = finiteNonNegative(Double(maxCount) / mean)
        metrics.collapseAlert = isCollapseAlert(
            metrics,
            entropyThreshold: collapseEntropyThreshold,
            deadFractionThreshold: collapseDeadFractionThreshold
        )
        return metrics
    }

    public static func evaluate(
        dispatchMap: [Int: Int],
        expertCount: Int,
        collapseEntropyThreshold: Double = 0.5,
        collapseDeadFractionThreshold: Double = 0.25
    ) -> MoELoadBalanceMetrics {
        guard expertCount > 0 else {
            return evaluate(
                counts: [],
                collapseEntropyThreshold: collapseEntropyThreshold,
                collapseDeadFractionThreshold: collapseDeadFractionThreshold
            )
        }
        var dense = [Int](repeating: 0, count: expertCount)
        for (expert, count) in dispatchMap where expert >= 0 && expert < expertCount && count > 0 {
            dense[expert] = count
        }
        return evaluate(
            counts: dense,
            collapseEntropyThreshold: collapseEntropyThreshold,
            collapseDeadFractionThreshold: collapseDeadFractionThreshold
        )
    }

    private static func shannonEntropy(_ counts: [Int], total: Double) -> Double {
        var entropy = 0.0
        for count in counts where count > 0 {
            let probability = Double(count) / total
            entropy -= probability * Foundation.log(probability)
        }
        return finiteNonNegative(entropy)
    }

    private static func squaredCoefficientOfVariation(_ counts: [Int], mean: Double) -> Double {
        let varianceSum = counts.reduce(0.0) { acc, count in
            let delta = Double(count) - mean
            return acc + delta * delta
        }
        let denominator = Double(counts.count) * mean * mean
        guard denominator > 0.0 else { return 0.0 }
        return finiteNonNegative(varianceSum / denominator)
    }

    private static func giniCoefficient(_ counts: [Int], total: Double) -> Double {
        let sorted = counts.sorted()
        let countDouble = Double(sorted.count)
        var weightedSum = 0.0
        for (offset, count) in sorted.enumerated() {
            weightedSum += Double(offset + 1) * Double(count)
        }
        let raw = (2.0 * weightedSum) / (countDouble * total) - (countDouble + 1.0) / countDouble
        return clampUnit(raw)
    }

    private static func isCollapseAlert(
        _ metrics: MoELoadBalanceMetrics,
        entropyThreshold: Double,
        deadFractionThreshold: Double
    ) -> Bool {
        guard metrics.totalDispatches > 0, metrics.expertCount > 1 else { return false }
        return metrics.normalizedRouterEntropy < entropyThreshold
            || metrics.deadExpertFraction >= deadFractionThreshold
            || metrics.maxToMeanRatio >= Double(metrics.expertCount) * 0.5
    }

    private static func clampUnit(_ value: Double) -> Double {
        guard value.isFinite else { return 0.0 }
        return min(1.0, max(0.0, value))
    }

    private static func finiteNonNegative(_ value: Double) -> Double {
        value.isFinite && value > 0.0 ? value : 0.0
    }
}
