import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct ZeRO2AdamWConfig: Sendable, Codable, Equatable {
    public var learningRate: Float
    public var beta1: Float
    public var beta2: Float
    public var eps: Float
    public var weightDecay: Float
    public var emaDecay: Float?

    public init(
        learningRate: Float = 1e-3,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        eps: Float = 1e-8,
        weightDecay: Float = 0.01,
        emaDecay: Float? = nil
    ) {
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weightDecay = weightDecay
        self.emaDecay = emaDecay
    }
}

public struct CandidateRolloutWire: Sendable, Codable, Equatable {
    public let rank: Int
    public let promptID: String
    public let tokenIDs: [Int32]
    public let reward: Float
    public let sumLogProb: Float

    public init(rank: Int, promptID: String, tokenIDs: [Int32], reward: Float, sumLogProb: Float) {
        self.rank = rank
        self.promptID = promptID
        self.tokenIDs = tokenIDs
        self.reward = reward
        self.sumLogProb = sumLogProb
    }
}

public struct PreferenceRolloutPair: Sendable, Codable, Equatable {
    public let promptID: String
    public let chosenTokens: [Int32]
    public let rejectedTokens: [Int32]
    public let rewardMargin: Float
    public let normalizedAdvantages: [Float]
}

public struct ZeRO2RankOptimizerShard: Sendable {
    public let rank: Int
    public let ownedSlices: [ZeRO2ParameterSlice]
    public private(set) var stepCount: Int = 0
    private var firstMoment: [[Float]]
    private var secondMoment: [[Float]]
    private var emaShadow: [[Float]]?

    public init(rank: Int, plan: ZeRO2PartitionPlan, enableEMA: Bool = false) {
        let owned = plan.slices(ownedBy: rank)
        self.rank = rank
        self.ownedSlices = owned
        self.firstMoment = owned.map { [Float](repeating: 0, count: $0.elementCount) }
        self.secondMoment = owned.map { [Float](repeating: 0, count: $0.elementCount) }
        self.emaShadow = enableEMA ? [] : nil
    }

    public var momentElementCount: Int {
        firstMoment.reduce(0) { $0 + $1.count }
    }

    public mutating func stepOwnedSlices(
        currentParams: [String: [Float]],
        scatteredGrads: [[Float]],
        config: ZeRO2AdamWConfig
    ) -> [(slice: ZeRO2ParameterSlice, updated: [Float], ema: [Float]?)] {
        stepCount += 1
        let bc1 = 1.0 - Float(pow(Double(config.beta1), Double(stepCount)))
        let bc2 = 1.0 - Float(pow(Double(config.beta2), Double(stepCount)))
        if emaShadow != nil, emaShadow?.isEmpty == true {
            emaShadow = ownedSlices.map { slice in
                guard let full = currentParams[slice.paramKey] else {
                    return [Float](repeating: 0, count: slice.elementCount)
                }
                return Array(full[slice.elementStart ..< (slice.elementStart + slice.elementCount)])
            }
        }
        var results: [(ZeRO2ParameterSlice, [Float], [Float]?)] = []
        results.reserveCapacity(ownedSlices.count)
        for (sIdx, slice) in ownedSlices.enumerated() {
            guard let fullParam = currentParams[slice.paramKey], sIdx < scatteredGrads.count else { continue }
            let grad = scatteredGrads[sIdx]
            var updated = [Float](repeating: 0, count: slice.elementCount)
            var emaOut: [Float]? = emaShadow != nil ? [Float](repeating: 0, count: slice.elementCount) : nil
            for i in 0 ..< slice.elementCount {
                let g = grad[i]
                let p = fullParam[slice.elementStart + i]
                let m = config.beta1 * firstMoment[sIdx][i] + (1.0 - config.beta1) * g
                let v = config.beta2 * secondMoment[sIdx][i] + (1.0 - config.beta2) * g * g
                firstMoment[sIdx][i] = m
                secondMoment[sIdx][i] = v
                let mHat = m / max(1e-12, bc1)
                let vHat = v / max(1e-12, bc2)
                let decayed = p * (1.0 - config.learningRate * config.weightDecay)
                let nextP = decayed - config.learningRate * (mHat / (vHat.squareRoot() + config.eps))
                updated[i] = nextP
                if let mu = config.emaDecay, emaShadow != nil {
                    let prevEMA = emaShadow![sIdx][i]
                    let nextEMA = mu * prevEMA + (1.0 - mu) * nextP
                    emaShadow![sIdx][i] = nextEMA
                    emaOut?[i] = nextEMA
                }
            }
            results.append((slice, updated, emaOut))
        }
        return results
    }
}

public enum ZeRO2CollectiveCore {
    public static func reduceScatterGradients(
        rankGradients: [(microbatchCount: Int, grads: [String: [Float]])],
        plan: ZeRO2PartitionPlan
    ) -> [Int: [[Float]]] {
        let totalSamples = max(1, rankGradients.reduce(0) { $0 + max(0, $1.microbatchCount) })
        var out: [Int: [[Float]]] = [:]
        for r in 0 ..< plan.worldSize {
            let owned = plan.slices(ownedBy: r)
            var rankSlices: [[Float]] = []
            rankSlices.reserveCapacity(owned.count)
            for slice in owned {
                var accum = [Float](repeating: 0, count: slice.elementCount)
                for contrib in rankGradients where contrib.microbatchCount > 0 {
                    guard let gFull = contrib.grads[slice.paramKey] else { continue }
                    let weight = Float(contrib.microbatchCount) / Float(totalSamples)
                    for i in 0 ..< slice.elementCount {
                        accum[i] += weight * gFull[slice.elementStart + i]
                    }
                }
                rankSlices.append(accum)
            }
            out[r] = rankSlices
        }
        return out
    }

    public static func allGatherParameters(
        baseShapes: [(key: String, shape: [Int])],
        rankUpdates: [[(slice: ZeRO2ParameterSlice, updated: [Float], ema: [Float]?)]]
    ) -> (weights: [String: [Float]], emaWeights: [String: [Float]]?) {
        var assembled: [String: [Float]] = [:]
        var assembledEMA: [String: [Float]] = [:]
        var hasEMA = false
        for entry in baseShapes {
            let count = max(1, entry.shape.reduce(1, *))
            assembled[entry.key] = [Float](repeating: 0, count: count)
            assembledEMA[entry.key] = [Float](repeating: 0, count: count)
        }
        for updates in rankUpdates {
            for item in updates {
                let start = item.slice.elementStart
                for i in 0 ..< item.slice.elementCount {
                    assembled[item.slice.paramKey]?[start + i] = item.updated[i]
                    if let ema = item.ema {
                        hasEMA = true
                        assembledEMA[item.slice.paramKey]?[start + i] = ema[i]
                    }
                }
            }
        }
        return (assembled, hasEMA ? assembledEMA : nil)
    }

    public static func collateRolloutGroup(
        _ rollouts: [CandidateRolloutWire]
    ) -> [PreferenceRolloutPair] {
        let grouped = Dictionary(grouping: rollouts, by: \.promptID)
        var pairs: [PreferenceRolloutPair] = []
        for (promptID, items) in grouped.sorted(by: { $0.key < $1.key }) where items.count >= 2 {
            let rewards = items.map(\.reward)
            let meanR = rewards.reduce(0, +) / Float(rewards.count)
            let variance = rewards.reduce(Float(0)) { $0 + ($1 - meanR) * ($1 - meanR) } / Float(rewards.count)
            let stdR = max(1e-6, variance.squareRoot())
            let advantages = rewards.map { ($0 - meanR) / stdR }
            let sortedByReward = items.sorted {
                if abs($0.reward - $1.reward) > 1e-7 { return $0.reward > $1.reward }
                return $0.rank < $1.rank
            }
            guard let best = sortedByReward.first, let worst = sortedByReward.last else { continue }
            pairs.append(PreferenceRolloutPair(
                promptID: promptID,
                chosenTokens: best.tokenIDs,
                rejectedTokens: worst.tokenIDs,
                rewardMargin: best.reward - worst.reward,
                normalizedAdvantages: advantages
            ))
        }
        return pairs
    }
}

public actor DistributedZeRO2CoordinatorActor {
    public let plan: ZeRO2PartitionPlan
    public let config: ZeRO2AdamWConfig
    private var shards: [ZeRO2RankOptimizerShard]
    private var currentParams: [String: [Float]]
    private let paramShapes: [(key: String, shape: [Int])]
    private var epochToken: UInt64 = 0

    public init(
        initialParameters: [(key: String, shape: [Int], values: [Float])],
        worldSize: Int,
        config: ZeRO2AdamWConfig = ZeRO2AdamWConfig()
    ) {
        let shapes = initialParameters.map { ($0.key, $0.shape) }
        let builtPlan = ZeRO2PartitionPlan.build(parameterShapes: shapes, worldSize: worldSize)
        self.plan = builtPlan
        self.config = config
        self.paramShapes = shapes
        self.shards = (0 ..< builtPlan.worldSize).map {
            ZeRO2RankOptimizerShard(rank: $0, plan: builtPlan, enableEMA: config.emaDecay != nil)
        }
        self.currentParams = Dictionary(uniqueKeysWithValues: initialParameters.map { ($0.key, $0.values) })
    }

    public func perRankMomentElementCounts() -> [Int] {
        shards.map(\.momentElementCount)
    }

    public func currentParameterSnapshot() -> [String: [Float]] {
        currentParams
    }

    public func stepHeterogeneousBatch(
        rankGradients: [(microbatchCount: Int, grads: [String: [Float]])]
    ) -> (weights: [String: [Float]], emaWeights: [String: [Float]]?) {
        let scattered = ZeRO2CollectiveCore.reduceScatterGradients(rankGradients: rankGradients, plan: plan)
        var updatesPerRank: [[(slice: ZeRO2ParameterSlice, updated: [Float], ema: [Float]?)]] = []
        updatesPerRank.reserveCapacity(shards.count)
        for idx in shards.indices {
            let r = shards[idx].rank
            let ownedGrads = scattered[r] ?? []
            let updated = shards[idx].stepOwnedSlices(
                currentParams: currentParams,
                scatteredGrads: ownedGrads,
                config: config
            )
            updatesPerRank.append(updated)
        }
        let gathered = ZeRO2CollectiveCore.allGatherParameters(baseShapes: paramShapes, rankUpdates: updatesPerRank)
        currentParams = gathered.weights
        epochToken &+= 1
        return gathered
    }

    public func shutdown() {
        withTaskCancellationShield {
            epochToken &+= 1
            currentParams.removeAll()
            shards.removeAll()
        }
    }
}
