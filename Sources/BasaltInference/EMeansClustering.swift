import BasaltCore
import MLX

public extension EMeansPrecision {
    var mlxDType: DType {
        switch self {
        case .float32: .float32
        case .bfloat16: .bfloat16
        case .float16: .float16
        }
    }
}

public struct EMeansResult {
    public let centroids: MLXArray
    public let counts: MLXArray
    public let metrics: EMeansMetrics

    public init(centroids: MLXArray, counts: MLXArray, metrics: EMeansMetrics) {
        self.centroids = centroids
        self.counts = counts
        self.metrics = metrics
    }
}

struct EMeansLoopState {
    var centroids: MLXArray
    var emaCounts: MLXArray
    var emaSums: MLXArray
    var slowCounts: MLXArray?
    var slowSums: MLXArray?
    var finalCounts: MLXArray?
    var finalSums: MLXArray?
    var revivals: Int = 0

    func evalAll() {
        var arrays = [centroids, emaCounts, emaSums]
        if let slowCounts, let slowSums {
            arrays.append(slowCounts)
            arrays.append(slowSums)
        }
        if let finalCounts, let finalSums {
            arrays.append(finalCounts)
            arrays.append(finalSums)
        }
        eval(arrays)
    }
}

struct EMeansSchedule {
    let batchSize: Int
    let batchesPerEpoch: Int
    let totalSteps: Int
    let peakAlpha: Float

    var finalEpochStart: Int {
        max(0, totalSteps - batchesPerEpoch)
    }
}

struct EMeansSplitUpdate {
    let allIdx: MLXArray
    let newCentroids: MLXArray
    let newCounts: MLXArray
}

public enum EMeans {
    public static func nearestIndices(
        points: MLXArray,
        centroids: MLXArray,
        precision: EMeansPrecision = .float32
    ) -> MLXArray {
        let dtype = precision.mlxDType
        let x = points.asType(dtype)
        let c = centroids.asType(dtype)
        let cNormSq = centroids.asType(.float32).square().sum(axis: -1, keepDims: true).swappedAxes(-1, -2)
        let dots = matmul(x, c.swappedAxes(-1, -2)).asType(.float32)
        return argMin(cNormSq - Float(2) * dots, axis: -1).asType(.int32)
    }

    public static func meanSquaredError(
        points: MLXArray,
        centroids: MLXArray,
        precision: EMeansPrecision = .float32
    ) -> Float {
        evaluateFinal(points: points, centroids: centroids, precision: precision).mse
    }

    public static func fit(_ points: MLXArray, config: EMeansConfig) throws(EMeansError) -> EMeansResult {
        guard points.ndim == 2 else {
            throw .invalidRank(expected: 2, actual: points.ndim)
        }
        let n = points.dim(0)
        let d = points.dim(1)
        try config.validate(points: n, dimensions: d)
        let batchSize = config.resolvedBatchSize(points: n)
        let schedule = EMeansSchedule(
            batchSize: batchSize,
            batchesPerEpoch: EMeansScalingLaw.batchesPerEpoch(points: n, batchSize: batchSize),
            totalSteps: EMeansScalingLaw.totalSteps(points: n, batchSize: batchSize, epochs: config.epochs),
            peakAlpha: config.resolvedPeakStep(points: n, batchSize: batchSize)
        )
        var rng = SplitMix64(seed: config.seed)
        let xF32 = points.asType(.float32)
        var state = makeInitialState(points: xF32, config: config, rng: &rng)
        runEpochs(points: xF32, state: &state, config: config, schedule: schedule, rng: &rng)
        let finalCentroids = finalizedCentroids(state: state, finalizer: config.finalizer, countAxisExpanded: false)
        let summary = evaluateFinal(points: xF32, centroids: finalCentroids, precision: config.assignmentDType)
        eval(finalCentroids, summary.counts)
        let metrics = EMeansMetrics(
            trainMSE: summary.mse,
            deadClusters: summary.deadClusters,
            revivals: state.revivals,
            steps: schedule.totalSteps,
            batchSize: schedule.batchSize,
            peakStep: schedule.peakAlpha
        )
        return EMeansResult(centroids: finalCentroids, counts: summary.counts, metrics: metrics)
    }

    public static func reviveDeadClusters(
        centroids: inout MLXArray,
        counts: inout MLXArray,
        sums: inout MLXArray,
        threshold: Float,
        delta: Float = EMeansConfig.defaultRevivalDelta,
        rng: inout SplitMix64
    ) -> Int {
        eval(centroids, counts, sums)
        let hostCounts = counts.asArray(Float.self)
        let pairs = EMeansCore.planRevivals(counts: hostCounts, threshold: threshold, rng: &rng)
        guard !pairs.isEmpty else { return 0 }
        let signs = alternatingSigns(dim: centroids.dim(1))
        let split = splitPairs(centroids: centroids, counts: counts, pairs: pairs, signs: signs, delta: delta)
        centroids = scatterReplace(centroids, at: split.allIdx, with: split.newCentroids)
        counts = scatterReplace(counts, at: split.allIdx, with: split.newCounts)
        let newSums = split.newCentroids * expandedDimensions(split.newCounts, axis: -1)
        sums = scatterReplace(sums, at: split.allIdx, with: newSums)
        eval(centroids, counts, sums)
        return pairs.count
    }
}

extension EMeans {
    static func scatterReplace(_ target: MLXArray, at indices: MLXArray, with values: MLXArray) -> MLXArray {
        let copy = target.reshaped(target.shape)
        copy[indices] = values
        return copy
    }

    static func alternatingSigns(dim: Int) -> MLXArray {
        MLXArray((0 ..< dim).map { $0.isMultiple(of: 2) ? Float(1) : Float(-1) }).reshaped(1, dim)
    }

    static func halvedCounts(counts: MLXArray, pairs: [EMeansRevivalPair]) -> MLXArray {
        let deadIdx = MLXArray(pairs.map { Int32($0.dead) })
        let donorIdx = MLXArray(pairs.map { Int32($0.donor) })
        let half = (counts.take(donorIdx, axis: 0) + counts.take(deadIdx, axis: 0)) * Float(0.5)
        return concatenated([half, half], axis: 0)
    }

    static func splitPairs(
        centroids: MLXArray,
        counts: MLXArray,
        pairs: [EMeansRevivalPair],
        signs: MLXArray,
        delta: Float
    ) -> EMeansSplitUpdate {
        let deadIdx = MLXArray(pairs.map { Int32($0.dead) })
        let donorIdx = MLXArray(pairs.map { Int32($0.donor) })
        let donorCentroids = centroids.take(donorIdx, axis: 0)
        let plusScale = MLXArray(Float(1)) + MLXArray(delta) * signs
        let minusScale = MLXArray(Float(1)) - MLXArray(delta) * signs
        let nonZero = donorCentroids.square().sum(axis: -1, keepDims: true) .> MLXArray(Float(1e-12))
        let donorChild = which(nonZero, donorCentroids * plusScale, MLXArray(delta) * signs)
        let deadChild = which(nonZero, donorCentroids * minusScale, MLXArray(-delta) * signs)
        let allIdx = concatenated([donorIdx, deadIdx], axis: 0)
        let newCentroids = concatenated([donorChild, deadChild], axis: 0)
        let newCounts = halvedCounts(counts: counts, pairs: pairs)
        return EMeansSplitUpdate(allIdx: allIdx, newCentroids: newCentroids, newCounts: newCounts)
    }

    static func shouldRevive(step: Int, totalSteps: Int, config: EMeansConfig) -> Bool {
        guard config.revivalInterval > 0 && config.revivalThreshold > 0 else { return false }
        guard (step + 1).isMultiple(of: config.revivalInterval) else { return false }
        return step + 1 < totalSteps || totalSteps <= config.revivalInterval
    }

    static func updateFinalizers(
        state: inout EMeansLoopState,
        batchCounts: MLXArray,
        batchSums: MLXArray,
        step: Int,
        config: EMeansConfig,
        schedule: EMeansSchedule
    ) {
        switch config.finalizer {
        case .none:
            break
        case .lastEpochMean:
            guard step >= schedule.finalEpochStart, let prevN = state.finalCounts, let prevS = state.finalSums else {
                return
            }
            state.finalCounts = prevN + batchCounts
            state.finalSums = prevS + batchSums
        case .slowEMA(let speed):
            guard let prevN = state.slowCounts, let prevS = state.slowSums else { return }
            let slowAlpha = config.resolvedSlowAlpha(peakStep: schedule.peakAlpha, speedMultiplier: speed)
            let slowGamma = Float(1) - slowAlpha
            state.slowCounts = prevN * slowGamma + batchCounts * slowAlpha
            state.slowSums = prevS * slowGamma + batchSums * slowAlpha
        }
    }

    static func finalizedCentroids(
        state: EMeansLoopState,
        finalizer: EMeansFinalizer,
        countAxisExpanded: Bool
    ) -> MLXArray {
        switch finalizer {
        case .none:
            return state.centroids
        case .lastEpochMean:
            guard let finalCounts = state.finalCounts, let finalSums = state.finalSums else {
                return state.centroids
            }
            let denom = countAxisExpanded ? finalCounts : expandedDimensions(finalCounts, axis: -1)
            let safeDenom = maximum(denom, MLXArray(Float(1)))
            return which(denom .> MLXArray(Float(0)), finalSums / safeDenom, state.centroids)
        case .slowEMA:
            guard let slowCounts = state.slowCounts, let slowSums = state.slowSums else {
                return state.centroids
            }
            let denom = countAxisExpanded ? slowCounts : expandedDimensions(slowCounts, axis: -1)
            let safeDenom = maximum(denom, MLXArray(Float(1e-20)))
            return which(denom .> MLXArray(Float(1e-20)), slowSums / safeDenom, state.centroids)
        }
    }
}

private extension EMeans {
    static func makeInitialState(
        points: MLXArray,
        config: EMeansConfig,
        rng: inout SplitMix64
    ) -> EMeansLoopState {
        let k = config.clusters
        let d = points.dim(1)
        let initIdx = EMeansCore.sampleUniqueIndices(count: points.dim(0), k: k, rng: &rng)
        let centroids = points.take(MLXArray(initIdx), axis: 0)
        let emaCounts = MLXArray.ones([k], dtype: .float32) * Float(1e-10)
        let emaSums = centroids * expandedDimensions(emaCounts, axis: -1)
        var state = EMeansLoopState(centroids: centroids, emaCounts: emaCounts, emaSums: emaSums)
        switch config.finalizer {
        case .none:
            break
        case .lastEpochMean:
            state.finalCounts = MLXArray.zeros([k], dtype: .float32)
            state.finalSums = MLXArray.zeros([k, d], dtype: .float32)
        case .slowEMA:
            let slowN = MLXArray.ones([k], dtype: .float32) * Float(1e-10)
            state.slowCounts = slowN
            state.slowSums = centroids * expandedDimensions(slowN, axis: -1)
        }
        state.evalAll()
        return state
    }

    static func runEpochs(
        points: MLXArray,
        state: inout EMeansLoopState,
        config: EMeansConfig,
        schedule: EMeansSchedule,
        rng: inout SplitMix64
    ) {
        let n = points.dim(0)
        let k = config.clusters
        let clusterIds = k <= 64 ? MLXArray((0 ..< k).map(Int32.init)).reshaped(k, 1) : nil
        let signs = alternatingSigns(dim: points.dim(1))
        var permutation: [Int32] = []
        for step in 0 ..< schedule.totalSteps {
            let batchInEpoch = step % schedule.batchesPerEpoch
            if batchInEpoch == 0 {
                permutation = EMeansCore.shuffledIndices(count: n, rng: &rng)
            }
            let start = batchInEpoch * schedule.batchSize
            let end = min(n, start + schedule.batchSize)
            let batch = points.take(MLXArray(Array(permutation[start ..< end])), axis: 0)
            stepBatch2D(
                batch: batch, state: &state, step: step,
                config: config, schedule: schedule, clusterIds: clusterIds)
            state.evalAll()
            if shouldRevive(step: step, totalSteps: schedule.totalSteps, config: config) {
                reviveLoopState(state: &state, config: config, signs: signs, rng: &rng)
                state.evalAll()
            }
        }
    }

    static func stepBatch2D(
        batch: MLXArray,
        state: inout EMeansLoopState,
        step: Int,
        config: EMeansConfig,
        schedule: EMeansSchedule,
        clusterIds: MLXArray?
    ) {
        let assign = nearestIndices(points: batch, centroids: state.centroids, precision: config.assignmentDType)
        let (batchCounts, batchSums) = accumulateBatch2D(
            batch: batch, assign: assign, k: config.clusters, clusterIds: clusterIds)
        let alpha = EMeansScalingLaw.stepAlpha(
            step: step, totalSteps: schedule.totalSteps,
            peakAlpha: schedule.peakAlpha, floor: config.cosineFloor)
        let gamma = Float(1) - alpha
        state.emaCounts = state.emaCounts * gamma + batchCounts * alpha
        state.emaSums = state.emaSums * gamma + batchSums * alpha
        let expandedN = expandedDimensions(state.emaCounts, axis: -1)
        let safeN = maximum(expandedN, MLXArray(Float(1e-20)))
        state.centroids = which(expandedN .> MLXArray(Float(1e-20)), state.emaSums / safeN, state.centroids)
        updateFinalizers(
            state: &state, batchCounts: batchCounts, batchSums: batchSums,
            step: step, config: config, schedule: schedule)
    }

    static func accumulateBatch2D(
        batch: MLXArray,
        assign: MLXArray,
        k: Int,
        clusterIds: MLXArray?
    ) -> (counts: MLXArray, sums: MLXArray) {
        if let clusterIds {
            let mask = (expandedDimensions(assign, axis: 0) .== clusterIds).asType(.float32)
            return (mask.sum(axis: -1), matmul(mask, batch))
        }
        let counts = MLXArray.zeros([k], dtype: .float32).at[assign].add(MLXArray(Float(1)))
        let sums = MLXArray.zeros([k, batch.dim(1)], dtype: .float32).at[assign].add(batch)
        return (counts, sums)
    }
}

private extension EMeans {
    static func reviveLoopState(
        state: inout EMeansLoopState,
        config: EMeansConfig,
        signs: MLXArray,
        rng: inout SplitMix64
    ) {
        let hostCounts = state.emaCounts.asArray(Float.self)
        let pairs = EMeansCore.planRevivals(counts: hostCounts, threshold: config.revivalThreshold, rng: &rng)
        guard !pairs.isEmpty else { return }
        let split = splitPairs(
            centroids: state.centroids, counts: state.emaCounts,
            pairs: pairs, signs: signs, delta: config.revivalDelta)
        let newSums = split.newCentroids * expandedDimensions(split.newCounts, axis: -1)
        state.centroids = scatterReplace(state.centroids, at: split.allIdx, with: split.newCentroids)
        state.emaCounts = scatterReplace(state.emaCounts, at: split.allIdx, with: split.newCounts)
        state.emaSums = scatterReplace(state.emaSums, at: split.allIdx, with: newSums)
        if let slowN = state.slowCounts, let slowS = state.slowSums {
            let slowNew = halvedCounts(counts: slowN, pairs: pairs)
            let slowNewSums = split.newCentroids * expandedDimensions(slowNew, axis: -1)
            state.slowCounts = scatterReplace(slowN, at: split.allIdx, with: slowNew)
            state.slowSums = scatterReplace(slowS, at: split.allIdx, with: slowNewSums)
        }
        if let finalN = state.finalCounts, let finalS = state.finalSums {
            let count = split.allIdx.dim(0)
            state.finalCounts = scatterReplace(finalN, at: split.allIdx, with: MLXArray.zeros([count], dtype: .float32))
            let zeroSums = MLXArray.zeros([count, state.centroids.dim(1)], dtype: .float32)
            state.finalSums = scatterReplace(finalS, at: split.allIdx, with: zeroSums)
        }
        state.revivals += pairs.count
    }

    static func evaluateFinal(
        points: MLXArray,
        centroids: MLXArray,
        precision: EMeansPrecision
    ) -> (mse: Float, deadClusters: Int, counts: MLXArray) {
        let n = points.dim(0)
        let k = centroids.dim(0)
        let chunkSize = max(1, min(n, 4_194_304 / max(1, k)))
        var totalSqErr: Double = 0
        var counts = MLXArray.zeros([k], dtype: .float32)
        let xF32 = points.asType(.float32)
        let cF32 = centroids.asType(.float32)
        for start in stride(from: 0, to: n, by: chunkSize) {
            let end = min(n, start + chunkSize)
            let chunk = xF32[start ..< end]
            let idx = nearestIndices(points: chunk, centroids: cF32, precision: precision)
            let assigned = cF32.take(idx, axis: 0)
            let sqErr = (chunk - assigned).square().sum()
            counts = counts.at[idx].add(MLXArray(Float(1)))
            eval(sqErr, counts)
            totalSqErr += Double(sqErr.item(Float.self))
        }
        let hostCounts = counts.asArray(Float.self)
        let dead = hostCounts.reduce(0) { $0 + ($1 == 0 ? 1 : 0) }
        return (Float(totalSqErr / Double(max(1, n))), dead, counts)
    }
}
