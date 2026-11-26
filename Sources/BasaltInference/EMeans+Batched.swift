import BasaltCore
import MLX

extension EMeans {
    package static func fitBatched(_ keys: MLXArray, config: EMeansConfig) -> (centroids: MLXArray, counts: MLXArray) {
        precondition(keys.ndim == 4 && keys.dim(2) > 0 && keys.dim(3) > 0, "Batched e-means requires [B, H, T, D]")
        precondition(config.clusters >= 1 && config.epochs.isFinite && config.epochs > 0, "Invalid batched e-means config")
        let x = keys.asType(.float32)
        let t = x.dim(2)
        let k = config.clusters
        let batchSize = config.resolvedBatchSize(points: t)
        let schedule = EMeansSchedule(
            batchSize: batchSize,
            batchesPerEpoch: EMeansScalingLaw.batchesPerEpoch(points: t, batchSize: batchSize),
            totalSteps: EMeansScalingLaw.totalSteps(points: t, batchSize: batchSize, epochs: config.epochs),
            peakAlpha: config.resolvedPeakStep(points: t, batchSize: batchSize)
        )
        var rng = SplitMix64(seed: config.seed)
        var state = makeBatchedInitialState(keys: x, config: config, rng: &rng)
        let clusterIds = MLXArray((0 ..< k).map(Int32.init)).reshaped(1, 1, k, 1)
        let signs = alternatingSigns(dim: x.dim(3))
        var permutation: [Int32] = []
        let useLastEpoch = config.finalizer == .lastEpochMean
        for step in 0 ..< schedule.totalSteps {
            let batchInEpoch: Int
            if useLastEpoch && step >= schedule.finalEpochStart {
                batchInEpoch = step - schedule.finalEpochStart
                if batchInEpoch == 0 {
                    permutation = EMeansCore.shuffledIndices(count: t, rng: &rng)
                }
            } else {
                batchInEpoch = step % schedule.batchesPerEpoch
                if batchInEpoch == 0 {
                    permutation = EMeansCore.shuffledIndices(count: t, rng: &rng)
                }
            }
            let start = batchInEpoch * schedule.batchSize
            let end = min(t, start + schedule.batchSize)
            let batch = x.take(MLXArray(Array(permutation[start ..< end])), axis: 2)
            stepBatched(
                batch: batch, state: &state, step: step,
                config: config, schedule: schedule, clusterIds: clusterIds)
            state.evalAll()
            if shouldRevive(step: step, totalSteps: schedule.totalSteps, config: config) {
                reviveBatchedState(state: &state, config: config, signs: signs, rng: &rng)
                state.evalAll()
            }
        }
        let finalCentroids = finalizedCentroids(state: state, finalizer: config.finalizer, countAxisExpanded: true)
        let assign = nearestIndices(points: x, centroids: finalCentroids, precision: config.assignmentDType)
        let mask = (expandedDimensions(assign, axis: 2) .== clusterIds).asType(.float32)
        let finalCounts = mask.sum(axis: -1, keepDims: true)
        eval(finalCentroids, finalCounts)
        return (finalCentroids, finalCounts)
    }

    package static func reviveBatched(
        centroids: MLXArray,
        counts: MLXArray,
        sums: MLXArray?,
        threshold: Float,
        delta: Float = EMeansConfig.defaultRevivalDelta,
        rng: inout SplitMix64
    ) -> (centroids: MLXArray, counts: MLXArray, sums: MLXArray?, revived: Int) {
        let (b, h, k, d) = (centroids.dim(0), centroids.dim(1), centroids.dim(2), centroids.dim(3))
        let groups = b * h
        let pairs = collectBatchedRevivalPairs(
            counts: counts, groups: groups, clusters: k, threshold: threshold, rng: &rng)
        guard !pairs.isEmpty else { return (centroids, counts, sums, 0) }
        let flatC = centroids.reshaped(groups * k, d)
        let flatN = counts.reshaped(groups * k)
        let split = splitPairs(
            centroids: flatC, counts: flatN, pairs: pairs, signs: alternatingSigns(dim: d), delta: delta)
        let updatedC = scatterReplace(flatC, at: split.allIdx, with: split.newCentroids)
        let updatedN = scatterReplace(flatN, at: split.allIdx, with: split.newCounts)
        let newSums = split.newCentroids * expandedDimensions(split.newCounts, axis: -1)
        let outSums = sums.map {
            scatterReplace($0.reshaped(groups * k, d), at: split.allIdx, with: newSums).reshaped(b, h, k, d)
        }
        return (updatedC.reshaped(b, h, k, d), updatedN.reshaped(b, h, k, 1), outSums, pairs.count)
    }
}

private extension EMeans {
    static func makeBatchedInitialState(
        keys: MLXArray,
        config: EMeansConfig,
        rng: inout SplitMix64
    ) -> EMeansLoopState {
        let (b, h, t, d) = (keys.dim(0), keys.dim(1), keys.dim(2), keys.dim(3))
        let k = config.clusters
        var indices: [Int32] = []
        indices.reserveCapacity(b * h * k)
        for _ in 0 ..< b * h {
            if t >= k {
                indices.append(contentsOf: EMeansCore.sampleUniqueIndices(count: t, k: k, rng: &rng))
            } else {
                for slot in 0 ..< k {
                    let pick = slot < t ? slot : Int(rng.next() % UInt64(t))
                    indices.append(Int32(pick))
                }
            }
        }
        let centroids = takeAlong(keys, MLXArray(indices).reshaped(b, h, k, 1), axis: 2)
        let emaCounts = MLXArray.ones([b, h, k, 1], dtype: .float32) * Float(1e-10)
        let emaSums = centroids * emaCounts
        var state = EMeansLoopState(centroids: centroids, emaCounts: emaCounts, emaSums: emaSums)
        switch config.finalizer {
        case .none:
            break
        case .lastEpochMean:
            state.finalCounts = MLXArray.zeros([b, h, k, 1], dtype: .float32)
            state.finalSums = MLXArray.zeros([b, h, k, d], dtype: .float32)
        case .slowEMA:
            let slowN = MLXArray.ones([b, h, k, 1], dtype: .float32) * Float(1e-10)
            state.slowCounts = slowN
            state.slowSums = centroids * slowN
        }
        state.evalAll()
        return state
    }

    static func stepBatched(
        batch: MLXArray,
        state: inout EMeansLoopState,
        step: Int,
        config: EMeansConfig,
        schedule: EMeansSchedule,
        clusterIds: MLXArray
    ) {
        let assign = nearestIndices(points: batch, centroids: state.centroids, precision: config.assignmentDType)
        let mask = (expandedDimensions(assign, axis: 2) .== clusterIds).asType(.float32)
        let batchCounts = mask.sum(axis: -1, keepDims: true)
        let batchSums = matmul(mask, batch)
        let alpha = EMeansScalingLaw.stepAlpha(
            step: step, totalSteps: schedule.totalSteps,
            peakAlpha: schedule.peakAlpha, floor: config.cosineFloor)
        let gamma = Float(1) - alpha
        state.emaCounts = state.emaCounts * gamma + batchCounts * alpha
        state.emaSums = state.emaSums * gamma + batchSums * alpha
        let safeN = maximum(state.emaCounts, MLXArray(Float(1e-20)))
        state.centroids = which(state.emaCounts .> MLXArray(Float(1e-20)), state.emaSums / safeN, state.centroids)
        updateFinalizers(
            state: &state, batchCounts: batchCounts, batchSums: batchSums,
            step: step, config: config, schedule: schedule)
    }

    static func collectBatchedRevivalPairs(
        counts: MLXArray,
        groups: Int,
        clusters: Int,
        threshold: Float,
        rng: inout SplitMix64
    ) -> [EMeansRevivalPair] {
        let hostCounts = counts.reshaped(groups, clusters).asArray(Float.self)
        var pairs: [EMeansRevivalPair] = []
        for group in 0 ..< groups {
            let offset = group * clusters
            let slice = Array(hostCounts[offset ..< (offset + clusters)])
            for pair in EMeansCore.planRevivals(counts: slice, threshold: threshold, rng: &rng) {
                pairs.append(EMeansRevivalPair(dead: offset + pair.dead, donor: offset + pair.donor))
            }
        }
        return pairs
    }

    static func reviveBatchedState(
        state: inout EMeansLoopState,
        config: EMeansConfig,
        signs: MLXArray,
        rng: inout SplitMix64
    ) {
        let (b, h, k, d) = (
            state.centroids.dim(0), state.centroids.dim(1),
            state.centroids.dim(2), state.centroids.dim(3)
        )
        let groups = b * h
        let pairs = collectBatchedRevivalPairs(
            counts: state.emaCounts, groups: groups, clusters: k,
            threshold: config.revivalThreshold, rng: &rng)
        guard !pairs.isEmpty else { return }
        let flatC = state.centroids.reshaped(groups * k, d)
        let flatN = state.emaCounts.reshaped(groups * k)
        let flatS = state.emaSums.reshaped(groups * k, d)
        let split = splitPairs(centroids: flatC, counts: flatN, pairs: pairs, signs: signs, delta: config.revivalDelta)
        let newSums = split.newCentroids * expandedDimensions(split.newCounts, axis: -1)
        state.centroids = scatterReplace(flatC, at: split.allIdx, with: split.newCentroids).reshaped(b, h, k, d)
        state.emaCounts = scatterReplace(flatN, at: split.allIdx, with: split.newCounts).reshaped(b, h, k, 1)
        state.emaSums = scatterReplace(flatS, at: split.allIdx, with: newSums).reshaped(b, h, k, d)
        syncBatchedFinalizers(state: &state, pairs: pairs, split: split, shape: [b, h, k, d])
        state.revivals += pairs.count
    }

    static func syncBatchedFinalizers(
        state: inout EMeansLoopState,
        pairs: [EMeansRevivalPair],
        split: EMeansSplitUpdate,
        shape: [Int]
    ) {
        let (b, h, k, d) = (shape[0], shape[1], shape[2], shape[3])
        let total = b * h * k
        if let flatSlowN = state.slowCounts?.reshaped(total), let flatSlowS = state.slowSums?.reshaped(total, d) {
            let slowNew = halvedCounts(counts: flatSlowN, pairs: pairs)
            let slowNewSums = split.newCentroids * expandedDimensions(slowNew, axis: -1)
            state.slowCounts = scatterReplace(flatSlowN, at: split.allIdx, with: slowNew).reshaped(b, h, k, 1)
            state.slowSums = scatterReplace(flatSlowS, at: split.allIdx, with: slowNewSums).reshaped(b, h, k, d)
        }
        if let flatFinalN = state.finalCounts?.reshaped(total), let flatFinalS = state.finalSums?.reshaped(total, d) {
            let count = split.allIdx.dim(0)
            let zerosN = MLXArray.zeros([count], dtype: .float32)
            let zerosS = MLXArray.zeros([count, d], dtype: .float32)
            state.finalCounts = scatterReplace(flatFinalN, at: split.allIdx, with: zerosN).reshaped(b, h, k, 1)
            state.finalSums = scatterReplace(flatFinalS, at: split.allIdx, with: zerosS).reshaped(b, h, k, d)
        }
    }
}
