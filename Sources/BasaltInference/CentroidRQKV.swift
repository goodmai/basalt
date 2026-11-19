import BasaltCore
import MLX

struct CentroidCodebook {
    let clusters: Int
    private(set) var centroids: MLXArray
    private(set) var counts: MLXArray

    init(clusters: Int, centroids: MLXArray, counts: MLXArray) {
        precondition(KVPrecision.supportedCentroidClusters.contains(clusters), "RQ-KV supports 4, 8, or 16 centroids")
        self.clusters = clusters
        self.centroids = centroids.asType(.float32)
        self.counts = counts.asType(.float32)
    }

    static func nearestIndices(keys: MLXArray, centroids: MLXArray) -> MLXArray {
        let cNormSq = centroids.square().sum(axis: -1, keepDims: true).swappedAxes(-1, -2)
        let dots = matmul(keys, centroids.swappedAxes(-1, -2))
        return argMin(cNormSq - 2 * dots, axis: -1).asType(.int32)
    }

    func gather(_ assignments: MLXArray) -> MLXArray {
        takeAlong(centroids, expandedDimensions(assignments.asType(.int32), axis: -1), axis: 2)
    }

    static func fitKMeansPlusPlus(
        keys: MLXArray, clusters: Int = 8, iterations: Int = 4, seed: UInt64 = 0
    ) -> CentroidCodebook {
        precondition(KVPrecision.supportedCentroidClusters.contains(clusters), "RQ-KV supports 4, 8, or 16 centroids")
        precondition(keys.ndim == 4 && keys.dim(2) > 0 && keys.dim(3) > 0, "RQ-KV keys require shape [B, H, T, D]")
        let x = keys.asType(.float32)
        let initial = seedKMeansPlusPlus(keys: x, clusters: clusters, seed: seed)
        let (centers, memberCounts) = refineLloyd(keys: x, initial: initial, clusters: clusters, iterations: iterations)
        return CentroidCodebook(clusters: clusters, centroids: centers, counts: memberCounts)
    }

    mutating func updateOnline(_ keys: MLXArray, decay: Float = 1.0) {
        precondition(decay.isFinite && decay > 0 && decay <= 1, "RQ-KV decay must lie in (0, 1]")
        guard keys.dim(2) > 0 else { return }
        let x = keys.asType(.float32)
        let assign = Self.nearestIndices(keys: x, centroids: centroids)
        let clusterIds = MLXArray((0 ..< clusters).map(Int32.init)).reshaped(1, 1, clusters, 1)
        let mask = (expandedDimensions(assign, axis: 2) .== clusterIds).asType(.float32)
        let batchCounts = mask.sum(axis: -1, keepDims: true)
        let batchSums = matmul(mask, x)
        let priorCounts = counts * decay
        let totalCounts = priorCounts + batchCounts
        let blended = (centroids * priorCounts + batchSums) / maximum(totalCounts, MLXArray(Float(1e-6)))
        let updated = batchCounts .> MLXArray(Float(0))
        centroids = which(updated, blended, centroids)
        counts = which(updated, totalCounts, maximum(priorCounts, MLXArray(Float(1e-3))))
        eval(centroids, counts)
    }

    private static func seedKMeansPlusPlus(keys x: MLXArray, clusters: Int, seed: UInt64) -> MLXArray {
        let (b, h, t) = (x.dim(0), x.dim(1), x.dim(2))
        let xNormSq = x.square().sum(axis: -1)
        var rng = SplitMix64(seed: seed)
        let firstIndices = (0 ..< b * h).map { _ in Int32(rng.next() % UInt64(t)) }
        var chosen: [MLXArray] = [takeAlong(x, MLXArray(firstIndices).reshaped(b, h, 1, 1), axis: 2)]
        var minDistSq = MLXArray.zeros([b, h, t], dtype: .float32)
        let ramp = MLXArray((0 ..< t).map { Int32(t - $0) }).reshaped(1, 1, t)
        for step in 1 ..< clusters {
            let prev = chosen[step - 1]
            let dots = matmul(x, prev.swappedAxes(-1, -2)).squeezed(axis: -1)
            let distSq = maximum(xNormSq + prev.square().sum(axis: -1) - 2 * dots, MLXArray(Float(0)))
            minDistSq = step == 1 ? distSq : minimum(minDistSq, distSq)
            let draws = (0 ..< b * h).map { _ in Float(rng.next() >> 40) / Float(1 << 24) }
            let thresholds = MLXArray(draws).reshaped(b, h, 1)
            let cdf = cumsum(minDistSq, axis: -1)
            let total = cdf[.ellipsis, (t - 1)...]
            let reached = (cdf .>= thresholds * total) .&& (minDistSq .> MLXArray(Float(0)))
            let fallback = broadcast(MLXArray(Int32(step % t)), to: [b, h])
            let picked = argMax(reached.asType(.int32) * ramp, axis: -1).asType(.int32)
            let nextIdx = which(total.squeezed(axis: -1) .> MLXArray(Float(1e-12)), picked, fallback)
            chosen.append(takeAlong(x, nextIdx.reshaped(b, h, 1, 1), axis: 2))
        }
        return concatenated(chosen, axis: 2)
    }

    private static func refineLloyd(
        keys x: MLXArray, initial: MLXArray, clusters: Int, iterations: Int
    ) -> (MLXArray, MLXArray) {
        var centers = initial
        let clusterIds = MLXArray((0 ..< clusters).map(Int32.init)).reshaped(1, 1, clusters, 1)
        let passes = max(0, iterations)
        var memberCounts = MLXArray.ones([x.dim(0), x.dim(1), clusters, 1], dtype: .float32)
        for _ in 0 ..< passes {
            let assign = nearestIndices(keys: x, centroids: centers)
            let mask = (expandedDimensions(assign, axis: 2) .== clusterIds).asType(.float32)
            memberCounts = mask.sum(axis: -1, keepDims: true)
            let updated = matmul(mask, x) / maximum(memberCounts, MLXArray(Float(1)))
            centers = which(memberCounts .> MLXArray(Float(0)), updated, centers)
        }
        if passes == 0 {
            let assign = nearestIndices(keys: x, centroids: centers)
            let mask = (expandedDimensions(assign, axis: 2) .== clusterIds).asType(.float32)
            memberCounts = mask.sum(axis: -1, keepDims: true)
        }
        return (centers, maximum(memberCounts, MLXArray(Float(1))))
    }
}

struct CentroidRQKVRows {
    static let supportedClusters = KVPrecision.supportedCentroidClusters

    enum ResidualStage {
        case affine(ZonedKVCache.QuantizedRows)
        case polar(PolarKeyRows)
    }

    let clusters: Int
    let iterations: Int
    let seed: UInt64
    private let pinnedCodebook: CentroidCodebook?
    private(set) var codebook: CentroidCodebook?
    private(set) var assignments: MLXArray?
    private(set) var residual: ResidualStage

    init(
        clusters: Int = 8, bits: Int = 2, groupSize: Int = 64,
        codebook: CentroidCodebook? = nil, iterations: Int = 4, seed: UInt64 = 0
    ) {
        precondition(Self.supportedClusters.contains(clusters), "RQ-KV supports 4, 8, or 16 centroids")
        self.clusters = clusters
        self.iterations = iterations
        self.seed = seed
        self.pinnedCodebook = codebook
        self.codebook = codebook
        self.residual = .affine(ZonedKVCache.QuantizedRows(bits: bits, groupSize: groupSize))
    }

    init(
        clusters: Int = 8, polar: PolarKeyRows,
        codebook: CentroidCodebook? = nil, iterations: Int = 4, seed: UInt64 = 0
    ) {
        precondition(Self.supportedClusters.contains(clusters), "RQ-KV supports 4, 8, or 16 centroids")
        self.clusters = clusters
        self.iterations = iterations
        self.seed = seed
        self.pinnedCodebook = codebook
        self.codebook = codebook
        self.residual = .polar(polar)
    }

    var count: Int { assignments?.dim(2) ?? 0 }

    var arrays: [MLXArray] {
        let stageArrays: [MLXArray] = switch residual {
        case .affine(let rows): rows.arrays
        case .polar(let rows): rows.arrays
        }
        return [codebook?.centroids, assignments].compactMap { $0 } + stageArrays
    }

    mutating func append(_ rows: MLXArray) {
        guard rows.dim(2) > 0 else { return }
        let active = codebook ?? CentroidCodebook.fitKMeansPlusPlus(
            keys: rows, clusters: clusters, iterations: iterations, seed: seed)
        codebook = active
        let codes = CentroidCodebook.nearestIndices(keys: rows.asType(.float32), centroids: active.centroids).asType(.uint8)
        let chosen = active.gather(codes)
        let residualRows = (rows.asType(.float32) - chosen).asType(rows.dtype)
        assignments = assignments.map { concatenated([$0, codes], axis: 2) } ?? codes
        switch residual {
        case .affine(var stage):
            stage.append(residualRows)
            residual = .affine(stage)
        case .polar(var stage):
            stage.append(residualRows)
            residual = .polar(stage)
        }
    }

    mutating func trim(_ n: Int) -> Int {
        let removed = min(max(0, n), count)
        guard removed > 0, let assignments else { return 0 }
        let kept = count - removed
        self.assignments = kept == 0 ? nil : assignments[0..., 0..., ..<kept]
        if kept == 0 { codebook = pinnedCodebook }
        switch residual {
        case .affine(var stage):
            _ = stage.trim(removed)
            residual = .affine(stage)
        case .polar(var stage):
            _ = stage.trim(removed)
            residual = .polar(stage)
        }
        return removed
    }

    func dequantizedCentroids(dtype: DType = .float32) -> MLXArray? {
        guard let codebook, let assignments else { return nil }
        return codebook.gather(assignments).asType(dtype)
    }

    func dequantizedResiduals(dtype: DType = .float32) -> MLXArray? {
        switch residual {
        case .affine(let rows): rows.dequantizedRows()?.asType(dtype)
        case .polar(let rows): rows.dequantizedRows(dtype: dtype)
        }
    }

    func dequantizedRows(dtype: DType = .float32) -> MLXArray? {
        guard let codebook, let assignments, let rHat = dequantizedResiduals(dtype: .float32) else { return nil }
        return (codebook.gather(assignments) + rHat).asType(dtype)
    }

    func scores(_ grouped: MLXArray) -> MLXArray? {
        guard let codebook, let assignments else { return nil }
        let centers = expandedDimensions(codebook.centroids.asType(grouped.dtype), axis: 2)
        let tableScores = matmul(grouped, centers.swappedAxes(-1, -2))
        let (b, h, g, q, t) = (grouped.dim(0), grouped.dim(1), grouped.dim(2), grouped.dim(3), assignments.dim(2))
        let index = broadcast(assignments.asType(.int32).reshaped(b, h, 1, 1, t), to: [b, h, g, q, t])
        let centroidScores = takeAlong(tableScores, index, axis: -1)
        let residualScores: MLXArray? = switch residual {
        case .affine(let rows):
            rows.parts.map { parts in
                quantizedMM(
                    grouped, expandedDimensions(parts.0, axis: 2), scales: expandedDimensions(parts.1, axis: 2),
                    biases: parts.2.map { expandedDimensions($0, axis: 2) }, transpose: true,
                    groupSize: rows.groupSize, bits: rows.bits)
            }
        case .polar(let rows):
            rows.dequantizedRows(dtype: grouped.dtype).map { rHat in
                matmul(grouped, expandedDimensions(rHat, axis: 2).swappedAxes(-1, -2))
            }
        }
        return residualScores.map { centroidScores + $0 }
    }
}
