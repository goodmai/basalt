import Foundation

public enum EMeansPrecision: String, Sendable, Equatable, Hashable, CaseIterable {
    case float32
    case bfloat16
    case float16
}

public enum EMeansFinalizer: Sendable, Equatable, Hashable {
    case none
    case lastEpochMean
    case slowEMA(speedMultiplier: Float = 0.25)
}

public enum EMeansError: Error, Sendable, Equatable, LocalizedError {
    case invalidRank(expected: Int, actual: Int)
    case invalidClusterCount(clusters: Int, points: Int)
    case invalidDimension(Int)
    case invalidEpochs(Double)
    case invalidParameter(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRank(let expected, let actual):
            "Expected \(expected)-D tensor, got \(actual)-D"
        case .invalidClusterCount(let clusters, let points):
            "Invalid cluster count \(clusters) for \(points) points (require 1 <= K <= N)"
        case .invalidDimension(let dim):
            "Invalid vector dimension \(dim) (require D >= 1)"
        case .invalidEpochs(let epochs):
            "Invalid epochs \(epochs) (require finite value > 0)"
        case .invalidParameter(let detail):
            "Invalid e-means parameter: \(detail)"
        case .invalidInput(let detail):
            "Invalid e-means input: \(detail)"
        }
    }
}

public struct EMeansConfig: Sendable, Equatable {
    public static let defaultRevivalDelta: Float = 1.0 / 128.0

    public var clusters: Int
    public var epochs: Double
    public var batchSize: Int?
    public var peakStep: Float? = nil
    public var cosineFloor: Float = 0.1
    public var revivalInterval: Int = 5
    public var revivalThreshold: Float = 0.1
    public var revivalDelta: Float = EMeansConfig.defaultRevivalDelta
    public var finalizer: EMeansFinalizer
    public var seed: UInt64
    public var assignmentDType: EMeansPrecision

    public init(
        clusters: Int,
        epochs: Double = 2.5,
        batchSize: Int? = nil,
        finalizer: EMeansFinalizer = .lastEpochMean,
        seed: UInt64 = 0,
        assignmentDType: EMeansPrecision = .float32
    ) {
        self.clusters = clusters
        self.epochs = epochs
        self.batchSize = batchSize
        self.finalizer = finalizer
        self.seed = seed
        self.assignmentDType = assignmentDType
    }

    public func validate(points: Int, dimensions: Int) throws(EMeansError) {
        guard clusters >= 1 && clusters <= points else {
            throw .invalidClusterCount(clusters: clusters, points: points)
        }
        guard dimensions >= 1 else {
            throw .invalidDimension(dimensions)
        }
        guard epochs.isFinite && epochs > 0 else {
            throw .invalidEpochs(epochs)
        }
        try validateStepSchedule()
        try validateRevivalAndFinalizer()
    }

    public func resolvedBatchSize(points: Int) -> Int {
        if let batchSize {
            return min(max(1, points), max(1, batchSize))
        }
        return EMeansScalingLaw.kneeBatchSize(points: points, clusters: clusters, epochs: epochs)
    }

    public func resolvedPeakStep(points: Int, batchSize: Int) -> Float {
        if let peakStep {
            return min(1.0, max(Float.leastNormalMagnitude, peakStep))
        }
        return EMeansScalingLaw.peakStep(points: points, clusters: clusters, batchSize: batchSize, epochs: epochs)
    }

    public func resolvedSlowAlpha(peakStep: Float, speedMultiplier: Float) -> Float {
        let raw = speedMultiplier * cosineFloor * peakStep
        return min(1.0, max(1e-5, raw))
    }

    private func validateStepSchedule() throws(EMeansError) {
        if let batchSize, batchSize < 1 {
            throw .invalidParameter("batchSize must be >= 1")
        }
        if let peakStep, !(peakStep.isFinite && peakStep > 0 && peakStep <= 1) {
            throw .invalidParameter("peakStep must be in (0, 1]")
        }
        guard cosineFloor.isFinite && cosineFloor >= 0 && cosineFloor <= 1 else {
            throw .invalidParameter("cosineFloor must be in [0, 1]")
        }
    }

    private func validateRevivalAndFinalizer() throws(EMeansError) {
        guard revivalInterval >= 0 else {
            throw .invalidParameter("revivalInterval must be >= 0")
        }
        guard revivalThreshold.isFinite && revivalThreshold >= 0 else {
            throw .invalidParameter("revivalThreshold must be finite and >= 0")
        }
        guard revivalDelta.isFinite && revivalDelta > 0 && revivalDelta < 1 else {
            throw .invalidParameter("revivalDelta must be in (0, 1)")
        }
        if case .slowEMA(let speed) = finalizer, !(speed.isFinite && speed > 0) {
            throw .invalidParameter("slowEMA speedMultiplier must be finite and > 0")
        }
    }
}

public enum EMeansScalingLaw {
    public static let largeDatasetThreshold = 2_000_000

    public static func kneeBatchSize(points n: Int, clusters k: Int, epochs t: Double) -> Int {
        guard n > 0, k > 0, t.isFinite, t > 0 else { return max(1, n) }
        let coefficient = n > largeDatasetThreshold ? 0.0044 : 0.035
        let raw = coefficient * pow(Double(k), 0.13) * pow(Double(n), 0.88) * pow(t, 0.81)
        let rounded = nearestPowerOfTwo(raw)
        return min(n, max(1, rounded))
    }

    public static func peakStep(points n: Int, clusters k: Int, batchSize b: Int, epochs t: Double) -> Float {
        let safeN = Double(max(1, n))
        let safeK = Double(max(1, k))
        let safeB = Double(max(1, b))
        let safeT = t.isFinite && t > 0 ? t : 1.0
        let raw = 0.824 * pow(safeK, -0.46) * pow(safeB, 0.90) * pow(safeN, -0.45) * pow(safeT, -0.11)
        return min(1.0, max(Float.leastNormalMagnitude, Float(raw)))
    }

    public static func stepAlpha(step t: Int, totalSteps m: Int, peakAlpha a0: Float, floor eps: Float = 0.1) -> Float {
        let denom = max(1, m)
        let clampedStep = min(denom, max(0, t))
        let clampedEps = min(1.0, max(0.0, Double(eps)))
        let cosine = 0.5 * (1.0 + cos(Double.pi * Double(clampedStep) / Double(denom)))
        let scaled = Double(a0) * (clampedEps + (1.0 - clampedEps) * cosine)
        return min(1.0, max(0.0, Float(scaled)))
    }

    public static func batchesPerEpoch(points n: Int, batchSize b: Int) -> Int {
        max(1, max(1, n) / max(1, b))
    }

    public static func totalSteps(points n: Int, batchSize b: Int, epochs t: Double) -> Int {
        guard t.isFinite, t > 0 else { return 1 }
        let perEpoch = batchesPerEpoch(points: n, batchSize: b)
        return max(1, Int((t * Double(perEpoch)).rounded(.down)))
    }

    private static func nearestPowerOfTwo(_ value: Double) -> Int {
        guard value.isFinite, value > 1.0 else { return 1 }
        let lowExp = min(60, max(0, Int(floor(log2(value)))))
        let low = pow(2.0, Double(lowExp))
        let high = low * 2.0
        let picked = (value - low) < (high - value) ? low : high
        return Int(picked)
    }
}

public struct EMeansMetrics: Sendable, Equatable {
    public let trainMSE: Float
    public let deadClusters: Int
    public let revivals: Int
    public let steps: Int
    public let batchSize: Int
    public let peakStep: Float

    public init(
        trainMSE: Float,
        deadClusters: Int,
        revivals: Int,
        steps: Int,
        batchSize: Int,
        peakStep: Float
    ) {
        self.trainMSE = trainMSE
        self.deadClusters = deadClusters
        self.revivals = revivals
        self.steps = steps
        self.batchSize = batchSize
        self.peakStep = peakStep
    }
}

public struct EMeansFileReport: Sendable, Equatable {
    public let points: Int
    public let dimensions: Int
    public let clusters: Int
    public let metrics: EMeansMetrics
    public let elapsedSeconds: Double
    public let outputPath: String

    public var vectorsPerSecond: Double {
        Double(points) / max(elapsedSeconds, 1e-9)
    }

    public init(
        points: Int,
        dimensions: Int,
        clusters: Int,
        metrics: EMeansMetrics,
        elapsedSeconds: Double,
        outputPath: String
    ) {
        self.points = points
        self.dimensions = dimensions
        self.clusters = clusters
        self.metrics = metrics
        self.elapsedSeconds = elapsedSeconds
        self.outputPath = outputPath
    }
}

public struct EMeansRevivalPair: Sendable, Equatable {
    public let dead: Int
    public let donor: Int

    public init(dead: Int, donor: Int) {
        self.dead = dead
        self.donor = donor
    }
}

public enum EMeansCore {
    private struct EmbeddingRecord: Decodable {
        let embedding: [Float]
    }

    public static func sampleUniqueIndices(count: Int, k: Int, rng: inout SplitMix64) -> [Int32] {
        let limit = min(max(0, k), max(0, count))
        var swaps: [Int: Int] = [:]
        swaps.reserveCapacity(limit)
        var result: [Int32] = []
        result.reserveCapacity(limit)
        for i in 0 ..< limit {
            let j = i + Int(rng.next() % UInt64(count - i))
            let valI = swaps[i] ?? i
            let valJ = swaps[j] ?? j
            swaps[j] = valI
            result.append(Int32(valJ))
        }
        return result
    }

    public static func shuffledIndices(count: Int, rng: inout SplitMix64) -> [Int32] {
        guard count > 0 else { return [] }
        var indices = (0 ..< count).map(Int32.init)
        if count == 1 { return indices }
        for i in stride(from: count - 1, through: 1, by: -1) {
            let j = Int(rng.next() % UInt64(i + 1))
            if i != j {
                indices.swapAt(i, j)
            }
        }
        return indices
    }

    public static func planRevivals(
        counts: [Float],
        threshold: Float,
        rng: inout SplitMix64
    ) -> [EMeansRevivalPair] {
        guard counts.count >= 2, threshold.isFinite, threshold > 0 else { return [] }
        let total = counts.reduce(0.0) { $0 + Double(max(0, $1)) }
        guard total > 0 else { return [] }
        let cutoff = Double(threshold) * (total / Double(counts.count))
        var dead: [Int] = []
        var donors: [Int] = []
        for index in counts.indices {
            let value = Double(max(0, counts[index]))
            if value < cutoff {
                dead.append(index)
            } else if value > 0 {
                donors.append(index)
            }
        }
        guard !dead.isEmpty, !donors.isEmpty else { return [] }
        var pairs: [EMeansRevivalPair] = []
        pairs.reserveCapacity(min(dead.count, donors.count))
        for deadIndex in dead {
            guard let pos = drawWeightedDonor(donors: donors, counts: counts, rng: &rng) else { break }
            pairs.append(EMeansRevivalPair(dead: deadIndex, donor: donors.remove(at: pos)))
        }
        return pairs
    }

    public static func parseJSONLVectors(
        _ text: String
    ) throws(EMeansError) -> (rows: Int, cols: Int, values: [Float]) {
        let decoder = JSONDecoder()
        var rows = 0
        var cols = 0
        var values: [Float] = []
        for (offset, rawLine) in text.split(whereSeparator: \.isNewline).enumerated() {
            let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let vector = try decodeVectorRow(trimmed, line: offset + 1, decoder: decoder)
            guard !vector.isEmpty else {
                throw .invalidInput("JSONL line \(offset + 1) contains an empty vector")
            }
            guard vector.allSatisfy(\.isFinite) else {
                throw .invalidInput("JSONL line \(offset + 1) contains non-finite values")
            }
            if cols == 0 {
                cols = vector.count
            } else if vector.count != cols {
                throw .invalidInput("JSONL line \(offset + 1) has dimension \(vector.count), expected \(cols)")
            }
            values.append(contentsOf: vector)
            rows += 1
        }
        guard rows > 0 else {
            throw .invalidInput("JSONL input contains no vector rows")
        }
        return (rows, cols, values)
    }

    private static func drawWeightedDonor(
        donors: [Int],
        counts: [Float],
        rng: inout SplitMix64
    ) -> Int? {
        guard !donors.isEmpty else { return nil }
        var totalWeight = 0.0
        for index in donors { totalWeight += Double(max(0, counts[index])) }
        guard totalWeight > 0 else { return nil }
        let draw = Double(rng.next() >> 11) * (1.0 / Double(1 << 53)) * totalWeight
        var cumulative = 0.0
        for position in donors.indices {
            cumulative += Double(max(0, counts[donors[position]]))
            if cumulative >= draw { return position }
        }
        return donors.count - 1
    }

    private static func decodeVectorRow(
        _ line: String,
        line number: Int,
        decoder: JSONDecoder
    ) throws(EMeansError) -> [Float] {
        let data = Data(line.utf8)
        if let direct = try? decoder.decode([Float].self, from: data) {
            return direct
        }
        if let wrapped = try? decoder.decode(EmbeddingRecord.self, from: data) {
            return wrapped.embedding
        }
        throw .invalidInput("JSONL line \(number) must be a JSON array of floats or {\"embedding\": [...]}")
    }
}
