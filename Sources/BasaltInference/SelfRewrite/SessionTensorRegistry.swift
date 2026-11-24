import BasaltCore
import Foundation
import MLX

public struct TensorStatsSummary: Sendable, Codable, Equatable {
    public let mean: Float?
    public let std: Float?
    public let minValue: Float
    public let maxValue: Float
    public let l2Norm: Float
    public let absMedian: Float?

    public init(
        mean: Float? = nil,
        std: Float? = nil,
        minValue: Float,
        maxValue: Float,
        l2Norm: Float,
        absMedian: Float? = nil
    ) {
        self.mean = mean
        self.std = std
        self.minValue = minValue
        self.maxValue = maxValue
        self.l2Norm = l2Norm
        self.absMedian = absMedian
    }
}

public struct TensorHandleSummary: Sendable, Codable, Equatable {
    public let key: String
    public let shape: [Int]
    public let dtype: String
    public let byteCount: Int
    public let stats: TensorStatsSummary
    public let topAbsIndices: [Int]

    public init(
        key: String,
        shape: [Int],
        dtype: String,
        byteCount: Int,
        stats: TensorStatsSummary,
        topAbsIndices: [Int]
    ) {
        self.key = key
        self.shape = shape
        self.dtype = dtype
        self.byteCount = byteCount
        self.stats = stats
        self.topAbsIndices = topAbsIndices
    }
}

public enum TensorRegistryError: Error, Sendable, Equatable, CustomStringConvertible {
    case capacityExceeded(limit: Int)
    case byteBudgetExceeded(required: Int64, limit: Int64)
    case handleNotFound(key: String)
    case nonFiniteTensor(key: String)
    case invalidShape(key: String, detail: String)

    public var description: String {
        switch self {
        case .capacityExceeded(let limit):
            return "session tensor registry exceeded capacity limit \(limit)"
        case .byteBudgetExceeded(let required, let limit):
            return "session tensor registry requires \(required) bytes, exceeding limit \(limit)"
        case .handleNotFound(let key):
            return "tensor handle '\(key)' not found in session registry"
        case .nonFiniteTensor(let key):
            return "tensor handle '\(key)' contains NaN or Inf values"
        case .invalidShape(let key, let detail):
            return "tensor handle '\(key)' has invalid shape: \(detail)"
        }
    }
}

public struct SessionTensorRegistry: Sendable {
    public let maxEntries: Int
    public let maxTotalBytes: Int64
    private var entries: [String: MaterializedArray] = [:]
    private var summaries: [String: TensorHandleSummary] = [:]

    public init(maxEntries: Int = 64, maxTotalBytes: Int64 = 64 * 1024 * 1024) {
        self.maxEntries = max(1, maxEntries)
        self.maxTotalBytes = max(1024, maxTotalBytes)
    }

    public var keys: [String] { entries.keys.sorted() }

    public var totalBytes: Int64 {
        entries.values.reduce(Int64(0)) { $0 + Int64($1.nbytes) }
    }

    public func summary(forKey key: String) -> TensorHandleSummary? {
        summaries[key]
    }

    @discardableResult
    public mutating func register(_ array: MLXArray, forKey key: String) throws -> TensorHandleSummary {
        try EditProgramValidator.validateIdentifier(key, field: "tensor_handle")
        guard array.size > 0 else {
            throw TensorRegistryError.invalidShape(key: key, detail: "tensor is empty")
        }
        let frozen = materialize(array)
        let floats = frozen.asType(.float32).flattened().asArray(Float.self)
        guard floats.allSatisfy(\.isFinite) else {
            throw TensorRegistryError.nonFiniteTensor(key: key)
        }
        let existingBytes = Int64(entries[key]?.nbytes ?? 0)
        if entries[key] == nil, entries.count >= maxEntries {
            throw TensorRegistryError.capacityExceeded(limit: maxEntries)
        }
        let projectedBytes = totalBytes - existingBytes + Int64(frozen.nbytes)
        guard projectedBytes <= maxTotalBytes else {
            throw TensorRegistryError.byteBudgetExceeded(required: projectedBytes, limit: maxTotalBytes)
        }
        let summary = Self.summarize(key: key, array: frozen, floats: floats)
        entries[key] = frozen
        summaries[key] = summary
        return summary
    }

    @discardableResult
    public mutating func registerActivationContrast(
        key: String,
        positive: [MLXArray],
        negative: [MLXArray]
    ) throws -> TensorHandleSummary {
        guard !positive.isEmpty, !negative.isEmpty else {
            throw TensorRegistryError.invalidShape(key: key, detail: "positive and negative activation lists must be non-empty")
        }
        let posMean = stacked(positive.map { $0.asType(.float32).flattened() }).mean(axis: 0)
        let negMean = stacked(negative.map { $0.asType(.float32).flattened() }).mean(axis: 0)
        let diff = posMean - negMean
        let norm = sqrt((diff * diff).sum())
        eval(norm)
        let normScalar = norm.item(Float.self)
        guard normScalar.isFinite, normScalar > 0 else {
            throw TensorRegistryError.nonFiniteTensor(key: key)
        }
        return try register(diff / normScalar, forKey: key)
    }

    @discardableResult
    public mutating func registerTopLeftSingularVector(
        key: String,
        matrix: MLXArray,
        iterations: Int = 16
    ) throws -> TensorHandleSummary {
        guard matrix.ndim == 2, matrix.dim(0) > 0, matrix.dim(1) > 0 else {
            throw TensorRegistryError.invalidShape(key: key, detail: "expected 2D non-empty matrix, got \(matrix.shape)")
        }
        let w = matrix.asType(.float32)
        let cols = w.dim(1)
        var v = withRandomState(MLXRandom.RandomState(seed: 770)) {
            MLXRandom.normal([cols]).asType(.float32)
        }
        v = v / maximum(sqrt((v * v).sum()), Float(1e-12))
        var u = MLXArray.zeros([w.dim(0)], dtype: .float32)
        for _ in 0 ..< max(1, iterations) {
            u = matmul(w, v.reshaped(cols, 1)).squeezed(axis: 1)
            u = u / maximum(sqrt((u * u).sum()), Float(1e-12))
            v = matmul(u.reshaped(1, w.dim(0)), w).squeezed(axis: 0)
            v = v / maximum(sqrt((v * v).sum()), Float(1e-12))
        }
        eval(u)
        return try register(u, forKey: key)
    }

    public func resolveOperands(for program: EditProgram) throws -> [String: MLXArray] {
        var resolved: [String: MLXArray] = [:]
        for key in program.referencedTensorKeys {
            guard let stored = entries[key] else {
                throw TensorRegistryError.handleNotFound(key: key)
            }
            resolved[key] = stored.reshaped(stored.shape)
        }
        return resolved
    }

    public mutating func remove(key: String) {
        entries.removeValue(forKey: key)
        summaries.removeValue(forKey: key)
    }

    public mutating func clear() {
        entries.removeAll()
        summaries.removeAll()
    }

    private static func summarize(key: String, array: MaterializedArray, floats: [Float]) -> TensorHandleSummary {
        var minVal = Float.greatestFiniteMagnitude
        var maxVal = -Float.greatestFiniteMagnitude
        var sumSq: Double = 0
        var absVals: [Float] = []
        absVals.reserveCapacity(floats.count)
        for v in floats {
            if v < minVal { minVal = v }
            if v > maxVal { maxVal = v }
            sumSq += Double(v) * Double(v)
            absVals.append(abs(v))
        }
        let sortedAbs = absVals.sorted()
        let mid = sortedAbs.count / 2
        let median: Float = sortedAbs.count % 2 == 1
            ? sortedAbs[mid]
            : (sortedAbs[mid - 1] + sortedAbs[mid]) * 0.5
        let topIdx = floats.enumerated()
            .sorted { abs($0.element) > abs($1.element) }
            .prefix(5)
            .map(\.offset)
        let stats = TensorStatsSummary(
            minValue: minVal,
            maxValue: maxVal,
            l2Norm: Float(sqrt(sumSq)),
            absMedian: median
        )
        return TensorHandleSummary(
            key: key,
            shape: array.shape,
            dtype: "\(array.dtype)",
            byteCount: array.nbytes,
            stats: stats,
            topAbsIndices: topIdx
        )
    }
}
