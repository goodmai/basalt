import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct MaterializedTensorSnapshot: Sendable {
    public let storage: MaterializedArray
    public let shape: [Int]
    public let dtype: DType
    public let elementCount: Int

    public init(materialized: MaterializedArray) {
        self.storage = materialized
        self.shape = materialized.shape
        self.dtype = materialized.dtype
        self.elementCount = materialized.size
    }

    public init(freezing array: MLXArray, contiguousLayout: Bool = true) {
        let prepared = contiguousLayout ? contiguous(array) : array
        self.init(materialized: MLX.materialize(prepared))
    }

    public func withRawSpan<R, E: Error>(
        _ body: (RawSpan) throws(E) -> R
    ) throws(E) -> R {
        let bytes = storage.asData(access: .noCopyIfContiguous).data
        defer { withExtendedLifetime(storage) {} }
        return try body(bytes.bytes)
    }

    public func withSpan<Scalar: HasDType & BitwiseCopyable, R, E: Error>(
        as _: Scalar.Type = Scalar.self,
        _ body: (Span<Scalar>) throws(E) -> R
    ) throws(E) -> R {
        guard dtype == Scalar.dtype else {
            return try UnifiedMemoryBridge.withContiguousSpan(of: storage, as: Scalar.self, body)
        }
        let bytes = storage.asData(access: .noCopyIfContiguous).data
        defer { withExtendedLifetime(storage) {} }
        return try bytes.withUnsafeBytes { raw throws(E) -> R in
            let count = raw.count / MemoryLayout<Scalar>.stride
            guard let base = raw.baseAddress else {
                return try body(UnsafeBufferPointer<Scalar>(start: nil, count: 0).span)
            }
            let bound = base.assumingMemoryBound(to: Scalar.self)
            return try body(UnsafeBufferPointer(start: bound, count: count).span)
        }
    }

    public func toArray<Scalar: HasDType & BitwiseCopyable>(
        as _: Scalar.Type = Scalar.self
    ) -> [Scalar] {
        withSpan(as: Scalar.self) { span in
            span.withUnsafeBufferPointer { Array($0) }
        }
    }
}

public struct MaterializedEmbeddingBatch: Sendable {
    public let matrix: MaterializedTensorSnapshot
    public let promptTokens: Int

    public init(matrix: MaterializedTensorSnapshot, promptTokens: Int) {
        self.matrix = matrix
        self.promptTokens = promptTokens
    }

    public var count: Int {
        matrix.shape.first ?? 0
    }

    public var dimensions: Int {
        matrix.shape.count >= 2 ? matrix.shape[1] : (matrix.shape.isEmpty ? 0 : matrix.elementCount)
    }

    public func withRowSpan<R, E: Error>(
        _ row: Int,
        _ body: (Span<Float>) throws(E) -> R
    ) throws(E) -> R {
        let width = dimensions
        precondition(row >= 0 && row < count, "embedding row \(row) out of bounds 0..<\(count)")
        return try matrix.withSpan(as: Float.self) { span throws(E) -> R in
            let start = row * width
            return try body(span.extracting(start ..< (start + width)))
        }
    }

    public func toEmbeddings() -> Embeddings {
        let rows = count
        let width = dimensions
        guard rows > 0, width > 0 else {
            return Embeddings(vectors: Array(repeating: [], count: rows), promptTokens: promptTokens)
        }
        let vectors = matrix.withSpan(as: Float.self) { span in
            (0 ..< rows).map { row in
                let start = row * width
                return span.extracting(start ..< (start + width)).withUnsafeBufferPointer { Array($0) }
            }
        }
        return Embeddings(vectors: vectors, promptTokens: promptTokens)
    }
}

public struct MaterializedLogitSnapshot: Sendable {
    public let logits: MaterializedTensorSnapshot
    public let stepIndex: Int

    public init(logits: MaterializedTensorSnapshot, stepIndex: Int = 0) {
        self.logits = logits
        self.stepIndex = stepIndex
    }

    public var vocabularySize: Int {
        logits.shape.last ?? 0
    }

    public var rowCount: Int {
        let vocab = vocabularySize
        return vocab > 0 ? logits.elementCount / vocab : 0
    }

    public func greedyToken(row: Int = 0) -> Int {
        let vocab = vocabularySize
        guard vocab > 0, row >= 0, row < rowCount else { return 0 }
        return logits.withSpan(as: Float.self) { span in
            let start = row * vocab
            let slice = span.extracting(start ..< (start + vocab))
            var bestIndex = 0
            var bestScore = slice[0]
            for index in 1 ..< vocab where slice[index] > bestScore {
                bestScore = slice[index]
                bestIndex = index
            }
            return bestIndex
        }
    }
}

public struct MaterializedXRayReadout: Sendable {
    public let info: TensorInfo
    public let slice: MaterializedTensorSnapshot

    public init(info: TensorInfo, slice: MaterializedTensorSnapshot) {
        self.info = info
        self.slice = slice
    }
}

public enum MaterializedTensorBridge {
    public static func freeze(_ array: MLXArray, contiguousLayout: Bool = true) -> MaterializedTensorSnapshot {
        MaterializedTensorSnapshot(freezing: array, contiguousLayout: contiguousLayout)
    }

    public static func freezeBatch(_ arrays: [MLXArray], contiguousLayout: Bool = true) -> [MaterializedTensorSnapshot] {
        guard !arrays.isEmpty else { return [] }
        let prepared = contiguousLayout ? arrays.map { contiguous($0) } : arrays
        return MLX.materialize(prepared).map(MaterializedTensorSnapshot.init(materialized:))
    }

    public static func freezeEmbeddings(_ embeddings: MLXArray, promptTokens: Int) -> MaterializedEmbeddingBatch {
        let floatMatrix = embeddings.dtype == .float32 ? embeddings : embeddings.asType(.float32)
        return MaterializedEmbeddingBatch(matrix: freeze(floatMatrix), promptTokens: promptTokens)
    }

    public static func freezeLogits(_ logits: MLXArray, stepIndex: Int = 0) -> MaterializedLogitSnapshot {
        let floatLogits = logits.dtype == .float32 ? logits : logits.asType(.float32)
        return MaterializedLogitSnapshot(logits: freeze(floatLogits), stepIndex: stepIndex)
    }

    public static func freezeXRay(info: TensorInfo, tensor: MLXArray) -> MaterializedXRayReadout {
        MaterializedXRayReadout(info: info, slice: freeze(tensor))
    }
}

extension MLXInferenceEngine {
    public func materializedTensor(_ request: TensorRequest) async throws(BasaltError) -> MaterializedXRayReadout {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }
        let result = await container.performEvaluating(nonSendable: ()) { context, _ in
            Result { () throws(BasaltError) -> MaterializedXRayReadout in
                try MLXInferenceEngine.extractMaterializedXRay(request, model: context.model)
            }
        }
        switch result {
        case .success(let readout): return readout
        case .failure(let error): throw error
        }
    }

    static func extractMaterializedXRay(_ request: TensorRequest, model: Module) throws(BasaltError) -> MaterializedXRayReadout {
        let views = WeightView.all(in: model)
        guard let view = views.first(where: { $0.name == request.name }) else {
            let matches = closeMatches(request.name, in: views.map(\.name)).joined(separator: ", ")
            throw .invalidRequestStructure(details: "no tensor named \(request.name); close matches: \(matches)")
        }
        let ranges = try validated(request.slice, extent: view.extent, name: view.name)
        guard let first = ranges.first, !first.isEmpty else {
            return MaterializedTensorBridge.freezeXRay(info: view.info, tensor: MLXArray.zeros([0], dtype: .float32))
        }
        var indices: [any MLXArrayIndex] = [0 ..< first.count]
        indices.append(contentsOf: ranges.dropFirst().map { $0 as any MLXArrayIndex })
        let sliced = view.rows(first)[indices]
        return MaterializedTensorBridge.freezeXRay(info: view.info, tensor: sliced)
    }
}
