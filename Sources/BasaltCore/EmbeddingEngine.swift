import Foundation

public enum EmbeddingInput: Sendable, Equatable {
    case text(String)
    case tokens([Int])
}

public struct Embeddings: Sendable, Equatable {
    public let vectors: [[Float]]
    public let promptTokens: Int

    public init(vectors: [[Float]], promptTokens: Int) {
        self.vectors = vectors
        self.promptTokens = promptTokens
    }
}

public struct EmbeddingModelInfo: Sendable, Equatable {
    public let id: String
    public let task: ModelTask
    public let truncatesDimensions: Bool

    public init(id: String, task: ModelTask, truncatesDimensions: Bool = false) {
        self.id = id
        self.task = task
        self.truncatesDimensions = truncatesDimensions
    }
}

public protocol EmbeddingEngine: Sendable {
    var models: [EmbeddingModelInfo] { get }
    func embed(_ inputs: [EmbeddingInput], dimensions: Int?, yieldingTo gate: GenerationGate?) async throws(BasaltError) -> Embeddings
    func rerank(query: String, documents: [String], yieldingTo gate: GenerationGate?) async throws(BasaltError) -> [Double]
}

public enum EmbeddingLimits {
    public static let maxInputs = 512
    public static let maxTokens = 131_072
    public static let maxDocuments = 1_000
}

public struct BesideDecodeEmbeddingEngine: EmbeddingEngine {
    public let inner: any EmbeddingEngine

    public var models: [EmbeddingModelInfo] { inner.models }

    public static func serving(_ engine: (any EmbeddingEngine)?, besideDecode: Bool) -> (any EmbeddingEngine)? {
        guard let engine, besideDecode, !(engine is BesideDecodeEmbeddingEngine) else { return engine }
        return BesideDecodeEmbeddingEngine(inner: engine)
    }

    public func embed(_ inputs: [EmbeddingInput], dimensions: Int?, yieldingTo _: GenerationGate?) async throws(BasaltError) -> Embeddings {
        try await inner.embed(inputs, dimensions: dimensions, yieldingTo: nil)
    }

    public func rerank(query: String, documents: [String], yieldingTo gate: GenerationGate?) async throws(BasaltError) -> [Double] {
        try await inner.rerank(query: query, documents: documents, yieldingTo: gate)
    }
}
