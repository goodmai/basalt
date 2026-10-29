import Foundation

// MARK: — Protocol

public protocol InferenceEngine: Sendable {
    func load(modelPath: String) async throws(BasaltError)
    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse
    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk>
    /// async getter — compatible with actor-isolated implementations.
    var isLoaded: Bool { get async }
    var kvMemoryBudgetBytes: Int64? { get async }
    var supportsKVZones: Bool { get async }
    var supportsKVEviction: Bool { get async }
    func promptTokenCount(for request: GenerationRequest) async -> Int?
}

extension InferenceEngine {
    public var kvMemoryBudgetBytes: Int64? { get async { nil } }
    public var supportsKVZones: Bool { get async { true } }
    public var supportsKVEviction: Bool { get async { true } }
    public func promptTokenCount(for request: GenerationRequest) async -> Int? { nil }
}
