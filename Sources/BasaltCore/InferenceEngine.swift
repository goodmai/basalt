import Foundation

// MARK: — Protocol

public protocol InferenceEngine: Sendable {
    func load(modelPath: String) async throws(BasaltError)
    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse
    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk>
    /// async getter — compatible with actor-isolated implementations.
    var isLoaded: Bool { get async }
    var kvMemoryBudget: ByteCount? { get async }
    var supportsKVZones: Bool { get async }
    var supportsKVEviction: Bool { get async }
    var supportsKVQuantization: Bool { get async }
    var hybridAnchorUsage: (count: Int, bytes: Int64)? { get async }
    var activeDeltas: [OverlayInfo] { get async }
    func promptTokenCount(for request: GenerationRequest) async -> Int?
    func persistPromptCache(within deadline: Duration) async
    func apply(_ directive: MemoryDirective) async -> MemoryLimits?
}

extension InferenceEngine {
    public var kvMemoryBudget: ByteCount? { get async { nil } }
    public var supportsKVZones: Bool { get async { true } }
    public var supportsKVEviction: Bool { get async { true } }
    public var supportsKVQuantization: Bool { get async { true } }
    public var hybridAnchorUsage: (count: Int, bytes: Int64)? { get async { nil } }
    public var activeDeltas: [OverlayInfo] { get async { [] } }
    public func promptTokenCount(for request: GenerationRequest) async -> Int? { nil }
    public func persistPromptCache(within deadline: Duration) async {}
    public func apply(_ directive: MemoryDirective) async -> MemoryLimits? { nil }
}
