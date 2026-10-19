import Foundation

// MARK: — Protocol

public protocol InferenceEngine: Sendable {
    func load(modelPath: String) async throws(BasaltError)
    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse
    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk>
    /// async getter — compatible with actor-isolated implementations.
    var isLoaded: Bool { get async }
}
