import Foundation
@testable import GemmaServerCore

/// Управляемый mock для тестирования оркестратора без реального MLX.
actor MockInferenceEngine: InferenceEngine {

    private var loaded = false

    // Управляющие флаги
    var shouldFailOnLoad     = false
    var shouldFailOnGenerate = false
    var loadError: GemmaServerError    = .modelNotFound(identifier: "mock")
    var generateError: GemmaServerError = .inferenceHardwareFailure(reason: "mock failure")
    var mockText = "Mock response"
    var mockTPS  = 42.0
    var artificialDelay: TimeInterval = 0.0 // Artificial delay in seconds for testing timeouts
    
    // Setter methods for actor-safe configuration
    func setArtificialDelay(_ delay: TimeInterval) {
        self.artificialDelay = delay
    }
    
    func setShouldFailOnGenerate(_ shouldFail: Bool) {
        self.shouldFailOnGenerate = shouldFail
    }
    
    func setShouldFailOnLoad(_ shouldFail: Bool) {
        self.shouldFailOnLoad = shouldFail
    }

    func load(modelPath: String) async throws(GemmaServerError) {
        if shouldFailOnLoad { throw loadError }
        loaded = true
    }

    func generate(request: GenerationRequest) async throws(GemmaServerError) -> GenerationResponse {
        let stream = try await generateStream(request: request)
        var metadata: GenerationResponse?
        for await chunk in stream {
            if case .metadata(let m) = chunk { metadata = m }
        }
        return metadata!
    }

    func generateStream(request: GenerationRequest) async throws(GemmaServerError) -> AsyncStream<StreamChunk> {
        guard loaded else {
            throw .inferenceHardwareFailure(reason: "not loaded")
        }
        if shouldFailOnGenerate { throw generateError }
        
        return AsyncStream { continuation in
            Task {
                let clock = ContinuousClock()
                let start = clock.now
                
                // Apply artificial delay if configured
                if artificialDelay > 0 {
                    try? await Task.sleep(for: .seconds(artificialDelay))
                }
                
                // simulate TTFT
                try? await Task.sleep(for: .milliseconds(5))
                let firstTokenTime = clock.now
                continuation.yield(.text("Mock response"))
                
                // simulate total generation time
                try? await Task.sleep(for: .milliseconds(5))
                
                let totalDuration = clock.now - start
                let genTime = Double(totalDuration.components.seconds) + Double(totalDuration.components.attoseconds) / 1e18
                
                let ttftDuration = firstTokenTime - start
                let ttft = Double(ttftDuration.components.seconds) + Double(ttftDuration.components.attoseconds) / 1e18
                
                let metadata = GenerationResponse(
                    generatedText: "Mock response",
                    promptTokens: request.prompt.split(separator: " ").count,
                    completionTokens: 10,
                    tokensPerSecond: 10.0 / genTime,
                    generationTime: genTime,
                    timeToFirstToken: ttft,
                    memory: .init(peakBytes: 1024, activeBytes: 512, cacheBytes: 256),
                    finishReason: .stop
                )
                continuation.yield(.metadata(metadata))
                continuation.finish()
            }
        }
    }

    var isLoaded: Bool { loaded }
}
