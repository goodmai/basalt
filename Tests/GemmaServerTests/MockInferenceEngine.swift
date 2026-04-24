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

    func load(modelPath: String) async throws(GemmaServerError) {
        if shouldFailOnLoad { throw loadError }
        loaded = true
    }

    func generate(request: GenerationRequest) async throws(GemmaServerError) -> GenerationResponse {
        guard loaded else {
            throw .inferenceHardwareFailure(reason: "not loaded")
        }
        if shouldFailOnGenerate { throw generateError }
        
        let clock = ContinuousClock()
        let start = clock.now
        
        // simulate TTFT
        try? await Task.sleep(for: .milliseconds(5))
        let firstTokenTime = clock.now
        
        // simulate total generation time
        try? await Task.sleep(for: .milliseconds(5))
        
        let totalDuration = clock.now - start
        let genTime = Double(totalDuration.components.seconds) + Double(totalDuration.components.attoseconds) / 1e18
        
        let ttftDuration = firstTokenTime - start
        let ttft = Double(ttftDuration.components.seconds) + Double(ttftDuration.components.attoseconds) / 1e18
        
        return GenerationResponse(
            generatedText: mockText,
            promptTokens: request.prompt.split(separator: " ").count,
            completionTokens: 10,
            tokensPerSecond: 10.0 / genTime,
            generationTime: genTime,
            timeToFirstToken: ttft,
            memory: .init(peakBytes: 1024, activeBytes: 512, cacheBytes: 256),
            finishReason: .stop
        )
    }

    var isLoaded: Bool { loaded }
}
