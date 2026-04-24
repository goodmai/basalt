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
        
        // simulate some generation time
        try? await Task.sleep(for: .milliseconds(10))
        
        let duration = clock.now - start
        let genTime = Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
        
        return GenerationResponse(
            generatedText: mockText,
            promptTokens: request.prompt.split(separator: " ").count,
            completionTokens: 10,
            tokensPerSecond: 10.0 / genTime,
            generationTime: genTime,
            finishReason: .stop
        )
    }

    var isLoaded: Bool { loaded }
}
