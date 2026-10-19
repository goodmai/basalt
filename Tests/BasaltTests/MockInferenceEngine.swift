import Foundation
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

/// Управляемый mock для тестирования оркестратора без реального MLX.
actor MockInferenceEngine: InferenceEngine {

    private var loaded = false

    // Управляющие флаги
    var shouldFailOnLoad     = false
    var shouldFailOnGenerate = false
    var loadError: BasaltError    = .modelNotFound(identifier: "mock")
    var generateError: BasaltError = .inferenceHardwareFailure(reason: "mock failure")
    var mockText = "Mock response"
    var mockTPS  = 42.0
    var artificialDelay: TimeInterval = 0.0 // Artificial delay in seconds for testing timeouts
    var mockReasoning: String? = nil        // yielded as a .reasoning chunk before the text
    var mockToolCall: ToolCallOutput? = nil // yielded as a .toolCall chunk after the text
    var mockTextAfterToolCall: String? = nil
    var lastRequest: GenerationRequest?
    var mockCachedPromptTokens = 0
    var mockFinishReason: GenerationResponse.FinishReason = .stop

    func setMockFinishReason(_ reason: GenerationResponse.FinishReason) {
        self.mockFinishReason = reason
    }

    func setMockCachedPromptTokens(_ tokens: Int) {
        self.mockCachedPromptTokens = tokens
    }
    
    // Setter methods for actor-safe configuration
    func setArtificialDelay(_ delay: TimeInterval) {
        self.artificialDelay = delay
    }
    
    func getLastRequest() -> GenerationRequest? {
        self.lastRequest
    }
    
    func setMockReasoning(_ reasoning: String?) {
        self.mockReasoning = reasoning
    }

    func setMockToolCall(_ call: ToolCallOutput?, text: String = "Mock response", after: String? = nil) {
        self.mockToolCall = call
        self.mockText = text
        self.mockTextAfterToolCall = after
    }

    func setShouldFailOnGenerate(_ shouldFail: Bool) {
        self.shouldFailOnGenerate = shouldFail
    }
    
    func setGenerateError(_ error: BasaltError) {
        self.generateError = error
    }

    func setShouldFailOnLoad(_ shouldFail: Bool) {
        self.shouldFailOnLoad = shouldFail
    }

    func load(modelPath: String) async throws(BasaltError) {
        if shouldFailOnLoad { throw loadError }
        loaded = true
    }

    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
        self.lastRequest = request
        let stream = try await generateStream(request: request)
        // Assemble the response from the stream, like MLXInferenceEngine.generate does.
        var metadata: GenerationResponse?
        var text = ""
        var calls: [ToolCallOutput] = []
        for await chunk in stream {
            switch chunk {
            case .metadata(let m): metadata = m
            case .text(let t): text += t
            case .toolCall(let c): calls.append(c)
            case .reasoning: break
            }
        }
        guard let metadata else {
            throw .inferenceHardwareFailure(reason: "No metadata received from stream")
        }
        return GenerationResponse(
            generatedText: text, promptTokens: metadata.promptTokens,
            cachedPromptTokens: metadata.cachedPromptTokens, completionTokens: metadata.completionTokens,
            tokensPerSecond: metadata.tokensPerSecond, generationTime: metadata.generationTime,
            timeToFirstToken: metadata.timeToFirstToken, memory: metadata.memory,
            finishReason: metadata.finishReason, toolCalls: calls.isEmpty ? nil : calls
        )
    }

    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        self.lastRequest = request
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
                if let reasoning = mockReasoning { continuation.yield(.reasoning(reasoning)) }
                if !mockText.isEmpty { continuation.yield(.text(mockText)) }
                if let call = mockToolCall { continuation.yield(.toolCall(call)) }
                if let after = mockTextAfterToolCall { continuation.yield(.text(after)) }
                
                // simulate total generation time
                try? await Task.sleep(for: .milliseconds(5))
                
                let totalDuration = clock.now - start
                let genTime = Double(totalDuration.components.seconds) + Double(totalDuration.components.attoseconds) / 1e18
                
                let ttftDuration = firstTokenTime - start
                let ttft = Double(ttftDuration.components.seconds) + Double(ttftDuration.components.attoseconds) / 1e18
                
                let metadata = GenerationResponse(
                    generatedText: "Mock response",
                    promptTokens: request.prompt.split(separator: " ").count + mockCachedPromptTokens,
                    cachedPromptTokens: mockCachedPromptTokens,
                    completionTokens: 10,
                    tokensPerSecond: 10.0 / genTime,
                    generationTime: genTime,
                    timeToFirstToken: ttft,
                    memory: .init(peakBytes: 1024, activeBytes: 512, cacheBytes: 256),
                    finishReason: mockFinishReason
                )
                continuation.yield(.metadata(metadata))
                continuation.finish()
            }
        }
    }

    var isLoaded: Bool { loaded }
}
