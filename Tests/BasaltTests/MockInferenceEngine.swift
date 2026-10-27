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
    var mockToolCalls: [ToolCallOutput] = []
    var mockTextAfterToolCall: String? = nil
    var lastRequest: GenerationRequest?
    var mockCachedPromptTokens = 0
    var mockFinishReason: GenerationResponse.FinishReason = .stop
    var mockKVBudget: Int64?
    var mockPromptTokens: Int?

    func setMemory(kvBudget: Int64?, promptTokens: Int? = nil) {
        mockKVBudget = kvBudget
        mockPromptTokens = promptTokens
    }

    var kvMemoryBudgetBytes: Int64? { mockKVBudget }

    func promptTokenCount(for request: GenerationRequest) async -> Int? { mockPromptTokens }

    func setMockFinishReason(_ reason: GenerationResponse.FinishReason) {
        self.mockFinishReason = reason
    }

    var responder: (@Sendable (GenerationRequest) -> String)?
    var requests: [GenerationRequest] = []

    func setResponder(_ responder: @escaping @Sendable (GenerationRequest) -> String) {
        self.responder = responder
    }

    var failure: (@Sendable (GenerationRequest) -> BasaltError?)?

    func setFailure(_ failure: @escaping @Sendable (GenerationRequest) -> BasaltError?) {
        self.failure = failure
    }

    func getRequests() -> [GenerationRequest] {
        requests
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
        setMockToolCalls(call.map { [$0] } ?? [], text: text, after: after)
    }

    func setMockToolCalls(_ calls: [ToolCallOutput], text: String = "Mock response", after: String? = nil) {
        self.mockToolCalls = calls
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
        var reasoning: String?
        for await chunk in stream {
            switch chunk {
            case .metadata(let m): metadata = m
            case .text(let t): text += t
            case .toolCall(let c): calls.append(c)
            case .reasoning(let r): reasoning = (reasoning ?? "") + r
            }
        }
        guard let metadata else {
            throw .inferenceHardwareFailure(reason: "No metadata received from stream")
        }
        return GenerationResponse(
            generatedText: text, reasoningText: reasoning, promptTokens: metadata.promptTokens,
            cachedPromptTokens: metadata.cachedPromptTokens, completionTokens: metadata.completionTokens,
            tokensPerSecond: metadata.tokensPerSecond, generationTime: metadata.generationTime,
            timeToFirstToken: metadata.timeToFirstToken, memory: metadata.memory,
            finishReason: metadata.finishReason, toolCalls: calls.isEmpty ? nil : calls,
            reasoningTokens: metadata.reasoningTokens
        )
    }

    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        self.lastRequest = request
        guard loaded else {
            throw .inferenceHardwareFailure(reason: "not loaded")
        }
        if shouldFailOnGenerate { throw generateError }
        requests.append(request)
        if let error = failure?(request) { throw error }
        let mockText = (request.assistantPrefix ?? "") + (responder?(request) ?? self.mockText)
        
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
                var gate = ToolCallGate(request: request)
                for call in mockToolCalls {
                    if let chunk = gate.admit(call) { continuation.yield(chunk) }
                }
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
                    finishReason: mockFinishReason,
                    reasoningTokens: mockReasoning.map { $0.split(separator: " ").count }
                )
                continuation.yield(.metadata(metadata))
                continuation.finish()
            }
        }
    }

    var isLoaded: Bool { loaded }
}
