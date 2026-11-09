import Foundation
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

actor MockInferenceEngine: InferenceEngine {
    private var loaded = false
    var shouldFailOnLoad = false
    var shouldFailOnGenerate = false
    var loadError: BasaltError = .modelNotFound(identifier: "mock")
    var generateError: BasaltError = .inferenceHardwareFailure(reason: "mock failure")
    var mockText = "Mock response"
    var mockTPS = 42.0
    var artificialDelay: TimeInterval = 0.0
    var mockReasoning: String? = nil
    var mockToolCalls: [ToolCallOutput] = []
    var mockTextAfterToolCall: String? = nil
    var lastRequest: GenerationRequest?
    var mockCachedPromptTokens = 0
    var mockFinishReason: GenerationResponse.FinishReason = .stop
    var mockKVBudget: Int64?
    var mockPromptTokens: Int?
    var mockConfidence: ResponseConfidence?
    var confidenceResponder: (@Sendable (GenerationRequest) -> ResponseConfidence?)?
    var finishReasonResponder: (@Sendable (GenerationRequest) -> GenerationResponse.FinishReason)?
    var mockSupportsKVZones = true
    var mockSupportsKVEviction = true
    var supportsKVQuantization = true
    var appliedDirectives: [MemoryDirective] = []
    var pacer: PrefillPacer?
    var responder: (@Sendable (GenerationRequest) -> String)?
    var requests: [GenerationRequest] = []
    var failure: (@Sendable (GenerationRequest) -> BasaltError?)?
    var heldStreams: [CheckedContinuation<Void, Never>]?
    var heldTokenCounts: [CheckedContinuation<Void, Never>]?
    private(set) var persistCount = 0
    var mockTranscription: Transcription?
    var mockTakesImages = true

    var kvMemoryBudget: ByteCount? { mockKVBudget.map { ByteCount($0) } }
    var supportsKVZones: Bool { mockSupportsKVZones }
    var supportsKVEviction: Bool { mockSupportsKVEviction }
    var isLoaded: Bool { loaded }
    var transcribesAudio: Bool { mockTranscription != nil }
    var takesImages: Bool { mockTakesImages }

    func promptTokenCount(for request: GenerationRequest) async -> Int? {
        if heldTokenCounts != nil { await withCheckedContinuation { heldTokenCounts?.append($0) } }
        return mockPromptTokens
    }

    func holdTokenCounts() { heldTokenCounts = [] }

    func releaseTokenCounts() {
        let held = heldTokenCounts ?? []
        heldTokenCounts = nil
        held.forEach { $0.resume() }
    }

    func apply(_ directive: MemoryDirective) async -> MemoryLimits? {
        appliedDirectives.append(directive)
        return MemoryLimits(cacheLimitBytes: Int(Double(512 << 20) * directive.cacheLimitShare),
                            promptCacheBudgetBytes: Int(Double(1 << 30) * directive.promptCacheShare), promptCacheEvictions: 0)
    }

    func pace(with pacer: PrefillPacer) async { self.pacer = pacer }
    func persistPromptCache(within deadline: Duration) async { persistCount += 1 }

    func load(modelPath: String) async throws(BasaltError) {
        if shouldFailOnLoad { throw loadError }
        loaded = true
    }

    func transcribe(_ request: TranscriptionRequest) async throws(BasaltError) -> Transcription {
        guard let mockTranscription else { throw .noAudioEncoder("") }
        return mockTranscription
    }

    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
        self.lastRequest = request
        let stream = try await generateStream(request: request)
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
        ).withConfidence(metadata.confidence)
    }

    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        self.lastRequest = request
        guard loaded else { throw .inferenceHardwareFailure(reason: "not loaded") }
        if shouldFailOnGenerate { throw generateError }
        requests.append(request)
        if let error = failure?(request) { throw error }
        let mockText = (request.assistantPrefix ?? "") + (responder?(request) ?? self.mockText)
        return AsyncStream { continuation in
            Task {
                let clock = ContinuousClock()
                let start = clock.now
                await waitWhileHeld()
                if artificialDelay > 0 { try? await Task.sleep(for: .seconds(artificialDelay)) }
                try? await Task.sleep(for: .milliseconds(5))
                let firstTokenTime = clock.now
                if let reasoning = mockReasoning { continuation.yield(.reasoning(reasoning)) }
                if !mockText.isEmpty { continuation.yield(.text(mockText)) }
                var gate = ToolCallGate(request: request)
                for call in mockToolCalls {
                    if let chunk = gate.admit(call) { continuation.yield(chunk) }
                }
                if let after = mockTextAfterToolCall { continuation.yield(.text(after)) }
                try? await Task.sleep(for: .milliseconds(5))
                let totalDuration = clock.now - start
                let genTime = Double(totalDuration.components.seconds) + Double(totalDuration.components.attoseconds) / 1e18
                let ttftDuration = firstTokenTime - start
                let ttft = Double(ttftDuration.components.seconds) + Double(ttftDuration.components.attoseconds) / 1e18
                let confidence = request.recordConfidence ? (confidenceResponder?(request) ?? mockConfidence) : nil
                let finishReason = finishReasonResponder?(request) ?? mockFinishReason
                let metadata = GenerationResponse(
                    generatedText: "Mock response",
                    promptTokens: request.prompt.split(separator: " ").count + mockCachedPromptTokens,
                    cachedPromptTokens: mockCachedPromptTokens,
                    completionTokens: 10,
                    tokensPerSecond: 10.0 / genTime,
                    generationTime: genTime,
                    timeToFirstToken: ttft,
                    memory: .init(peakBytes: 1024, activeBytes: 512, cacheBytes: 256),
                    finishReason: finishReason,
                    reasoningTokens: mockReasoning.map { $0.split(separator: " ").count }
                ).withConfidence(confidence)
                continuation.yield(.metadata(metadata))
                continuation.finish()
            }
        }
    }
}

extension MockInferenceEngine {
    func setMockConfidence(_ confidence: ResponseConfidence?) { mockConfidence = confidence }
    func setConfidenceResponder(_ responder: @escaping @Sendable (GenerationRequest) -> ResponseConfidence?) { confidenceResponder = responder }
    func setFinishReasonResponder(_ responder: @escaping @Sendable (GenerationRequest) -> GenerationResponse.FinishReason) { finishReasonResponder = responder }
    func setMemory(kvBudget: Int64?, promptTokens: Int? = nil) { mockKVBudget = kvBudget; mockPromptTokens = promptTokens }
    func setSupportsKVZones(_ supported: Bool) { mockSupportsKVZones = supported }
    func setSupportsKVEviction(_ supported: Bool) { mockSupportsKVEviction = supported }
    func setSupportsKVQuantization(_ supported: Bool) { supportsKVQuantization = supported }
    func setMockFinishReason(_ reason: GenerationResponse.FinishReason) { mockFinishReason = reason }
    func setResponder(_ responder: @escaping @Sendable (GenerationRequest) -> String) { self.responder = responder }
    func setFailure(_ failure: @escaping @Sendable (GenerationRequest) -> BasaltError?) { self.failure = failure }
    func getRequests() -> [GenerationRequest] { requests }
    func setMockCachedPromptTokens(_ tokens: Int) { mockCachedPromptTokens = tokens }
    func setArtificialDelay(_ delay: TimeInterval) { artificialDelay = delay }
    func holdStreams() { heldStreams = heldStreams ?? [] }
    func releaseStreams() {
        let held = heldStreams ?? []
        heldStreams = nil
        for stream in held { stream.resume() }
    }
    func waitWhileHeld() async {
        guard heldStreams != nil else { return }
        await withCheckedContinuation { heldStreams?.append($0) }
    }
    func getLastRequest() -> GenerationRequest? { lastRequest }
    func setMockReasoning(_ reasoning: String?) { mockReasoning = reasoning }
    func setMockToolCall(_ call: ToolCallOutput?, text: String = "Mock response", after: String? = nil) {
        setMockToolCalls(call.map { [$0] } ?? [], text: text, after: after)
    }
    func setMockToolCalls(_ calls: [ToolCallOutput], text: String = "Mock response", after: String? = nil) {
        mockToolCalls = calls; mockText = text; mockTextAfterToolCall = after
    }
    func setShouldFailOnGenerate(_ shouldFail: Bool) { shouldFailOnGenerate = shouldFail }
    func setGenerateError(_ error: BasaltError) { generateError = error }
    func setShouldFailOnLoad(_ shouldFail: Bool) { shouldFailOnLoad = shouldFail }
    func setMockTranscription(_ transcription: Transcription?) { mockTranscription = transcription }
    func setMockTakesImages(_ takes: Bool) { mockTakesImages = takes }
}

extension GenerationRequest {
    func with(_ edit: (inout GenerationRequest) throws -> Void) rethrows -> GenerationRequest {
        var copy = self
        try edit(&copy)
        return copy
    }
}
