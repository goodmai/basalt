import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCloud

actor StubCloud: CloudFallback {
    var assessment: CloudAssessment?
    var failsMidStream = false
    var failsOnCall = false
    private(set) var calls = 0

    init(_ assessment: CloudAssessment?) { self.assessment = assessment }

    func failing(midStream: Bool = false, onCall: Bool = false) {
        failsMidStream = midStream
        failsOnCall = onCall
    }

    func assess(_ request: GenerationRequest) async -> CloudAssessment? { assessment }

    func generate(_ request: GenerationRequest) async throws -> GenerationResponse {
        calls += 1
        if failsOnCall { throw BasaltError.modelInferenceError(details: "cloud down") }
        return GenerationResponse(
            generatedText: "cloud answer", promptTokens: 1, completionTokens: 2, tokensPerSecond: 1,
            generationTime: 1, timeToFirstToken: 0, memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
            finishReason: .stop)
    }

    func generateStream(_ request: GenerationRequest) async throws -> AsyncThrowingStream<StreamChunk, Error> {
        calls += 1
        if failsOnCall { throw BasaltError.modelInferenceError(details: "cloud down") }
        let failsMidStream = failsMidStream
        return AsyncThrowingStream { continuation in
            continuation.yield(.text("cloud "))
            if failsMidStream {
                continuation.finish(throwing: BasaltError.modelInferenceError(details: "disconnected"))
            } else {
                continuation.finish()
            }
        }
    }
}

@Suite("Hybrid local ↔ cloud fallback (#22)")
struct HybridFallbackTests {
    static let plenty: @Sendable () -> Int64 = { 64 << 30 }
    static let cheap = CloudAssessment(capabilityGap: 0, estimatedCost: 0.01, remainingBudget: 10, latencySeconds: 1)

    func orchestrator(
        _ cloud: StubCloud, arbiter: HybridArbiter = HybridArbiter(), freeMemory: @escaping @Sendable () -> Int64 = plenty
    ) async throws -> ModelOrchestratorActor {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "local")
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        await orchestrator.setCloudFallback(cloud, arbiter: arbiter, freeMemory: freeMemory)
        return orchestrator
    }

    @Test("A request whose KV does not fit in the generation memory budget is answered by the cloud")
    func memory() async throws {
        let cloud = StubCloud(Self.cheap)
        let orchestrator = try await orchestrator(cloud, freeMemory: { 1 << 20 })
        let request = GenerationRequest(prompt: "hi", maxTokens: 4_096)
        #expect(await orchestrator.cloudFallbackReason(for: request) == .memory)
        #expect(try await orchestrator.generate(request: request).generatedText == "cloud answer")
        #expect(await cloud.calls == 1)
    }

    @Test("A prompt beyond the local context limit is answered by the cloud")
    func contextDegradation() async throws {
        let cloud = StubCloud(Self.cheap)
        let orchestrator = try await orchestrator(cloud, arbiter: HybridArbiter(localContextLimit: 100))
        let long = GenerationRequest(prompt: String(repeating: "word ", count: 200))
        #expect(await orchestrator.cloudFallbackReason(for: long) == .contextDegradation)
        #expect(await orchestrator.cloudFallbackReason(for: GenerationRequest(prompt: "short")) == nil)
    }

    @Test("A task the router rates beyond the local model goes to the cloud when the gain beats the cost")
    func capability() async throws {
        let hard = StubCloud(CloudAssessment(capabilityGap: 0.9, estimatedCost: 0.5, remainingBudget: 10, latencySeconds: 1))
        let hardOrchestrator = try await orchestrator(hard)
        #expect(await hardOrchestrator.cloudFallbackReason(for: GenerationRequest(prompt: "prove it")) == .capability)
        let pricey = StubCloud(CloudAssessment(capabilityGap: 0.9, estimatedCost: 9.5, remainingBudget: 10, latencySeconds: 1))
        let priceyOrchestrator = try await orchestrator(pricey)
        #expect(await priceyOrchestrator.cloudFallbackReason(for: GenerationRequest(prompt: "prove it")) == nil)
    }

    @Test("Without a reason the local model answers and the cloud is not called")
    func staysLocal() async throws {
        let cloud = StubCloud(Self.cheap)
        let orchestrator = try await orchestrator(cloud)
        #expect(try await orchestrator.generate(request: GenerationRequest(prompt: "hi")).generatedText == "Mock response")
        #expect(await cloud.calls == 0)
    }

    @Test("Over budget the cloud is not called, even when the local model does not fit")
    func budgetExceeded() async throws {
        let over = StubCloud(CloudAssessment(capabilityGap: 0.9, estimatedCost: 2, remainingBudget: 1, latencySeconds: 1))
        let orchestrator = try await orchestrator(over, freeMemory: { 1 << 20 })
        let request = GenerationRequest(prompt: "hi", maxTokens: 4_096)
        #expect(await orchestrator.cloudFallbackReason(for: request) == nil)
        _ = try await orchestrator.generate(request: request)
        #expect(await over.calls == 0)

        let exhausted = StubCloud(nil)
        _ = try await self.orchestrator(exhausted, freeMemory: { 1 << 20 }).generate(request: request)
        #expect(await exhausted.calls == 0)
    }

    @Test("Service requests (titles, summaries) never go to the cloud")
    func serviceRequestsStayLocal() async throws {
        let orchestrator = try await orchestrator(StubCloud(Self.cheap), freeMemory: { 1 << 20 })
        #expect(await orchestrator.cloudFallbackReason(for: GenerationRequest(prompt: "t", maxTokens: 4_096, isServiceRequest: true)) == nil)
    }

    @Test("A failing cloud call is answered locally")
    func cloudFailureFallsBackLocally() async throws {
        let cloud = StubCloud(Self.cheap)
        await cloud.failing(onCall: true)
        let orchestrator = try await orchestrator(cloud, freeMemory: { 1 << 20 })
        let request = GenerationRequest(prompt: "hi", maxTokens: 4_096)
        #expect(try await orchestrator.generate(request: request).generatedText == "Mock response")
        var text = ""
        for await chunk in try await orchestrator.generateStream(request: request) {
            if case .text(let t) = chunk { text += t }
        }
        #expect(text == "Mock response")
    }

    @Test("A cloud stream that drops mid-way ends with an error finish, which the APIs report as truncated")
    func midStreamFailureIsVisible() async throws {
        let cloud = StubCloud(Self.cheap)
        await cloud.failing(midStream: true)
        let orchestrator = try await orchestrator(cloud, freeMemory: { 1 << 20 })
        var chunks: [StreamChunk] = []
        for await chunk in try await orchestrator.generateStream(request: GenerationRequest(prompt: "hi", maxTokens: 4_096)) {
            chunks.append(chunk)
        }
        guard case .text("cloud ") = chunks.first, case .metadata(let meta) = chunks.last else {
            Issue.record("unexpected chunks \(chunks)")
            return
        }
        #expect(meta.finishReason == .error)
        #expect(meta.finishReason.isTruncated)
    }

    @Test("Utility follows w_cap·Φ·(1−δ) − w_lat·τ/τ_max − w_cost·C/B_rem")
    func utilityFormula() {
        let arbiter = HybridArbiter(capabilityWeight: 2, latencyWeight: 1, costWeight: 3, maxLatencySeconds: 10)
        let candidate = HybridCandidate(capability: 0.5, degradation: 0.2, latencySeconds: 5, cost: 1)
        #expect(abs(arbiter.utility(candidate, remainingBudget: 4) - (2 * 0.5 * 0.8 - 0.5 - 3 * 0.25)) < 1e-12)
        #expect(arbiter.utility(candidate, remainingBudget: 0) == -.infinity)
    }
}

@Suite("ModelRouter as the cloud fallback (#22)")
struct ModelRouterCloudFallbackTests {
    func router(client: Bool = true, tracker: CostTracker? = nil) throws -> ModelRouter {
        ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()),
            cloudClient: client ? try OpenRouterClient(config: .init(
                apiKey: "k", baseURL: URL(string: "http://127.0.0.1:9")!, timeout: 2, maxRetries: 0)) : nil,
            strategy: .cloudOnly, costTracker: tracker, fallbackCloudModel: "openai/gpt-4-turbo")
    }

    @Test("No assessment without a cloud client, with tools, a response format or an assistant prefill, or once the budget is exhausted")
    func noAssessment() async throws {
        let offline = try router(client: false)
        let online = try router()
        #expect(await offline.assess(GenerationRequest(prompt: "hi")) == nil)
        #expect(await online.assess(GenerationRequest(prompt: "hi", tools: [ToolDefinition(name: "t")])) == nil)
        let prefilled = [ChatMessageInput(role: .user, text: "hi"), ChatMessageInput(role: .assistant, text: "{")]
        #expect(await online.assess(GenerationRequest(prompt: "hi", messages: prefilled)) == nil)
        #expect(await online.assess(GenerationRequest(prompt: "hi", messages: [prefilled[0]])) != nil)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("cost-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let tracker = try CostTracker(storageURL: file, budget: .init(dailyLimit: 1, monthlyLimit: 10))
        _ = try? await tracker.recordUsage(model: "m", cost: 1)
        let broke = try router(tracker: tracker)
        #expect(await broke.assess(GenerationRequest(prompt: "hi")) == nil)
    }

    @Test("The assessment prices the request and reports the remaining budget")
    func assessment() async throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("cost-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let tracker = try CostTracker(storageURL: file, budget: .init(dailyLimit: 5, monthlyLimit: 50))
        try await tracker.recordUsage(model: "m", cost: 1)
        let funded = try router(tracker: tracker)
        let assessment = try #require(await funded.assess(GenerationRequest(prompt: "hi", maxTokens: 1_000)))
        #expect(abs(assessment.estimatedCost - 1_000 / 1_000_000.0 * 30) < 1e-9)
        #expect(assessment.remainingBudget == 4)
        #expect(assessment.capabilityGap == 0)
    }

    @Test("A model without a known price is reported, so an unmetered budget is not silent")
    func priceKnowledge() async throws {
        let router = try router()
        #expect(await router.knowsPrice(of: "openai/gpt-4-turbo"))
        #expect(await !router.knowsPrice(of: "qwen3-235b"))
    }

    @Test("Chat turns reach the cloud with their roles; tool turns become readable text, an empty trailing assistant turn is dropped as it is locally")
    func cloudMessages() {
        let request = GenerationRequest(prompt: "flat", messages: [
            ChatMessageInput(role: .system, text: "be brief"),
            ChatMessageInput(role: .user, text: "weather?"),
            ChatMessageInput(role: .assistant, text: "", toolCalls: [ToolCallOutput(id: "1", name: "get_weather", arguments: "{}")]),
            ChatMessageInput(role: .tool, text: "sunny", toolCallId: "1", toolName: "get_weather"),
        ])
        let messages = ModelRouter.cloudMessages(request)
        #expect(messages.map(\.role) == ["system", "user", "assistant", "user"])
        #expect(messages[2].content == "get_weather({})")
        #expect(messages[3].content == "Result of get_weather: sunny")
        #expect(ModelRouter.cloudMessages(GenerationRequest(prompt: "flat")).map(\.content) == ["flat"])
        let emptyPrefill = GenerationRequest(prompt: "flat", messages: [
            ChatMessageInput(role: .user, text: "weather?"), ChatMessageInput(role: .assistant, text: "")])
        #expect(ModelRouter.cloudMessages(emptyPrefill).map(\.role) == ["user"])
    }
}
