import Testing
import Foundation
import ArgumentParser
@testable import BasaltCore
@testable import BasaltCloud
@testable import BasaltCLI

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

    func assess(_ request: GenerationRequest, localModel: String) async -> CloudAssessment? { assessment }
    func recordLocalOutcome(_ request: GenerationRequest, response: GenerationResponse, localModel: String) async {}
    var routingBias: [String: [String: Double]]? { nil }

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
        #expect(await orchestrator.cloudFallbackReason(for: GenerationRequest(prompt: "t", maxTokens: 4_096).with { $0.isServiceRequest = true }) == nil)
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
            cloudClient: client ? try OpenRouterClient(
                config: .init(apiKey: "k", baseURL: URL(string: "http://127.0.0.1:9")!, timeout: 2, maxRetries: 0),
                configuration: StubURLProtocol.configuration) : nil,
            options: .init(strategy: .cloudOnly, costTracker: tracker, fallbackCloudModel: "openai/gpt-4-turbo"))
    }

    @Test("No assessment without a cloud client, with tools, a response format or an assistant prefill, or once the budget is exhausted")
    func noAssessment() async throws {
        let offline = try router(client: false)
        let online = try router()
        #expect(await offline.assess(GenerationRequest(prompt: "hi"), localModel: "local") == nil)
        #expect(await online.assess(GenerationRequest(prompt: "hi", tools: [ToolDefinition(name: "t")]), localModel: "local") == nil)
        let prefilled = [ChatMessageInput(role: .user, text: "hi"), ChatMessageInput(role: .assistant, text: "{")]
        #expect(await online.assess(GenerationRequest(prompt: "hi", messages: prefilled), localModel: "local") == nil)
        #expect(await online.assess(GenerationRequest(prompt: "hi", messages: [prefilled[0]]), localModel: "local") != nil)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("cost-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let tracker = try CostTracker(storageURL: file, budget: .init(dailyLimit: 1, monthlyLimit: 10))
        _ = try? await tracker.recordUsage(model: "m", cost: 1)
        let broke = try router(tracker: tracker)
        #expect(await broke.assess(GenerationRequest(prompt: "hi"), localModel: "local") == nil)
    }

    @Test("The assessment prices the request and reports the remaining budget")
    func assessment() async throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("cost-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let tracker = try CostTracker(storageURL: file, budget: .init(dailyLimit: 5, monthlyLimit: 50))
        try await tracker.recordUsage(model: "m", cost: 1)
        let funded = try router(tracker: tracker)
        let assessment = try #require(await funded.assess(GenerationRequest(prompt: "hi", maxTokens: 1_000), localModel: "local"))
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

@Suite("Ordered fallback tiers: a free LAN endpoint before the paid cloud (#142)")
struct FallbackTierTests {
    static let usage = #""usage":{"prompt_tokens":1000,"completion_tokens":1000,"total_tokens":2000}"#

    static func answering(_ text: String) -> StubURLProtocol.Handler {
        { _, body in
            guard let body, String(decoding: body, as: UTF8.self).contains(#""stream":true"#) else {
                let reply = #"{"id":"r","model":"m","choices":[{"message":{"role":"assistant","content":"\#(text)"},"finish_reason":"stop"}],\#(usage)}"#
                return (200, [:], Data(reply.utf8))
            }
            let chunk = #"{"id":"r","model":"m","choices":[{"delta":{"content":"\#(text)"},"finish_reason":"stop"}],\#(usage)}"#
            return (200, ["Content-Type": "text/event-stream"], Data("data: \(chunk)\n\ndata: [DONE]\n\n".utf8))
        }
    }

    static func failing(_ status: Int) -> StubURLProtocol.Handler {
        { _, _ in (status, [:], Data("tier says no".utf8)) }
    }

    static func host(_ handler: StubURLProtocol.Handler?) -> String {
        handler.map(StubURLProtocol.host) ?? "unreachable-\(UUID().uuidString.lowercased()).test"
    }

    static func url(_ host: String) -> URL { URL(string: "http://\(host)/v1")! }

    struct Served {
        let orchestrator: ModelOrchestratorActor
        let lan: String
        let cloud: String
        let costURL: URL

        func spent() async throws -> Double {
            try await CostTracker(storageURL: costURL, budget: .init(dailyLimit: 5, monthlyLimit: 50)).getStats().dailyUsage
        }
    }

    static func localOrchestrator(_ fallback: some CloudFallback) async throws -> ModelOrchestratorActor {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "local")
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        await orchestrator.setCloudFallback(fallback, freeMemory: { 1 << 20 })
        return orchestrator
    }

    func serve(lan: StubURLProtocol.Handler?, cloud: StubURLProtocol.Handler?) async throws -> Served {
        let lanHost = Self.host(lan)
        let cloudHost = Self.host(cloud)
        let costURL = FileManager.default.temporaryDirectory.appendingPathComponent("cost-\(UUID().uuidString).json")
        var settings = ServeCommand.CloudFallbackSettings(
            model: "qwen3-235b", baseURL: Self.url(lanHost), apiKey: "", localContextLimit: nil)
        settings.tiers = [
            .init(model: "qwen3-235b", baseURL: Self.url(lanHost), apiKey: "", free: true),
            .init(model: "openai/gpt-4-turbo", baseURL: Self.url(cloudHost), apiKey: "sk-cloud"),
        ]
        let tiers = try await ServeCommand.cloudFallbackRouter(
            settings, orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()), costURL: costURL,
            session: StubURLProtocol.configuration)
        return Served(orchestrator: try await Self.localOrchestrator(tiers), lan: lanHost, cloud: cloudHost, costURL: costURL)
    }

    static func streamedText(_ orchestrator: ModelOrchestratorActor, _ request: GenerationRequest) async throws -> String {
        var text = ""
        for await chunk in try await orchestrator.generateStream(request: request) {
            if case .text(let piece) = chunk { text += piece }
        }
        return text
    }

    let tooBig = GenerationRequest(prompt: "hi", maxTokens: 4_096)

    @Test("A request that does not fit locally goes to the free LAN tier, costs $0 and never sees the cloud key")
    func lanTierAnswers() async throws {
        let served = try await serve(lan: Self.answering("lan answer"), cloud: Self.answering("cloud answer"))
        defer { try? FileManager.default.removeItem(at: served.costURL) }
        #expect(try await served.orchestrator.generate(request: tooBig).generatedText == "lan answer")
        #expect(try await Self.streamedText(served.orchestrator, tooBig) == "lan answer")
        #expect(StubURLProtocol.requests(for: served.cloud).isEmpty)
        let lanRequests = StubURLProtocol.requests(for: served.lan)
        #expect(lanRequests.count == 2)
        #expect(lanRequests.allSatisfy { $0.0.value(forHTTPHeaderField: "Authorization") == nil })
        #expect(try await served.spent() == 0)
    }

    @Test("A LAN tier that cannot be reached or answers 5xx hands over to the cloud on its first failure, and the cloud is charged",
          arguments: [nil, 503] as [Int?])
    func lanDownReachesCloud(lanStatus: Int?) async throws {
        let served = try await serve(lan: lanStatus.map(Self.failing), cloud: Self.answering("cloud answer"))
        defer { try? FileManager.default.removeItem(at: served.costURL) }
        #expect(try await served.orchestrator.generate(request: tooBig).generatedText == "cloud answer")
        #expect(StubURLProtocol.requests(for: served.lan).count == 1)
        #expect(abs(try await served.spent() - 0.04) < 1e-9)
        #expect(try await Self.streamedText(served.orchestrator, tooBig) == "cloud answer")
        #expect(StubURLProtocol.requests(for: served.lan).count == 2)
        #expect(StubURLProtocol.requests(for: served.cloud).count == 2)
        #expect(StubURLProtocol.requests(for: served.cloud).allSatisfy {
            $0.0.value(forHTTPHeaderField: "Authorization") == "Bearer sk-cloud"
        })
    }

    @Test("A 4xx from the LAN tier is not re-sent to the paid cloud; the local model answers", arguments: [400, 401, 429])
    func clientErrorStaysOffCloud(status: Int) async throws {
        let served = try await serve(lan: Self.failing(status), cloud: Self.answering("cloud answer"))
        defer { try? FileManager.default.removeItem(at: served.costURL) }
        #expect(try await served.orchestrator.generate(request: tooBig).generatedText == "Mock response")
        #expect(try await Self.streamedText(served.orchestrator, tooBig) == "Mock response")
        #expect(StubURLProtocol.requests(for: served.cloud).isEmpty)
        #expect(try await served.spent() == 0)
    }

    @Test("With every tier down the local model answers")
    func allTiersDown() async throws {
        let tiers = try [Self.host(nil), Self.host(Self.failing(502))].map { host in
            ModelRouter(
                localOrchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()),
                cloudClient: try OpenRouterClient(
                    config: .init(apiKey: "", baseURL: Self.url(host), timeout: 2, maxRetries: 1),
                    configuration: StubURLProtocol.configuration),
                options: .init(strategy: .cloudOnly, fallbackCloudModel: "m"))
        }
        let orchestrator = try await Self.localOrchestrator(FallbackTiers(tiers))
        #expect(try await orchestrator.generate(request: tooBig).generatedText == "Mock response")
        #expect(try await Self.streamedText(orchestrator, tooBig) == "Mock response")
    }

    @Test("Only connection errors and server errors fail over")
    func failoverClassification() {
        #expect(FallbackTiers.failsOver(URLError(.cannotConnectToHost)))
        #expect(FallbackTiers.failsOver(BasaltError.modelInferenceError(details: "HTTP 503")))
        #expect(!FallbackTiers.failsOver(BasaltError.invalidRequestStructure(details: "HTTP 400")))
        #expect(!FallbackTiers.failsOver(BasaltError.authenticationFailed(details: "Invalid API key")))
        #expect(!FallbackTiers.failsOver(BasaltError.rateLimitExceeded(retryAfter: nil)))
    }

    func resolve(_ json: String, model: String? = nil, env: [String: String] = [:]) throws -> ServeCommand.CloudFallbackSettings? {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-\(UUID().uuidString).json")
        try Data(json.utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        return try ServeCommand.resolveCloudFallback(
            .init(model: model), configFile: file.path, environment: env, savedOpenRouterKey: { nil })
    }

    @Test("cloud_fallback.tiers resolves in order with a key per tier; the OpenRouter key never reaches the LAN; a flag replaces the list")
    func tiersFromConfig() throws {
        let json = #"{"cloud_fallback": {"tiers": [{"base_url": "http://192.168.1.20:8000/v1", "model": "qwen3-235b", "free": true}, {"base_url": "https://llm.lan/v1", "model": "mid", "api_key_env": "LAN_KEY"}, {"base_url": "https://openrouter.ai/api/v1", "model": "anthropic/claude-sonnet-4"}]}}"#
        let settings = try #require(try resolve(json, env: ["OPENROUTER_API_KEY": "sk-or", "LAN_KEY": "lan"]))
        #expect(settings.endpoints == [
            .init(model: "qwen3-235b", baseURL: URL(string: "http://192.168.1.20:8000/v1")!, apiKey: "", free: true),
            .init(model: "mid", baseURL: URL(string: "https://llm.lan/v1")!, apiKey: "lan"),
            .init(model: "anthropic/claude-sonnet-4", baseURL: URL(string: "https://openrouter.ai/api/v1")!, apiKey: "sk-or"),
        ])
        #expect(settings.model == "qwen3-235b")
        let flagged = try #require(try resolve(json, model: "other", env: ["OPENROUTER_API_KEY": "sk-or"]))
        #expect(flagged.endpoints.map(\.model) == ["other"])
        #expect(flagged.tiers.isEmpty)
    }

    @Test("Bad tiers are errors, and the error never repeats a secret")
    func badTiers() throws {
        for json in [
            #"{"cloud_fallback": {"tiers": []}}"#,
            #"{"cloud_fallback": {"tiers": ["http://x/v1"]}}"#,
            #"{"cloud_fallback": {"tiers": [{"model": "m"}]}}"#,
            #"{"cloud_fallback": {"tiers": [{"base_url": "http://x/v1"}]}}"#,
            #"{"cloud_fallback": {"tiers": [{"base_url": "ftp://x/v1", "model": "m"}]}}"#,
            #"{"cloud_fallback": {"tiers": [{"base_url": "http://x/v1", "model": "m", "free": "yes"}]}}"#,
            #"{"cloud_fallback": {"tiers": [{"base_url": "https://openrouter.ai/api/v1", "model": "m"}]}}"#,
        ] {
            #expect(throws: ValidationError.self, "\(json)") { try resolve(json) }
        }
        for json in [
            #"{"cloud_fallback": {"tiers": [{"base_url": "http://user:hunter2-secret@x/v1", "model": "m"}]}}"#,
            #"{"cloud_fallback": {"tiers": [{"base_url": "http://x/v1", "model": "m", "api_key": "hunter2-secret\r\nX-Evil: 1"}]}}"#,
            #"{"cloud_fallback": {"base_url": "http://x/v1", "model": "m", "api_key": "hunter2 secret"}}"#,
        ] {
            let error = #expect(throws: ValidationError.self) { try resolve(json) }
            #expect(error.map { !"\($0)".contains("hunter2") } == true)
        }
    }
}
