import Testing
import Foundation
import ArgumentParser
import Synchronization
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
        let withPrefill = GenerationRequest(prompt: "flat", messages: [
            ChatMessageInput(role: .user, text: "weather?"), ChatMessageInput(role: .assistant, text: "Sunny, ")])
        let prefilled = ModelRouter.cloudMessages(withPrefill)
        #expect(prefilled.map(\.role) == ["user", "assistant"])
        #expect(prefilled.last?.content == "Sunny, ")
    }
}

enum FallbackTierTestFixtures {
    static func openAIServer(models: [String], answer: String, promptTokens: Int = 100, completionTokens: Int = 50) -> String {
        StubURLProtocol.host { request, _ in
            let path = request.url?.path ?? ""
            if path.hasSuffix("/models") {
                let items = models.map { "{\"id\":\"\($0)\",\"object\":\"model\"}" }.joined(separator: ",")
                return (200, ["Content-Type": "application/json"], Data("{\"object\":\"list\",\"data\":[\(items)]}".utf8))
            }
            let body = """
            {"id":"c1","model":"\(models.first ?? "m")","choices":[{"message":{"role":"assistant","content":"\(answer)"},"finish_reason":"stop"}],\
            "usage":{"prompt_tokens":\(promptTokens),"completion_tokens":\(completionTokens),"total_tokens":\(promptTokens + completionTokens)}}
            """
            return (200, ["Content-Type": "application/json"], Data(body.utf8))
        }
    }

    static func tier(host: String, model: String, isFree: Bool) throws -> FallbackTier {
        let baseURL = URL(string: "http://\(host)/v1")!
        let client = try OpenRouterClient(
            config: .init(apiKey: isFree ? "" : "sk-cloud", baseURL: baseURL, timeout: 5, maxRetries: 1),
            configuration: StubURLProtocol.configuration)
        return FallbackTier(client: client, baseURL: baseURL, model: model, isFree: isFree)
    }

    static func stack(
        tiers: [FallbackTier], dailyLimit: Double = 5.0, probe: FallbackTierProbeConfig = FallbackTierProbeConfig(),
        sessionLogDir: URL? = nil
    ) async throws -> (ModelOrchestratorActor, CostTracker, TelemetryTracker, URL) {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "local")
        let telemetry = TelemetryTracker()
        let logWriter = sessionLogDir.map { SessionLogWriter(mode: .meta, directory: $0) }
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: telemetry, sessionLog: logWriter)
        let costURL = FileManager.default.temporaryDirectory.appendingPathComponent("cost-\(UUID().uuidString).json")
        let tracker = try CostTracker(storageURL: costURL, budget: .init(dailyLimit: dailyLimit, monthlyLimit: dailyLimit * 10))
        let options = ModelRouter.Options(strategy: .cloudOnly, costTracker: tracker).withFallbackTiers(tiers, probe: probe)
        let router = ModelRouter(localOrchestrator: orchestrator, cloudClient: nil, options: options)
        await orchestrator.setCloudFallback(router, arbiter: HybridArbiter(), freeMemory: { 0 })
        return (orchestrator, tracker, telemetry, costURL)
    }
}

@Suite("Ordered LAN-before-cloud fallback tiers (#142)")
struct OrderedFallbackTiersTests {
    @Test("Fits-nowhere request goes to free LAN tier, records stats and session log, and charges $0")
    func routesToFreeLanFirst() async throws {
        let lanHost = FallbackTierTestFixtures.openAIServer(models: ["qwen3-235b"], answer: "lan-answer", promptTokens: 200, completionTokens: 80)
        let cloudHost = FallbackTierTestFixtures.openAIServer(models: ["openai/gpt-4-turbo"], answer: "cloud-answer")
        let logDir = FileManager.default.temporaryDirectory.appendingPathComponent("slog-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: logDir) }
        let tiers = [
            try FallbackTierTestFixtures.tier(host: lanHost, model: "qwen3-235b", isFree: true),
            try FallbackTierTestFixtures.tier(host: cloudHost, model: "openai/gpt-4-turbo", isFree: false)
        ]
        let (orchestrator, tracker, telemetry, costURL) = try await FallbackTierTestFixtures.stack(tiers: tiers, sessionLogDir: logDir)
        defer { try? FileManager.default.removeItem(at: costURL) }

        let response = try await orchestrator.generate(request: GenerationRequest(prompt: "big task", maxTokens: 4_096))
        await orchestrator.shutdown()
        #expect(response.generatedText == "lan-answer")
        #expect(StubURLProtocol.requests(for: cloudHost).isEmpty)
        #expect(await tracker.getStats().dailyUsage == 0)
        #expect(await tracker.remaining(scopes: []) == 5.0)
        let stats = await telemetry.snapshot()
        #expect(stats.totalRequests == 1)
        #expect(stats.totalTokensGenerated == 80)
        #expect(stats.windows?.first?.tokens == 80)
        let logFiles = (try? FileManager.default.contentsOfDirectory(atPath: logDir.path)) ?? []
        #expect(!logFiles.isEmpty)
    }

    @Test("Hanging LAN tier times out at 200 ms, reaches paid cloud within 250 ms of probe timeout, and caches backoff")
    func deadLanReachesCloudFastAndBacksOff() async throws {
        let nowBox = Mutex(ContinuousClock.now)
        let probe = FallbackTierProbeConfig(timeout: .milliseconds(200), now: { nowBox.withLock { $0 } })
        let deadLanHost = StubURLProtocol.unresponsiveHost()
        let cloudHost = FallbackTierTestFixtures.openAIServer(
            models: ["openai/gpt-4-turbo"], answer: "cloud-answer", promptTokens: 1_000, completionTokens: 1_000)
        let tiers = [
            try FallbackTierTestFixtures.tier(host: deadLanHost, model: "qwen3-235b", isFree: true),
            try FallbackTierTestFixtures.tier(host: cloudHost, model: "openai/gpt-4-turbo", isFree: false)
        ]
        let (orchestrator, tracker, _, costURL) = try await FallbackTierTestFixtures.stack(tiers: tiers, probe: probe)
        defer { try? FileManager.default.removeItem(at: costURL) }

        let start = ContinuousClock.now
        let first = try await orchestrator.generate(request: GenerationRequest(prompt: "hi", maxTokens: 4_096))
        let firstElapsed = start.duration(to: ContinuousClock.now)
        #expect(first.generatedText == "cloud-answer")
        #expect(firstElapsed >= .milliseconds(180))
        #expect(firstElapsed < .milliseconds(450))
        #expect(StubURLProtocol.requests(for: deadLanHost).count == 1)
        #expect(await tracker.getStats().dailyUsage > 0)

        let cachedStart = ContinuousClock.now
        let second = try await orchestrator.generate(request: GenerationRequest(prompt: "again", maxTokens: 4_096))
        let cachedElapsed = cachedStart.duration(to: ContinuousClock.now)
        #expect(second.generatedText == "cloud-answer")
        #expect(cachedElapsed < .milliseconds(150))
        #expect(StubURLProtocol.requests(for: deadLanHost).count == 1)

        StubURLProtocol.setHandler(for: deadLanHost) { request, _ in
            if (request.url?.path ?? "").hasSuffix("/models") {
                return (200, ["Content-Type": "application/json"], Data(#"{"data":[{"id":"qwen3-235b"}]}"#.utf8))
            }
            return (200, ["Content-Type": "application/json"], Data(
                #"{"id":"1","model":"qwen3-235b","choices":[{"message":{"role":"assistant","content":"lan-recovered"},"finish_reason":"stop"}]}"#.utf8))
        }
        nowBox.withLock { $0 = $0.advanced(by: .seconds(2)) }
        let third = try await orchestrator.generate(request: GenerationRequest(prompt: "recovered", maxTokens: 4_096))
        #expect(third.generatedText == "lan-recovered")
    }

    @Test("LAN tier without the configured model is skipped; when both tiers are down the local error is returned")
    func missingModelAndBothDownReturnLocalError() async throws {
        let wrongModelHost = FallbackTierTestFixtures.openAIServer(models: ["other-7b"], answer: "wrong")
        let cloudHost = FallbackTierTestFixtures.openAIServer(models: ["openai/gpt-4-turbo"], answer: "cloud-ok")
        let tiers = [
            try FallbackTierTestFixtures.tier(host: wrongModelHost, model: "qwen3-235b", isFree: true),
            try FallbackTierTestFixtures.tier(host: cloudHost, model: "openai/gpt-4-turbo", isFree: false)
        ]
        let (orchestrator, _, _, costURL) = try await FallbackTierTestFixtures.stack(tiers: tiers)
        defer { try? FileManager.default.removeItem(at: costURL) }
        #expect(try await orchestrator.generate(request: GenerationRequest(prompt: "hi", maxTokens: 4_096)).generatedText == "cloud-ok")

        let bothDown = [
            try FallbackTierTestFixtures.tier(host: "missing-lan-\(UUID().uuidString).test", model: "qwen3-235b", isFree: true),
            try FallbackTierTestFixtures.tier(host: "missing-cloud-\(UUID().uuidString).test", model: "openai/gpt-4-turbo", isFree: false)
        ]
        let (offlineOrchestrator, _, _, offlineCostURL) = try await FallbackTierTestFixtures.stack(tiers: bothDown)
        defer { try? FileManager.default.removeItem(at: offlineCostURL) }
        await #expect(throws: BasaltError.self) {
            try await offlineOrchestrator.generate(request: GenerationRequest(prompt: "hi", maxTokens: 4_096))
        }
    }
}

@Suite("Ordered fallback stickiness, budgets & config (#142)")
struct OrderedFallbackStickinessAndConfigTests {
    @Test("Free LAN tier still answers when paid daily and virtual-key budgets are exhausted")
    func freeTierBypassesExhaustedBudgets() async throws {
        let lanHost = FallbackTierTestFixtures.openAIServer(models: ["qwen3-235b"], answer: "free-lan")
        let cloudHost = FallbackTierTestFixtures.openAIServer(models: ["openai/gpt-4-turbo"], answer: "paid-cloud")
        let tiers = [
            try FallbackTierTestFixtures.tier(host: lanHost, model: "qwen3-235b", isFree: true),
            try FallbackTierTestFixtures.tier(host: cloudHost, model: "openai/gpt-4-turbo", isFree: false)
        ]
        let (orchestrator, tracker, _, costURL) = try await FallbackTierTestFixtures.stack(tiers: tiers, dailyLimit: 1.0)
        defer { try? FileManager.default.removeItem(at: costURL) }
        await tracker.setMaxBudget(0.5, for: .virtualKey("vk-1"))
        try await tracker.recordUsage(model: "openai/gpt-4-turbo", cost: 1.0, scopes: [.virtualKey("vk-1")])
        #expect(await tracker.isExhausted(scopes: [.virtualKey("vk-1")]))

        let request = GenerationRequest(prompt: "still free", maxTokens: 4_096).withRouting(sessionId: nil, budgetKey: "vk-1")
        let reply = try await orchestrator.generate(request: request)
        #expect(reply.generatedText == "free-lan")
        #expect(await tracker.getStats().dailyUsage == 1.0)
    }

    @Test("Session stickiness keeps a session on the tier it started on while that tier stays healthy")
    func sessionStickinessAcrossTierRecovery() async throws {
        let nowBox = Mutex(ContinuousClock.now)
        let probe = FallbackTierProbeConfig(timeout: .milliseconds(200), now: { nowBox.withLock { $0 } })
        let lanHost = StubURLProtocol.host { _, _ in (503, [:], Data("down".utf8)) }
        let cloudHost = FallbackTierTestFixtures.openAIServer(models: ["openai/gpt-4-turbo"], answer: "cloud-sticky")
        let tiers = [
            try FallbackTierTestFixtures.tier(host: lanHost, model: "qwen3-235b", isFree: true),
            try FallbackTierTestFixtures.tier(host: cloudHost, model: "openai/gpt-4-turbo", isFree: false)
        ]
        let (orchestrator, _, _, costURL) = try await FallbackTierTestFixtures.stack(tiers: tiers, probe: probe)
        defer { try? FileManager.default.removeItem(at: costURL) }

        let s1Turn1 = GenerationRequest(prompt: "turn 1", maxTokens: 4_096).withRouting(sessionId: "session-1", budgetKey: nil)
        #expect(try await orchestrator.generate(request: s1Turn1).generatedText == "cloud-sticky")

        StubURLProtocol.setHandler(for: lanHost) { request, _ in
            if (request.url?.path ?? "").hasSuffix("/models") {
                return (200, ["Content-Type": "application/json"], Data(#"{"data":[{"id":"qwen3-235b"}]}"#.utf8))
            }
            return (200, ["Content-Type": "application/json"], Data(
                #"{"id":"1","model":"qwen3-235b","choices":[{"message":{"role":"assistant","content":"lan-fresh"},"finish_reason":"stop"}]}"#.utf8))
        }
        nowBox.withLock { $0 = $0.advanced(by: .seconds(5)) }

        let s1Turn2 = GenerationRequest(prompt: "turn 2", maxTokens: 4_096).withRouting(sessionId: "session-1", budgetKey: nil)
        #expect(try await orchestrator.generate(request: s1Turn2).generatedText == "cloud-sticky")
        let s2Turn1 = GenerationRequest(prompt: "fresh session", maxTokens: 4_096).withRouting(sessionId: "session-2", budgetKey: nil)
        #expect(try await orchestrator.generate(request: s2Turn1).generatedText == "lan-fresh")
    }

    @Test("cloud_fallback.tiers parses ordered LAN and cloud endpoints without leaking keys")
    func multiTierSettingsResolution() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("tiers-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let json = """
        {
          "cloud_fallback": {
            "local_context_limit": 16384,
            "tiers": [
              {"base_url": "http://192.168.1.20:8000/v1", "model": "qwen3-235b", "free": true},
              {"model": "anthropic/claude-sonnet-4"}
            ]
          }
        }
        """
        try Data(json.utf8).write(to: url)
        let settings = try #require(try ServeCommand.resolveCloudFallback(
            .init(), disabled: false, configFile: url.path,
            environment: ["OPENROUTER_API_KEY": "sk-or"], savedOpenRouterKey: { nil }))
        #expect(settings.tiers.count == 2)
        #expect(settings.tiers[0] == .init(baseURL: URL(string: "http://192.168.1.20:8000/v1")!, model: "qwen3-235b", apiKey: "", free: true))
        #expect(settings.tiers[1] == .init(baseURL: URL(string: "https://openrouter.ai/api/v1")!, model: "anthropic/claude-sonnet-4", apiKey: "sk-or", free: false))
        #expect(settings.paidModels == ["anthropic/claude-sonnet-4"])

        let emptyURL = FileManager.default.temporaryDirectory.appendingPathComponent("tiers-empty-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: emptyURL) }
        try Data(#"{"cloud_fallback": {"tiers": []}}"#.utf8).write(to: emptyURL)
        #expect(throws: ValidationError.self) {
            try ServeCommand.resolveCloudFallback(.init(), disabled: false, configFile: emptyURL.path, environment: [:], savedOpenRouterKey: { nil })
        }
    }
}
