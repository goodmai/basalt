import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("ModelRouterTests")
struct ModelRouterTests {
    
    // MARK: - Test Fixtures
    
    private func createTestRouter(
        strategy: ModelRouter.RoutingStrategy = .auto,
        mockRAM: Int64 = 16 * 1024 * 1024 * 1024 // 16 GB
    ) async throws -> ModelRouter {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)
        
        let config = OpenRouterClient.Config(
            apiKey: "test-key",
            baseURL: URL(string: "https://openrouter.ai/api/v1")!,
            timeout: 60.0,
            maxRetries: 3
        )
        let client = try OpenRouterClient(config: config)
        
        return ModelRouter(
            localOrchestrator: orchestrator,
            cloudClient: client,
            strategy: strategy,
            mockAvailableRAM: mockRAM
        )
    }
    
    private func createTestRouterWithoutCloud() async throws -> ModelRouter {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)
        
        return ModelRouter(
            localOrchestrator: orchestrator,
            cloudClient: nil,
            strategy: .auto
        )
    }

    // MARK: - Tests
    
    @Test("Strategy .cloudOnly forces cloud model mapping")
    func testCloudOnlyStrategy() async throws {
        let router = try await createTestRouter(strategy: .cloudOnly)
        
        let decision = try await router.makeRoutingDecision(modelId: "gpt-4")
        
        if case .cloud(let provider, let models) = decision {
            #expect(provider == .openrouter)
            #expect(models.first == "openai/gpt-4-turbo")
        } else {
            Issue.record("Expected cloud decision")
        }
    }
    
    @Test("Strategy .localOnly forces local model")
    func testLocalOnlyStrategy() async throws {
        let router = try await createTestRouter(strategy: .localOnly)
        
        let decision = try await router.makeRoutingDecision(modelId: "gpt-4")
        
        if case .local(let path) = decision {
            #expect(path == "gpt-4")
        } else {
            Issue.record("Expected local decision")
        }
    }
    
    @Test("Auto strategy routes cloud-only model to cloud")
    func testAutoStrategyCloudOnlyModel() async throws {
        let router = try await createTestRouter(strategy: .auto)
        
        let decision = try await router.makeRoutingDecision(modelId: "claude-3.5")
        
        if case .cloud(let provider, let models) = decision {
            #expect(provider == .openrouter)
            #expect(models.first == "anthropic/claude-3.5-sonnet")
        } else {
            Issue.record("Expected cloud decision")
        }
    }
    
    @Test("Auto strategy routes unknown model to cloud if cloud configured")
    func testAutoStrategyUnknownModelWithCloud() async throws {
        let router = try await createTestRouter(strategy: .auto)
        
        let decision = try await router.makeRoutingDecision(modelId: "unknown-model-xyz")
        
        if case .cloud(_, let models) = decision {
            #expect(models.first == "unknown-model-xyz")
        } else {
            Issue.record("Expected cloud decision")
        }
    }
    
    @Test("Auto strategy throws error for unknown model if cloud not configured")
    func testAutoStrategyUnknownModelWithoutCloud() async throws {
        let router = try await createTestRouterWithoutCloud()
        
        await #expect(throws: BasaltError.self) {
            _ = try await router.makeRoutingDecision(modelId: "unknown-model-xyz")
        }
    }
    
    @Test("Auto strategy falls back to cloud on insufficient RAM for local model")
    func testAutoStrategyFallbackOnOOM() async throws {
        // Mock 1GB RAM, model needs more
        let router = try await createTestRouter(strategy: .auto, mockRAM: 1 * 1024 * 1024 * 1024)
        
        // Register a local model that requires 4GB
        await router.registerLocalModel(id: "large-local", estimatedRAM: 4000)
        
        let decision = try await router.makeRoutingDecision(modelId: "large-local")
        
        if case .cloud(let provider, let models) = decision {
            #expect(provider == .openrouter)
            #expect(models.first == "large-local") // Fallback uses same ID unless mapped
        } else {
            Issue.record("Expected cloud fallback decision due to OOM")
        }
    }
    
    @Test("Auto strategy uses local model when RAM is sufficient")
    func testAutoStrategySufficientRAM() async throws {
        // Mock 16GB RAM, model needs 4GB
        let router = try await createTestRouter(strategy: .auto, mockRAM: 16 * 1024 * 1024 * 1024)
        
        await router.registerLocalModel(id: "small-local", estimatedRAM: 4000)
        
        let decision = try await router.makeRoutingDecision(modelId: "small-local")
        
        if case .local(let path) = decision {
            #expect(path == "small-local")
        } else {
            Issue.record("Expected local decision")
        }
    }
    
    @Test("Auto strategy throws on OOM when cloud is not configured")
    func testAutoStrategyThrowsOOMWithoutCloud() async throws {
        // Mock 1GB RAM, model needs more
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)
        
        let router = ModelRouter(
            localOrchestrator: orchestrator,
            cloudClient: nil,
            strategy: .auto,
            mockAvailableRAM: 1 * 1024 * 1024 * 1024
        )
        
        await router.registerLocalModel(id: "large-local", estimatedRAM: 4000)
        
        await #expect(throws: BasaltError.self) {
            _ = try await router.makeRoutingDecision(modelId: "large-local")
        }
    }
    
    @Test("Cost estimation uses model specific rates")
    func testCostEstimation() async throws {
        let router = try await createTestRouter()
        
        // GPT-4: 1M tokens = $10 input, $30 output. Let's say 1000 input, 1000 output.
        // That's $0.01 + $0.03 = $0.04
        let gpt4Cost = await router.estimateCost(model: "gpt-4", promptTokens: 1000, completionTokens: 1000)
        #expect(abs(gpt4Cost - 0.04) < 0.0001)
        
        // Claude 3.5: 1M = $3 input, $15 output. 1000/1000 = $0.003 + $0.015 = $0.018
        let claudeCost = await router.estimateCost(model: "claude-3.5", promptTokens: 1000, completionTokens: 1000)
        #expect(abs(claudeCost - 0.018) < 0.0001)
    }

    @Test("routeStream hands the routed local model to the orchestrator")
    func routeStreamForwardsLocalModelId() async throws {
        let router = try await createTestRouter(strategy: .localOnly)
        do {
            _ = try await router.routeStream(request: GenerationRequest(prompt: "hi"), preferredModel: "uncached-model-xyz")
            Issue.record("Expected the orchestrator to try switching to uncached-model-xyz")
        } catch BasaltError.modelNotCached(let identifier) {
            #expect(identifier == "uncached-model-xyz")
        }
    }

    @Test("route hands the routed local model to the orchestrator")
    func routeForwardsLocalModelId() async throws {
        let router = try await createTestRouter(strategy: .localOnly)
        do {
            _ = try await router.route(request: GenerationRequest(prompt: "hi"), preferredModel: "uncached-model-xyz")
            Issue.record("Expected the orchestrator to try switching to uncached-model-xyz")
        } catch BasaltError.modelNotCached(let identifier) {
            #expect(identifier == "uncached-model-xyz")
        }
    }

    @Test("A cloud stream that fails mid-way surfaces the error instead of finishing cleanly")
    func cloudStreamFailurePropagates() async throws {
        struct Disconnected: Error {}
        let chunk = try JSONDecoder().decode(
            OpenRouterClient.ChatStreamResponse.self,
            from: Data(#"{"id":"1","model":"m","choices":[{"delta":{"content":"partial"}}]}"#.utf8)
        )
        let source = AsyncThrowingStream<OpenRouterClient.ChatStreamResponse, Error> { continuation in
            continuation.yield(chunk)
            continuation.finish(throwing: Disconnected())
        }
        var texts: [String] = []
        await #expect(throws: Disconnected.self) {
            for try await item in ModelRouter.relayCloudStream(source, startTime: .now) {
                if case .text(let text) = item { texts.append(text) }
            }
        }
        #expect(texts == ["partial"])
    }

    @Test("Available system memory is the free part of physical RAM, not all of it")
    func availableSystemMemoryIsBelowPhysical() {
        let available = SystemProfiler.availableSystemMemory()
        #expect(available > 0)
        #expect(available < Int64(ProcessInfo.processInfo.physicalMemory))
    }

    private func unreachableCloudClient() throws -> OpenRouterClient {
        try OpenRouterClient(config: .init(
            apiKey: "test-key", baseURL: URL(string: "http://127.0.0.1:9")!, timeout: 2, maxRetries: 0
        ))
    }

    private func cloudCalls(_ client: OpenRouterClient) async -> Int {
        let metrics = await client.getMetrics()
        return metrics.requests + metrics.errors
    }

    private func costRouting() throws -> ModelRouter.CostRouting {
        ModelRouter.CostRouting(
            policy: StickyRoutingPolicy(strongModel: "gpt-4", weakModel: "small-local", threshold: 0.5, cacheStickinessPenalty: 1),
            scorer: try BradleyTerryRouterTests.scorer()
        )
    }

    @Test("Inside one warm session the router keeps the model when a short hard step does not beat the switch penalty")
    func warmSessionKeepsModel() async throws {
        let router = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()),
            cloudClient: nil, costRouting: try costRouting()
        )
        let session = ModelRouter.RoutingContext(sessionId: "agent-1")
        let longEasy = GenerationRequest(prompt: Array(repeating: BradleyTerryRouterTests.easyWord, count: 8_000).joined(separator: " "))
        let shortHard = GenerationRequest(prompt: BradleyTerryRouterTests.hardWord)

        #expect(await router.resolveModel(nil, request: longEasy, context: session) == "small-local")
        #expect(await router.resolveModel(nil, request: shortHard, context: session) == "small-local")
        #expect(await router.resolveModel(nil, request: shortHard, context: .init(sessionId: "fresh")) == "gpt-4")
        #expect(await router.resolveModel("claude-3.5", request: shortHard, context: session) == "claude-3.5")
    }

    @Test("Once a session budget is exhausted the cloud is not called and the answer is served locally")
    func exhaustedSessionBudgetServesLocally() async throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("cost-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let tracker = try CostTracker(storageURL: file, budget: .init(dailyLimit: 100, monthlyLimit: 1_000))
        await tracker.setMaxBudget(0.5, for: .session("agent-1"))
        try await tracker.recordUsage(model: "openai/gpt-4-turbo", cost: 0.5, scopes: [.session("agent-1")])

        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "small-local")
        let client = try unreachableCloudClient()
        let router = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: engine), cloudClient: client,
            costTracker: tracker, localFallbackModel: "small-local"
        )
        let request = GenerationRequest(prompt: "hi")
        let context = ModelRouter.RoutingContext(sessionId: "agent-1")

        let response = try await router.route(request: request, preferredModel: "gpt-4", context: context)
        #expect(response.generatedText == "Mock response")
        for try await _ in try await router.routeStream(request: request, preferredModel: "gpt-4", context: context) {}
        #expect(await cloudCalls(client) == 0)
    }

    @Test("An exhausted virtual-key budget without a local fallback refuses instead of calling the cloud")
    func exhaustedKeyBudgetWithoutFallbackRefuses() async throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("cost-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let tracker = try CostTracker(storageURL: file, budget: .init(dailyLimit: 100, monthlyLimit: 1_000))
        await tracker.setMaxBudget(0, for: .virtualKey("team-a"))
        let client = try unreachableCloudClient()
        let router = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()), cloudClient: client, costTracker: tracker
        )
        await #expect(throws: BasaltError.self) {
            _ = try await router.route(request: GenerationRequest(prompt: "hi"), preferredModel: "gpt-4", context: .init(virtualKey: "team-a"))
        }
        #expect(await cloudCalls(client) == 0)
    }

    @Test("When every model in the cloud chain fails the request falls through to the local fallback")
    func failedCloudChainFallsBackLocally() async throws {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "small-local")
        let client = try unreachableCloudClient()
        let router = ModelRouter(
            localOrchestrator: ModelOrchestratorActor(engine: engine), cloudClient: client, localFallbackModel: "small-local"
        )
        let response = try await router.route(request: GenerationRequest(prompt: "hi"), preferredModel: "gpt-4")
        #expect(response.generatedText == "Mock response")
        #expect(await cloudCalls(client) == 2)
    }
}
