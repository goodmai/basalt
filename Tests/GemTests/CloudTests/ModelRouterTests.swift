import Testing
import Foundation
@testable import GemCore

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
        
        await #expect(throws: GemError.self) {
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
        
        await #expect(throws: GemError.self) {
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
}
