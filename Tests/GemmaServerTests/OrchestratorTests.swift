import Testing
import Foundation
@testable import GemmaServerCore

@Suite("ModelOrchestratorActor")
struct OrchestratorTests {

    // MARK: — generate before load

    @Test("generate before load throws inferenceHardwareFailure")
    func generateBeforeLoad() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)

        await #expect(throws: GemmaServerError.self) {
            _ = try await orch.generate(request: .init(prompt: "Hello"))
        }
    }

    // MARK: — load success → generate

    @Test("generate after successful load returns mock response")
    func generateAfterLoad() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)

        try await orch.loadModel(path: "test-model")
        let response = try await orch.generate(request: .init(prompt: "Hello", maxTokens: 10))

        #expect(response.generatedText == "Mock response")
        #expect(response.tokensPerSecond > 0)
        #expect(response.generationTime > 0)
        #expect(response.timeToFirstToken > 0)
        #expect(response.memory.peakBytes > 0)
    }

    // MARK: — load failure

    @Test("load failure propagates GemmaServerError")
    func loadFailure() async throws {
        let engine = MockInferenceEngine()
        await engine.update { $0.shouldFailOnLoad = true }
        let orch = ModelOrchestratorActor(engine: engine)

        await #expect(throws: GemmaServerError.self) {
            try await orch.loadModel(path: "bad-path")
        }
    }

    // MARK: — generate failure

    @Test("generate failure propagates GemmaServerError")
    func generateFailure() async throws {
        let engine = MockInferenceEngine()
        await engine.update { $0.shouldFailOnGenerate = true }
        let orch = ModelOrchestratorActor(engine: engine)

        try await orch.loadModel(path: "model")

        await #expect(throws: GemmaServerError.self) {
            _ = try await orch.generate(request: .init(prompt: "Hi"))
        }
    }

    // MARK: — validation rejection

    @Test("empty prompt is rejected by orchestrator before reaching engine")
    func emptyPromptRejected() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "model")

        await #expect(throws: GemmaServerError.self) {
            _ = try await orch.generate(request: .init(prompt: ""))
        }
    }

    // MARK: — health snapshot

    @Test("healthSnapshot.isReady is false before load")
    func healthBeforeLoad() async {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)
        let health = await orch.healthSnapshot(modelId: nil)
        #expect(health.isReady == false)
        #expect(health.status  == "initializing")
    }

    @Test("healthSnapshot.isReady is true after successful load")
    func healthAfterLoad() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "model")
        let health = await orch.healthSnapshot(modelId: "model")
        #expect(health.isReady == true)
        #expect(health.status  == "ok")
        #expect(health.modelId == "model")
    }

    @Test("healthSnapshot version matches HealthResponse.version")
    func healthVersion() async {
        let orch   = ModelOrchestratorActor(engine: MockInferenceEngine())
        let health = await orch.healthSnapshot(modelId: nil)
        #expect(health.version == HealthResponse.version)
    }

    // MARK: — FIFO / concurrency

    @Test("multiple concurrent generates all succeed")
    func concurrentGenerates() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "model")

        // Swift actor serializes calls — no race, all should succeed
        async let r1 = orch.generate(request: .init(prompt: "A"))
        async let r2 = orch.generate(request: .init(prompt: "B"))
        async let r3 = orch.generate(request: .init(prompt: "C"))

        let results = try await [r1, r2, r3]
        #expect(results.count == 3)
        #expect(results.allSatisfy { $0.generatedText == "Mock response" })
    }
}

// MARK: — Helpers

extension MockInferenceEngine {
    /// Convenience: mutate state from test (since actor requires isolation)
    func update(_ block: (inout MockState) -> Void) {
        var state = MockState(
            shouldFailOnLoad: shouldFailOnLoad,
            shouldFailOnGenerate: shouldFailOnGenerate
        )
        block(&state)
        shouldFailOnLoad     = state.shouldFailOnLoad
        shouldFailOnGenerate = state.shouldFailOnGenerate
    }

    struct MockState {
        var shouldFailOnLoad: Bool
        var shouldFailOnGenerate: Bool
    }
}
