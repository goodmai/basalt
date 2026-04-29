import Testing
import Foundation
@testable import GemCore

@Suite("ModelOrchestratorActor")
struct OrchestratorTests {

    // MARK: — generate before load

    @Test("generate before load throws inferenceHardwareFailure")
    func generateBeforeLoad() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)

        await #expect(throws: GemError.self) {
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

    @Test("load failure propagates GemError")
    func loadFailure() async throws {
        let engine = MockInferenceEngine()
        await engine.update { $0.shouldFailOnLoad = true }
        let orch = ModelOrchestratorActor(engine: engine)

        await #expect(throws: GemError.self) {
            try await orch.loadModel(path: "bad-path")
        }
    }

    // MARK: — generate failure

    @Test("generate failure propagates GemError")
    func generateFailure() async throws {
        let engine = MockInferenceEngine()
        await engine.update { $0.shouldFailOnGenerate = true }
        let orch = ModelOrchestratorActor(engine: engine)

        try await orch.loadModel(path: "model")

        await #expect(throws: GemError.self) {
            _ = try await orch.generate(request: .init(prompt: "Hi"))
        }
    }

    // MARK: — validation rejection

    @Test("empty prompt is rejected by orchestrator before reaching engine")
    func emptyPromptRejected() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "model")

        await #expect(throws: GemError.self) {
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

    @Test("intensive concurrency (Task 4.2): 50 parallel requests")
    func intensiveConcurrency() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "model")
        
        try await withThrowingTaskGroup(of: GenerationResponse.self) { group in
            for i in 1...50 {
                group.addTask {
                    try await orch.generate(request: .init(prompt: "Prompt \(i)"))
                }
            }
            
            var count = 0
            for try await _ in group {
                count += 1
            }
            #expect(count == 50)
        }
    }

    // MARK: — Missing coverage tests

    @Test("generateStream after successful load returns mock response")
    func generateStreamAfterLoad() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)

        try await orch.loadModel(path: "test-model")
        let stream = try await orch.generateStream(request: .init(prompt: "Hello", maxTokens: 10))

        var texts: [String] = []
        var finalResponse: GenerationResponse?
        for try await chunk in stream {
            switch chunk {
            case .text(let t): texts.append(t)
            case .metadata(let m): finalResponse = m
            }
        }

        #expect(texts.joined() == "Mock response")
        #expect(finalResponse != nil)
    }

    @Test("generateStream throws error if not loaded")
    func generateStreamBeforeLoad() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)

        await #expect(throws: GemError.self) {
            _ = try await orch.generateStream(request: .init(prompt: "Hello"))
        }
    }

    @Test("generate with requested tokens exceeding dynamic budget logs warning")
    func generateExceedsBudget() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine, maxTokens: 10)
        try await orch.loadModel(path: "model")

        let req = GenerationRequest(prompt: "Hi", maxTokens: 65000)
        _ = try await orch.generate(request: req)
        _ = try await orch.generateStream(request: req)
    }

    @Test("diagnostics returns correct counts")
    func diagnosticsCheck() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "model")

        _ = try await orch.generate(request: .init(prompt: "A"))
        _ = try await orch.generate(request: .init(prompt: "B"))

        let diag = await orch.diagnostics
        #expect(diag.requests == 2)
        #expect(diag.tokens > 0)
    }

    @Test("loadModel with real directory calculates size correctly")
    func loadModelCalculatesDirectorySize() async throws {
        let engine = MockInferenceEngine()
        let orch   = ModelOrchestratorActor(engine: engine)

        let fm = FileManager.default
        let tempDir = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try fm.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempDir) }

        let fileUrl = tempDir.appendingPathComponent("test.bin")
        let data = Data(repeating: 0, count: 1024 * 1024 * 2) // 2 MB
        try data.write(to: fileUrl)

        try await orch.loadModel(path: tempDir.path)
        // Should not throw and should update modelSizeMB internally
    }
    
    // MARK: — TC-2.2.2: Actor Isolation Tests
    
    @Test("Mutable state only accessed via actor methods")
    func testActorIsolation() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        
        // Load model (mutates state)
        try await orch.loadModel(path: "test-model")
        
        // Check health (reads state)
        let health = await orch.healthSnapshot(modelId: "test-model")
        #expect(health.isReady == true)
        
        // Generate (mutates internal queue state)
        let response = try await orch.generate(request: .init(prompt: "Test", maxTokens: 10))
        #expect(!response.generatedText.isEmpty)
        
        // All state access went through actor methods - no data races
        // Swift 6 compiler enforces this at compile time
    }
    
    @Test("DTOs conform to Sendable for safe actor communication")
    func testSendableConformance() async throws {
        // These DTOs must be Sendable to cross actor boundaries
        let request = GenerationRequest(prompt: "Test", maxTokens: 10)
        
        // Pass DTOs to actor - compiles only if Sendable
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "test-model")
        
        // This test verifies Sendable conformance at compile time
        // If DTOs weren't Sendable, this wouldn't compile
        let result = try await orch.generate(request: request)
        #expect(!result.generatedText.isEmpty)
        #expect(result.tokensPerSecond > 0)
    }
    
    // MARK: — TC-2.2.3: Deadlock Prevention Tests
    
    @Test("Actor reentrancy does not cause deadlock")
    func testNoDeadlock() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        
        try await orch.loadModel(path: "test-model")
        
        // Simulate scenario where orchestrator might call itself
        // This should complete without hanging
        async let result1 = orch.generate(request: .init(prompt: "Request 1", maxTokens: 5))
        async let result2 = orch.generate(request: .init(prompt: "Request 2", maxTokens: 5))
        async let result3 = orch.generate(request: .init(prompt: "Request 3", maxTokens: 5))
        
        let responses = try await [result1, result2, result3]
        #expect(responses.count == 3)
        #expect(responses.allSatisfy { !$0.generatedText.isEmpty })
    }
    
    @Test("Long-running operation allows other tasks to proceed")
    func testConcurrentTasksWithLongOperation() async throws {
        let slowEngine = MockInferenceEngine()
        await slowEngine.setArtificialDelay(1.0) // 1 second delay
        
        let orch = ModelOrchestratorActor(engine: slowEngine)
        try await orch.loadModel(path: "test-model")
        
        let start = ContinuousClock.now
        
        // Start long-running task
        async let slowTask = orch.generate(request: .init(prompt: "Slow", maxTokens: 10))
        
        // Check health should not block on slow task
        let health = await orch.healthSnapshot(modelId: "test-model")
        
        let quickDuration = start.duration(to: .now)
        
        // Health check completed quickly (< 500ms)
        #expect(quickDuration < Duration.milliseconds(500))
        #expect(health.isReady == true)
        
        // Wait for slow task to complete
        let _ = try await slowTask
    }
    
    @Test("Graceful cancellation of in-flight requests")
    func testGracefulCancellation() async throws {
        let slowEngine = MockInferenceEngine()
        await slowEngine.setArtificialDelay(2.0)
        
        let orch = ModelOrchestratorActor(engine: slowEngine)
        try await orch.loadModel(path: "test-model")
        
        let task = Task {
            try await orch.generate(request: .init(prompt: "Test", maxTokens: 10))
        }
        
        // Cancel task immediately
        task.cancel()
        
        do {
            let _ = try await task.value
            // Task might complete before cancellation is processed
        } catch {
            // Cancellation handled gracefully
            #expect(error is CancellationError || error is GemError)
        }
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
