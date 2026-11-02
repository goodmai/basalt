import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("MemoryCoordinator on a real model: eviction, critical admission, step-by-step restore, deferred apply (#220)", .serialized)
struct MemoryCoordinatorE2ETests {
    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_MEMORY_MODEL_PATH"]
    }

    static let budget = 2 << 30

    static func document(_ seed: Int, words: Int) -> String {
        let vocabulary = "kernel buffer cache tensor stream decode anchor memory budget window".split(separator: " ")
        return "Document \(seed). Summarise it in one sentence.\n"
            + (0..<words).map { "\(vocabulary[($0 * 7 + seed) % vocabulary.count])\(($0 + seed) % 97)" }.joined(separator: " ")
    }

    static func cached(_ prompt: String, maxTokens: Int = 16) -> GenerationRequest {
        GenerationRequest(prompt: prompt, maxTokens: maxTokens, temperature: 0, cacheKey: "agent")
    }

    static func cold(_ prompt: String, maxTokens: Int = 16) -> GenerationRequest {
        GenerationRequest(prompt: prompt, maxTokens: maxTokens, temperature: 0)
    }

    struct Rig {
        let engine: MLXInferenceEngine
        let orchestrator: ModelOrchestratorActor
        let source: FakeMemoryPressure
        let time: FakeElapsed
        let coordinator: MemoryCoordinator

        func pressure(_ pressure: MemoryPressure, available: Int64 = 16 << 30) async -> Bool {
            source.set(pressure, available: available)
            let directive = await coordinator.sample().directive
            return await GenerationGateTests.eventually { await engine.appliedMemoryDirective == directive }
        }

        func calm(for duration: Duration) async {
            source.set(.normal)
            await coordinator.sample()
            time.advance(duration)
            await coordinator.sample()
        }

        var tree: PromptCacheTree { get async { await engine.promptCaches } }
    }

    static func rig() async throws -> Rig {
        let engine = MLXInferenceEngine(promptCacheBytes: budget)
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: try #require(modelPath))
        let source = FakeMemoryPressure()
        let time = FakeElapsed()
        let coordinator = MemoryCoordinatorTests.coordinator(source, time)
        await orchestrator.coordinateMemory(with: coordinator)
        return Rig(engine: engine, orchestrator: orchestrator, source: source, time: time, coordinator: coordinator)
    }

    @Test("A warning evicts least recently used prompts down to a quarter of the budget; the kept prompt answers as before the pressure and the evicted one as when it was first stored",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_MEMORY_MODEL_PATH"))
    func warningEvictsLeastRecentlyUsed() async throws {
        let rig = try await Self.rig()
        let documents = (0..<4).map { Self.document($0, words: 600) }
        var filledAnswers: [String] = []
        for document in documents { filledAnswers.append(try await rig.orchestrator.generate(request: Self.cached(document)).generatedText) }
        let filled = await rig.tree
        try #require(filled.byteCount > Self.budget / 4, "four prompts must outgrow the tight budget, got \(filled.byteCount) bytes")
        let evictionsBefore = filled.evictions
        let warmReuse = try await rig.orchestrator.generate(request: Self.cached(documents[3])).generatedText

        #expect(await rig.pressure(.warning))
        let tight = await rig.tree
        #expect(tight.byteBudget == Self.budget / 4)
        #expect(tight.byteCount <= Self.budget / 4)
        #expect(tight.evictions > evictionsBefore)
        let reported = try #require(await rig.orchestrator.telemetrySnapshot().memory?.limits)
        #expect(reported.promptCacheBytes == tight.byteCount && reported.promptCacheBudgetBytes == tight.byteBudget)

        var reusedTokens: [Int] = []
        for (document, expected) in [(documents[3], warmReuse), (documents[0], filledAnswers[0])] {
            let reused = try await rig.orchestrator.generate(request: Self.cached(document))
            reusedTokens.append(reused.cachedPromptTokens)
            #expect(reused.generatedText == expected, "cached \(reused.cachedPromptTokens)/\(reused.promptTokens): \(reused.generatedText.debugDescription) vs \(expected.debugDescription)")
            #expect(await rig.tree.byteCount <= Self.budget / 4)
        }
        #expect(reusedTokens[0] > reusedTokens[1], "newest kept \(reusedTokens[0]) tokens, oldest \(reusedTokens[1])")
        await rig.orchestrator.shutdown()
    }

    @Test("Critical drops the whole prompt cache, refuses an 8K-token prompt with 503 and still answers a short one exactly",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_MEMORY_MODEL_PATH"))
    func criticalDropsCacheAndRefusesLongPrompts() async throws {
        let rig = try await Self.rig()
        let short = Self.document(7, words: 200)
        let expected = try await rig.orchestrator.generate(request: Self.cold(short)).generatedText
        _ = try await rig.orchestrator.generate(request: Self.cached(Self.document(1, words: 1_200)))
        #expect(await rig.tree.byteCount > 0)

        #expect(await rig.pressure(.critical))
        #expect(await rig.tree.byteBudget == 0)
        #expect(await rig.tree.byteCount == 0)

        let long = Self.cached(Self.document(2, words: 4_000))
        #expect(try #require(await rig.engine.promptTokenCount(for: long)) >= MemoryDirective.longContextTokens)
        #expect(await MemoryCoordinatorTests.refusedForMemory(rig.orchestrator, long))

        let answered = try await rig.orchestrator.generate(request: Self.cached(short))
        #expect(answered.generatedText == expected)
        #expect(answered.cachedPromptTokens == 0)
        #expect(await rig.tree.byteCount == 0)
        #expect(try #require(await rig.orchestrator.telemetrySnapshot().memory).refusedAdmissions == 1)
        await rig.orchestrator.shutdown()
    }

    @Test("After critical, every 30 s of normal pressure restores one step: budget 0 → ¼ → ½ → full, and a restored cache is reused again",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_MEMORY_MODEL_PATH"))
    func restoresOneStepPerCalmPeriod() async throws {
        let rig = try await Self.rig()
        #expect(await rig.pressure(.critical))

        await rig.calm(for: .seconds(29))
        #expect(await rig.coordinator.state.directive == .critical)
        for (step, share) in [(MemoryDirective.tight, 4), (.restoring, 2), (.relaxed, 1)] {
            rig.time.advance(.seconds(1))
            await rig.coordinator.sample()
            #expect(await GenerationGateTests.eventually { await rig.engine.appliedMemoryDirective == step })
            #expect(await rig.tree.byteBudget == Self.budget / share)
            rig.time.advance(.seconds(29))
        }

        let document = Self.document(3, words: 1_200)
        let first = try await rig.orchestrator.generate(request: Self.cached(document))
        let second = try await rig.orchestrator.generate(request: Self.cached(document))
        #expect(first.cachedPromptTokens == 0)
        #expect(second.cachedPromptTokens > first.promptTokens * 9 / 10, "cached \(second.cachedPromptTokens) of \(first.promptTokens)")
        #expect(second.generatedText == first.generatedText)
        await rig.orchestrator.shutdown()
    }

    @Test("A warning during a generation waits for it to finish, leaves its output unchanged, and is applied when it completes",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_MEMORY_MODEL_PATH"))
    func directiveDuringGenerationIsDeferred() async throws {
        let rig = try await Self.rig()
        let document = Self.document(5, words: 1_200)
        let expected = try await rig.orchestrator.generate(request: Self.cold(document, maxTokens: 200)).generatedText

        let orchestrator = rig.orchestrator
        let running = Task { try await orchestrator.generate(request: Self.cached(document, maxTokens: 200)) }
        #expect(await GenerationGateTests.eventually { await orchestrator.gate.inFlight == 1 })
        rig.source.set(.warning)
        #expect(await rig.coordinator.sample().directive == .tight)
        #expect(await rig.engine.appliedMemoryDirective == .relaxed)

        let response = try await running.value
        #expect(response.generatedText == expected)
        #expect(await rig.engine.appliedMemoryDirective == .tight)
        #expect(await rig.tree.byteCount <= Self.budget / 4)
        await rig.orchestrator.shutdown()
    }
}
