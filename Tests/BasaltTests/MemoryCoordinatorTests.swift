import Testing
import Foundation
import Synchronization
import Hummingbird
import HummingbirdTesting
import NIOCore
import MLX
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

final class FakeMemoryPressure: MemoryPressureSource {
    private let current: Mutex<MemorySample>

    init(_ pressure: MemoryPressure = .normal, available: Int64 = 16 << 30) {
        current = Mutex(MemorySample(pressure: pressure, availableBytes: available))
    }

    func set(_ pressure: MemoryPressure, available: Int64 = 16 << 30) {
        current.withLock { $0 = MemorySample(pressure: pressure, availableBytes: available) }
    }

    func sample() -> MemorySample { current.withLock { $0 } }
}

final class FakeElapsed: Sendable {
    private let elapsed = Mutex(Duration.zero)
    var now: Duration { elapsed.withLock { $0 } }
    func advance(_ by: Duration) { elapsed.withLock { $0 += by } }
}

@Suite("MemoryCoordinator: runtime reaction to macOS memory pressure (#220)")
struct MemoryCoordinatorTests {
    static let parked: @Sendable (Duration) async throws -> Void = { _ in
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        for await _ in stream {}
        continuation.finish()
        throw CancellationError()
    }

    static func coordinator(_ source: FakeMemoryPressure, _ time: FakeElapsed = FakeElapsed()) -> MemoryCoordinator {
        MemoryCoordinator(source: source, now: { time.now }, sleep: parked)
    }

    static func orchestrator(_ engine: MockInferenceEngine) async throws -> ModelOrchestratorActor {
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "m")
        await engine.setMemory(kvBudget: 1 << 40, promptTokens: 100)
        return orchestrator
    }

    static func refusedForMemory(_ orchestrator: ModelOrchestratorActor, _ request: GenerationRequest) async -> Bool {
        do {
            _ = try await orchestrator.generate(request: request)
            return false
        } catch {
            guard case .memoryPressure(let retryAfter) = error else { return false }
            return retryAfter == MemoryDirective.retryAfterSeconds && error.httpStatus == 503
        }
    }

    @Test("The kernel's pressure levels map 1 → normal, 2 → warning, 4 → critical")
    func kernelLevels() {
        #expect(MemoryPressure(kernelLevel: 0) == .normal)
        #expect(MemoryPressure(kernelLevel: 1) == .normal)
        #expect(MemoryPressure(kernelLevel: 2) == .warning)
        #expect(MemoryPressure(kernelLevel: 4) == .critical)
    }

    @Test("At normal pressure the directive stays relaxed: full limits, no cache clear, no refusals, nothing published")
    func normalChangesNothing() async {
        let time = FakeElapsed()
        let coordinator = Self.coordinator(FakeMemoryPressure(), time)
        for _ in 0..<120 {
            #expect(await coordinator.sample().directive == .relaxed)
            time.advance(.seconds(1))
        }
        let relaxed = MemoryDirective.relaxed
        #expect(relaxed.cacheLimitShare == 1 && relaxed.promptCacheShare == 1)
        #expect(!relaxed.clearsCache && !relaxed.limitsAdmission)
    }

    @Test("A warning sample tightens at once: clear the allocator cache, lower its limit, shrink the prompt cache, still admit")
    func warningTightens() async {
        let source = FakeMemoryPressure()
        let coordinator = Self.coordinator(source)
        source.set(.warning, available: 3 << 30)
        let state = await coordinator.sample()
        #expect(state.directive == .tight)
        #expect(state.sample.availableBytes == 3 << 30)
        #expect(state.directive.clearsCache)
        #expect(state.directive.cacheLimitShare < 1 && state.directive.promptCacheShare < 1)
        #expect(!state.directive.limitsAdmission)
    }

    @Test("A critical sample limits admission and drops the whole prompt-cache budget; a later warning does not loosen it")
    func criticalLimitsAdmission() async {
        let source = FakeMemoryPressure(.critical)
        let coordinator = Self.coordinator(source)
        #expect(await coordinator.sample().directive == .critical)
        #expect(MemoryDirective.critical.limitsAdmission)
        #expect(MemoryDirective.critical.promptCacheShare == 0)
        source.set(.warning)
        #expect(await coordinator.sample().directive == .critical)
    }

    @Test("Normal pressure held under 30 s keeps the limits; every further 30 s restores one step")
    func restoresStepwiseAfterThirtySeconds() async {
        let source = FakeMemoryPressure(.critical)
        let time = FakeElapsed()
        let coordinator = Self.coordinator(source, time)
        await coordinator.sample()
        source.set(.normal)
        var seen: [MemoryDirective] = []
        for _ in 0...95 {
            seen.append(await coordinator.sample().directive)
            time.advance(.seconds(1))
        }
        #expect(seen[0..<30].allSatisfy { $0 == .critical })
        #expect(seen[30..<60].allSatisfy { $0 == .tight })
        #expect(seen[60..<90].allSatisfy { $0 == .restoring })
        #expect(seen[90...].allSatisfy { $0 == .relaxed })
    }

    @Test("Alternating normal and warning samples never loosen the limits: no flapping")
    func alternatingSamplesDoNotFlap() async {
        let source = FakeMemoryPressure()
        let time = FakeElapsed()
        let coordinator = Self.coordinator(source, time)
        for second in 0..<300 {
            source.set(second.isMultiple(of: 2) ? .warning : .normal)
            #expect(await coordinator.sample().directive == .tight)
            time.advance(.seconds(1))
        }
        source.set(.warning)
        await coordinator.sample()
        source.set(.normal)
        time.advance(.seconds(29))
        #expect(await coordinator.sample().directive == .tight)
    }

    @Test("The loop samples once per 1 s tick, publishes the newest directive and finishes its stream when the sleep is cancelled")
    func loopSamplesOnEveryTick() async {
        let source = FakeMemoryPressure()
        let script = Mutex<[MemoryPressure]>([.normal, .warning, .warning, .critical])
        let intervals = Mutex<[Duration]>([])
        let coordinator = MemoryCoordinator(source: source, now: { .zero }, sleep: { interval in
            intervals.withLock { $0.append(interval) }
            guard let next = script.withLock({ $0.isEmpty ? nil : $0.removeFirst() }) else { throw CancellationError() }
            source.set(next)
        })
        await coordinator.run()
        var published: [MemoryDirective] = []
        for await state in coordinator.changes { published.append(state.directive) }
        #expect(published == [.critical])
        #expect(intervals.withLock { $0 } == Array(repeating: MemoryCoordinator.interval, count: 5))
        #expect(MemoryCoordinator.interval == .seconds(1))
    }

    @Test("At normal pressure a request goes through exactly as without the coordinator, and the engine is only handed relaxed")
    func orchestratorAtNormalIsUnchanged() async throws {
        let plainEngine = MockInferenceEngine()
        let plain = try await Self.orchestrator(plainEngine)
        _ = try await plain.generate(request: GenerationRequest(prompt: "p", maxTokens: 30_000))

        let engine = MockInferenceEngine()
        let orchestrator = try await Self.orchestrator(engine)
        await orchestrator.coordinateMemory(with: Self.coordinator(FakeMemoryPressure()))
        let request = GenerationRequest(prompt: "p", maxTokens: 30_000)
        #expect(await orchestrator.admissionRefusal(for: request) == nil)
        _ = try await orchestrator.generate(request: request)

        #expect(await engine.lastRequest?.maxTokens == plainEngine.lastRequest?.maxTokens)
        #expect(await engine.lastRequest?.maxTokens == 30_000)
        #expect(await engine.appliedDirectives.allSatisfy { $0 == .relaxed })
        let memory = try #require(await orchestrator.telemetrySnapshot().memory)
        #expect(memory.pressure == .normal && memory.step == .relaxed && memory.refusedAdmissions == 0)
        await orchestrator.shutdown()
    }

    @Test("A warning reaches the engine as a tight directive at the next admission and, while idle, without one")
    func warningIsAppliedByTheEngine() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = try await Self.orchestrator(engine)
        let source = FakeMemoryPressure()
        let coordinator = Self.coordinator(source)
        await orchestrator.coordinateMemory(with: coordinator)

        source.set(.warning)
        await coordinator.sample()
        #expect(await GenerationGateTests.eventually { await engine.appliedDirectives.last == .tight })

        source.set(.critical)
        _ = try await orchestrator.generate(request: GenerationRequest(prompt: "short", maxTokens: 10))
        #expect(await engine.appliedDirectives.last == .critical)
        await orchestrator.shutdown()
    }

    @Test("At critical pressure a long prompt is refused with 503 and Retry-After, and a short one is capped by free memory")
    func criticalRefusesLongAndCapsShort() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = try await Self.orchestrator(engine)
        let source = FakeMemoryPressure(.critical, available: 64 << 20)
        await orchestrator.coordinateMemory(with: Self.coordinator(source))

        await engine.setMemory(kvBudget: 1 << 40, promptTokens: MemoryDirective.longContextTokens)
        let long = GenerationRequest(prompt: "long", maxTokens: 100)
        let refusal = try #require(await orchestrator.admissionRefusal(for: long))
        guard case .memoryPressure(let retryAfter) = refusal else { Issue.record("expected memoryPressure, got \(refusal)"); return }
        #expect(retryAfter == MemoryDirective.retryAfterSeconds)
        #expect(refusal.httpStatus == 503)
        #expect(await Self.refusedForMemory(orchestrator, long))

        await engine.setMemory(kvBudget: 1 << 40, promptTokens: 100)
        _ = try await orchestrator.generate(request: GenerationRequest(prompt: "short", maxTokens: 60_000))
        let capped = try #require(await engine.lastRequest?.maxTokens)
        #expect(capped > 0 && capped < 60_000)

        source.set(.critical, available: 0)
        #expect(await Self.refusedForMemory(orchestrator, GenerationRequest(prompt: "short", maxTokens: 100)))
        #expect(try #require(await orchestrator.telemetrySnapshot().memory).refusedAdmissions == 3)
        await orchestrator.shutdown()
    }

    @Test("REST answers a long prompt at critical pressure with 503 and Retry-After 30, streaming or not")
    func restRefusesWithRetryAfter() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = try await Self.orchestrator(engine)
        await engine.setMemory(kvBudget: 1 << 40, promptTokens: MemoryDirective.longContextTokens)
        await orchestrator.coordinateMemory(with: Self.coordinator(FakeMemoryPressure(.critical)))
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock-model"))
        let app = Application(router: server.makeRouter())
        try await app.test(.router) { client in
            for stream in [false, true] {
                let body = ByteBuffer(string: #"{"model": "org/mock-model", "stream": \#(stream), "messages": [{"role": "user", "content": "hi"}]}"#)
                try await client.execute(uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"], body: body) { res in
                    #expect(res.status == .serviceUnavailable, "stream \(stream)")
                    #expect(res.headers[.retryAfter] == "30", "stream \(stream)")
                }
            }
        }
        await orchestrator.shutdown()
    }

    @Test("Stats carry the pressure level, the step, the applied limits and the eviction and refusal counts")
    func statsExposeTheMemoryFields() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = try await Self.orchestrator(engine)
        await orchestrator.coordinateMemory(with: Self.coordinator(FakeMemoryPressure(.warning, available: 2 << 30)))
        let snapshot = await orchestrator.telemetrySnapshot()
        let memory = try #require(snapshot.memory)
        #expect(memory.pressure == .warning)
        #expect(memory.step == .tight)
        #expect(memory.availableBytes == 2 << 30)
        #expect(memory.limits == MemoryLimits(cacheLimitBytes: 128 << 20, promptCacheBudgetBytes: 256 << 20, promptCacheEvictions: 0))

        let json = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(snapshot)) as? [String: Any])
        let encoded = try #require(json["memory"] as? [String: Any])
        #expect(encoded["pressure"] as? String == "warning")
        #expect(encoded["step"] as? Int == 2)
        #expect((encoded["limits"] as? [String: Any])?["promptCacheEvictions"] as? Int == 0)
        #expect(encoded["refusedAdmissions"] as? Int == 0)
        #expect(snapshot.formattedStatsLine.contains("Memory: warning step 2, cache limit 128.0 MB, prompt cache 0 B of 256.0 MB, evicted 0"))
        #expect(try JSONDecoder().decode(ServerTelemetrySnapshot.self, from: JSONEncoder().encode(snapshot)) == snapshot)

        let uncoordinated = try await Self.orchestrator(MockInferenceEngine())
        let off = await uncoordinated.telemetrySnapshot()
        #expect(off.memory == nil)
        #expect(!off.formattedStatsLine.contains("Memory:"))
        await orchestrator.shutdown()
    }

    @Test("The MLX engine applies a directive on its own actor: the reported cache limit and prompt-cache budget follow the step, and relaxed restores them; the global Memory.cacheLimit is not asserted because parallel tests that load a model set it too")
    func engineAppliesTheDirective() async {
        let engine = MLXInferenceEngine(EngineConfiguration(gpuCacheLimit: 512 << 20, promptCacheBytes: 1 << 30))
        #expect(await engine.apply(.relaxed) == MemoryLimits(cacheLimitBytes: 512 << 20, promptCacheBudgetBytes: 1 << 30, promptCacheEvictions: 0))

        let tight = await engine.apply(.tight)
        #expect(tight?.cacheLimitBytes == 128 << 20)
        #expect(tight?.promptCacheBudgetBytes == 256 << 20)
        #expect(await engine.appliedMemoryDirective == .tight)

        #expect(await engine.apply(.critical)?.promptCacheBudgetBytes == 0)
        let relaxed = await engine.apply(.relaxed)
        #expect(relaxed?.cacheLimitBytes == 512 << 20)
        #expect(relaxed?.promptCacheBudgetBytes == 1 << 30)
    }
}
