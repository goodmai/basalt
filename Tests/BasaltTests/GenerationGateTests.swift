import Testing
import Foundation
@testable import BasaltCore

@Suite("Bounded generations, single-flight switch, shutdown drain (#74)", .serialized)
struct GenerationGateTests {
    actor CountingEngine: InferenceEngine {
        let inner = MockInferenceEngine()
        private(set) var loads: [String] = []
        private var heldLoads: [CheckedContinuation<Void, Never>]?

        func holdLoads() { heldLoads = heldLoads ?? [] }

        func releaseLoads() {
            let held = heldLoads ?? []
            heldLoads = nil
            for load in held { load.resume() }
        }

        func load(modelPath: String) async throws(BasaltError) {
            loads.append(modelPath)
            if heldLoads != nil { await withCheckedContinuation { heldLoads?.append($0) } }
            try await inner.load(modelPath: modelPath)
        }
        func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
            try await inner.generate(request: request)
        }
        func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
            try await inner.generateStream(request: request)
        }
        var isLoaded: Bool { get async { await inner.isLoaded } }
    }

    static func modelDirectory(_ name: String) throws -> String {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
        return dir.path
    }

    static func eventually(within limit: Duration = .seconds(60), _ condition: () async -> Bool) async -> Bool {
        let deadline = ContinuousClock.now + limit
        while ContinuousClock.now < deadline {
            if await condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return await condition()
    }

    static func drain(_ stream: AsyncStream<StreamChunk>) async -> String {
        var text = ""
        for await chunk in stream { if case .text(let t) = chunk { text += t } }
        return text
    }

    @Test("maxConcurrent 1, maxQueue 1: the second request waits for the first, the third gets 503 with a retry hint")
    func boundedQueue() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(
            engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 1, maxQueue: 1))
        try await orchestrator.loadModel(path: "m")
        await engine.holdStreams()

        let first = try await orchestrator.generateStream(request: GenerationRequest(prompt: "1"))
        async let second = orchestrator.generate(request: GenerationRequest(prompt: "2"))
        #expect(await Self.eventually { await orchestrator.gate.queued == 1 })
        #expect(await orchestrator.gate.checkInvariants() == [])
        do {
            _ = try await orchestrator.generate(request: GenerationRequest(prompt: "3"))
            Issue.record("expected serverBusy")
        } catch {
            guard case .serverBusy(let retryAfter) = error else { Issue.record("unexpected \(error)"); return }
            #expect(retryAfter == 1)
            #expect(error.httpStatus == 503)
        }
        #expect(await engine.getRequests().map(\.prompt) == ["1"])
        await engine.releaseStreams()
        #expect(await Self.drain(first) == "Mock response")
        #expect(try await second.generatedText == "Mock response")
        #expect(await engine.getRequests().map(\.prompt) == ["1", "2"])
        #expect(await orchestrator.gate.inFlight == 0)
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])
    }

    @Test("A stream holds its slot until the consumer drops it")
    func droppedStreamReleasesSlot() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(
            engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 1, maxQueue: 0))
        try await orchestrator.loadModel(path: "m")
        await engine.holdStreams()
        var stream: AsyncStream<StreamChunk>? = try await orchestrator.generateStream(request: GenerationRequest(prompt: "1"))
        #expect(await orchestrator.gate.inFlight == 1)
        _ = stream
        stream = nil
        #expect(await Self.eventually { await orchestrator.gate.inFlight == 0 })
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])
    }

    @Test("A queued request whose client goes away leaves the queue")
    func cancelledWaiterLeaves() async throws {
        let gate = GenerationGate(maxConcurrent: 1, maxQueue: 2)
        let holder = try await gate.admit()
        let waiter = Task { try await gate.admit() }
        #expect(await Self.eventually { await gate.queued == 1 })
        #expect(await gate.checkInvariants() == [])
        waiter.cancel()
        await #expect(throws: CancellationError.self) { _ = try await waiter.value }
        #expect(await gate.queued == 0)
        #expect(await gate.checkInvariants() == [])
        await gate.release(holder)
        #expect(await gate.inFlight == 0)
        #expect(await gate.checkInvariants(idle: true) == [])
    }

    @Test("Two concurrent switches to the same model load it once; a switch to another model meanwhile is 503")
    func singleFlightSwitch() async throws {
        let engine = CountingEngine()
        await engine.holdLoads()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        let b = try Self.modelDirectory("B")
        let c = try Self.modelDirectory("C")
        defer { [b, c].forEach { try? FileManager.default.removeItem(atPath: $0) } }
        async let one: Void = orchestrator.switchModel(to: b)
        async let two: Void = orchestrator.switchModel(to: b)
        #expect(await Self.eventually { await engine.loads.count == 1 })
        do {
            try await orchestrator.switchModel(to: c)
            Issue.record("expected modelSwitching")
        } catch {
            guard case .modelSwitching(_, let to) = error else { Issue.record("unexpected \(error)"); return }
            #expect(to == b)
            #expect(error.httpStatus == 503)
        }
        await engine.releaseLoads()
        _ = try await (one, two)
        #expect(await engine.loads.count == 1)
        #expect(await orchestrator.currentModelId == b)
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])
    }

    @Test("A switch issued during a running stream loads only after the stream ends, and new requests queue behind it")
    func switchWaitsForStreams() async throws {
        let engine = CountingEngine()
        let orchestrator = ModelOrchestratorActor(
            engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 2, maxQueue: 4))
        try await orchestrator.loadModel(path: "A")
        await engine.inner.holdStreams()
        let b = try Self.modelDirectory("B")
        defer { try? FileManager.default.removeItem(atPath: b) }

        let stream = try await orchestrator.generateStream(request: GenerationRequest(prompt: "running"))
        let switching = Task { try await orchestrator.switchModel(to: b) }
        #expect(await Self.eventually { await orchestrator.gate.exclusiveWaiting == 1 })
        #expect(await orchestrator.gate.checkInvariants() == [])
        #expect(await engine.loads == ["A"])
        let behind = Task { try await orchestrator.generate(request: GenerationRequest(prompt: "behind")) }
        #expect(await Self.eventually { await orchestrator.gate.queued == 1 })
        #expect(await engine.loads == ["A"])
        #expect(await orchestrator.gate.checkInvariants() == [])

        await engine.inner.releaseStreams()
        _ = await Self.drain(stream)
        try await switching.value
        _ = try await behind.value
        #expect(await engine.loads.count == 2)
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])
    }

    @Test("Shutdown stops admitting, and a stream still open after the deadline is cancelled and ends", .timeLimit(.minutes(2)))
    func shutdownDrainsWithDeadline() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "m")
        await engine.holdStreams()
        let stream = try await orchestrator.generateStream(request: GenerationRequest(prompt: "long"))
        let consumer = Task { await Self.drain(stream) }

        await orchestrator.shutdown(deadline: .milliseconds(300))
        #expect(await consumer.value.isEmpty)
        #expect(await orchestrator.gate.inFlight == 0)
        await #expect(throws: BasaltError.self) {
            _ = try await orchestrator.generate(request: GenerationRequest(prompt: "late"))
        }
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])
    }
}
