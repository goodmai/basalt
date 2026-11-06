import Foundation
import Testing
@testable import BasaltCore

actor PausingEngine: InferenceEngine {
    enum Point { case countingPrompt, generating }

    nonisolated let pauses: AsyncStream<CheckedContinuation<Void, Never>>
    private let pause: AsyncStream<CheckedContinuation<Void, Never>>.Continuation
    private var armed: Set<Point> = []
    private var path = ""

    init() {
        (pauses, pause) = AsyncStream.makeStream(of: CheckedContinuation<Void, Never>.self)
    }

    func arm(_ points: Set<Point>) {
        armed.formUnion(points)
    }

    private func pass(_ point: Point) async {
        guard armed.remove(point) != nil else { return }
        await withCheckedContinuation { pause.yield($0) }
    }

    func load(modelPath: String) async throws(BasaltError) {
        path = modelPath
    }

    var isLoaded: Bool { !path.isEmpty }

    var kvMemoryBudget: ByteCount? { ByteCount(1 << 40) }

    func promptTokenCount(for request: GenerationRequest) async -> Int? {
        let counted = path == "b" ? 3000 : 100
        await pass(.countingPrompt)
        return counted
    }

    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
        await pass(.generating)
        return GenerationResponse(
            generatedText: "ok", promptTokens: 0, completionTokens: 0, tokensPerSecond: 0, generationTime: 0, timeToFirstToken: 0,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .stop)
    }

    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        AsyncStream { $0.finish() }
    }
}

@Suite("Model orchestrator state: one reducer, and plans that cross a model load (#300)")
struct ModelOrchestratorActorTests {
    static let request = GenerationRequest(prompt: "p", maxTokens: 100)

    static func reserved(loading path: String) async throws -> ByteCount? {
        let engine = PausingEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: path)
        var pauses = engine.pauses.makeAsyncIterator()
        await engine.arm([.generating])
        let response = Task { try await orchestrator.generate(request: request) }
        let generating = try #require(await pauses.next())
        let reserved = await orchestrator.kvReservations.values.first
        generating.resume()
        _ = try await response.value
        return reserved
    }

    @Test("A request planned under one model and admitted after another loads reserves KV for the loaded model")
    func planCrossingALoadIsRedone() async throws {
        let engine = PausingEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "a")
        var pauses = engine.pauses.makeAsyncIterator()
        await engine.arm([.countingPrompt, .generating])
        let response = Task { try await orchestrator.generate(request: Self.request) }
        let counting = try #require(await pauses.next())
        try await orchestrator.loadModel(path: "b")
        counting.resume()
        let generating = try #require(await pauses.next())
        let reserved = await orchestrator.kvReservations.values.first
        generating.resume()
        #expect(try await response.value.generatedText == "ok")

        let underB = try await Self.reserved(loading: "b")
        let underA = try await Self.reserved(loading: "a")
        #expect(underB != underA)
        #expect(reserved == underB)
        #expect(await orchestrator.kvReservations.isEmpty)
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])
    }

    func summary(_ state: OrchestratorState) -> String {
        "\(state.modelEpoch) \(state.modelId) \(state.modelPath) \(state.reservations.bytes.keys.sorted()) \(state.memoryRefusals) \(state.pendingSwitch?.modelId ?? "-")"
    }

    func loaded(_ path: String, as id: String? = nil) -> OrchestratorEvent {
        .loaded(path: path, modelId: id, geometry: KVCacheGeometry(parameterProxy: 4), takes: (true, true, true, false))
    }

    @Test("Each event changes only its own part of the state, and a reservation planned under an older model is dropped")
    func reduceTable() {
        let empty = OrchestratorState(kv: KVPlanner(maxTokens: 1000, precisionLadder: [nil], zones: nil))
        let start = OrchestratorState.reduce(
            OrchestratorState.reduce(empty, loaded("/a", as: "x")),
            .reserved(ticket: 7, bytes: ByteCount(50), budget: ByteCount(100), epoch: 1))
        #expect(summary(start) == "1 x /a [7] 0 -")
        let switching = Task<Void, Error> {}
        let table: [(event: OrchestratorEvent, expected: String)] = [
            (loaded("/b"), "2 x /b [7] 0 -"),
            (loaded("/b", as: "y"), "2 y /b [7] 0 -"),
            (.named("z"), "1 z /a [7] 0 -"),
            (.reserved(ticket: 8, bytes: ByteCount(50), budget: ByteCount(100), epoch: 1), "1 x /a [7, 8] 0 -"),
            (.reserved(ticket: 8, bytes: ByteCount(60), budget: ByteCount(100), epoch: 1), "1 x /a [7] 0 -"),
            (.reserved(ticket: 8, bytes: ByteCount(1), budget: ByteCount(100), epoch: 0), "1 x /a [7] 0 -"),
            (.released(7), "1 x /a [] 0 -"),
            (.refusedForMemory, "1 x /a [7] 1 -"),
            (.switchStarted("y", switching), "1 x /a [7] 0 y"),
            (.switchEnded("y"), "1 x /a [7] 0 -"),
        ]
        for (event, expected) in table {
            let next = OrchestratorState.reduce(start, event)
            #expect(summary(next) == expected)
            #expect(next.checkInvariants(after: start) == [])
        }
        let pending = OrchestratorState.reduce(start, .switchStarted("y", switching))
        #expect(summary(OrchestratorState.reduce(pending, .switchEnded("z"))) == "1 x /a [7] 0 y")
        #expect(summary(OrchestratorState.reduce(pending, .switchEnded("y"))) == "1 x /a [7] 0 -")
        let limits = MemoryLimits(cacheLimitBytes: 1, promptCacheBudgetBytes: 2, promptCacheEvictions: 0)
        #expect(OrchestratorState.reduce(start, .limitsApplied(limits)).memoryLimits == limits)
        #expect(start.checkInvariants(after: OrchestratorState.reduce(start, loaded("/b"))) != [])
    }
}
