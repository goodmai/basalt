import Foundation
import Testing
@testable import BasaltCore

@Suite("Pre-prefill KV reservations for concurrent requests (#204)")
struct KVReservationTests {
    static func orchestrator(_ engine: MockInferenceEngine) -> ModelOrchestratorActor {
        ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 2, maxQueue: 4))
    }

    @Test("A request that fits alone but not next to a running one is refused with 503 Retry-After, and admitted once that one ends")
    func refusesWhileOthersHoldTheRoom() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = Self.orchestrator(engine)
        try await orchestrator.loadModel(path: "m")
        await engine.setMemory(kvBudget: 1 << 40, promptTokens: 100)
        await engine.setArtificialDelay(1)

        let first = try await orchestrator.generateStream(request: GenerationRequest(prompt: "first", maxTokens: 100))
        let reserved = try #require(await orchestrator.kvReservations.values.first)
        await engine.setMemory(kvBudget: reserved * 3 / 2, promptTokens: 100)

        do {
            _ = try await orchestrator.generate(request: GenerationRequest(prompt: "second", maxTokens: 100))
            Issue.record("the second request was admitted next to the first")
        } catch {
            guard case .serverBusy(let retryAfter) = error else { Issue.record("expected 503, got \(error)"); return }
            #expect(retryAfter == 1)
            #expect(error.httpStatus == 503)
        }
        #expect(await orchestrator.kvReservations.count == 1)
        #expect(await orchestrator.gate.inFlight == 1)

        _ = await GenerationGateTests.drain(first)
        #expect(await GenerationGateTests.eventually { await orchestrator.kvReservations.isEmpty })
        await engine.setArtificialDelay(0)
        _ = try await orchestrator.generate(request: GenerationRequest(prompt: "third", maxTokens: 100))
        #expect(await orchestrator.kvReservations.isEmpty)
        #expect(await orchestrator.gate.inFlight == 0)
    }

    @Test("Requests that fit together run together, and a lone request is never refused by its own reservation")
    func fittingRequestsShareTheRoom() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = Self.orchestrator(engine)
        try await orchestrator.loadModel(path: "m")
        await engine.setMemory(kvBudget: 1 << 40, promptTokens: 100)
        await engine.setArtificialDelay(1)
        let first = try await orchestrator.generateStream(request: GenerationRequest(prompt: "first", maxTokens: 100))
        let reserved = try #require(await orchestrator.kvReservations.values.first)
        await engine.setMemory(kvBudget: reserved * 2, promptTokens: 100)
        let second = try await orchestrator.generateStream(request: GenerationRequest(prompt: "second", maxTokens: 100))
        #expect(await orchestrator.kvReservations.count == 2)
        _ = await GenerationGateTests.drain(first)
        _ = await GenerationGateTests.drain(second)
        #expect(await GenerationGateTests.eventually { await orchestrator.kvReservations.isEmpty })
    }
}
