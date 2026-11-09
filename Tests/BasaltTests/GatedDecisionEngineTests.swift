import Testing
import Foundation
import Synchronization
@testable import BasaltCore

struct BesideDecodeDecisionEngine: DecisionEngine {
    let modelID = "org/ane-jev"
    let runsBesideDecode = true

    func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        SystemOneResponse(model: modelID, answers: [:], usage: SystemOneUsage(prefixTokens: 0, candidateBranches: 0, prefixEvaluations: 0, branchForwardCalls: 0, suffixTokens: 0), calibrated: true)
    }
}

@Suite("System One on a GPU engine waits for the generation gate; an engine that runs beside decode is not wrapped (#234, #399)")
struct GatedDecisionEngineTests {
    static let request = SystemOneRequest(state: "s", questions: ["q": SystemOneQuestion(type: .noul, instructions: "It rains.")])

    @Test("A decision on a GPU engine starts only after the running generation releases its slot")
    func waitsForGeneration() async throws {
        let gate = GenerationGate(maxConcurrent: 1, maxQueue: 4)
        let stub = StubDecisionEngine()
        let gated = try #require(GatedDecisionEngine.serving(stub, gate: gate))
        #expect(gated is GatedDecisionEngine)
        let generation = try await gate.admit()
        let decision = Task { try await gated.decide(Self.request) }
        while await gate.exclusiveWaiting < 1 { await Task.yield() }
        #expect(stub.calls.requests.withLock { $0.isEmpty })
        await gate.release(generation)
        _ = try await decision.value
        #expect(stub.calls.requests.withLock { $0.count } == 1)
        #expect(await gate.checkInvariants(idle: true).isEmpty)
    }

    @Test("An engine that runs beside decode is served as is, and no engine stays nil")
    func besideDecodeIsNotWrapped() {
        let gate = GenerationGate()
        #expect(GatedDecisionEngine.serving(BesideDecodeDecisionEngine(), gate: gate) is BesideDecodeDecisionEngine)
        #expect(GatedDecisionEngine.serving(nil, gate: gate) == nil)
    }
}
