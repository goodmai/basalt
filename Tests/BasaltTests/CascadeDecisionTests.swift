import Testing
import Foundation
import Synchronization
@testable import BasaltCore

struct ScriptedRemoteEngine: DecisionEngine {
    let modelID = "cloud/jev"
    let runsBesideDecode = true
    var confidence: [String: Double] = [:]
    var failure: BasaltError?
    var waitsForCancellation = false
    let calls = DecisionCalls()

    var onDevice: (any DecisionEngine)? { nil }

    func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        calls.requests.withLock { $0.append(request) }
        while waitsForCancellation, !Task.isCancelled { await Task.yield() }
        if let failure { throw failure }
        let answers = request.questions.reduce(into: [String: SystemOneAnswer]()) { result, entry in
            var answer = SystemOneCore.answer(entry.value, scores: SystemOneCore.options(entry.value).keys.indices.map { -Double($0) * 4 }, temperature: 1)
            answer.confidence = confidence[entry.key] ?? 0.95
            result[entry.key] = SystemOneCore.gated(answer, abstainBelow: request.abstainBelow, minMargin: request.minMargin)
        }
        var usage = SystemOneUsage(prefixTokens: 0, candidateBranches: 0, prefixEvaluations: 0, branchForwardCalls: 0, suffixTokens: 0)
        usage.cost = 0.00002
        return SystemOneResponse(model: "cloud/jev-1.14", answers: answers, usage: usage, calibrated: true)
    }
}

@Suite("Cascade System One: remote first, local below the confidence threshold, whole-request fallback, the local stage gated (#407)", .timeLimit(.minutes(1)))
struct CascadeDecisionTests {
    static let questions: [String: SystemOneQuestion] = [
        "color": SystemOneQuestion(type: .choice, instructions: "Which color?", criteria: .named(["red": "Red.", "blue": "Blue."])),
        "rain": SystemOneQuestion(type: .noul, instructions: "It rains."),
        "size": SystemOneQuestion(type: .score, instructions: "How big?", criteria: .levels(["Small.", "Medium.", "Large."])),
    ]
    static let request = SystemOneRequest(state: "The car is red.", questions: questions)

    static func answer(confidence: Double?, noul: Double? = nil) -> SystemOneAnswer {
        var answer = SystemOneAnswer()
        (answer.confidence, answer.noul) = (confidence, noul)
        return answer
    }

    static func response(_ model: String, _ answers: [String: SystemOneAnswer], cost: Double? = nil, prefix: Int = 0) -> SystemOneResponse {
        var usage = SystemOneUsage(prefixTokens: prefix, candidateBranches: 0, prefixEvaluations: 0, branchForwardCalls: 0, suffixTokens: 0)
        usage.cost = cost
        return SystemOneResponse(model: model, answers: answers, usage: usage, calibrated: true)
    }

    @Test("A question escalates when its confidence is below the threshold or it has no answer; at the threshold it is kept; noul without confidence uses |2p − 1|")
    func split() {
        let first = Self.response("r", ["color": Self.answer(confidence: 0.6), "rain": Self.answer(confidence: nil, noul: 0.75)])
        let split = CascadeCore.split(Self.request, first, below: 0.6)
        #expect(split.kept == ["color"])
        #expect(split.escalated == ["rain", "size"])
        #expect(CascadeCore.split(Self.request, first, below: 0.5).escalated == ["size"])
        #expect(CascadeCore.subrequest(Self.request, ["rain"]).questions.keys.sorted() == ["rain"])
        #expect(CascadeCore.subrequest(Self.request, ["rain"]).state == Self.request.state)
    }

    @Test("Merging keeps each stage's own questions, sums the cost, takes the local usage counts and names both models")
    func merged() {
        let remote = Self.response("cloud", ["color": Self.answer(confidence: 0.9), "rain": Self.answer(confidence: 0.1), "stray": Self.answer(confidence: 1)], cost: 0.001)
        let local = Self.response("self", ["rain": Self.answer(confidence: 0.8), "size": Self.answer(confidence: 0.7)], prefix: 42)
        let merged = CascadeCore.merged(Self.request, [
            (remote, SystemOneStage(model: "cloud", questions: ["color"], latencyMs: 300, cost: 0.001)),
            (local, SystemOneStage(model: "self", questions: ["rain", "size"], latencyMs: 900)),
        ])
        #expect(merged.answers.keys.sorted() == ["color", "rain", "size"])
        #expect(merged.answers["rain"]?.confidence == 0.8)
        #expect(merged.model == "cloud+self")
        #expect(merged.usage.prefixTokens == 42)
        #expect(merged.usage.cost == 0.001)
        #expect(merged.usage.stages?.map(\.questions) == [["color"], ["rain", "size"]])
        let fallback = CascadeCore.merged(Self.request, [(local, SystemOneStage(model: "self", questions: ["rain", "size"], latencyMs: 1))], failed: SystemOneStage(model: "cloud", questions: [], latencyMs: 3000, fallback: "timeout"))
        #expect(fallback.usage.cost == nil)
        #expect(fallback.usage.stages?.first?.fallback == "timeout")
        #expect(fallback.model == "self")
    }

    @Test("Confident remote answers are final; only the unsure questions go to the local engine, in one call; abstain_below still gates the final answer")
    func escalatesOnlyUnsure() async throws {
        let local = StubDecisionEngine()
        let confident = CascadeDecisionEngine(remote: ScriptedRemoteEngine(), local: local, escalateBelow: 0.6, gate: GenerationGate())
        let all = try await confident.decide(Self.request)
        #expect(local.calls.requests.withLock { $0.isEmpty })
        #expect(all.usage.stages?.map(\.model) == ["cloud/jev-1.14"])
        #expect(all.usage.cost == 0.00002)
        let unsure = CascadeDecisionEngine(remote: ScriptedRemoteEngine(confidence: ["rain": 0.3, "size": 0.59]), local: local, escalateBelow: 0.6, gate: GenerationGate())
        var gated = Self.request
        gated.abstainBelow = 0.99
        let mixed = try await unsure.decide(gated)
        #expect(local.calls.requests.withLock { $0.map { $0.questions.keys.sorted() } } == [["rain", "size"]])
        #expect(local.calls.requests.withLock { $0.first?.abstainBelow } == 0.99)
        #expect(mixed.usage.stages?.map(\.questions) == [["color"], ["rain", "size"]])
        #expect(mixed.usage.stages?.last?.model == "org/stub-jev")
        #expect(mixed.answers["color"]?.abstain == true)
        #expect(mixed.model == "cloud/jev-1.14+org/stub-jev")
    }

    @Test("When the remote engine fails, the whole request goes to the local engine and the record names the failure")
    func fallsBack() async throws {
        let local = StubDecisionEngine()
        let remote = ScriptedRemoteEngine(failure: .modelInferenceError(details: "cloud decisions: timeout"))
        let response = try await CascadeDecisionEngine(remote: remote, local: local, escalateBelow: 0.6, gate: GenerationGate()).decide(Self.request)
        #expect(local.calls.requests.withLock { $0.map { $0.questions.count } } == [3])
        #expect(response.answers.count == 3)
        #expect(response.usage.stages?.map(\.model) == ["cloud/jev", "org/stub-jev"])
        #expect(response.usage.stages?.first?.questions == [])
        #expect(response.usage.stages?.first?.fallback?.contains("timeout") == true)
    }

    @Test("A cancelled request is not handed to the local engine after the remote call fails")
    func cancellationDoesNotFallBack() async {
        let local = StubDecisionEngine()
        let remote = ScriptedRemoteEngine(failure: .modelInferenceError(details: "cloud decisions: network error"), waitsForCancellation: true)
        let cascade = CascadeDecisionEngine(remote: remote, local: local, escalateBelow: 0.6, gate: GenerationGate())
        let task = Task { try await cascade.decide(Self.request) }
        while remote.calls.requests.withLock({ $0.isEmpty }) { await Task.yield() }
        task.cancel()
        await #expect(throws: BasaltError.self) { try await task.value }
        #expect(remote.calls.requests.withLock { $0.count } == 1)
        #expect(local.calls.requests.withLock { $0.isEmpty })
    }

    @Test("With a generation running, a confident cascade answers at once while an escalated question waits for the gate")
    func localStageIsGated() async throws {
        let gate = GenerationGate(maxConcurrent: 1, maxQueue: 4)
        let local = StubDecisionEngine()
        let generation = try await gate.admit()
        let confident = CascadeDecisionEngine(remote: ScriptedRemoteEngine(), local: local, escalateBelow: 0.6, gate: gate)
        #expect(try await confident.decide(Self.request).answers.count == 3)
        let unsure = CascadeDecisionEngine(remote: ScriptedRemoteEngine(confidence: ["rain": 0.1]), local: local, escalateBelow: 0.6, gate: gate)
        let escalated = Task { try await unsure.decide(Self.request) }
        while await gate.exclusiveWaiting < 1 { try Task.checkCancellation(); await Task.yield() }
        #expect(local.calls.requests.withLock { $0.isEmpty })
        await gate.release(generation)
        #expect(try await escalated.value.usage.stages?.last?.questions == ["rain"])
        #expect(await gate.checkInvariants(idle: true).isEmpty)
    }

    @Test("The cascade itself is never wrapped by the gate; its local stage is, unless it runs beside decode; pipeline states use the local stage")
    func wrapping() throws {
        let gate = GenerationGate()
        let cascade = CascadeDecisionEngine(remote: ScriptedRemoteEngine(), local: StubDecisionEngine(), escalateBelow: 0.6, gate: gate)
        #expect(GatedDecisionEngine.serving(cascade, gate: gate) is CascadeDecisionEngine)
        #expect(GatedDecisionEngine.serving(cascade, gate: gate, besideDecode: false) is CascadeDecisionEngine)
        #expect(cascade.local is GatedDecisionEngine)
        #expect(cascade.onDevice is GatedDecisionEngine)
        #expect(ScriptedRemoteEngine().onDevice == nil)
        #expect(CascadeDecisionEngine(remote: ScriptedRemoteEngine(), local: BesideDecodeDecisionEngine(), escalateBelow: 0.6, gate: gate).local is BesideDecodeDecisionEngine)
        #expect(CascadeDecisionEngine(remote: ScriptedRemoteEngine(), local: BesideDecodeDecisionEngine(), escalateBelow: 0.6, gate: gate, besideDecode: false).local is GatedDecisionEngine)
        #expect(cascade.modelID == "cloud/jev+org/stub-jev")
    }
}
