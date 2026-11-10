import Testing
import Foundation
import Synchronization
@testable import BasaltCore
@testable import BasaltCloud

final class TestClock: Sendable {
    let date = Mutex(Date(timeIntervalSince1970: 1_794_300_000))
    func now() -> Date { date.withLock { $0 } }
    func advance(_ seconds: TimeInterval) { date.withLock { $0 = $0.addingTimeInterval(seconds) } }
}

@Suite("Cloud gate: daily budgets per model and in total, kept across restarts; a refused key disables the model until restart, a rate limit pauses it; cascades of any length skip a refused stage (#412)", .timeLimit(.minutes(1)))
struct CloudGateTests {
    @Test("The core refuses over budget, rolls the day, disables on 401/403 and pauses on 429 for Retry-After")
    func core() {
        var core = CloudGateCore(day: "2026-11-10", limits: ["jev": 0.01, CloudGateCore.total: 0.05])
        let now = Date(timeIntervalSince1970: 0)
        #expect(core.refusal("jev", now: now) == nil)
        core.record("jev", cost: 0.006)
        core.record("jev", cost: .nan)
        #expect(core.refusal("jev", now: now) == nil)
        core.record("jev", cost: 0.006)
        #expect(core.refusal("jev", now: now) == .budget(spent: 0.012, limit: 0.01))
        core.record("frontier", cost: 0.04)
        #expect(core.refusal("frontier", now: now)?.reason.hasPrefix("daily budget spent") == true)
        core.roll(to: "2026-11-11")
        #expect(core.refusal("jev", now: now) == nil && core.spent.isEmpty)
        core.failed("jev", .serverBusy(retryAfterSeconds: 7), now: now)
        #expect(core.refusal("jev", now: now) == .paused(until: now.addingTimeInterval(7)))
        #expect(core.refusal("jev", now: now.addingTimeInterval(8)) == nil)
        core.failed("jev", .authenticationFailed(details: "HTTP 401"), now: now)
        #expect(core.refusal("jev", now: now.addingTimeInterval(1e6)) == .disabled("HTTP 401"))
        core.failed("other", .modelInferenceError(details: "timeout"), now: now)
        #expect(core.refusal("other", now: now) == nil)
    }

    @Test("Spend survives a restart on the same day and resets on the next")
    func persistence() async throws {
        let file = FileManager.default.temporaryDirectory.appending(path: "cloud-spend-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let clock = TestClock()
        let first = CloudGate(limits: ["jev": 0.01], storage: file, now: clock.now)
        await first.record("jev", cost: 0.02)
        #expect(await first.refusal("jev") != nil)
        let restarted = CloudGate(limits: ["jev": 0.01], storage: file, now: clock.now)
        #expect(await restarted.spent("jev") == 0.02)
        clock.advance(86_400)
        let tomorrow = CloudGate(limits: ["jev": 0.01], storage: file, now: clock.now)
        #expect(await tomorrow.refusal("jev") == nil)
    }

    @Test("Three stages: each answers its confident questions and passes the rest on; a failed middle stage is skipped and named; onDevice drops the cloud stage")
    func threeStages() async throws {
        let local = StubDecisionEngine()
        let request = SystemOneRequest(state: "Sky is grey.", questions: [
            "rain": SystemOneQuestion(type: .noul, instructions: "It will rain."),
            "color": SystemOneQuestion(type: .choice, instructions: "Which color?", criteria: .named(["grey": "Grey.", "blue": "Blue."])),
        ])
        let cascade = CascadeDecisionEngine(stages: [ScriptedRemoteEngine(confidence: ["rain": 0.2]), ScriptedRemoteEngine(failure: .modelInferenceError(details: "boom")), local], escalateBelow: 0.6)
        let response = try await cascade.decide(request)
        #expect(response.answers.keys.sorted() == ["color", "rain"])
        #expect(response.usage.stages?.map(\.questions) == [[], ["color"], ["rain"]])
        #expect(response.usage.stages?.first?.fallback?.contains("boom") == true)
        #expect(local.calls.requests.withLock { $0.map { $0.questions.keys.sorted() } } == [["rain"]])
        #expect(cascade.onDevice.map { $0 is StubDecisionEngine } == true)
        #expect(CascadeDecisionEngine(stages: [ScriptedRemoteEngine(), local, StubDecisionEngine()], escalateBelow: 0.6).onDevice is CascadeDecisionEngine)
    }

    @Test("Against the stub: 401 becomes an authentication error and disables the stage without another call; 429 pauses it for Retry-After; the cascade answers locally with the reason")
    func guardedRemote() async throws {
        try await RemoteStub.serve { base, log in
            let gate = CloudGate(limits: [:], storage: nil)
            let refused = CascadeDecisionEngine(stages: [GuardedDecisionEngine(try RemoteStub.engine(base, "auth"), gate: gate), StubDecisionEngine()], escalateBelow: 0.6)
            let first = try await refused.decide(RemoteDecisionEngineTests.request)
            #expect(first.usage.stages?.first?.fallback?.contains("HTTP 401") == true)
            let second = try await refused.decide(RemoteDecisionEngineTests.request)
            #expect(second.usage.stages?.first?.fallback?.contains("disabled until restart") == true)
            #expect(log.calls.withLock { $0.count } == 1)
            do {
                _ = try await RemoteStub.engine(base, "limited").decide(RemoteDecisionEngineTests.request)
                Issue.record("a 429 did not throw")
            } catch let error as BasaltError {
                guard case .serverBusy(let seconds) = error else {
                    Issue.record("expected serverBusy, got \(error)")
                    return
                }
                #expect(seconds == 7)
            }
            let pausing = CloudGate(limits: [:], storage: nil)
            let limited = GuardedDecisionEngine(try RemoteStub.engine(base, "limited"), gate: pausing)
            _ = try? await limited.decide(RemoteDecisionEngineTests.request)
            #expect(await pausing.refusal(limited.modelID).map { if case .paused = $0 { true } else { false } } == true)
        }
    }

    @Test("A budget stops the cloud stage once spent; the cascade then decides locally")
    func budget() async throws {
        try await RemoteStub.serve { base, log in
            let gate = CloudGate(limits: ["~typesafe/jev-latest": 0.000015], storage: nil)
            let cascade = CascadeDecisionEngine(stages: [GuardedDecisionEngine(try RemoteStub.engine(base, "ok"), gate: gate), StubDecisionEngine()], escalateBelow: 0.6)
            _ = try await cascade.decide(RemoteDecisionEngineTests.request)
            _ = try await cascade.decide(RemoteDecisionEngineTests.request)
            let third = try await cascade.decide(RemoteDecisionEngineTests.request)
            #expect(log.calls.withLock { $0.count } == 2)
            #expect(third.usage.stages?.first?.fallback?.contains("daily budget spent") == true)
        }
    }
}
