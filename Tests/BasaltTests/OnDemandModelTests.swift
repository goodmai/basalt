import Testing
import Foundation
import Synchronization
@testable import BasaltCore
@testable import BasaltServer

final class SteppingClock: Clock, Sendable {
    struct Instant: InstantProtocol, Sendable {
        let offset: Duration

        func advanced(by duration: Duration) -> Instant {
            Instant(offset: offset + duration)
        }

        func duration(to other: Instant) -> Duration {
            other.offset - offset
        }

        static func < (lhs: Instant, rhs: Instant) -> Bool {
            lhs.offset < rhs.offset
        }
    }

    private struct Waiter: Sendable {
        let deadline: Instant
        let continuation: CheckedContinuation<Void, any Error>
    }

    private struct State: Sendable {
        var now = Instant(offset: .zero)
        var nextID = 0
        var pending: Set<Int> = []
        var waiters: [Int: Waiter] = [:]
        var cancelled: Set<Int> = []
    }

    private let state = Mutex(State())

    var now: Instant { state.withLock { $0.now } }
    var minimumResolution: Duration { .nanoseconds(1) }
    var pendingWaiters: Int { state.withLock { $0.waiters.count } }

    func sleep(until deadline: Instant, tolerance: Duration? = nil) async throws {
        try Task.checkCancellation()
        let id = state.withLock { state -> Int? in
            if deadline <= state.now { return nil }
            let id = state.nextID
            state.nextID += 1
            state.pending.insert(id)
            return id
        }
        guard let id else { return }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                let action: Result<Void, any Error>? = state.withLock { state in
                    state.pending.remove(id)
                    if state.cancelled.remove(id) != nil { return .failure(CancellationError()) }
                    if deadline <= state.now { return .success(()) }
                    state.waiters[id] = Waiter(deadline: deadline, continuation: continuation)
                    return nil
                }
                switch action {
                case .success: continuation.resume()
                case .failure(let error): continuation.resume(throwing: error)
                case nil: break
                }
            }
        } onCancel: {
            let continuation = state.withLock { state -> CheckedContinuation<Void, any Error>? in
                if let waiter = state.waiters.removeValue(forKey: id) {
                    return waiter.continuation
                }
                if state.pending.contains(id) {
                    state.cancelled.insert(id)
                }
                return nil
            }
            continuation?.resume(throwing: CancellationError())
        }
    }

    func advance(by duration: Duration) {
        guard duration > .zero else { return }
        let due: [CheckedContinuation<Void, any Error>] = state.withLock { state in
            state.now = state.now.advanced(by: duration)
            let expiredIDs = state.waiters.compactMap { $0.value.deadline <= state.now ? $0.key : nil }
            return expiredIDs.compactMap { state.waiters.removeValue(forKey: $0)?.continuation }
        }
        for continuation in due {
            continuation.resume()
        }
    }
}

@Suite("On-demand models: loaded by the first use, kept while in use, unloaded after the idle time, reloaded on the next use (#412)", .timeLimit(.minutes(1)))
struct OnDemandModelTests {
    static func model(_ engine: MockInferenceEngine, _ clock: SteppingClock) async throws -> OnDemandModel {
        let directory = try OCRAPITests.modelDirectory("GLM-OCR", modelType: "glm_ocr")
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        return OnDemandModel(model: orchestrator, path: directory.path, idle: .seconds(120), clock: clock)
    }

    static func waitUntil(_ model: OnDemandModel, _ phase: OnDemandCore.Phase) async {
        guard await model.phase != phase else { return }
        for await next in model.phases where next == phase {
            if await model.phase == phase { return }
        }
    }

    @Test("The core: one load for concurrent users, an expiry only for the last release, and a stale expiry is ignored")
    func core() {
        var core = OnDemandCore()
        #expect(core.enter() == .load && core.enter() == .wait)
        core.loaded(true)
        #expect(core.enter() == .ready && core.enter() == .ready && core.users == 2)
        #expect(core.leave() == nil)
        let last = core.leave()
        #expect(last != nil && core.enter() == .ready)
        #expect(core.expire(last ?? 0) == false)
        let now = core.leave() ?? 0
        #expect(core.expire(now) && core.phase == .unloading && core.enter() == .wait)
        core.unloaded()
        #expect(core.enter() == .load)
        core.loaded(true)
        let currentEpoch = core.loadEpoch
        #expect(core.enter() == .ready && core.leave(epoch: currentEpoch - 1) == nil && core.users == 1)
        #expect(core.leave(epoch: currentEpoch) != nil && core.users == 0)
    }

    @Test("First acquire loads and reports the load time, the next reuses; the idle timer unloads, and the next acquire reloads")
    func lifecycle() async throws {
        let (engine, clock) = (MockInferenceEngine(), SteppingClock())
        let model = try await Self.model(engine, clock)
        #expect(await model.phase == .unloaded)
        _ = try await model.acquire()
        #expect(try await model.acquire() == .zero)
        await model.release()
        clock.advance(by: .seconds(120))
        let warmPhase = await model.phase
        #expect(clock.pendingWaiters == 0 && warmPhase == .loaded)
        await model.release()
        clock.advance(by: .seconds(119))
        let (loaded, loads) = (await engine.isLoaded, await engine.loadCount)
        #expect(loaded && loads == 1)
        clock.advance(by: .seconds(1))
        await Self.waitUntil(model, .unloaded)
        #expect(await !engine.isLoaded)
        _ = try await model.acquire()
        #expect(await engine.loadCount == 2)
        await model.release()
    }

    @Test("A release from before an explicit unload is ignored, so it cannot drop the hold of a user that acquired after the reload")
    func staleReleaseKeepsNewHold() async throws {
        let (engine, clock) = (MockInferenceEngine(), SteppingClock())
        let model = try await Self.model(engine, clock)
        let stale = try await model.acquireGrant()
        await model.unload()
        let fresh = try await model.acquireGrant()
        #expect(fresh.epoch != stale.epoch)
        await model.release(epoch: stale.epoch)
        clock.advance(by: .seconds(600))
        let (phase, loaded) = (await model.phase, await engine.isLoaded)
        #expect(phase == .loaded && loaded)
        await model.release(epoch: fresh.epoch)
    }

    @Test("A use before the idle time cancels the pending unload")
    func useCancelsExpiry() async throws {
        let (engine, clock) = (MockInferenceEngine(), SteppingClock())
        let model = try await Self.model(engine, clock)
        _ = try await model.acquire()
        await model.release()
        clock.advance(by: .seconds(60))
        _ = try await model.acquire()
        clock.advance(by: .seconds(120))
        let (phase, loaded) = (await model.phase, await engine.isLoaded)
        #expect(phase == .loaded && loaded)
        await model.release()
    }

    @Test("A failed load is thrown, leaves the model unloaded, and the next acquire tries again")
    func failedLoad() async throws {
        let (engine, clock) = (MockInferenceEngine(), SteppingClock())
        let model = try await Self.model(engine, clock)
        await engine.failLoads(1)
        await #expect(throws: BasaltError.self) { _ = try await model.acquire() }
        #expect(await model.phase == .unloaded)
        _ = try await model.acquire()
        let (phase, loads) = (await model.phase, await engine.loadCount)
        #expect(phase == .loaded && loads == 2)
        await model.release()
    }

    @Test("End to end on /v1/pipeline: an on-demand OCR model loads on the first document, the receipt shows the load time, it unloads after the idle time and reloads for the next document")
    func pipeline() async throws {
        let (ocrEngine, clock) = (MockInferenceEngine(), SteppingClock())
        await ocrEngine.setResponder { _ in "Invoice total: 42" }
        let held = try await Self.model(ocrEngine, clock)
        var stages = PipelineStages(ocr: held.model, abstainBelow: 0.5)
        stages.onDemand = [.ocrModel: held]
        let (chatEngine, chat) = try await PipelineSwapEdgeTests.chat()
        let body: [String: Any] = ["attachments": [PipelineSwapEdgeTests.attachments[1]], "task": "What is the total?"]
        let first = try await PipelineAPITests.post(chat, decisions: nil, stages: stages, body)
        #expect(first.status == .ok)
        #expect(((first.json["perceived"] as? [String: Any])?["pages"] as? [[String: Any]])?.first?["markdown"] as? String == "Invoice total: 42")
        #expect((first.json["receipt"] as? [String: Any])?["swap_ms"] is Int)
        #expect(await ocrEngine.isLoaded)
        clock.advance(by: .seconds(120))
        await Self.waitUntil(held, .unloaded)
        let second = try await PipelineAPITests.post(chat, decisions: nil, stages: stages, body)
        #expect(second.status == .ok)
        let (loads, chatLoaded) = (await ocrEngine.loadCount, await chatEngine.isLoaded)
        #expect(loads == 2 && chatLoaded)
    }
}
