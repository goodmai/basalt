import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltServer

actor ManualSleep {
    private var waiting: [Int: CheckedContinuation<Void, Error>] = [:]
    private var watchers: [CheckedContinuation<Void, Never>] = []
    private var next = 0

    var pending: Int { waiting.count }

    func sleep() async throws {
        let id = next
        next += 1
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                waiting[id] = continuation
                for watcher in watchers { watcher.resume() }
                watchers.removeAll()
            }
        } onCancel: {
            Task { await self.cancel(id) }
        }
    }

    func cancel(_ id: Int) {
        waiting.removeValue(forKey: id)?.resume(throwing: CancellationError())
    }

    func fire() {
        let all = waiting
        waiting.removeAll()
        for continuation in all.values { continuation.resume() }
    }

    func waitForSleeper() async {
        guard waiting.isEmpty else { return }
        await withCheckedContinuation { watchers.append($0) }
    }
}

@Suite("On-demand models: loaded by the first use, kept while in use, unloaded after the idle time, reloaded on the next use (#412)", .timeLimit(.minutes(1)))
struct OnDemandModelTests {
    static func model(_ engine: MockInferenceEngine, _ clock: ManualSleep) async throws -> OnDemandModel {
        let directory = try OCRAPITests.modelDirectory("GLM-OCR", modelType: "glm_ocr")
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        return OnDemandModel(model: orchestrator, path: directory.path, idle: .seconds(120)) { _ in try await clock.sleep() }
    }

    static func waitUntil(_ model: OnDemandModel, _ phase: OnDemandCore.Phase) async {
        guard await model.phase != phase else { return }
        for await next in model.phases where next == phase { return }
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
    }

    @Test("First acquire loads and reports the load time, the next reuses; the idle timer unloads, and the next acquire reloads")
    func lifecycle() async throws {
        let (engine, clock) = (MockInferenceEngine(), ManualSleep())
        let model = try await Self.model(engine, clock)
        #expect(await model.phase == .unloaded)
        _ = try await model.acquire()
        #expect(try await model.acquire() == .zero)
        await model.release()
        #expect(await clock.pending == 0)
        await model.release()
        await clock.waitForSleeper()
        let (loaded, loads) = (await engine.isLoaded, await engine.loadCount)
        #expect(loaded && loads == 1)
        await clock.fire()
        await Self.waitUntil(model, .unloaded)
        #expect(await !engine.isLoaded)
        _ = try await model.acquire()
        #expect(await engine.loadCount == 2)
        await model.release()
    }

    @Test("A use before the idle time cancels the pending unload")
    func useCancelsExpiry() async throws {
        let (engine, clock) = (MockInferenceEngine(), ManualSleep())
        let model = try await Self.model(engine, clock)
        _ = try await model.acquire()
        await model.release()
        await clock.waitForSleeper()
        _ = try await model.acquire()
        await clock.fire()
        let (phase, loaded) = (await model.phase, await engine.isLoaded)
        #expect(phase == .loaded && loaded)
        await model.release()
    }

    @Test("A failed load is thrown, leaves the model unloaded, and the next acquire tries again")
    func failedLoad() async throws {
        let (engine, clock) = (MockInferenceEngine(), ManualSleep())
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
        let (ocrEngine, clock) = (MockInferenceEngine(), ManualSleep())
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
        await clock.waitForSleeper()
        #expect(await ocrEngine.isLoaded)
        await clock.fire()
        await Self.waitUntil(held, .unloaded)
        let second = try await PipelineAPITests.post(chat, decisions: nil, stages: stages, body)
        #expect(second.status == .ok)
        let (loads, chatLoaded) = (await ocrEngine.loadCount, await chatEngine.isLoaded)
        #expect(loads == 2 && chatLoaded)
    }
}
