import Foundation
import Testing
import MLX
import MLXLMCommon
import Synchronization
@testable import BasaltCore
@testable import BasaltInference

@Suite("Preemptible distillation guards: arrivals hold the next step, a changed model stops the run, one run per output, resume refuses what does not fit (#399 phase 2)", .serialized, .timeLimit(.minutes(1)))
struct PreemptibleTrainingGuardTests {
    typealias Fixture = PreemptibleTrainingTests

    static func failure(_ body: @escaping @Sendable () async throws -> DistillRun) async -> String {
        do {
            _ = try await Task { try await body() }.value
            return "no error"
        } catch {
            return error.localizedDescription
        }
    }

    static func interrupted(_ engine: MLXInferenceEngine, _ output: URL) async throws -> URL {
        let cancelling = StepLease(enter: {
            if Fixture.savedStep(output) == 2 { withUnsafeCurrentTask { $0?.cancel() } }
            return 0
        }, leave: { _ in })
        _ = await failure { try await Fixture.run(engine, Fixture.request(output: output), lease: cancelling) }
        let backup = output.deletingLastPathComponent().appending(path: "\(output.lastPathComponent)-backup")
        try FileManager.default.copyItem(at: AdapterCheckpoint.directory(in: output), to: backup)
        return backup
    }

    static func reset(_ output: URL, from backup: URL) throws {
        try? FileManager.default.removeItem(at: AdapterCheckpoint.directory(in: output))
        try? FileManager.default.removeItem(at: AdapterCheckpoint.staging(in: output))
        try FileManager.default.copyItem(at: backup, to: AdapterCheckpoint.directory(in: output))
    }

    static func resumed(_ output: URL) -> DistillRequest {
        var request = Fixture.request(output: output)
        request.resume = true
        return request
    }

    @Test("A request announced before admission holds the next training step until it is admitted")
    func arrivalHoldsTheStep() async throws {
        let polls = Mutex(0)
        let gate = GenerationGate(sleep: { _ in
            polls.withLock { $0 += 1 }
            await Task.yield()
        })
        let orchestrator = ModelOrchestratorActor(engine: try await TinyModel.engine(.qwen3), telemetry: TelemetryTracker(), gate: gate)
        let lease = try await orchestrator.preemptibly { _, lease in lease }
        await gate.arrive()
        let entered = Mutex(false)
        let entering = Task {
            defer { entered.withLock { $0 = true } }
            return try await lease.enter()
        }
        while polls.withLock({ $0 }) < 3, !entered.withLock({ $0 }) { try Task.checkCancellation(); await Task.yield() }
        #expect(!entered.withLock { $0 }, "the step waits while a request is arriving")
        #expect(await gate.inFlight == 0)
        await gate.arrived()
        let ticket = try await entering.value
        #expect(await gate.inFlight == 1)
        await lease.leave(ticket)
        #expect(await gate.checkInvariants(idle: true).isEmpty)
    }

    @Test("A weights delta applied between steps stops the run at the next step, and a modified model is refused at the start")
    func modifiedModelStopsTheRun() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let before = await DistillTests.baseWeights(container)
        let entered = Mutex(0)
        let modifying = StepLease(enter: {
            if entered.withLock({ $0 += 1; return $0 }) == 4 { await engine.record(.learned(await engine.modelEpoch) { $0.deltas["probe"] = AppliedDelta() }) }
            return 0
        }, leave: { _ in })
        let output = try DistillTests.scratch().appending(path: "modified")
        let message = await Self.failure { try await Fixture.run(engine, Fixture.request(output: output), lease: modifying) }
        #expect(message.contains("overlay or weights delta"), "\(message)")
        #expect(Fixture.savedStep(output) == 0, "the step-0 checkpoint stays for a resume")
        await #expect(throws: BasaltError.self) { try await engine.adapterSteps(layers: nil, rank: 4, lease: nil, what: "distillation") }
        await engine.record(.learned(await engine.modelEpoch) { $0.deltas["probe"] = nil })
        #expect(await DistillTests.baseWeights(container) == before)
    }

    @Test("Two runs cannot reserve one output, and resume without a checkpoint is refused")
    func oneRunPerOutput() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let output = try DistillTests.scratch().appending(path: "adapter").path
        _ = try await engine.reservedDirectory(output)
        await #expect(throws: BasaltError.self) { try await engine.reservedDirectory(output) }
        await #expect(throws: BasaltError.self) { try await engine.resumedDirectory(output) }
        await #expect(throws: BasaltError.self) { try await engine.resumedDirectory(nil) }
    }

    @Test("Resume refuses another model's checkpoint, foreign or corrupt keys and a lone checkpoint.next, and leaves the base weights; a stale checkpoint.next beside a good checkpoint is ignored and cleaned")
    func resumeRefusals() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let before = await DistillTests.baseWeights(container)
        let output = try DistillTests.scratch().appending(path: "run")
        let backup = try await Self.interrupted(engine, output)
        let checkpoint = AdapterCheckpoint.directory(in: output)

        var progress = try AdapterCheckpoint.saved(in: output)
        progress.model = "another-model"
        try JSONEncoder().encode(progress).write(to: checkpoint.appending(component: "progress.json"))
        #expect(await Self.failure { try await Fixture.run(engine, Self.resumed(output)) }.contains("another model"))

        try Self.reset(output, from: backup)
        var weights = try MLX.loadArrays(url: checkpoint.appending(component: "adapter.safetensors"))
        weights["model.embed_tokens.weight"] = MLXArray.zeros([2, 2])
        try MLX.save(arrays: weights, url: checkpoint.appending(component: "adapter.safetensors"))
        #expect(await Self.failure { try await Fixture.run(engine, Self.resumed(output)) }.contains("not LoRA weights"))

        try Self.reset(output, from: backup)
        try MLX.save(arrays: ["layers.x.0.m": MLXArray.zeros([2])], url: checkpoint.appending(component: "optimizer.safetensors"))
        #expect(await Self.failure { try await Fixture.run(engine, Self.resumed(output)) }.contains("optimizer state"))

        try Self.reset(output, from: backup)
        try FileManager.default.moveItem(at: checkpoint, to: AdapterCheckpoint.staging(in: output))
        #expect(await Self.failure { try await Fixture.run(engine, Self.resumed(output)) }.contains("missing or unreadable"))
        #expect(await DistillTests.baseWeights(container) == before)

        try Self.reset(output, from: backup)
        try FileManager.default.createDirectory(at: AdapterCheckpoint.staging(in: output), withIntermediateDirectories: false)
        try Data("garbage".utf8).write(to: AdapterCheckpoint.staging(in: output).appending(component: "progress.json"))
        let finished = try await Fixture.run(engine, Self.resumed(output))
        #expect(finished.resumedAt == 2)
        #expect(!FileManager.default.fileExists(atPath: checkpoint.path) && !FileManager.default.fileExists(atPath: AdapterCheckpoint.staging(in: output).path))
        #expect(await DistillTests.baseWeights(container) == before)
    }
}
