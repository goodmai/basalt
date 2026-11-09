import Foundation
import Testing
import MLX
import MLXLMCommon
import Synchronization
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

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

    @Test("A weights delta that lands while a step runs discards that step: its checkpoint is not written and the run stops")
    func deltaDuringAStepIsNotCheckpointed() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let before = await DistillTests.baseWeights(container)
        let left = Mutex(0)
        let editing = StepLease(enter: { 0 }, leave: { _ in
            if left.withLock({ $0 += 1; return $0 }) == 6 { await engine.record(.learned(await engine.modelEpoch) { $0.deltas["probe"] = AppliedDelta() }) }
        })
        let output = try DistillTests.scratch().appending(path: "edited")
        let message = await Self.failure { try await Fixture.run(engine, Fixture.request(output: output), lease: editing) }
        #expect(message.contains("overlay or weights delta"), "\(message)")
        #expect(Fixture.savedStep(output) == 0, "step 2 ran on the edited weights, so only the step-0 checkpoint stays")
        await engine.record(.learned(await engine.modelEpoch) { $0.deltas["probe"] = nil })
        #expect(await DistillTests.baseWeights(container) == before)
    }

    @Test("Unloading the model between steps stops the run with the switched-model error")
    func switchedModelStopsTheRun() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let entered = Mutex(0)
        let unloading = StepLease(enter: {
            if entered.withLock({ $0 += 1; return $0 }) == 6 { await engine.record(.unloaded) }
            return 0
        }, leave: { _ in })
        let output = try DistillTests.scratch().appending(path: "switched")
        let message = await Self.failure { try await Fixture.run(engine, Fixture.request(output: output), lease: unloading) }
        #expect(message.contains("was switched"), "\(message)")
        #expect(Fixture.savedStep(output) == 0)
    }

    @Test("Two runs cannot reserve one output: of concurrent reservations exactly one wins, the atomic create refuses an existing directory, and resume without a checkpoint is refused")
    func oneRunPerOutput() async throws {
        let engines = [try await TinyModel.engine(.qwen3), try await TinyModel.engine(.qwen3), try await TinyModel.engine(.qwen3), try await TinyModel.engine(.qwen3)]
        let output = try DistillTests.scratch().appending(path: "adapter").path
        let wins = await withTaskGroup(of: Bool.self) { group in
            for engine in engines { group.addTask { (try? await engine.reservedDirectory(output)) != nil } }
            return await group.reduce(0) { $0 + ($1 ? 1 : 0) }
        }
        #expect(wins == 1)
        #expect(throws: BasaltError.self) { try MLXInferenceEngine.reserve(URL(fileURLWithPath: output)) }
        await #expect(throws: BasaltError.self) { try await engines[0].resumedDirectory(output) }
        await #expect(throws: BasaltError.self) { try await engines[0].resumedDirectory(nil) }
    }

    @Test("The row digest follows every token and prompt boundary, and resume over changed rows is refused")
    func changedRowsAreRefused() async throws {
        let rows = [DistillRow(tokens: [1, 2, 3], promptCount: 1), DistillRow(tokens: [4, 5], promptCount: 1)]
        let digest = AdapterProgress.digest(rows)
        #expect(digest == AdapterProgress.digest(rows))
        #expect(digest != AdapterProgress.digest([DistillRow(tokens: [1, 2, 3], promptCount: 2), rows[1]]))
        #expect(digest != AdapterProgress.digest([DistillRow(tokens: [1, 2], promptCount: 1), DistillRow(tokens: [3, 4, 5], promptCount: 1)]))
        #expect(digest != AdapterProgress.digest(Array(rows.reversed())))

        let engine = try await TinyModel.engine(.qwen3)
        let output = try DistillTests.scratch().appending(path: "rows")
        _ = try await Self.interrupted(engine, output)
        var changed = Fixture.train
        changed[0].answer = "The colour is green."
        let message = await Self.failure { [changed] in try await Fixture.run(engine, Self.resumed(output), train: changed) }
        #expect(message.contains("training rows differ"), "\(message)")
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

    @Test("MCP distill holds no gate ticket for the run: with the only slot taken and no queue it still starts, where a research call is refused; checkpointEvery and resume parse")
    func mcpDistillIsPreemptible() async throws {
        let gate = GenerationGate(maxConcurrent: 1, maxQueue: 0)
        let orchestrator = ModelOrchestratorActor(engine: try await TinyModel.engine(.qwen3), telemetry: TelemetryTracker(), gate: gate)
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orchestrator) { capturer.capture($0) }.withResearchTools(true)
        let held = try await gate.admit()
        let arguments: [String: Any] = [
            "prompts": "/nonexistent-\(UUID().uuidString).jsonl", "teacherUrl": "https://teacher.invalid/v1", "teacherModel": "t",
            "heldOut": ["Hello."], "checkpointEvery": 7, "resume": true, "output": "/tmp/x",
        ]
        let call: [String: Any] = ["jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": ["name": "distill", "arguments": arguments]]
        await server.dispatch(String(decoding: try JSONSerialization.data(withJSONObject: call), as: UTF8.self))
        await server.dispatch(#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"overlays_list"}}"#)
        await gate.release(held)
        let lines = capturer.lines
        #expect(lines.count == 2)
        #expect(lines.first?.contains("could not read the prompts") == true, "\(lines.first ?? "")")
        #expect(lines.last?.contains("busy") == true || lines.last?.contains("503") == true, "\(lines.last ?? "")")
        let request = try DistillRequest(mcp: arguments)
        #expect(request.checkpointEvery == 7 && request.resume && request.output == "/tmp/x")
        #expect(DistillRequest(prompts: "", teacher: TeacherSpec(url: "u", model: "m"), gate: GateRequest(heldOut: [])).checkpointEvery == 50)
    }
}
