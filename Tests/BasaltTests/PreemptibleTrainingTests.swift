import Foundation
import Testing
import MLX
import MLXLMCommon
import Synchronization
@testable import BasaltCore
@testable import BasaltInference

@Suite("Preemptible distillation: one model hold per optimizer step, requests served between steps, checkpoint and resume (#399 phase 2)", .serialized, .timeLimit(.minutes(1)))
struct PreemptibleTrainingTests {
    static let train = (0 ..< 8).map { TrainingTask(prompt: "Name the colour number \($0).", answer: "The colour is blue, as in \($0 % 2 == 0 ? "sky" : "sea").") }
    static let heldOut = (8 ..< 10).map { TrainingTask(prompt: "Name the colour number \($0).", answer: "The colour is blue.") }

    static func request(output: URL, iterations: Int = 6) -> DistillRequest {
        var request = DistillRequest(prompts: "", teacher: TeacherSpec(url: "https://teacher.invalid/v1", model: "teacher"), gate: GateRequest(heldOut: ["Hello."]))
        (request.iterations, request.rank, request.learningRate, request.maxSeqLen, request.batchSize, request.checkpointEvery) = (iterations, 4, 3e-3, 256, 2, 2)
        request.output = output.path
        return request
    }

    static func run(_ engine: MLXInferenceEngine, _ request: DistillRequest, lease: StepLease? = nil, train: [TrainingTask] = train) async throws -> DistillRun {
        let directory = request.resume ? URL(fileURLWithPath: try #require(request.output)) : try await engine.reservedDirectory(request.output)
        let steps = try await engine.adapterSteps(layers: request.layers, rank: request.rank, lease: lease, what: "distillation")
        return try await MLXInferenceEngine.distillAdapter(train, heldOut: Self.heldOut, request: request, directory: directory, steps: steps)
    }

    static func savedStep(_ output: URL) -> Int? { try? AdapterCheckpoint.saved(in: output).step }

    @Test("Requests queued during a step are served before the next step; training then finishes every step on unchanged base weights")
    func requestsServedBetweenSteps() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        let container = try #require(await engine.container)
        let before = await DistillTests.baseWeights(container)
        let request = Self.request(output: try DistillTests.scratch().appending(path: "adapter"), iterations: 8)
        let entered = Mutex(0)
        let gateAtFourth = Mutex<(queued: Int, inFlight: Int)?>(nil)
        let replies = Mutex<[Task<GenerationResponse, any Error>]>([])
        let count = 3
        let run = try await orchestrator.preemptibly { _, lease throws(BasaltError) in
            let counting = StepLease(enter: { () async throws(BasaltError) -> Int in
                let ticket = try await lease.enter()
                switch entered.withLock({ $0 += 1; return $0 }) {
                case 3:
                    let tasks = (0 ..< count).map { _ in Task { try await orchestrator.generate(request: GenerationRequest(prompt: "Hi", maxTokens: 4, temperature: 0)) } }
                    replies.withLock { $0 = tasks }
                    let gate = orchestrator.gate
                    while await gate.arriving + gate.queued < count, !Task.isCancelled { await Task.yield() }
                case 4:
                    let gate = orchestrator.gate
                    let state = (await gate.queued + gate.arriving, await gate.inFlight)
                    gateAtFourth.withLock { $0 = state }
                default: break
                }
                return ticket
            }, leave: lease.leave)
            do {
                return try await Self.run(engine, request, lease: counting)
            } catch {
                throw error as? BasaltError ?? .inferenceError("\(error)")
            }
        }
        for reply in replies.withLock({ $0 }) { #expect(try await reply.value.completionTokens > 0) }
        let state = try #require(gateAtFourth.withLock { $0 })
        #expect(state.queued == 0 && state.inFlight == 1, "the requests that arrived in section 3 left the gate before section 4 entered")
        #expect(entered.withLock { $0 } > 3 + request.iterations)
        #expect(run.losses.count == request.iterations)
        #expect(await orchestrator.gate.checkInvariants(idle: true).isEmpty)
        #expect(await DistillTests.baseWeights(container) == before)
    }

    @Test("Resumed from its step-2 checkpoint, a run continues the loss curve of the same run left uninterrupted")
    func resumeContinuesTheCurve() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let scratch = try DistillTests.scratch()
        let (full, copy) = (scratch.appending(path: "full"), scratch.appending(path: "resumed"))
        let copied = Mutex(false)
        let snapshot = StepLease(enter: {
            let source = AdapterCheckpoint.directory(in: full)
            if Self.savedStep(full) == 2, !copied.withLock({ $0 }) {
                try? FileManager.default.createDirectory(at: copy, withIntermediateDirectories: true)
                try? FileManager.default.copyItem(at: source, to: AdapterCheckpoint.directory(in: copy))
                copied.withLock { $0 = true }
            }
            return 0
        }, leave: { _ in })
        let reference = try await Self.run(engine, Self.request(output: full), lease: snapshot)
        #expect(!FileManager.default.fileExists(atPath: AdapterCheckpoint.directory(in: full).path), "a finished run removes its checkpoint")

        var resume = Self.request(output: copy)
        resume.resume = true
        let saved = try AdapterCheckpoint.saved(in: copy)
        #expect(saved.step == 2 && saved.losses == Array(reference.losses.prefix(2)))
        let resumed = try await Self.run(engine, resume)
        #expect(resumed.resumedAt == 2)
        #expect(resumed.losses == reference.losses)
        #expect(resumed.nllAfter == reference.nllAfter)
    }

    @Test("Cancelled inside the step after the step-2 checkpoint: base weights stay bit-identical and the checkpoint holds step 2")
    func cancelKeepsBaseWeights() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let before = await DistillTests.baseWeights(container)
        let output = try DistillTests.scratch().appending(path: "interrupted")
        let progress = AdapterCheckpoint.directory(in: output).appending(component: "progress.json")
        let cancelling = StepLease(enter: {
            if Self.savedStep(output) == 2 { withUnsafeCurrentTask { $0?.cancel() } }
            return 0
        }, leave: { _ in })
        let failure = await #expect(throws: BasaltError.self) {
            try await Task { try await Self.run(engine, Self.request(output: output), lease: cancelling) }.value
        }
        #expect(failure?.localizedDescription.contains("cancelled") == true, "\(String(describing: failure))")
        #expect(await DistillTests.baseWeights(container) == before)
        let saved = try JSONDecoder().decode(AdapterProgress.self, from: Data(contentsOf: progress))
        #expect(saved.step == 2 && saved.losses.count == 2)
        #expect(!FileManager.default.fileExists(atPath: output.appending(component: "adapters.safetensors").path))
    }

    @Test("Checkpoints fall every N steps but never on the last; resume refuses another model, other rows, other settings or a finished step count")
    func checkpointRules() {
        let request = Self.request(output: URL(fileURLWithPath: "/"))
        let identity = AdapterProgress(request, model: "m", data: "d", layerCount: 4)
        var progress = identity
        let due = (1 ... 6).map { _ in
            progress = progress.advanced(by: 1)
            return progress.checkpointDue(every: 2, of: 6)
        }
        #expect(due == [false, true, false, true, false, false])
        #expect(!progress.checkpointDue(every: 0, of: 100))
        #expect(identity.layers == [0, 1, 2, 3])
        #expect(progress.resumable(as: identity, iterations: 9) == nil)
        #expect(progress.resumable(as: identity, iterations: 6)?.contains("iterations must be larger") == true)
        #expect(progress.resumable(as: AdapterProgress(request, model: "other", data: "d", layerCount: 4), iterations: 9)?.contains("another model") == true)
        #expect(progress.resumable(as: AdapterProgress(request, model: "m", data: "other", layerCount: 4), iterations: 9)?.contains("training rows differ") == true)
        #expect(progress.resumable(as: AdapterProgress(request, model: "m", data: "d", layerCount: 5), iterations: 9)?.contains("same settings") == true)
        var otherRank = request
        otherRank.rank = 8
        #expect(progress.resumable(as: AdapterProgress(otherRank, model: "m", data: "d", layerCount: 4), iterations: 9)?.contains("same settings") == true)
        var resume = request
        (resume.resume, resume.onPolicy) = (true, true)
        #expect(throws: BasaltError.self) { try MLXInferenceEngine.validate(resume) }
    }

    @Test("Restore takes exactly the LoRA keys and shapes and Adam's m, v and step per key; anything else is refused before it touches the model")
    func restoreChecksKeys() {
        let trainable = ["a.lora_a": [4, 8], "a.lora_b": [8, 4]]
        let optimizer = trainable.reduce(into: [String: [Int]]()) { all, entry in
            (all["\(entry.key).m"], all["\(entry.key).v"], all["\(entry.key).step"]) = (entry.value, entry.value, [])
        }
        #expect(AdapterCheckpoint.mismatch(weights: trainable, optimizer: optimizer, trainable: trainable) == nil)
        #expect(AdapterCheckpoint.mismatch(weights: trainable, optimizer: [:], trainable: trainable) == nil)
        #expect(AdapterCheckpoint.mismatch(weights: trainable.merging(["model.embed_tokens.weight": [16, 8]]) { $1 }, optimizer: optimizer, trainable: trainable)?.contains("model.embed_tokens.weight") == true)
        #expect(AdapterCheckpoint.mismatch(weights: ["a.lora_a": [4, 8], "a.lora_b": [4, 4]], optimizer: optimizer, trainable: trainable) != nil)
        #expect(AdapterCheckpoint.mismatch(weights: trainable, optimizer: optimizer.merging(["layers.0.m": [1]]) { $1 }, trainable: trainable) != nil)
        #expect(AdapterCheckpoint.mismatch(weights: trainable, optimizer: optimizer.filter { $0.key != "a.lora_b.step" }, trainable: trainable) != nil)
        #expect(AdapterCheckpoint.mismatch(weights: trainable, optimizer: optimizer.merging(["a.lora_a.step": [2]]) { $1 }, trainable: trainable) != nil)
    }
}
