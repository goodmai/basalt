import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXOptimizers
import Synchronization
@testable import BasaltCore
@testable import BasaltInference

@Suite("Preemptible distillation on a real model: TTFT idle and during training, step time, throughput against the exclusive run (#399 phase 2)", .serialized)
struct PreemptibleTrainingRealModelTests {
    final class Sections: Sendable {
        let list = Mutex<[(start: ContinuousClock.Instant, end: ContinuousClock.Instant?)]>([])
    }

    static let env = ProcessInfo.processInfo.environment
    static let models = NF4Run.models("BASALT_E2E_PREEMPT_MODELS")
    static let steps = Int(env["BASALT_E2E_PREEMPT_STEPS"] ?? "") ?? 30
    static let requests = Int(env["BASALT_E2E_PREEMPT_REQUESTS"] ?? "") ?? 20

    static func tasks(_ range: Range<Int>) -> [TrainingTask] {
        range.map { seed in
            let (a, b, c) = (12 + seed % 80, 3 + seed % 17, 40 + seed % 300)
            return TrainingTask(
                prompt: "A shop receives \(a) boxes with \(b) items in each and sells \(c) items. How many items are left? Show brief working, then end with a final line 'Answer: <number>'.",
                answer: "The shop starts with \(a) boxes of \(b) items, which is \(a) times \(b) = \(a * b) items. It sells \(c) of them, so \(a * b) minus \(c) leaves \(a * b - c) items on the shelves and in storage together.\nAnswer: \(a * b - c)")
        }
    }

    static func prompt(_ seed: Int) -> String {
        let words = "kernel buffer cache tensor stream decode anchor memory budget window thermal clock".split(separator: " ")
        return "Request \(seed). Name the most frequent word in this list in one word.\n" + (0 ..< 160).map { "\(words[($0 * 7 + seed) % words.count])\(($0 * 13 + seed) % 997)" }.joined(separator: " ")
    }

    static func ttft(_ orchestrator: ModelOrchestratorActor, _ seed: Int, log: ModelRunLog? = nil) async throws -> Double {
        let start = ContinuousClock.now
        var first: Duration?
        var engine: GenerationResponse?
        for await chunk in try await orchestrator.generateStream(request: GenerationRequest(prompt: prompt(seed), maxTokens: 8, temperature: 0)) {
            switch chunk {
            case .text, .reasoning: first = first ?? ContinuousClock.now - start
            case .metadata(let response): engine = response
            default: break
            }
        }
        let client = try #require(first).inSeconds
        log?.record("request", "\(seed) client TTFT \(format(client)) s, engine TTFT \(format(engine?.timeToFirstToken ?? .nan)) s, prompt \(engine?.promptTokens ?? 0) cached \(engine?.cachedPromptTokens ?? 0), memory \(engine.map { "\($0.memory)" } ?? "-")")
        return client
    }

    static func checksum(_ container: ModelContainer) async -> [String: UInt64] {
        await container.perform { context in
            Dictionary(uniqueKeysWithValues: context.model.parameters().flattened().map { name, array in
                (name, array.view(dtype: .uint8).asType(.uint64).sum().item(UInt64.self))
            })
        }
    }

    static func exclusiveSeconds(_ engine: MLXInferenceEngine, _ container: ModelContainer, rows: [DistillRow], steps: Int) async throws -> Double {
        let adapter = try await engine.adapterSteps(layers: nil, rank: 16, lease: nil, what: "distillation")
        return try await container.performEvaluating(nonSendable: ()) { context, _ in
            var state = AdapterState(optimizer: Adam(learningRate: 1e-4))
            let start = ContinuousClock.now
            _ = try adapter.withLoRA(&state, context: context) { context, optimizer, _ in
                try (0 ..< steps).map { try MLXInferenceEngine.distillStep(rows, index: $0, batchSize: 2, optimizer: optimizer, model: context.model) }
            }
            return (ContinuousClock.now - start).inSeconds
        }
    }

    static func recorded(_ lease: StepLease, _ sections: Sections) -> StepLease {
        StepLease(enter: { () async throws(BasaltError) -> Int in
            let ticket = try await lease.enter()
            sections.list.withLock { $0.append((ContinuousClock.now, nil)) }
            return ticket
        }, leave: { ticket in
            sections.list.withLock { $0[$0.count - 1].end = ContinuousClock.now }
            await lease.leave(ticket)
        })
    }

    static func request(iterations: Int) -> DistillRequest {
        var request = DistillRequest(prompts: "", teacher: TeacherSpec(url: "https://teacher.invalid/v1", model: "teacher"), gate: GateRequest(heldOut: ["Hello."]))
        (request.iterations, request.batchSize, request.maxSeqLen, request.checkpointEvery) = (iterations, 2, 1024, 0)
        request.output = FileManager.default.temporaryDirectory.appending(path: "preempt-\(UUID().uuidString)").path
        return request
    }

    static func distill(_ orchestrator: ModelOrchestratorActor, _ engine: MLXInferenceEngine, iterations: Int, sections: Sections) async throws(BasaltError) -> DistillRun {
        let request = request(iterations: iterations)
        let directory = try await engine.reservedDirectory(request.output)
        return try await orchestrator.preemptibly { _, lease throws(BasaltError) in
            let steps = try await engine.adapterSteps(layers: nil, rank: request.rank, lease: recorded(lease, sections), what: "distillation")
            return try await MLXInferenceEngine.distillAdapter(tasks(0 ..< 64), heldOut: tasks(64 ..< 68), request: request, directory: directory, steps: steps)
        }
    }

    static func stepSeconds(_ sections: [(start: ContinuousClock.Instant, end: ContinuousClock.Instant?)], first: Int, count: Int) -> (each: [Double], wall: Double) {
        let window = Array(sections.dropFirst(first).prefix(count))
        let each = window.compactMap { section in section.end.map { ($0 - section.start).inSeconds } }
        let wall = window.first.flatMap { start in window.last?.end.map { ($0 - start.start).inSeconds } } ?? .nan
        return (each, wall)
    }

    static func percentile(_ values: [Double], _ share: Double) -> Double { ThermalStressRealModelTests.percentile(values, share) }

    static func format(_ value: Double) -> String { String(format: "%.3f", value) }
}

extension PreemptibleTrainingRealModelTests {
    @Test("TTFT p95 idle and during a running distillation, step time, and steps/s with no requests against today's exclusive run",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_PREEMPT_MODELS"), arguments: models)
    func preemptible(entry: String) async throws {
        let log = try await ModelRunLog("preempt", modelPath: NF4Run.path(entry))
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 1 << 30))
        try await log.step("load") { try await engine.load(modelPath: NF4Run.path(entry)) }
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        let container = try #require(await engine.container)
        let base = await Self.checksum(container)
        let rows = try await container.performEvaluating(nonSendable: ()) { context, _ in
            try MLXInferenceEngine.distillRows(Self.tasks(0 ..< 64), heldOut: Self.tasks(64 ..< 68), maxSeqLen: 1024, context)
        }
        log.record("data", "\(rows.train.count) rows, \(rows.train.map(\.tokens.count).reduce(0, +) / rows.train.count) tokens each on average, batch 2, rank 16")
        _ = try await Self.ttft(orchestrator, 9_999)
        _ = try await Self.exclusiveSeconds(engine, container, rows: rows.train, steps: 2)

        var idle: [Double] = []
        for seed in 0 ..< Self.requests { idle.append(try await Self.ttft(orchestrator, seed, log: log)) }
        log.record("idle", "TTFT p50 \(Self.format(Self.percentile(idle, 0.5))) s, p95 \(Self.format(Self.percentile(idle, 0.95))) s over \(idle.count)")

        let exclusive = try await Self.exclusiveSeconds(engine, container, rows: rows.train, steps: Self.steps)
        log.record("exclusive", "\(Self.steps) steps in \(Self.format(exclusive)) s: \(Self.format(Double(Self.steps) / exclusive)) steps/s")

        let quiet = Sections()
        _ = try await Self.distill(orchestrator, engine, iterations: Self.steps, sections: quiet)
        let (quietSteps, quietWall) = Self.stepSeconds(quiet.list.withLock { $0 }, first: 1 + rows.heldOut.count, count: Self.steps)
        let ratio = exclusive / quietWall
        log.record("preemptible", "\(Self.steps) steps in \(Self.format(quietWall)) s: \(Self.format(Double(Self.steps) / quietWall)) steps/s, \(Self.format(ratio * 100))% of exclusive; step p50 \(Self.format(Self.percentile(quietSteps, 0.5))) s")
        #expect(await Self.checksum(container) == base, "base weights after a completed run")

        try await duringTraining(orchestrator, engine, idle: idle, ratio: ratio, base: base, log: log)
    }

    func duringTraining(
        _ orchestrator: ModelOrchestratorActor, _ engine: MLXInferenceEngine, idle: [Double], ratio: Double, base: [String: UInt64], log: ModelRunLog
    ) async throws {
        let container = try #require(await engine.container)
        let sections = Sections()
        let training = Task { try await Self.distill(orchestrator, engine, iterations: 10_000, sections: sections) }
        while sections.list.withLock({ $0.count }) < 8 { try await Task.sleep(for: .milliseconds(100)) }
        let stepTime = Self.percentile(Self.stepSeconds(sections.list.withLock { $0 }, first: 5, count: 3).each, 0.5)
        var generator = SplitMix64(seed: 399)
        var during: [Double] = []
        for seed in 0 ..< Self.requests {
            try await Task.sleep(for: .seconds(Double.random(in: 0 ..< stepTime * 1.5, using: &generator)))
            during.append(try await Self.ttft(orchestrator, 1000 + seed, log: log))
        }
        training.cancel()
        let outcome = await training.result
        let all = sections.list.withLock { $0 }
        let steps = Self.stepSeconds(all, first: 5, count: all.count - 5).each
        log.record("during", "TTFT p50 \(Self.format(Self.percentile(during, 0.5))) s, p95 \(Self.format(Self.percentile(during, 0.95))) s over \(during.count); \(all.count - 5) steps ran, step p50 \(Self.format(Self.percentile(steps, 0.5))) s, p95 \(Self.format(Self.percentile(steps, 0.95))) s, max \(Self.format(steps.max() ?? .nan)) s; training ended with \(String(describing: outcome).prefix(160))")
        let budget = Self.percentile(idle, 0.95) + (steps.max() ?? 0)
        log.record("RESULT", "TTFT p95 idle \(Self.format(Self.percentile(idle, 0.95))) s -> during \(Self.format(Self.percentile(during, 0.95))) s (budget idle p95 + max step \(Self.format(budget)) s); throughput \(Self.format(ratio * 100))% of exclusive")
        #expect(Self.percentile(during, 0.95) <= budget)
        #expect(ratio >= 0.9)
        #expect(await Self.checksum(container) == base, "base weights after a cancelled run")
    }
}
