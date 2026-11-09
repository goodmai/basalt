import Foundation
import Synchronization
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Concurrent sidecars by rule: Gemma 4 decode drop while 1000 chunks embed on the ANE, against the MLX embedder and Laya decisions (#399 phase 3)", .serialized)
struct SidecarDecodeDropBenchmarkTests {
    static let env = ProcessInfo.processInfo.environment
    static let enabled = env["BASALT_E2E"] == "1"
    static let chatModel = enabled ? env["BASALT_E2E_SIDECAR_CHAT_MODEL"].map(GoldenLogitsRealModelTests.path) : nil
    static let embeddingBundle = env["BASALT_E2E_COREAI_EMBEDDING_BUNDLE"]
    static let mlxEmbedder = env["BASALT_E2E_SIDECAR_MLX_EMBEDDER"].map(GoldenLogitsRealModelTests.path)
    static let decisionBundle = env["BASALT_E2E_COREAI_DECISION_BUNDLE"]
    static let chunkCount = 1000
    static let chunksPerRequest = 50

    static let chunks: [String] = (0 ..< chunkCount).map { chunk in
        (0 ..< 12).map { "Section \(chunk).\($0): the survey of mill \(chunk * 13 + $0) recorded silt depth, flow rate and the keeper's notes on the spring flood." }
            .joined(separator: " ")
    }

    static let decision = SystemOneRequest(state: "The color is red.", questions: [
        "color": SystemOneQuestion(type: .choice, instructions: "Which color?", criteria: .named(["red": "Red.", "blue": "Blue."])),
    ])

    struct Lanes {
        let ane: any EmbeddingEngine
        let mlx: any EmbeddingEngine
        let laya: any DecisionEngine
        let plan: PlanVerdict
    }

    final class Load: Sendable {
        let done = Mutex(0)
        let busy = Mutex(0.0)
    }

    static func chat(_ orchestrator: ModelOrchestratorActor) async throws -> Double {
        var request = GenerationRequest(prompt: SystemOneBudgetBenchmarkTests.longPrompt, maxTokens: 512, temperature: 0)
        request.reasoningEffort = "none"
        return try await orchestrator.generate(request: request).tokensPerSecond
    }

    static func embedLoop(_ engine: any EmbeddingEngine, gate: GenerationGate, duty: Double, load: Load) async throws {
        while !Task.isCancelled {
            for start in stride(from: 0, to: chunkCount, by: chunksPerRequest) where !Task.isCancelled {
                let begin = ContinuousClock.now
                let batch = chunks[start ..< start + chunksPerRequest].map(EmbeddingInput.text)
                _ = try await engine.embed(batch, dimensions: nil, yieldingTo: gate)
                let seconds = (ContinuousClock.now - begin).inSeconds
                load.done.withLock { $0 += chunksPerRequest }
                load.busy.withLock { $0 += seconds }
                if duty < 1 { try? await Task.sleep(for: .seconds(seconds * (1 - duty) / duty)) }
            }
        }
    }

    static func decideLoop(_ engine: any DecisionEngine, duty: Double, load: Load) async throws {
        while !Task.isCancelled {
            let begin = ContinuousClock.now
            _ = try await engine.decide(decision)
            let seconds = (ContinuousClock.now - begin).inSeconds
            load.done.withLock { $0 += 1 }
            load.busy.withLock { $0 += seconds }
            if duty < 1 { try? await Task.sleep(for: .seconds(seconds * (1 - duty) / duty)) }
        }
    }

    static func arm(
        _ name: String, solo: Double, orchestrator: ModelOrchestratorActor, log: ModelRunLog, minimumDone: Int = 0,
        _ sidecar: @escaping @Sendable (Load) async throws -> Void
    ) async throws {
        let load = Load()
        let worker = Task { try await sidecar(load) }
        let start = ContinuousClock.now
        var rates: [Double] = []
        try await log.step(name) {
            while rates.count < 2 || load.done.withLock({ $0 }) < minimumDone {
                rates.append(try await chat(orchestrator))
            }
        }
        let wall = (ContinuousClock.now - start).inSeconds
        worker.cancel()
        _ = try? await worker.value
        let (done, busy) = (load.done.withLock { $0 }, load.busy.withLock { $0 })
        let decode = rates.reduce(0, +) / Double(rates.count)
        log.record("RESULT", String(
            format: "SIDECAR %@: decode %.2f tok/s over %d chats vs solo %.2f (%+.1f%%), sidecar %d items in %.1f s (%.1f/s, duty %.0f%%), %@",
            name, decode, rates.count, solo, (decode / solo - 1) * 100, done, wall, Double(done) / wall, busy / wall * 100, log.host()))
    }

    static func solo(_ name: String, orchestrator: ModelOrchestratorActor, log: ModelRunLog) async throws -> Double {
        let rates = try await log.step(name) { [try await chat(orchestrator), try await chat(orchestrator)] }
        let mean = rates.reduce(0, +) / 2
        log.record("RESULT", String(format: "SIDECAR %@: decode %.2f tok/s (%.2f, %.2f), %@", name, mean, rates[0], rates[1], log.host()))
        return mean
    }

    static func lanes(gate: GenerationGate, log: ModelRunLog) async throws -> Lanes {
        let ane = try await log.step("load Core AI embeddings") { try await CoreAIEmbeddingService(directory: URL(fileURLWithPath: try #require(embeddingBundle))) }
        let mlxPath = try #require(mlxEmbedder)
        let mlx = try await log.step("load MLX embedder") { try await EmbeddingService(embeddingModel: (id: "mlx-embedder", directory: URL(fileURLWithPath: mlxPath)), rerankModel: nil) }
        let laya = try await log.step("load Laya") { try await CoreAIDecisionEngine(directory: URL(fileURLWithPath: try #require(decisionBundle))) }
        let plan = ComputePlan.admit([
            Workload(name: "chat", role: "chat", unit: .gpu, kind: .decode, residentBytes: 0),
            Workload(name: "embeddinggemma", role: "embed", unit: .ane, kind: .encoder, residentBytes: 0),
            Workload(name: "mlx-embedder", role: "embed-gpu", unit: .gpu, kind: .encoder, residentBytes: 0),
            Workload(name: "laya", role: "decide", unit: .ane, kind: .encoder, residentBytes: 0),
        ], budgetBytes: 0, power: PowerSource.current())
        for line in ComputePlan.describe(plan) { log.record("plan", line) }
        return Lanes(
            ane: try #require(BesideDecodeEmbeddingEngine.serving(ane, besideDecode: plan.runsBesideDecode("embed"))),
            mlx: try #require(BesideDecodeEmbeddingEngine.serving(mlx, besideDecode: plan.runsBesideDecode("embed-gpu"))),
            laya: try #require(GatedDecisionEngine.serving(laya, gate: gate, besideDecode: plan.runsBesideDecode("decide"))),
            plan: plan)
    }

    @Test("Gemma 4 decode beside 1000 Core AI chunks (continuous and 25% duty), the gated and ungated MLX embedder and Laya at 25% duty; drops, sidecar throughput and load are logged",
          .enabled(if: chatModel != nil && embeddingBundle != nil && mlxEmbedder != nil && decisionBundle != nil,
                   "set BASALT_E2E=1, BASALT_E2E_SIDECAR_CHAT_MODEL, BASALT_E2E_COREAI_EMBEDDING_BUNDLE, BASALT_E2E_SIDECAR_MLX_EMBEDDER and BASALT_E2E_COREAI_DECISION_BUNDLE"))
    func decodeDrop() async throws {
        let chatPath = try #require(Self.chatModel)
        let log = try await ModelRunLog("sidecar-decode-drop", modelPath: chatPath)
        defer { log.releaseGPU() }
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine(), telemetry: TelemetryTracker())
        try await log.step("load chat") { try await orchestrator.loadModel(path: chatPath) }
        let gate = orchestrator.gate
        let lanes = try await Self.lanes(gate: gate, log: log)
        #expect((lanes.ane is BesideDecodeEmbeddingEngine) == !lanes.plan.degraded && !(lanes.mlx is BesideDecodeEmbeddingEngine))
        _ = try await Self.chat(orchestrator)
        _ = try await lanes.ane.embed(Array(Self.chunks.prefix(4)).map(EmbeddingInput.text), dimensions: nil, yieldingTo: gate)
        _ = try await lanes.mlx.embed(Array(Self.chunks.prefix(4)).map(EmbeddingInput.text), dimensions: nil, yieldingTo: gate)
        let before = try await Self.solo("solo before", orchestrator: orchestrator, log: log)
        try await Self.arm("Core AI embeddings continuous", solo: before, orchestrator: orchestrator, log: log, minimumDone: Self.chunkCount) { [ane = lanes.ane] in
            try await Self.embedLoop(ane, gate: gate, duty: 1, load: $0)
        }
        try await Self.arm("Core AI embeddings 25% duty", solo: before, orchestrator: orchestrator, log: log, minimumDone: Self.chunkCount) { [ane = lanes.ane] in
            try await Self.embedLoop(ane, gate: gate, duty: 0.25, load: $0)
        }
        try await Self.arm("MLX embedder gated", solo: before, orchestrator: orchestrator, log: log) { [mlx = lanes.mlx] in
            try await Self.embedLoop(mlx, gate: gate, duty: 1, load: $0)
        }
        try await Self.arm("MLX embedder ungated", solo: before, orchestrator: orchestrator, log: log) { [mlx = lanes.mlx] in
            try await Self.embedLoop(BesideDecodeEmbeddingEngine(inner: mlx), gate: gate, duty: 1, load: $0)
        }
        try await Self.arm("Laya decisions 25% duty", solo: before, orchestrator: orchestrator, log: log) { [laya = lanes.laya] in
            try await Self.decideLoop(laya, duty: 0.25, load: $0)
        }
        let after = try await Self.solo("solo after", orchestrator: orchestrator, log: log)
        log.record("RESULT", String(format: "SIDECAR solo drift %+.1f%%", (after / before - 1) * 100))
    }
}
