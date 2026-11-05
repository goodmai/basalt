import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Per-request LoRA adapters on real models: exactness across switches, switch latency, decode speed, resident memory (#179)", .serialized)
struct AdapterServingRealModelTests {
    static let environment = ProcessInfo.processInfo.environment
    static var modelPath: String? { environment["BASALT_E2E"] == "1" ? environment["BASALT_E2E_LORA_MODEL_PATH"] : nil }
    static var refusedModelPath: String? { environment["BASALT_E2E"] == "1" ? environment["BASALT_E2E_LORA_REFUSED_MODEL_PATH"] : nil }
    static let prompt = "What is 17 * 23? Answer with the number only, then one sentence on how you got it."
    static let switches = 50

    static func trained(_ container: ModelContainer) async throws -> [String: URL] {
        if let a = environment["BASALT_E2E_LORA_ADAPTER_A"], let b = environment["BASALT_E2E_LORA_ADAPTER_B"] {
            return ["a": URL(fileURLWithPath: a), "b": URL(fileURLWithPath: b)]
        }
        let blocks = await container.perform { ($0.model as? LoRAModel)?.loraLayers.count ?? 0 }
        var directories: [String: URL] = [:]
        for (name, seed) in [("a", UInt64(1)), ("b", UInt64(2))] {
            let directory = FileManager.default.temporaryDirectory.appending(path: "basalt-lora-\(name)-\(UUID().uuidString)")
            var request = TrainRequest(tasks: .arithmetic(count: 16, seed: seed), gate: GateRequest(heldOut: ["x"]))
            (request.layers, request.rank, request.iterations, request.learningRate) = (Array(max(0, blocks - 4) ..< blocks), 8, 30, 1e-4)
            let pairs = TrainingTask.arithmetic(count: 16, seed: seed)
            _ = try await container.performEvaluating(nonSendable: ()) { [request] context, _ in
                try MLXInferenceEngine.trainAdapter(pairs, heldOut: Array(pairs.prefix(2)), request: request, directory: directory, context: context)
            }
            directories[name] = directory
        }
        return directories
    }

    static func ask(_ orchestrator: ModelOrchestratorActor, _ adapter: String?, maxTokens: Int = 48) async throws -> GenerationResponse {
        try await orchestrator.generate(request: greedy(prompt, key: nil, maxTokens: maxTokens).withAdapter(adapter))
    }

    static func percentile(_ values: [Duration], _ share: Double) -> Duration {
        let sorted = values.sorted()
        return sorted[min(sorted.count - 1, Int(Double(sorted.count) * share))]
    }

    @Test("A → B → base → A each equal their first, never-switched output; 50 alternating switches; unfused decode ≥ 0.9× fused; memory delta",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_LORA_MODEL_PATH"))
    func servesAdapters() async throws {
        let path = try #require(Self.modelPath)
        let engine = MLXInferenceEngine(promptCacheBytes: 0)
        try await engine.load(modelPath: path)
        let container = try #require(await engine.container)
        let directories = try await Self.trained(container)
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        let base = try await Self.ask(orchestrator, nil).generatedText
        let before = Memory.activeMemory
        try await orchestrator.loadAdapters(directories)
        let firstA = try await Self.ask(orchestrator, "a").generatedText
        let firstB = try await Self.ask(orchestrator, "b").generatedText
        let resident = Memory.activeMemory - before
        #expect(try await Self.ask(orchestrator, nil).generatedText == base)
        #expect(try await Self.ask(orchestrator, "a").generatedText == firstA)
        print("[#179] \(path): base \(base.debugDescription) | a \(firstA.debugDescription) | b \(firstB.debugDescription)")

        var latencies: [Duration] = []
        for index in 0 ..< Self.switches {
            let started = ContinuousClock.now
            try await engine.useAdapter(index.isMultiple(of: 2) ? "b" : "a")
            latencies.append(ContinuousClock.now - started)
        }
        #expect(try await Self.ask(orchestrator, "a").generatedText == firstA)

        let unfused = try await Self.ask(orchestrator, "a", maxTokens: 256).tokensPerSecond
        let plain = try await Self.ask(orchestrator, nil, maxTokens: 256).tokensPerSecond
        let weights = try LoRAContainer.from(directory: try #require(directories["a"]))
        try await container.perform { context in try weights.fuse(with: context.model) }
        let fused = try await engine.generate(request: greedy(Self.prompt, key: nil, maxTokens: 256)).tokensPerSecond
        print("[#179] \(path): switch median \(Self.percentile(latencies, 0.5)), p90 \(Self.percentile(latencies, 0.9)) over \(Self.switches); decode tok/s base \(plain), unfused \(unfused), fused \(fused) (\(unfused / fused)×); adapters resident +\(resident >> 20) MB")
        #expect(unfused >= 0.9 * fused)
    }

    @Test("A base whose architecture takes no LoRA (Gemma 4 through MLXVLM) refuses the adapter at startup, naming it",
          .enabled(if: refusedModelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_LORA_REFUSED_MODEL_PATH"))
    func refusesArchitecture() async throws {
        let engine = MLXInferenceEngine(promptCacheBytes: 0)
        try await engine.load(modelPath: try #require(Self.refusedModelPath))
        let directory = try LabGateTests.adapterDirectory(rank: 2, storedRank: 2)
        let refusal = await #expect(throws: BasaltError.self) { try await engine.loadAdapters(["a": directory]) }
        print("[#179] refused: \(refusal?.localizedDescription ?? "nothing")")
        #expect(refusal?.localizedDescription.contains("architecture takes no LoRA") == true)
        #expect(await engine.adapterNames.isEmpty)
    }
}
