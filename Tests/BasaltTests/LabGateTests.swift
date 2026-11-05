import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Quality gate and live adapters: rank-1 and directory LoRA, kept or undone by the gate, restored bit for bit (#332)", .serialized)
struct LabGateTests {
    static let downProj = "model.layers.0.mlp.down_proj"
    static let heldOut = ["The quick brown fox", "2 + 2 ="]

    static func engine() async throws -> MLXInferenceEngine {
        let engine = try await TinyModel.engine(.qwen3)
        await engine.skipJournalForDeltaTests()
        return engine
    }

    static func rankOne(scale: Float) -> AdapterSource {
        .rankOne(module: downProj, input: Array(repeating: 1, count: 128), output: Array(repeating: 1, count: 64), scale: scale)
    }

    static func loraLayers(_ container: ModelContainer) async -> Int {
        await container.perform { context in context.model.namedModules().filter { $0.1 is LoRALayer }.count }
    }

    static func adapterDirectory(rank: Int, storedRank: Int) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: "lab-adapter-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let config = #"{"fine_tune_type": "lora", "num_layers": 4, "lora_parameters": {"rank": \#(rank), "scale": 2.0, "keys": ["self_attn.q_proj"]}}"#
        try Data(config.utf8).write(to: directory.appending(path: "adapter_config.json"))
        var arrays: [String: MLXArray] = [:]
        for layer in 0 ..< 4 {
            arrays["model.layers.\(layer).self_attn.q_proj.lora_a"] = MLXRandom.normal([64, storedRank])
            arrays["model.layers.\(layer).self_attn.q_proj.lora_b"] = MLXRandom.normal([storedRank, 64])
        }
        try save(arrays: arrays, url: directory.appending(path: "adapters.safetensors"))
        return directory
    }

    @Test("A rank-1 adapter with scale 0 passes the gate, is kept without expiry and marks the model; undo restores the logits bit for bit")
    func identityPasses() async throws {
        let engine = try await Self.engine()
        let container = try #require(await engine.container)
        let logits = try await WeightDeltaTests.logits(container)
        let verdict = try await engine.applyAdapter(Self.rankOne(scale: 0), gate: GateRequest(heldOut: Self.heldOut))
        let report = try #require(verdict.report)
        #expect(report.passed, "\(report.checks)")
        #expect(Set(report.checks.map(\.name)).isSuperset(of: ["probe perplexity", "held-out mean KL", "held-out perplexity rise"]))
        #expect(report.compare?.rows.count == Self.heldOut.count)
        let overlay = try #require(verdict.overlay)
        #expect(overlay.expiresAt == nil)
        #expect(await engine.modelModified)
        #expect(await engine.activeDeltas.map(\.id) == [overlay.id])
        #expect(await Self.loraLayers(container) == 1)
        try await engine.undo(overlay.id)
        #expect(await Self.loraLayers(container) == 0)
        #expect(try await WeightDeltaTests.logits(container) == logits)
        #expect(await !engine.modelModified)
    }

    @Test("A rank-1 adapter with a huge scale fails the KL check and is undone; the logits come back bit for bit")
    func hugeScaleUndone() async throws {
        let engine = try await Self.engine()
        let container = try #require(await engine.container)
        let logits = try await WeightDeltaTests.logits(container)
        let verdict = try await engine.applyAdapter(Self.rankOne(scale: 1000), gate: GateRequest(heldOut: Self.heldOut, maxKL: 0.01, maxPerplexityRise: 0.01))
        #expect(verdict.overlay == nil)
        let report = try #require(verdict.report)
        #expect(!report.passed)
        let kl = try #require(report.checks.first { $0.name == "held-out mean KL" })
        #expect(!kl.passed && kl.value > kl.limit && kl.value.isFinite)
        #expect(try await WeightDeltaTests.logits(container) == logits)
        #expect(await Self.loraLayers(container) == 0)
        #expect(await !engine.modelModified)
        #expect(await engine.overlays().isEmpty)
    }

    @Test("gate_check lifts a live weight delta for its baseline and leaves it applied")
    func gateLiftsDeltas() async throws {
        let engine = try await Self.engine()
        let container = try #require(await engine.container)
        let info = try await engine.applyDelta(WeightDelta(tensor: "model.norm.weight", values: [5]))
        let applied = try await WeightDeltaTests.logits(container)
        let report = try await engine.gate(GateRequest(heldOut: Self.heldOut, maxKL: 0.0001))
        #expect(report.checks.first { $0.name == "held-out mean KL" }.map { $0.value > 0 } == true)
        #expect(try await WeightDeltaTests.logits(container) == applied)
        try await engine.undo(info.id)
    }

    @Test("An mlx-lm adapter directory applies live with a TTL and changes the logits; undo removes every LoRA layer and restores them bit for bit")
    func directoryRoundTrip() async throws {
        let engine = try await Self.engine()
        let container = try #require(await engine.container)
        let logits = try await WeightDeltaTests.logits(container)
        let directory = try Self.adapterDirectory(rank: 2, storedRank: 2)
        let verdict = try await engine.applyAdapter(.directory(directory.path), gate: nil)
        let overlay = try #require(verdict.overlay)
        #expect(verdict.report == nil && overlay.expiresAt != nil)
        #expect(try await WeightDeltaTests.logits(container) != logits)
        #expect(await Self.loraLayers(container) == 4)
        await #expect(throws: BasaltError.self) { try await engine.applyAdapter(.directory(directory.path), gate: nil) }
        await #expect(throws: BasaltError.self) { try await engine.applyAdapter(.rankOne(module: "model.layers.0.self_attn.q_proj", input: [1], output: [1], scale: 1), gate: nil) }
        try await engine.undo(overlay.id)
        #expect(await Self.loraLayers(container) == 0)
        #expect(try await WeightDeltaTests.logits(container) == logits)
        #expect(await !engine.modelModified)
    }

    @Test("An adapter whose tensors do not fit the model, an unknown layer or a wrong-sized vector is refused and leaves the model untouched")
    func refused() async throws {
        let engine = try await Self.engine()
        let container = try #require(await engine.container)
        let directory = try Self.adapterDirectory(rank: 2, storedRank: 3)
        await #expect(throws: BasaltError.self) { try await engine.applyAdapter(.directory(directory.path), gate: nil) }
        await #expect(throws: BasaltError.self) { try await engine.applyAdapter(.directory("/nonexistent"), gate: nil) }
        await #expect(throws: BasaltError.self) { try await engine.applyAdapter(.rankOne(module: "model.layers.0.mlp", input: [1], output: [1], scale: 1), gate: nil) }
        await #expect(throws: BasaltError.self) { try await engine.applyAdapter(.rankOne(module: Self.downProj, input: [1], output: [1], scale: 1), gate: nil) }
        await #expect(throws: BasaltError.self) { try await engine.applyAdapter(Self.rankOne(scale: 1), gate: GateRequest(heldOut: ["x"], maxKL: -1)) }
        #expect(await Self.loraLayers(container) == 0)
        #expect(await !engine.modelModified)
        #expect(await engine.overlays().isEmpty)
    }

    @Test("Probe checks count only a regression against the candidate; held-out checks fail on non-finite values and report them finite")
    func checkMath() {
        let missed = ModelProbeTests.facts { $0.perplexity = 200; $0.answers = [.init(expected: "8", got: "7")] }
        let close = ModelProbeTests.facts { $0.perplexity = 215; $0.answers = [.init(expected: "8", got: "7")] }
        let far = ModelProbeTests.facts { $0.perplexity = 230; $0.answers = [.init(expected: "8", got: "7")] }
        let answered = ModelProbeTests.facts { $0.perplexity = 200; $0.answers = [.init(expected: "8", got: " 8")] }
        #expect(MLXInferenceEngine.probeChecks(close, pristine: missed, family: nil, rise: 0.1).allSatisfy(\.passed))
        #expect(MLXInferenceEngine.probeChecks(far, pristine: missed, family: nil, rise: 0.1).map(\.passed) == [false, true])
        #expect(MLXInferenceEngine.probeChecks(missed, pristine: answered, family: nil, rise: 0.1).map(\.passed) == [true, false])
        let rows = [CompareRow(prompt: "p", baselineText: "", overlayText: "", klDivergence: .infinity, baselinePerplexity: 2, overlayPerplexity: .nan)]
        let checks = MLXInferenceEngine.heldOutChecks(rows, GateRequest(heldOut: ["p"]))
        #expect(checks.map(\.passed) == [false, false])
        #expect(checks.allSatisfy(\.value.isFinite))
    }

    @Test("adapter_apply and gate_check arguments are checked at the boundary")
    func mcpRejects() throws {
        let cases: [(ResearchTool, [String: Any], String)] = [
            (.gateCheck, [:], "heldOut must be an array of 1...16 strings"),
            (.gateCheck, ["heldOut": ["h"], "target": [["prompt": "p"]]], "Missing required argument: expected"),
            (.adapterApply, [:], "Missing required argument: module"),
            (.adapterApply, ["directory": "/a", "gate": 3], "gate must be an object"),
            (.adapterApply, ["module": "m", "input": [1.0], "output": [Double]()], "output must be a non-empty array of numbers"),
        ]
        for (tool, args, fragment) in cases {
            let failure = #expect(throws: MCPFailure.self) { try ResearchCall(tool, args) }
            #expect(failure?.code == -32602, "\(tool)")
            #expect(failure?.message.contains(fragment) == true, "\(tool): \(failure?.message ?? "no error")")
        }
        #expect(try ResearchCall(.adapterApply, ["directory": "/a"]) == .adapter(.directory("/a"), gate: nil))
    }
}
