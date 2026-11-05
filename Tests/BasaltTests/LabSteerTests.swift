import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Contrastive steering finds a direction, ranks heads, steers held-out prompts and builds the rank-1 commit (#332)", .serialized)
struct LabSteerTests {
    static var searchArguments: [String: Any] {
        ["positive": ["a", "b"], "negative": ["c", "d"], "heldOut": ["e"], "topHeads": 2, "scales": [1.0, 2.0]]
    }

    static var commitArguments: [String: Any] {
        ["layer": 1, "direction": [0.5, -1.0], "scale": 2.0, "positive": ["a"], "heldOut": ["e"], "maxKL": 0.25]
    }

    static var searchRequest: ContrastRequest {
        var request = ContrastRequest(positive: ["a", "b"], negative: ["c", "d"], heldOut: ["e"])
        request.topHeads = 2
        request.scales = [1, 2]
        return request
    }

    static var commitRequest: SteeringCommit {
        var commit = SteeringCommit(layer: 1, direction: [0.5, -1], scale: 2, gate: GateRequest(heldOut: ["e"], maxKL: 0.25))
        commit.positive = ["a"]
        return commit
    }

    static let positive = ["aaaa aaaa a", "a aaa aa aaaa", "aaaaaaaaaa a"]
    static let negative = ["zzzz zzzz z", "z zzz zz zzzz", "zzzzzzzzzz z"]

    static var contrast: ContrastRequest {
        var request = ContrastRequest(positive: positive, negative: negative, heldOut: ["the cat sat", "on the mat"])
        request.topHeads = 3
        request.scales = [1, 4, 16]
        return request
    }

    @Test("Two clearly different classes separate at some layer, the best layer has the top separation, and its direction is a unit vector of the hidden size")
    func directions() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let anatomy = try await engine.anatomy()
        let report = try await engine.contrast(Self.contrast)
        #expect(report.layers.map(\.index) == Array(0 ..< anatomy.layers.count))
        #expect(report.layers.contains { $0.separation > 0 && $0.norm > 0 })
        #expect(report.layers.first { $0.index == report.bestLayer }?.separation == report.layers.map(\.separation).max())
        #expect(report.direction.count == anatomy.hiddenSize)
        #expect(abs(report.direction.reduce(0) { $0 + $1 * $1 } - 1) < 1e-3)
    }

    @Test("Heads come from the best layer or the one before, largest effect first; trials report KL growing with the scale; no overlay is left behind")
    func headsAndTrials() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let report = try await engine.contrast(Self.contrast)
        #expect(report.heads.count == 3)
        #expect(report.heads.allSatisfy { [report.bestLayer - 1, report.bestLayer].contains($0.layer) && $0.effect.isFinite })
        #expect(zip(report.heads, report.heads.dropFirst()).allSatisfy { $0.effect >= $1.effect })
        #expect(report.trials.map(\.scale) == [1, 4, 16])
        #expect(report.trials.allSatisfy { $0.klDivergence > 0 && $0.projectionShift.isFinite })
        #expect(zip(report.trials, report.trials.dropFirst()).allSatisfy { $0.klDivergence < $1.klDivergence })
        #expect(await engine.overlays().isEmpty)
        var restricted = Self.contrast
        (restricted.layers, restricted.topHeads) = ([2, 1], 0)
        let narrow = try await engine.contrast(restricted)
        #expect(narrow.layers.map(\.index) == [1, 2] && narrow.heads.isEmpty && [1, 2].contains(narrow.bestLayer))
    }

    @Test("Requests outside the bounds are refused before any forward",
          arguments: [
            ContrastRequest(positive: ["a"], negative: ["b", "c"], heldOut: ["d"]),
            ContrastRequest(positive: ["a", "b"], negative: ["c", "d"], heldOut: []),
            ContrastRequest(positive: ["a", "b"], negative: ["c", "d"], heldOut: ["e"], layers: [9]),
            ContrastRequest(positive: ["a", "b"], negative: ["c", "d"], heldOut: ["e"], layers: []),
          ])
    func refused(request: ContrastRequest) async throws {
        let engine = try await TinyModel.engine(.qwen3)
        await #expect(throws: BasaltError.self) { try await engine.contrast(request) }
    }

    @Test("The rank-1 input has dot 1 with the mean input, the output is the direction, and a wrong size or a zero mean is refused")
    func rankOne() throws {
        let source = try MLXInferenceEngine.rankOne(module: "m", meanInput: [3, 4], direction: [1, 0, 0], scale: 2, outputs: 3)
        guard case .rankOne(let module, let input, let output, let scale) = source else {
            Issue.record("expected a rank-1 source, got \(source)")
            return
        }
        #expect(module == "m" && output == [1, 0, 0] && scale == 2)
        #expect(abs(input[0] * 3 + input[1] * 4 - 1) < 1e-6)
        #expect(throws: BasaltError.self) { try MLXInferenceEngine.rankOne(module: "m", meanInput: [3, 4], direction: [1, 0], scale: 2, outputs: 3) }
        #expect(throws: BasaltError.self) { try MLXInferenceEngine.rankOne(module: "m", meanInput: [0, 0], direction: [1, 0, 0], scale: 2, outputs: 3) }
    }

    @Test("The commit targets the dense down projection, falls back to the attention output on a MoE layer, and finds no layer past the end")
    func target() throws {
        let dense = try TinyModel.make(.qwen3)
        let moe = try TinyModel.make(.qwen3Moe)
        #expect(MLXInferenceEngine.decoderLayer(2, in: dense)?.path == "model.layers.2")
        #expect(MLXInferenceEngine.decoderLayer(2, in: dense).flatMap { MLXInferenceEngine.steeringTarget(in: $0.layer) }?.key == "mlp.down_proj")
        #expect(MLXInferenceEngine.decoderLayer(1, in: moe).flatMap { MLXInferenceEngine.steeringTarget(in: $0.layer) }?.key == "self_attn.o_proj")
        #expect(MLXInferenceEngine.decoderLayer(4, in: dense) == nil)
    }

    @Test("The commit source has the module's input and output sizes, and the measuring tap is gone afterwards")
    func commitSource() async throws {
        let container = try TinyModel.container(.qwen3)
        let direction = (0 ..< 64).map { Float($0 % 5) - 2 }
        var commit = SteeringCommit(layer: 2, direction: direction, scale: 3, gate: GateRequest(heldOut: ["x"]))
        commit.positive = Self.positive
        let source = try await container.performEvaluating(nonSendable: commit) { context, commit in
            try MLXInferenceEngine.steeringSource(commit, context: context)
        }
        guard case .rankOne(let module, let input, let output, let scale) = source else {
            Issue.record("expected a rank-1 source, got \(source)")
            return
        }
        #expect(module == "model.layers.2.mlp.down_proj")
        #expect(input.count == 128 && input.allSatisfy(\.isFinite) && input.contains { $0 != 0 })
        #expect(output == direction && scale == 3)
        let restored = await container.perform { context in
            context.model.namedModules().first { $0.0 == "model.layers.2.mlp.down_proj" }.map { String(describing: type(of: $0.1)) }
        }
        #expect(restored == "Linear")
    }

    @Test("A committed direction goes through the gate as a rank-1 adapter on the down projection, changes the logits, and undo restores them bit for bit")
    func commitApplies() async throws {
        let engine = try await LabGateTests.engine()
        let container = try #require(await engine.container)
        let logits = try await WeightDeltaTests.logits(container)
        let found = try await engine.contrast(Self.contrast)
        var commit = SteeringCommit(
            layer: found.bestLayer, direction: found.direction, scale: 4,
            gate: GateRequest(heldOut: LabGateTests.heldOut, maxKL: 100, maxPerplexityRise: 100))
        commit.positive = Self.positive
        let verdict = try await engine.commitSteering(commit)
        #expect(verdict.report?.passed == true, "\(verdict.report?.checks ?? [])")
        let overlay = try #require(verdict.overlay)
        #expect(await LabGateTests.loraLayers(container) == 1)
        #expect(try await WeightDeltaTests.logits(container) != logits)
        try await engine.undo(overlay.id)
        #expect(await LabGateTests.loraLayers(container) == 0)
        #expect(try await WeightDeltaTests.logits(container) == logits)
    }

    @Test("Malformed steer arguments are -32602 naming the argument")
    func rejects() {
        let cases: [(ResearchTool, [String: Any], String)] = [
            (.steerSearch, ["positive": ["a"], "negative": ["c", "d"], "heldOut": ["e"]], "positive must be an array of 2...32 strings"),
            (.steerSearch, ["positive": ["a", "b"], "negative": ["c", "d"], "heldOut": ["e"], "scales": Array(repeating: 1.0, count: 9)], "scales must hold 1...8"),
            (.steerCommit, ["layer": 1, "direction": [1.0], "scale": 1.0, "heldOut": ["e"]], "positive must be an array of 1...32 strings"),
            (.steerCommit, ["layer": 1, "direction": [1.0], "positive": ["a"], "heldOut": ["e"]], "Missing required argument: scale"),
        ]
        for (tool, args, fragment) in cases {
            let failure = #expect(throws: MCPFailure.self) { try ResearchCall(tool, args) }
            #expect(failure?.code == -32602, "\(tool)")
            #expect(failure?.message.contains(fragment) == true, "\(tool): \(failure?.message ?? "no error")")
        }
    }
}
