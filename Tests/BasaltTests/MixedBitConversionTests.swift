import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltInference

@Suite("models convert --quantize mixed: a noise-variance bit plan under a byte budget, written as mlx-lm per-layer quantization (#292)")
struct MixedBitConversionTests {
    static let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/ReferenceLogits/qwen3_moe")

    final class Scratch {
        let root = FileManager.default.temporaryDirectory.appending(component: "basalt-mixed-\(UUID().uuidString)")
        func url(_ name: String) -> URL { root.appending(component: name) }
        deinit { try? FileManager.default.removeItem(at: root) }
    }

    static func candidate(_ layer: String, _ score: Double?, elements: Int = 4096) -> BitPlan.Candidate {
        BitPlan.Candidate(layer: layer, elements: elements, scaleBytes: 2, score: score)
    }

    @Test("The plan upgrades the largest noise variance first, skips a layer that does not fit, never upgrades an unscored one and refuses a budget below all-low")
    func allocation() throws {
        let low = BitPlan.bytes(Self.candidate("x", nil), bits: 2, groupSize: 64)
        let extra = BitPlan.bytes(Self.candidate("x", nil), bits: 8, groupSize: 64) - low
        let candidates = [Self.candidate("a", 1), Self.candidate("b", 3), Self.candidate("c", nil), Self.candidate("d", 2)]
        var mixed = MixedBits(low: 2, high: 8, groupSize: 64, budgetBytes: 100 + 4 * low + 2 * extra)
        let plan = try BitPlan(candidates, fixedBytes: 100, mixed: mixed)
        #expect(plan.highLayers == ["b", "d"])
        #expect(plan.bytes == mixed.budgetBytes)
        #expect(plan.layers == 4 && plan.unscored == 1)

        mixed.budgetBytes = 100 + 4 * low + 10 * extra
        #expect(try BitPlan(candidates, fixedBytes: 100, mixed: mixed).highLayers == ["a", "b", "d"])

        let large = [Self.candidate("big", 9, elements: 8192), Self.candidate("small", 1)]
        mixed.budgetBytes = 3 * low + extra
        #expect(try BitPlan(large, fixedBytes: 0, mixed: mixed).highLayers == ["small"])

        mixed.budgetBytes = 100 + 4 * low - 1
        #expect(throws: WeightConverter.Failure.self) { try BitPlan(candidates, fixedBytes: 100, mixed: mixed) }
    }

    @Test("--quantize mixed needs --budget-gb, valid bits with low below high and a valid group; --budget-gb needs mixed")
    func parsing() throws {
        let mixed = try WeightConverter.precision(quantize: "mixed:3:6:64", dequantize: nil, format: .mlx, budgetGB: 12.5)
        #expect(mixed == .mixed(MixedBits(low: 3, high: 6, groupSize: 64, budgetBytes: 12_500_000_000)))
        #expect(try WeightConverter.precision(quantize: "affine:4:64", dequantize: nil, format: .mlx) == .affine(bits: 4, groupSize: 64))
        let refused: [(String?, String?, Double?)] = [
            ("mixed:3:6:64", nil, nil), ("mixed:6:3:64", nil, 10), ("mixed:3:7:64", nil, 10), ("mixed:3:6:48", nil, 10),
            ("mixed:3:6", nil, 10), ("mixed:3:6:64", nil, -1), ("mixed:3:6:64", nil, .infinity), ("affine:4:64", nil, 10),
            (nil, "bf16", 10), (nil, nil, 10),
        ]
        for (quantize, dequantize, budget) in refused {
            #expect(throws: WeightConverter.Failure.self, "\(quantize ?? "-") \(dequantize ?? "-") \(budget ?? 0)") {
                try WeightConverter.precision(quantize: quantize, dequantize: dequantize, format: .mlx, budgetGB: budget)
            }
        }
    }

    @Test("Only a per-expert key (experts.<n>.) is pinned to the low bits; stacked experts are planned")
    func perExpertKeys() {
        #expect(WeightConverter.isPerExpert("model.layers.0.mlp.experts.3.gate_proj.weight"))
        #expect(!WeightConverter.isPerExpert("model.layers.0.mlp.switch_mlp.gate_proj.weight"))
        #expect(!WeightConverter.isPerExpert("language_model.model.layers.0.experts.switch_glu.up_proj.weight"))
    }

    @Test("A dense source: the mixed checkpoint fits the budget to the byte, equals the uniform conversions layer by layer and loads through the factory")
    func denseSource() async throws {
        let scratch = Scratch()
        try await Self.checkMixed(source: Self.fixture, low: 2, high: 8, scratch: scratch)
    }

    @Test("An affine 8-bit source: high layers pass through bit for bit, low layers are requantized, and the result loads")
    func quantizedSource() async throws {
        let scratch = Scratch()
        try WeightConverter(precision: .affine(bits: 8, groupSize: 32)).run(source: Self.fixture, output: scratch.url("q8-source"))
        try await Self.checkMixed(source: scratch.url("q8-source"), low: 4, high: 8, scratch: scratch)
    }

    @Test("A budget below the all-low size is refused before the output directory is created")
    func refusesSmallBudget() throws {
        let scratch = Scratch()
        let converter = WeightConverter(precision: .mixed(MixedBits(low: 2, high: 8, groupSize: 32, budgetBytes: 1)))
        #expect(throws: WeightConverter.Failure.self) { try converter.run(source: Self.fixture, output: scratch.url("mixed")) }
        #expect(!FileManager.default.fileExists(atPath: scratch.url("mixed").path))
    }

    static func checkMixed(source: URL, low: Int, high: Int, scratch: Scratch) async throws {
        let uniformLow = try WeightConverter(precision: .affine(bits: low, groupSize: 32)).run(source: source, output: scratch.url("low"))
        let uniformHigh = try WeightConverter(precision: .affine(bits: high, groupSize: 32)).run(source: source, output: scratch.url("high"))
        let budget = (uniformLow.bytesAfter + uniformHigh.bytesAfter) / 2
        let mixed = MixedBits(low: low, high: high, groupSize: 32, budgetBytes: budget)
        let summary = try WeightConverter(precision: .mixed(mixed)).run(source: source, output: scratch.url("mixed"))
        let plan = try #require(summary.plan)
        #expect(summary.bytesAfter == plan.bytes)
        #expect(summary.bytesAfter <= budget && summary.bytesAfter > uniformLow.bytesAfter)
        #expect(!plan.highLayers.isEmpty && plan.highLayers.count < plan.layers && plan.unscored == 0)

        let config = try WeightConverterTests.config(scratch.url("mixed"))
        let block = try #require(config["quantization"] as? [String: Any])
        #expect(block["bits"] as? Int == low && block["group_size"] as? Int == 32)
        #expect(Set(block.keys).subtracting(["bits", "group_size", "mode"]) == plan.highLayers)
        let override = try #require(plan.highLayers.first.flatMap { block[$0] as? [String: Any] })
        #expect(override["bits"] as? Int == high && override["group_size"] as? Int == 32)

        let written = try WeightConverterTests.weights(scratch.url("mixed"))
        let lows = try WeightConverterTests.weights(scratch.url("low"))
        let highs = try WeightConverterTests.weights(scratch.url("high"))
        #expect(Set(written.keys) == Set(lows.keys))
        for (key, value) in written {
            let layer = key.split(separator: ".").dropLast().joined(separator: ".")
            let reference = try #require((plan.highLayers.contains(layer) ? highs : lows)[key])
            #expect(value.dtype == reference.dtype && value.shape == reference.shape && arrayEqual(value, reference).item(Bool.self), "\(key)")
        }

        let container = try await loadModelContainer(from: scratch.url("mixed"), using: ByteTokenizerLoader())
        let finite = try await container.perform { (context: ModelContext) throws -> Bool in
            let logits = context.model(MLXArray([1, 2, 3] as [Int32]).reshaped(1, -1), cache: try context.model.newCache(parameters: nil))
            return isNaN(logits).any().item(Bool.self) == false && isInf(logits).any().item(Bool.self) == false
        }
        #expect(finite)
    }
}
