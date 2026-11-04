import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Tensor statistics merge per-chunk GPU reductions into exact moments and sampled |x| percentiles (#328)")
struct TensorStatisticsTests {
    struct Case: Sendable, CustomTestStringConvertible {
        let label: String
        let values: [Float]
        let shape: [Int]
        let chunkElements: Int
        let sampleLimit: Int
        let mean: Float, std: Float, min: Float, max: Float, l2: Float
        let percentiles: [Float]
        var testDescription: String { label }
    }

    static let cases = [
        Case(
            label: "two chunks of one row each", values: [1, 2, 3, 4], shape: [2, 2], chunkElements: 1, sampleLimit: 1 << 20,
            mean: 2.5, std: Float(1.25).squareRoot(), min: 1, max: 4, l2: Float(30).squareRoot(), percentiles: [1, 1, 3, 4, 4]),
        Case(
            label: "signed values, one element per chunk", values: [-3, 0, 3], shape: [3], chunkElements: 1, sampleLimit: 1 << 20,
            mean: 0, std: Float(6).squareRoot(), min: -3, max: 3, l2: Float(18).squareRoot(), percentiles: [0, 0, 3, 3, 3]),
        Case(
            label: "percentiles from every second value", values: [1, 2, 3, 4], shape: [4], chunkElements: 1 << 24, sampleLimit: 2,
            mean: 2.5, std: Float(1.25).squareRoot(), min: 1, max: 4, l2: Float(30).squareRoot(), percentiles: [1, 1, 3, 3, 3]),
        Case(
            label: "one value", values: [-2], shape: [1], chunkElements: 1 << 24, sampleLimit: 1 << 20,
            mean: -2, std: 0, min: -2, max: -2, l2: 2, percentiles: [2, 2, 2, 2, 2]),
    ]

    @Test("Moments, extremes, L2 and percentiles match the hand-computed values", arguments: cases)
    func measures(_ example: Case) {
        let array = MLXArray(example.values, example.shape)
        let measured = TensorStatistics.measure(
            rows: { array[$0] }, count: example.shape[0], rowElements: example.shape.dropFirst().reduce(1, *),
            chunkElements: example.chunkElements, sampleLimit: example.sampleLimit)
        let statistics = measured.statistics
        #expect(statistics.count == example.values.count)
        #expect(abs(Float(statistics.mean) - example.mean) < 1e-5)
        #expect(abs(statistics.std - example.std) < 1e-5)
        #expect(statistics.min == example.min && statistics.max == example.max)
        #expect(abs(statistics.l2 - example.l2) < 1e-5)
        #expect(measured.percentiles == example.percentiles)
    }

    @Test("Merging with an empty side keeps the other side; nothing measured reports zeros and no percentiles")
    func emptySides() {
        let one = TensorStatistics(count: 2, mean: 1, squaredDeviations: 2, min: 0, max: 2)
        #expect(one + TensorStatistics() == one)
        #expect(TensorStatistics() + one == one)
        #expect(TensorStatistics().std == 0 && TensorStatistics().l2 == 0)
        #expect(TensorStatistics.percentileIndices(count: 0) == [])
        #expect(TensorStatistics.percentiles([]) == [])
    }

    @Test("A typo in a tensor name ranks the intended name first")
    func closeMatches() {
        let names = ["model.norm.weight", "model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.up_proj.weight"]
        #expect(MLXInferenceEngine.closeMatches("model.layers.0.mlp.down_prj.weight", in: names, count: 1) == ["model.layers.0.mlp.down_proj.weight"])
        #expect(MLXInferenceEngine.editDistance("kitten", "sitting") == 3)
        #expect(MLXInferenceEngine.editDistance("", "abc") == 3)
    }
}

@Suite("Model X-ray tensors on tiny Qwen3: listing, stats, bounded slices, dequantized views (#328)", .serialized)
struct XRayTensorTests {
    static let downProj = "model.layers.0.mlp.down_proj.weight"

    static func engine(_ model: any LanguageModel) async -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(promptCacheBytes: nil)
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-xray"), model: model, processor: ByteInputProcessor(), tokenizer: ByteTokenizer())))
        return engine
    }

    static func parameters(_ model: any LanguageModel) -> [String: MLXArray] {
        Dictionary(model.parameters().flattened(), uniquingKeysWith: { first, _ in first })
    }

    @Test("The listing names embedding, attention and MLP tensors with their shapes; a filter narrows it")
    func listing() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let shapes = Dictionary(try await engine.tensors(matching: nil).map { ($0.name, $0.shape) }, uniquingKeysWith: { first, _ in first })
        #expect(shapes["model.embed_tokens.weight"] == [256, 64])
        #expect(shapes["model.layers.0.self_attn.q_proj.weight"] == [64, 64])
        #expect(shapes["model.layers.0.self_attn.k_proj.weight"] == [32, 64])
        #expect(shapes["model.layers.3.mlp.gate_proj.weight"] == [128, 64])
        #expect(shapes[Self.downProj] == [64, 128])
        let filtered = try await engine.tensors(matching: "mlp.down_proj")
        #expect(filtered.count == 4 && filtered.allSatisfy { $0.name.hasSuffix("mlp.down_proj.weight") && !$0.quantized })
    }

    @Test("A float tensor reports its own stats and exactly the requested slice")
    func floatSlice() async throws {
        let model = try TinyModel.make(.qwen3)
        let original = try #require(Self.parameters(model)[Self.downProj])
        let engine = await Self.engine(model)
        let report = try await engine.tensor(TensorRequest(name: Self.downProj, slice: [1 ..< 3, 4 ..< 6], maxValues: 3))
        #expect(report.info.dtype == "\(original.dtype)" && !report.info.quantized)
        #expect(report.values == Array(original[1 ..< 3, 4 ..< 6].asType(.float32).asArray(Float.self).prefix(3)))
        #expect(abs(report.mean - original.mean().item(Float.self)) < 1e-5)
        #expect(report.max == original.asType(.float32).max().item(Float.self))
        #expect(report.percentiles.count == 5 && report.percentiles == report.percentiles.sorted())
    }

    @Test("An unknown name, an out-of-range slice and a negative maxValues are request errors", arguments: [
        TensorRequest(name: "model.layers.0.mlp.down_prj.weight", slice: nil, maxValues: 4),
        TensorRequest(name: downProj, slice: [0 ..< 65], maxValues: 4),
        TensorRequest(name: downProj, slice: [0 ..< 1, 0 ..< 1, 0 ..< 1], maxValues: 4),
        TensorRequest(name: downProj, slice: nil, maxValues: -1),
    ])
    func rejects(_ request: TensorRequest) async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let error = await #expect(throws: BasaltError.self) { _ = try await engine.tensor(request) }
        guard case .invalidRequestStructure(let details)? = error else { Issue.record("\(String(describing: error))"); return }
        if request.name != Self.downProj { #expect(details.contains(Self.downProj)) }
    }

    @Test("maxValues above the cap returns at most the cap")
    func capsValues() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let report = try await engine.tensor(TensorRequest(name: "model.embed_tokens.weight", slice: nil, maxValues: 1 << 20))
        #expect(report.values?.count == MLXInferenceEngine.tensorValuesCap)
    }

    @Test("An 8-bit model lists its weights as quantized at the float shape and reports stats and slices close to the float original")
    func dequantizedView() async throws {
        let model = try TinyModel.make(.qwen3)
        let original = try #require(Self.parameters(model)[Self.downProj]).asType(.float32)
        quantize(model: model, groupSize: 32, bits: 8)
        let engine = await Self.engine(model)
        let listed = try await engine.tensors(matching: "layers.0.mlp.down_proj")
        let weight = try #require(listed.first { $0.name == Self.downProj })
        #expect(weight.quantized && weight.shape == [64, 128] && weight.dtype == "8-bit affine g32")
        #expect(listed.contains { $0.name == "model.layers.0.mlp.down_proj.scales" && !$0.quantized })
        let embedding = try await engine.tensors(matching: "embed_tokens.weight")
        #expect(embedding.map(\.shape) == [[256, 64]] && embedding.allSatisfy(\.quantized))

        let report = try await engine.tensor(TensorRequest(name: Self.downProj, slice: [2 ..< 4], maxValues: 256))
        let step = (original.max() - original.min()).item(Float.self) / 255
        #expect(abs(report.mean - original.mean().item(Float.self)) < step)
        #expect(abs(report.std - sqrt(original.variance()).item(Float.self)) < step)
        #expect(abs(report.max - original.max().item(Float.self)) < step)
        let expected = original[2 ..< 4].asArray(Float.self)
        let values = try #require(report.values)
        #expect(values.count == expected.count)
        #expect(zip(values, expected).allSatisfy { abs($0 - $1) < step })
    }
}
