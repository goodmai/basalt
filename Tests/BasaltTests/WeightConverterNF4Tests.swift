import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("models convert --quantize nf4:<group>: native nf4 codes and scales on disk, mode nf4, and a load through the fork, MoE experts included (#357)")
struct WeightConverterNF4Tests {
    @Test("--quantize nf4 takes a group of 32, 64 or 128; anything else is an options failure naming the accepted forms")
    func nf4Options() throws {
        for group in [32, 64, 128] {
            #expect(try WeightConverter.precision(quantize: "nf4:\(group)", dequantize: nil, format: .mlx) == .nf4(groupSize: group))
        }
        for spec in ["nf4", "nf4:16", "nf4:48", "nf4:32:4"] {
            let failure = #expect(throws: WeightConverter.Failure.self) { try WeightConverter.precision(quantize: spec, dequantize: nil, format: .mlx) }
            guard case .options(let message)? = failure else { Issue.record("\(spec): \(String(describing: failure))"); continue }
            #expect(message.contains("nf4:<group>"), "\(message)")
        }
    }

    @Test("nf4:32 writes uint32 codes and fp16 scales without biases for linear weights, from a dense or an affine source; embeddings and norms stay dense; config.json carries mode nf4")
    func nf4Checkpoint() throws {
        let scratch = try WeightConverterTests.Scratch()
        let original = try loadArrays(url: scratch.source.appending(component: "model.safetensors"))
        try WeightConverter(precision: .nf4(groupSize: 32)).run(source: scratch.source, output: scratch.url("nf4"))
        let packed = try WeightConverterTests.weights(scratch.url("nf4"))
        let codes = try #require(packed[WeightConverterTests.attention + ".weight"])
        let scales = try #require(packed[WeightConverterTests.attention + ".scales"])
        #expect(codes.dtype == .uint32 && codes.shape == [64, 16] && scales.dtype == .float16 && scales.shape == [64, 4])
        #expect(packed[WeightConverterTests.attention + ".biases"] == nil && packed["model.layers.0.mlp.down_proj.scales"] != nil)
        #expect(packed["model.embed_tokens.scales"] == nil && packed["model.norm.scales"] == nil)
        let source = try #require(original[WeightConverterTests.attention + ".weight"])
        let (expectedCodes, expectedScales, _) = MLX.quantized(source, groupSize: 32, bits: 4, mode: .nf4)
        #expect(WeightConverterTests.maxDifference(codes, expectedCodes) == 0 && WeightConverterTests.maxDifference(scales, expectedScales) == 0)
        let block = try #require(try WeightConverterTests.config(scratch.url("nf4"))["quantization"] as? [String: Any])
        #expect(block["mode"] as? String == "nf4" && block["quant_method"] == nil && block["bits"] as? Int == 4 && block["group_size"] as? Int == 32)

        try WeightConverter(precision: .affine(bits: 4, groupSize: 64)).run(source: scratch.source, output: scratch.url("q4"))
        try WeightConverter(precision: .nf4(groupSize: 32)).run(source: scratch.url("q4"), output: scratch.url("q4-nf4"))
        let requantized = try WeightConverterTests.weights(scratch.url("q4-nf4"))
        #expect(requantized[WeightConverterTests.attention + ".biases"] == nil && requantized[WeightConverterTests.attention + ".weight"]?.dtype == .uint32)
        let affineBlock = try #require(try WeightConverterTests.config(scratch.url("q4-nf4"))["quantization"] as? [String: Any])
        #expect(affineBlock["mode"] as? String == "nf4" && affineBlock["group_size"] as? Int == 64)
    }

    @Test("A source with quantized layers that carry no biases (mxfp4) is refused for nf4, since the loader would read them as nf4")
    func nf4RefusesUnbiasedSource() throws {
        let scratch = try WeightConverterTests.Scratch()
        let mxfp4 = scratch.url("mxfp4")
        try FileManager.default.createDirectory(at: mxfp4, withIntermediateDirectories: true)
        let (weight, scales, _) = MLX.quantized(MLXRandom.normal([64, 128], key: MLXRandom.key(8)), groupSize: 32, bits: 4, mode: .mxfp4)
        try save(arrays: [WeightConverterTests.attention + ".weight": weight, WeightConverterTests.attention + ".scales": scales], url: mxfp4.appending(component: "model.safetensors"))
        try Data(#"{"model_type": "qwen3", "quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}}"#.utf8)
            .write(to: mxfp4.appending(component: "config.json"))
        #expect(throws: WeightConverter.Failure.self) { try WeightConverter(precision: .nf4(groupSize: 32)).run(source: mxfp4, output: scratch.url("out")) }
        #expect(!FileManager.default.fileExists(atPath: scratch.url("out").path))
    }

    static func decoded(_ weights: [String: MLXArray]) -> [(String, MLXArray)] {
        weights.compactMap { key, scales -> (String, MLXArray)? in
            let base = String(key.dropLast(".scales".count))
            guard key.hasSuffix(".scales"), weights[base + ".biases"] == nil, let codes = weights[base + ".weight"] else { return nil }
            let levels = MLXArray(NF4NativeTests.referenceCodebook().map(Float.init))
            let values = levels.take((codes.expandedDimensions(axis: -1) >> MLXArray((0 ..< 8).map { UInt32(4 * $0) })) & UInt32(15))
            let groups = values.reshaped(Array(scales.shape) + [-1]) * scales.asType(.float32).expandedDimensions(axis: -1)
            return (base + ".weight", groups.reshaped(Array(codes.shape.dropLast()) + [codes.dim(-1) * 8]))
        }
    }

    @Test("A tiny model converted to nf4:32 loads as native nf4 QuantizedLinear and QuantizedSwitchLinear, experts included, with the greedy tokens and logits of its decoded weights",
          arguments: [TinyModel.Family.qwen35, .qwen3Moe, .gemma4Moe])
    func nf4CheckpointLoads(family: TinyModel.Family) throws {
        let scratch = try WeightConverterTests.Scratch()
        let tiny = scratch.url("tiny")
        try FileManager.default.createDirectory(at: tiny, withIntermediateDirectories: true)
        let model = try TinyModel.make(family)
        try save(arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()), url: tiny.appending(component: "model.safetensors"))
        try Data(family.config.utf8).write(to: tiny.appending(component: "config.json"))
        try WeightConverter(precision: .nf4(groupSize: 32)).run(source: tiny, output: scratch.url("nf4"))
        let decoded = Self.decoded(try WeightConverterTests.weights(scratch.url("nf4")))
        #expect(decoded.contains { $0.0.hasSuffix("self_attn.q_proj.weight") })
        model.update(parameters: ModuleParameters.unflattened(decoded))
        let loaded = try TinyModel.make(family)
        let configuration = try JSONDecoder.json5().decode(BaseConfiguration.self, from: Data(contentsOf: scratch.url("nf4").appending(component: "config.json")))
        try loadWeights(modelDirectory: scratch.url("nf4"), model: loaded, perLayerQuantization: configuration.perLayerQuantization)
        let linears = loaded.namedModules().compactMap { $0.1 as? QuantizedLinear }
        #expect(!linears.isEmpty && linears.allSatisfy { $0.mode == .nf4 })
        if family != .qwen35 {
            let experts = loaded.namedModules().compactMap { $0.1 as? QuantizedSwitchLinear }
            #expect(!experts.isEmpty && experts.allSatisfy { $0.mode == .nf4 }, "\(family): experts \(experts.map(\.mode))")
        }
        let input = MLXArray(TinyModel.tokens("nf4 checkpoint").map(Int32.init)).reshaped(1, -1)
        let expected = model(input, cache: nil).asType(.float32)
        let actual = loaded(input, cache: nil).asType(.float32)
        #expect(actual.argMax(axis: -1).asArray(Int32.self) == expected.argMax(axis: -1).asArray(Int32.self))
        #expect(WeightConverterTests.maxDifference(actual, expected) <= 1e-3 * abs(expected).max().item(Float.self),
                "\(family): \(WeightConverterTests.maxDifference(actual, expected)) against max \(abs(expected).max().item(Float.self))")
    }
}
