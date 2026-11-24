import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("ModelOpt NVFP4 import: per-expert nvfp4 repacked losslessly, FP8 per-tensor to affine 8, experts stacked to the model's switch layers (#677 Gap 3)", .serialized)
struct ModelOptImporterTests {
    static var tokens: MLXArray { MLXArray((0 ..< 24).map { Int32(($0 * 29 + 5) % 200) })[.newAxis] }

    static func expertSource(_ stem: String) -> (prefix: String, projection: String) {
        var parts = stem.split(separator: ".").map(String.init)
        let projection = parts.removeLast()
        let parent = parts.joined(separator: ".")
        if parent.hasSuffix(".switch_glu") { return (String(parent.dropLast(".switch_glu".count)), projection) }
        return (parent.replacingOccurrences(of: ".switch_mlp", with: ".experts"), projection)
    }

    static func write(
        _ family: TinyModel.Family, at directory: URL, algorithm: String, fp8: (String) -> Bool
    ) throws -> [String: MLXArray] {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let model = try TinyModel.make(family)
        let switches = Set(model.leafModules().flattened().filter { $0.1 is SwitchLinear }.map(\.0))
        var (source, reference) = ([String: MLXArray](), [String: MLXArray]())
        for (key, value) in model.parameters().flattened() {
            let stem = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
            if switches.contains(stem) {
                let (prefix, projection) = expertSource(stem)
                var dense: [MLXArray] = []
                for expert in 0 ..< value.dim(0) {
                    let scale2 = Float(0.02 * Double(expert + 1))
                    let (wq, scales, _) = MLX.quantized(value[expert].asType(.float32) / scale2, groupSize: 16, bits: 4, mode: .nvfp4)
                    let name = "\(prefix).\(expert).\(projection)"
                    source[name + ".weight"] = wq.view(dtype: .uint8)
                    source[name + ".weight_scale"] = scales
                    source[name + ".weight_scale_2"] = MLXArray(scale2)
                    source[name + ".input_scale"] = MLXArray(Float(1))
                    dense.append(MLX.dequantized(wq, scales: scales, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4, dtype: .float32) * scale2)
                }
                reference[key] = stacked(dense).asType(value.dtype)
            } else if key.hasSuffix(".weight"), value.ndim == 2, fp8(stem) {
                let scale = abs(value.asType(.float32)).max() / 448
                let weight8 = toFP8(value.asType(.float32) / scale)
                source[key] = weight8
                source[stem + ".weight_scale"] = scale.reshaped([])
                reference[key] = (fromFP8(weight8, dtype: .float32) * scale).asType(value.dtype)
            } else {
                (source[key], reference[key]) = (value, value)
            }
        }
        try save(arrays: source, url: directory.appending(component: "model.safetensors"))
        try Data(family.config.utf8).write(to: directory.appending(component: "config.json"))
        let quant: [String: Any] = ["quantization": ["quant_algo": algorithm, "group_size": 16, "exclude_modules": ["lm_head", "model.embed_tokens*"]]]
        try JSONSerialization.data(withJSONObject: quant).write(to: directory.appending(component: "hf_quant_config.json"))
        try Data("{}".utf8).write(to: directory.appending(component: "tokenizer.json"))
        return reference
    }

    static func load(_ family: TinyModel.Family, from directory: URL) throws -> any LanguageModel {
        let config = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: config)
        let model = try family.model(config)
        try loadWeights(modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization)
        return model
    }

    static func temporary(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appending(component: "basalt-modelopt-\(name)-\(UUID().uuidString)")
    }

    @Test("Imported logits equal the ModelOpt dequantization (q·s·s₂ for nvfp4, w·s for FP8) and greedy ids match; nibble order and the 2688 factor hold",
          arguments: [(TinyModel.Family.gemma4Moe, "NVFP4", Float(2e-3)), (.qwen35Moe, "MIXED_PRECISION", Float(5e-2))])
    func importMatchesModelOptDequantization(family: TinyModel.Family, algorithm: String, tolerance: Float) throws {
        let root = Self.temporary("\(family)")
        defer { try? FileManager.default.removeItem(at: root) }
        let (source, output) = (root.appending(component: "source"), root.appending(component: "mlx"))
        let reference = try Self.write(family, at: source, algorithm: algorithm) { algorithm == "MIXED_PRECISION" && $0.hasSuffix("o_proj") }
        let summary = try ModelOptImporter(source: source).run(output: output, model: try TinyModel.make(family))
        #expect(summary.nvfp4 > 0 && summary.stackedExperts > 0)
        #expect((summary.fp8 > 0) == (algorithm == "MIXED_PRECISION"))
        #expect(summary.fp8MaxRelativeRMSE < 1e-2)
        let imported = try Self.load(family, from: output)
        let dense = try TinyModel.make(family)
        try dense.update(parameters: ModuleParameters.unflattened(reference), verify: [.all])
        let got = imported(Self.tokens, cache: nil).asType(.float32)
        let expected = dense(Self.tokens, cache: nil).asType(.float32)
        #expect(abs(got - expected).max().item(Float.self) < tolerance)
        #expect(argMax(got, axis: -1).asArray(Int32.self) == argMax(expected, axis: -1).asArray(Int32.self))
        let config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: output.appending(component: "config.json"))) as? [String: Any])
        #expect((config["quantization"] as? [String: Any])?["mode"] as? String == "nvfp4")
        #expect((config["basalt_import"] as? [String: Any])?["quant_algo"] as? String == algorithm)
    }

    @Test("Re-importing the same source writes identical weights")
    func reimportIsIdempotent() throws {
        let root = Self.temporary("idempotent")
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appending(component: "source")
        _ = try Self.write(.gemma4Moe, at: source, algorithm: "NVFP4") { _ in false }
        for name in ["a", "b"] {
            _ = try ModelOptImporter(source: source).run(output: root.appending(component: name), model: try TinyModel.make(.gemma4Moe))
        }
        let a = try loadArrays(url: root.appending(component: "a/model.safetensors"))
        let b = try loadArrays(url: root.appending(component: "b/model.safetensors"))
        #expect(Set(a.keys) == Set(b.keys))
        #expect(a.keys.allSatisfy { arrayEqual(a[$0]!, b[$0]!).item(Bool.self) })
    }

    @Test("Unknown quant_algo and non-ModelOpt directories are refused by name; exclude_modules globs match prefixes")
    func refusalsAndGlobs() throws {
        let root = Self.temporary("refuse")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let quant: [String: Any] = ["quantization": ["quant_algo": "W4A8_AWQ", "group_size": 128]]
        try JSONSerialization.data(withJSONObject: quant).write(to: root.appending(component: "hf_quant_config.json"))
        #expect(throws: ModelOptImporter.Refused.self) { try ModelOptImporter(source: root) }
        try FileManager.default.removeItem(at: root.appending(component: "hf_quant_config.json"))
        try Data(#"{"model_type": "qwen3"}"#.utf8).write(to: root.appending(component: "config.json"))
        #expect(throws: ModelOptImporter.Refused.self) { try ModelOptImporter(source: root) }
        #expect(ModelOptImporter.matches("model.language_model.layers.0.mlp*", "model.language_model.layers.0.mlp.down_proj"))
        #expect(!ModelOptImporter.matches("model.language_model.layers.0.mlp*", "model.language_model.layers.10.mlp.down_proj"))
        #expect(!ModelOptImporter.matches("lm_head", "lm_head.extra"))
    }
}
