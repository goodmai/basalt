import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("models convert: streaming requantize, dequantize, LoRA merge and HF export")
struct WeightConverterTests {
    static let attention = "model.layers.0.self_attn.q_proj"

    final class Scratch {
        let root = FileManager.default.temporaryDirectory.appending(component: "basalt-convert-\(UUID().uuidString)")
        var source: URL { root.appending(component: "source") }
        func url(_ name: String) -> URL { root.appending(component: name) }

        init() throws {
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            let tensors: [String: MLXArray] = [
                "model.embed_tokens.weight": MLXRandom.uniform(-1 ..< 1, [32, 128], key: MLXRandom.key(1)),
                WeightConverterTests.attention + ".weight": MLXRandom.uniform(-1 ..< 1, [64, 128], key: MLXRandom.key(2)),
                "model.layers.0.mlp.down_proj.weight": MLXRandom.uniform(-1 ..< 1, [128, 64], key: MLXRandom.key(3)),
                "model.layers.0.input_layernorm.weight": MLXArray.ones([128]),
                "model.norm.weight": MLXArray.ones([128]),
            ]
            try save(arrays: tensors, metadata: ["format": "mlx"], url: source.appending(component: "model.safetensors"))
            try Data(#"{"model_type": "qwen3", "hidden_size": 128}"#.utf8).write(to: source.appending(component: "config.json"))
            try Data(#"{"version": "1.0"}"#.utf8).write(to: source.appending(component: "tokenizer.json"))
            try Data("{{ messages }}".utf8).write(to: source.appending(component: "chat_template.jinja"))
        }

        deinit { try? FileManager.default.removeItem(at: root) }
    }

    static func weights(_ directory: URL) throws -> [String: MLXArray] {
        let index = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appending(component: "model.safetensors.index.json"))) as? [String: Any]
        let shards = Set((index?["weight_map"] as? [String: String] ?? [:]).values)
        return try shards.reduce(into: [:]) { all, shard in
            all.merge(try loadArrays(url: directory.appending(component: shard))) { _, new in new }
        }
    }

    static func config(_ directory: URL) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(contentsOf: directory.appending(component: "config.json"))) as? [String: Any])
    }

    static func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    @Test("f32 → affine 4-bit (sharded) → dequantized stays close to the original")
    func quantizeRoundTrip() throws {
        let scratch = try Scratch()
        let original = try loadArrays(url: scratch.source.appending(component: "model.safetensors"))
        var quantizer = WeightConverter(precision: .affine(bits: 4, groupSize: 64))
        quantizer.maxShardBytes = 16_384
        let summary = try quantizer.run(source: scratch.source, output: scratch.url("q4"))
        #expect(summary.shards > 1)
        #expect(FileManager.default.fileExists(atPath: scratch.url("q4").appending(component: "model-00001-of-0000\(summary.shards).safetensors").path))
        #expect(!FileManager.default.fileExists(atPath: scratch.url("q4").appending(component: "part-0.safetensors").path))
        let packed = try Self.weights(scratch.url("q4"))
        #expect(packed[Self.attention + ".scales"] != nil)
        #expect(packed["model.layers.0.mlp.down_proj.scales"] != nil)
        #expect(packed["model.embed_tokens.scales"] == nil)
        #expect(packed["model.norm.scales"] == nil)
        #expect(summary.bytesAfter < summary.bytesBefore)
        let block = try #require(try Self.config(scratch.url("q4"))["quantization"] as? [String: Any])
        #expect(block["bits"] as? Int == 4 && block["group_size"] as? Int == 64)
        #expect(FileManager.default.fileExists(atPath: scratch.url("q4").appending(component: "tokenizer.json").path))
        #expect(FileManager.default.fileExists(atPath: scratch.url("q4").appending(component: "chat_template.jinja").path))

        let split = scratch.url("split")
        let weightKey = Self.attention + ".weight"
        try FileManager.default.createDirectory(at: split, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: scratch.url("q4").appending(component: "config.json"), to: split.appending(component: "config.json"))
        try save(arrays: packed.filter { $0.key != weightKey }, url: split.appending(component: "model-00001-of-00002.safetensors"))
        try save(arrays: [weightKey: try #require(packed[weightKey])], url: split.appending(component: "model-00002-of-00002.safetensors"))

        try WeightConverter(precision: .dense(nil)).run(source: split, output: scratch.url("dense"))
        let dense = try Self.weights(scratch.url("dense"))
        for key in [Self.attention + ".weight", "model.layers.0.mlp.down_proj.weight"] {
            let restored = try #require(dense[key])
            #expect(restored.shape == original[key]?.shape)
            let source = try #require(original[key])
            let (w, scales, biases) = quantized(source, groupSize: 64, bits: 4)
            let reference = dequantized(w, scales: scales, biases: biases, groupSize: 64, bits: 4)
            #expect(Self.maxDifference(restored, reference) < 1e-6)
        }
        #expect(dense.keys.allSatisfy { !$0.hasSuffix(".scales") && !$0.hasSuffix(".biases") })
        #expect(try Self.config(scratch.url("dense"))["quantization"] == nil)
    }

    static func writeAdapter(at directory: URL, scale: Float) throws -> (a: MLXArray, b: MLXArray) {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let a = MLXRandom.uniform(-1 ..< 1, [128, 4], key: MLXRandom.key(4))
        let b = MLXRandom.uniform(-1 ..< 1, [4, 64], key: MLXRandom.key(5))
        try save(arrays: [attention + ".lora_a": a, attention + ".lora_b": b], url: directory.appending(component: "adapters.safetensors"))
        let config = #"{"fine_tune_type": "lora", "num_layers": 1, "lora_parameters": {"rank": 4, "scale": \#(scale), "dropout": 0.0}}"#
        try Data(config.utf8).write(to: directory.appending(component: "adapter_config.json"))
        return (a, b)
    }

    @Test("A merged LoRA adapter computes what the LoRA-applied linear computes")
    func adapterMerge() throws {
        let scratch = try Scratch()
        let (a, b) = try Self.writeAdapter(at: scratch.url("adapter"), scale: 2)
        let weight = try #require(try loadArrays(url: scratch.source.appending(component: "model.safetensors"))[Self.attention + ".weight"])
        let lora = LoRALinear(128, 64, rank: 4, scale: 2, linear: Linear(weight: weight))
        try lora.update(parameters: ModuleParameters.unflattened(["lora_a": a, "lora_b": b]), verify: .noUnusedKeys)
        let x = MLXRandom.normal([3, 128], key: MLXRandom.key(6))

        try WeightConverter(precision: .source, adapter: scratch.url("adapter")).run(source: scratch.source, output: scratch.url("merged"))
        let merged = try #require(try Self.weights(scratch.url("merged"))[Self.attention + ".weight"])
        #expect(Self.maxDifference(Linear(weight: merged)(x), lora(x)) < 1e-3)

        try WeightConverter(precision: .affine(bits: 8, groupSize: 64)).run(source: scratch.source, output: scratch.url("q8"))
        try WeightConverter(precision: .source, adapter: scratch.url("adapter")).run(source: scratch.url("q8"), output: scratch.url("q8-merged"))
        let requantized = try Self.weights(scratch.url("q8-merged"))
        let scales = try #require(requantized[Self.attention + ".scales"])
        let restored = MLX.dequantized(
            try #require(requantized[Self.attention + ".weight"]), scales: scales, biases: requantized[Self.attention + ".biases"],
            groupSize: 64, bits: 8)
        #expect(Self.maxDifference(restored, merged) < 0.1)
    }

    @Test("HF export writes dense tensors without .scales/.biases and a config without quantization")
    func hfExport() throws {
        let scratch = try Scratch()
        try WeightConverter(precision: .affine(bits: 4, groupSize: 64)).run(source: scratch.source, output: scratch.url("q4"))
        let precision = try WeightConverter.precision(quantize: nil, dequantize: nil, format: .hf)
        #expect(precision == .dense(nil))
        try WeightConverter(precision: precision, format: .hf).run(source: scratch.url("q4"), output: scratch.url("hf"))
        let exported = try Self.weights(scratch.url("hf"))
        #expect(!exported.isEmpty)
        #expect(exported.keys.allSatisfy { !$0.hasSuffix(".scales") && !$0.hasSuffix(".biases") })
        #expect(exported[Self.attention + ".weight"]?.shape == [64, 128])
        let config = try Self.config(scratch.url("hf"))
        #expect(config["quantization"] == nil && config["quantization_config"] == nil)
        let (_, metadata) = try loadArraysAndMetadata(url: scratch.url("hf").appending(component: "model.safetensors"))
        #expect(metadata["format"] == "pt")
        #expect(throws: WeightConverter.Failure.self) {
            try WeightConverter.precision(quantize: "affine:4:64", dequantize: nil, format: .hf)
        }
    }

    @Test("--out equal to the source, inside it, or already existing is refused before anything is written")
    func refusesUnsafeOutput() throws {
        let scratch = try Scratch()
        let converter = WeightConverter(precision: .affine(bits: 4, groupSize: 64))
        for output in [scratch.source, scratch.source.appending(component: "converted"), scratch.source.appending(path: "a/b"), scratch.root] {
            #expect(throws: WeightConverter.Failure.self) { try converter.run(source: scratch.source, output: output) }
        }
        #expect(!FileManager.default.fileExists(atPath: scratch.source.appending(component: "converted").path))
        #expect(!FileManager.default.fileExists(atPath: scratch.source.appending(component: "a").path))
    }

    @Test("The source files are byte-identical after every kind of conversion")
    func sourceUnchanged() throws {
        let scratch = try Scratch()
        _ = try Self.writeAdapter(at: scratch.url("adapter"), scale: 1)
        func snapshot() throws -> [String: Data] {
            try FileManager.default.contentsOfDirectory(atPath: scratch.source.path).reduce(into: [:]) { all, name in
                all[name] = try Data(contentsOf: scratch.source.appending(component: name))
            }
        }
        let before = try snapshot()
        try WeightConverter(precision: .affine(bits: 4, groupSize: 64), adapter: scratch.url("adapter")).run(source: scratch.source, output: scratch.url("a"))
        try WeightConverter(precision: .dense(.bfloat16)).run(source: scratch.source, output: scratch.url("b"))
        try WeightConverter(precision: .dense(nil), format: .hf).run(source: scratch.source, output: scratch.url("c"))
        #expect(try snapshot() == before)
    }
}
