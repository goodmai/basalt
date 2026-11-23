import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("MLX quantization config traps: metadata, omitted group_size, quantization_config fallback, and ignored-key safety net (#677 Gap 1)")
struct QuantizationConfigTrapsTests {
    private static func makeQuantizedCheckpoint(
        directory: URL,
        bits: Int,
        groupSize: Int,
        mode: QuantizationMode,
        configPatch: (inout [String: Any]) -> Void
    ) throws -> (Data, MLXArray) {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let model = try TinyModel.make(.qwen3Small)
        quantize(model: model) { _, module in
            module is Linear ? (groupSize, bits, mode) : nil
        }
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try save(arrays: weights, url: directory.appending(component: "model.safetensors"))
        var json = try #require(JSONSerialization.jsonObject(with: Data(TinyModel.Family.qwen3Small.config.utf8)) as? [String: Any])
        json["model_type"] = "qwen3"
        configPatch(&json)
        let configData = try JSONSerialization.data(withJSONObject: json)
        try configData.write(to: directory.appending(component: "config.json"))
        let tokens = MLXArray([1, 2, 3, 4] as [Int32]).reshaped([1, 4])
        let logits = model(tokens, cache: nil)
        eval(logits)
        return (configData, logits)
    }

    private static func loadLogits(from directory: URL) throws -> MLXArray {
        let configData = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        let model = try TinyModel.Family.qwen3Small.model(configData)
        try loadWeights(modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization)
        let tokens = MLXArray([1, 2, 3, 4] as [Int32]).reshaped([1, 4])
        let logits = model(tokens, cache: nil)
        eval(logits)
        return logits
    }

    @Test("TinyModel checkpoints with scalar, array, null, non-layer dict metadata, and quantization_config-only load with identical logits")
    func affineConfigTrapsMatchCleanLogits() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: "basalt-677-affine-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let cleanDir = root.appending(component: "clean")
        let (_, expectedLogits) = try Self.makeQuantizedCheckpoint(directory: cleanDir, bits: 4, groupSize: 64, mode: .affine) {
            $0["quantization"] = ["bits": 4, "group_size": 64]
        }
        let cleanLoaded = try Self.loadLogits(from: cleanDir)
        #expect(abs(cleanLoaded - expectedLogits).max().item(Float.self) == 0)

        let traps: [(String, [String: Any])] = [
            ("scalar-and-nested-meta", [
                "quantization": [
                    "bits": 4,
                    "group_size": 64,
                    "method": "dwq",
                    "calibration": "allenai/tulu-3-sft-mixture",
                    "num_samples": 1024,
                    "quantized_components": ["mlp", "self_attn"],
                    "modules": ["note": "non-layer"],
                    "nullable_meta": NSNull(),
                ] as [String: Any],
            ]),
            ("quantization-config-implicit-mlx", [
                "quantization_config": ["bits": 4, "group_size": 64] as [String: Any],
            ]),
            ("quantization-config-explicit-mlx", [
                "quantization_config": ["quant_method": "mlx", "bits": 4] as [String: Any],
            ]),
        ]

        for (name, patch) in traps {
            let dir = root.appending(component: name)
            _ = try Self.makeQuantizedCheckpoint(directory: dir, bits: 4, groupSize: 64, mode: .affine) { json in
                for (k, v) in patch { json[k] = v }
            }
            let loaded = try Self.loadLogits(from: dir)
            let maxDiff = abs(loaded - expectedLogits).max().item(Float.self)
            #expect(maxDiff == 0, "\(name) maxDiff=\(maxDiff)")
        }
    }

    @Test("MXFP4 checkpoint with omitted group_size defaults to 32 and matches clean logits")
    func mxfp4OmittedGroupSizeMatchesCleanLogits() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: "basalt-677-mxfp4-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let dir = root.appending(component: "mxfp4")
        let (_, expectedLogits) = try Self.makeQuantizedCheckpoint(directory: dir, bits: 4, groupSize: 32, mode: .mxfp4) {
            $0["quantization"] = ["bits": 4, "mode": "mxfp4"]
        }
        let loaded = try Self.loadLogits(from: dir)
        #expect(abs(loaded - expectedLogits).max().item(Float.self) == 0)
    }

    @Test("Packed-width mismatch surfaces ignored quantization keys in both loadWeights and ModelDeepCheck")
    func widthMismatchSurfacesIgnoredKeys() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: "basalt-677-mismatch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let dir = root.appending(component: "jang-trap")
        let (configData, _) = try Self.makeQuantizedCheckpoint(directory: dir, bits: 2, groupSize: 64, mode: .affine) {
            $0["quantization"] = [
                "bits": 4,
                "group_size": 64,
                "routed_expert_bits": 2,
                "method": "jang",
            ] as [String: Any]
        }

        let loadError = #expect(throws: (any Error).self) {
            _ = try Self.loadLogits(from: dir)
        }
        let message = String(describing: loadError)
        #expect(message.contains("routed_expert_bits"), "\(message)")
        #expect(message.contains("method"), "\(message)")

        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        let weights = try loadArrays(url: dir.appending(component: "model.safetensors"))
        let tensors = weights.mapValues { ModelInspector.Tensor(dtype: "U32", shape: $0.shape, end: 0) }
        let model = try TinyModel.Family.qwen3Small.model(configData)
        let problems = ModelDeepCheck.parameterProblems(model, tensors: tensors, metadata: [:], quantization: base.perLayerQuantization)
        #expect(problems.contains { $0.contains("routed_expert_bits") }, "\(problems)")
    }

    @Test("ModelInspector QuantizationConfig agrees with BaseConfiguration on quantization_config mode and group_size")
    func inspectorAgreesWithLoader() throws {
        let json: [String: Any] = [
            "model_type": "gemma4",
            "quantization_config": [
                "quant_method": "mlx",
                "bits": 4,
                "mode": "mxfp4",
                "model.embed_tokens": ["bits": 8, "group_size": 64] as [String: Any],
            ] as [String: Any],
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        let inspector = ModelInspector.QuantizationConfig(json: json)

        #expect(base.perLayerQuantization?.quantization(layer: "model.layers.0.mlp")?.groupSize == inspector.defaultGroupSize)
        #expect(inspector.mode == "mxfp4")
        #expect(inspector.bits == 4)
        #expect(inspector.layerGroupSizes["model.embed_tokens"] == 64)

        let foreign: [String: Any] = [
            "model_type": "qwen3",
            "quantization_config": ["quant_method": "modelopt", "bits": 4] as [String: Any],
        ]
        let foreignData = try JSONSerialization.data(withJSONObject: foreign)
        let foreignBase = try JSONDecoder.json5().decode(BaseConfiguration.self, from: foreignData)
        let foreignInspector = ModelInspector.QuantizationConfig(json: foreign)
        #expect(foreignBase.perLayerQuantization == nil)
        #expect(foreignInspector.mode == "modelopt")
    }
}
