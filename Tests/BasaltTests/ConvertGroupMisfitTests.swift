import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltInference

@Suite("Affine conversion falls back per tensor to the largest group size that divides the input dimension, records it per layer, and reports what no group fits (#841)")
struct ConvertGroupMisfitTests {
    static func convert(_ arrays: [String: MLXArray], group: Int) throws -> (WeightConverter.Summary, [String: MLXArray], [String: Any], URL) {
        let root = FileManager.default.temporaryDirectory.appending(path: "misfit-\(UUID().uuidString)")
        let (source, output) = (root.appending(component: "src"), root.appending(component: "out"))
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try save(arrays: arrays, url: source.appending(component: "model.safetensors"))
        try Data(#"{"model_type":"qwen3"}"#.utf8).write(to: source.appending(component: "config.json"))
        let summary = try WeightConverter(precision: .affine(bits: 4, groupSize: group)).run(source: source, output: output)
        let written = try loadArrays(url: output.appending(component: "model.safetensors"))
        let config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: output.appending(component: "config.json"))) as? [String: Any])
        return (summary, written, config, root)
    }

    @Test("A 96-input projection under group 64 is quantized with group 32 and listed per layer in config.json; the 128-input one keeps group 64")
    func fallsBackToDividingGroup() throws {
        let (summary, written, config, root) = try Self.convert([
            "model.layers.0.mlp.up_proj.weight": MLXArray.ones([64, 96]) * 0.01,
            "model.layers.0.mlp.down_proj.weight": MLXArray.ones([96, 128]) * 0.01,
        ], group: 64)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(summary.groupFallbacks == ["model.layers.0.mlp.up_proj": BaseConfiguration.Quantization(groupSize: 32, bits: 4)])
        #expect(summary.groupMisfits.isEmpty)
        #expect(written["model.layers.0.mlp.up_proj.scales"]?.shape == [64, 3])
        #expect(written["model.layers.0.mlp.down_proj.scales"]?.shape == [96, 2])
        let block = try #require(config["quantization"] as? [String: Any])
        let entry = try #require(block["model.layers.0.mlp.up_proj"] as? [String: Any])
        #expect(entry["group_size"] as? Int == 32 && entry["bits"] as? Int == 4 && block["group_size"] as? Int == 64)
        #expect((config["quantization_config"] as? [String: Any])?["language_model.model.layers.0.mlp.up_proj"] != nil)
    }

    @Test("A 100-input projection that no group divides stays in its source dtype and is named in the warning")
    func reportsWhatNoGroupFits() throws {
        let (summary, written, _, root) = try Self.convert(["model.layers.0.mlp.up_proj.weight": MLXArray.ones([64, 100]) * 0.01], group: 64)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(summary.groupMisfits == ["model.layers.0.mlp.up_proj.weight": GroupMisfit(inputs: 100, bytes: 64 * 100 * 4)])
        #expect(written["model.layers.0.mlp.up_proj.scales"] == nil)
        let warning = try #require(WeightConverter.groupMisfitWarning(summary.groupMisfits, group: 64))
        #expect(warning.contains("no group size of 32, 64 divides") && warning.contains("model.layers.0.mlp.up_proj.weight (input 100"))
        #expect(WeightConverter.groupMisfitWarning([:], group: 64) == nil)
    }
}
