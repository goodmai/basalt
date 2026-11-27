import Foundation
import MLX
import Testing
@testable import BasaltInference

@Suite("Affine conversion reports linear tensors left unquantized because their input dimension is not divisible by the group size (#841)")
struct ConvertGroupMisfitTests {
    @Test("A 96-input projection under group 64 is recorded as a misfit, the 128-input one is quantized, and the warning names a group size that fits")
    func recordsMisfit() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "misfit-\(UUID().uuidString)")
        let (source, output) = (root.appending(component: "src"), root.appending(component: "out"))
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try save(arrays: [
            "model.layers.0.mlp.up_proj.weight": MLXArray.ones([64, 96]) * 0.01,
            "model.layers.0.mlp.down_proj.weight": MLXArray.ones([96, 128]) * 0.01,
        ], url: source.appending(component: "model.safetensors"))
        try Data(#"{"model_type":"qwen3"}"#.utf8).write(to: source.appending(component: "config.json"))

        let summary = try WeightConverter(precision: .affine(bits: 4, groupSize: 64)).run(source: source, output: output)
        #expect(summary.groupMisfits == ["model.layers.0.mlp.up_proj.weight": GroupMisfit(inputs: 96, bytes: 64 * 96 * 4)])
        let warning = try #require(WeightConverter.groupMisfitWarning(summary.groupMisfits, group: 64))
        #expect(warning.contains("model.layers.0.mlp.up_proj.weight (input 96") && warning.contains("group size 32 quantizes all of them"))
    }

    @Test("No misfits give no warning; dimensions that no smaller group divides say so")
    func warningAdvice() {
        #expect(WeightConverter.groupMisfitWarning([:], group: 64) == nil)
        let odd = WeightConverter.groupMisfitWarning(["a.weight": GroupMisfit(inputs: 100, bytes: 400)], group: 64)
        #expect(odd?.contains("no group size of 32, 64 or 128 divides all of them") == true)
    }
}
