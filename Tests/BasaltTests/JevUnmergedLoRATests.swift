import Testing
import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
@testable import BasaltInference
@testable import BasaltCore

@Suite("OpenJev heads on any base: a PEFT LoRA applied unmerged matches the merged float32 path on tiny Qwen3 and Qwen3.5, and stays close on a 4-bit base (#423 J2 J3)", .serialized)
struct JevUnmergedLoRATests {
    static let rank = 4
    static let alpha: Float = 8

    static func adapter(for model: any LanguageModel) throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appending(path: "jev-lora-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let config = ["peft_type": "LORA", "r": rank, "lora_alpha": alpha, "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"], "use_dora": false] as [String: Any]
        try JSONSerialization.data(withJSONObject: config).write(to: folder.appending(path: "adapter_config.json"))
        var arrays: [String: MLXArray] = [:]
        let key = MLXRandom.key(7)
        for (index, (path, weight)) in model.parameters().flattened().filter({ $0.0.hasSuffix("q_proj.weight") || $0.0.hasSuffix("v_proj.weight") }).enumerated() {
            let stem = "base_model.model." + String(path.dropLast(".weight".count))
            let (out, input) = (weight.dim(0), weight.dim(1))
            let keys = MLXRandom.split(key: MLXRandom.key(UInt64(index + 1)), into: 2)
            arrays[stem + ".lora_A.weight"] = MLXRandom.normal([rank, input], key: keys[0]) * 0.05
            arrays[stem + ".lora_B.weight"] = MLXRandom.normal([out, rank], key: keys[1]) * 0.05
            _ = key
        }
        #expect(!arrays.isEmpty)
        try MLX.save(arrays: arrays, url: folder.appending(path: "adapter_model.safetensors"))
        return folder
    }

    static func hidden(_ model: any LanguageModel) throws -> MLXArray {
        let ids = MLXArray((0 ..< 12).map { Int32(($0 * 7) % 50 + 1) }).reshaped(1, -1)
        let output = try JevEvaluator.hidden(model, ids, cache: try model.newCache(parameters: nil)).asType(.float32)
        eval(output)
        return output
    }

    static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let (x, y) = (a.reshaped(-1), b.reshaped(-1))
        return ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))).item(Float.self)
    }

    @Test("Unmerged against merged float32: max |Δ| within 1e-3 on float weights; on a 4-bit base the hidden states keep cosine ≥ 0.98 with the reference",
          arguments: [TinyModel.Family.qwen3Small, .qwen35])
    func parity(family: TinyModel.Family) throws {
        let reference = try TinyModel.make(family)
        let folder = try Self.adapter(for: reference)
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appending(path: "adapter_model.safetensors")
        #expect(JevEvaluator.supports(reference))
        try JevEvaluator.apply(adapter: folder, file: file, to: reference, scale: Self.alpha / Float(Self.rank), float32: true)
        let expected = try Self.hidden(reference)
        let unmerged = try TinyModel.make(family)
        try JevEvaluator.apply(adapter: folder, file: file, to: unmerged, scale: Self.alpha / Float(Self.rank), float32: false)
        let actual = try Self.hidden(unmerged)
        #expect(abs(expected - actual).max().item(Float.self) < 1e-3, "\(family)")
        let plain = try Self.hidden(try TinyModel.make(family))
        #expect(abs(expected - plain).max().item(Float.self) > 1e-3, "the adapter changed nothing on \(family)")
        let quantized = try TinyModel.make(family)
        quantize(model: quantized, groupSize: 32, bits: 4) { _, module in module is Linear && !(module is QuantizedLinear) }
        try JevEvaluator.apply(adapter: folder, file: file, to: quantized, scale: Self.alpha / Float(Self.rank), float32: false)
        #expect(Self.cosine(expected, try Self.hidden(quantized)) >= 0.98, "\(family) 4-bit")
    }
}
