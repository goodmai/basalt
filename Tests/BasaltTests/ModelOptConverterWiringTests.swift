import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("nvfp4 as an any-to-any target and ModelOpt as a source (#677 Gap 3, #546)", .serialized)
struct ModelOptConverterWiringTests {
    static func denseDirectory(_ family: TinyModel.Family, at directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let model = try TinyModel.make(family)
        try save(arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()), url: directory.appending(component: "model.safetensors"))
        try Data(family.config.utf8).write(to: directory.appending(component: "config.json"))
        try Data("{}".utf8).write(to: directory.appending(component: "tokenizer.json"))
    }

    @Test("nvfp4 parses, prints and prices like the other targets")
    func targetSpec() throws {
        #expect(try QuantizationTarget.parse("nvfp4") == .nvfp4)
        #expect(QuantizationTarget.nvfp4.specString == "nvfp4")
        #expect(QuantizationTarget.nvfp4.effectiveBitsPerWeight == 4.5)
        #expect(QuantizationTarget.standardCatalogSpecs.contains("nvfp4"))
    }

    static func relativeRMSE(_ approx: MLXArray, _ exact: MLXArray) -> Float {
        let (a, e) = (approx.asType(.float32), exact.asType(.float32))
        return (sqrt(mean(square(a - e))) / maximum(sqrt(mean(square(e))), MLXArray(Float(1e-12)))).item(Float.self)
    }

    @Test("Dense tiny Gemma 4 MoE to nvfp4: every quantized tensor dequantizes (q·s·g/2688) to its original within the measured nvfp4 error, global scales are scalars on dense layers and [E] on experts, and the result loads")
    func denseToNVFP4() throws {
        let root = ModelOptImporterTests.temporary("to-nvfp4")
        defer { try? FileManager.default.removeItem(at: root) }
        let (dense, output) = (root.appending(component: "dense"), root.appending(component: "nvfp4"))
        try Self.denseDirectory(.gemma4Moe, at: dense)
        _ = try WeightConverter(precision: .nvfp4).run(source: dense, output: output)
        let original = try loadArrays(url: dense.appending(component: "model.safetensors"))
        let written = try ModelOptImporter.loadShards(in: output)
        var (worst, checked, expertScales) = (Float(0), 0, 0)
        for (key, weight) in written where key.hasSuffix(".weight") {
            let stem = String(key.dropLast(".weight".count))
            guard let scales = written[stem + ".scales"], let global = written[stem + ".global_scale"], let exact = original[key] else { continue }
            let deq = MLX.dequantized(weight, scales: scales, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4, dtype: .float32)
            let factor = (global.asType(.float32) / (448 * 6)).reshaped(global.ndim == 1 ? [-1, 1, 1] : [])
            worst = max(worst, Self.relativeRMSE(deq * factor, exact))
            #expect(global.ndim == (weight.ndim == 3 ? 1 : 0), "\(stem) global_scale shape \(global.shape)")
            expertScales += weight.ndim == 3 ? 1 : 0
            checked += 1
        }
        print("NVFP4 converter: \(checked) tensors, worst relative RMSE \(worst)")
        #expect(checked > 0 && expertScales > 0)
        #expect(worst < 0.12)
        let config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: output.appending(component: "config.json"))) as? [String: Any])
        let quantization = try #require(config["quantization"] as? [String: Any])
        #expect(quantization["mode"] as? String == "nvfp4" && quantization["group_size"] as? Int == 16)
        let quantized = try ModelOptImporterTests.load(.gemma4Moe, from: output)
        let logits = quantized(ModelOptImporterTests.tokens, cache: nil).asType(.float32)
        #expect(isNaN(logits).any().item(Bool.self) == false && abs(logits).max().item(Float.self).isFinite)
    }

    @Test("A ModelOpt directory is detected as modelopt:nvfp4 and its canonical decode equals ModelOpt's own dequantization tensor by tensor (the reference is fp16-rounded, so within 1e-3)")
    func modelOptSourceDecodes() async throws {
        let root = ModelOptImporterTests.temporary("source")
        defer { try? FileManager.default.removeItem(at: root) }
        let (source, output) = (root.appending(component: "modelopt"), root.appending(component: "dense"))
        let reference = try ModelOptImporterTests.write(.gemma4Moe, at: source, algorithm: "NVFP4") { _ in false }
        let detected = try UniversalSourceReader.detect(at: source)
        #expect(detected.format == "modelopt:nvfp4")
        #expect(detected.requiresCanonicalDecode)
        _ = try await UniversalSourceReader.exportCanonicalDense(from: source, to: output, dtype: .float32)
        let decoded = try ModelOptImporter.loadShards(in: output)
        #expect(Set(decoded.keys) == Set(reference.keys))
        let worst = reference.map { key, exact in decoded[key].map { abs($0.asType(.float32) - exact.asType(.float32)).max().item(Float.self) } ?? .infinity }.max() ?? 0
        #expect(worst < 1e-3, "worst element difference \(worst)")
    }

    @Test("convert --to nvfp4 from a ModelOpt source takes the lossless importer path: the output arrays are bit-identical to ModelOptImporter's, with no dense staging")
    func modelOptToNVFP4IsLossless() async throws {
        let root = ModelOptImporterTests.temporary("lossless")
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appending(component: "modelopt")
        _ = try ModelOptImporterTests.write(.gemma4Moe, at: source, algorithm: "NVFP4") { _ in false }
        let direct = root.appending(component: "direct")
        _ = try ModelOptImporter(source: source).run(output: direct, model: try TinyModel.make(.gemma4Moe))
        let converted = root.appending(component: "converted")
        let request = ModelConversionRequest(source: source.path, outputId: "local/nvfp4", target: .nvfp4, stream: false)
        let result = try await UniversalModelConverter.convertToDirectory(request: request, sourceURL: source, outputURL: converted) { _ in }
        #expect(result.shards >= 1)
        let a = try ModelOptImporter.loadShards(in: direct), b = try ModelOptImporter.loadShards(in: converted)
        #expect(Set(a.keys) == Set(b.keys))
        #expect(a.keys.allSatisfy { arrayEqual(a[$0]!, b[$0]!).item(Bool.self) })
    }
}
