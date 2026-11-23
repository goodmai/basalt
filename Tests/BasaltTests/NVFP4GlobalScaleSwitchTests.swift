import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("nvfp4 per-expert global_scale on QuantizedSwitchLinear, streaming, Inkling sanitize, X-ray, and converter (#677 Gap 2)")
struct NVFP4GlobalScaleSwitchTests {
    private static func makePerExpertNVFP4Checkpoint(
        at directory: URL,
        scalesByExpert: [Float]
    ) throws -> (Data, [String: MLXArray]) {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let model = try TinyModel.make(.qwen3Moe)
        let globalScale = MLXArray(scalesByExpert, [scalesByExpert.count]).asType(.float32)
        var updates: [(String, Module)] = []
        for (path, module) in model.leafModules().flattened() {
            if let sw = module as? SwitchLinear {
                updates.append((path, QuantizedSwitchLinear(sw, groupSize: 16, bits: 4, mode: .nvfp4, globalScale: globalScale)))
            }
        }
        model.update(modules: ModuleChildren.unflattened(updates))
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try save(arrays: weights, url: directory.appending(component: "model.safetensors"))
        var json = try #require(JSONSerialization.jsonObject(with: Data(TinyModel.Family.qwen3Moe.config.utf8)) as? [String: Any])
        json["model_type"] = "qwen3_moe"
        json["quantization"] = ["bits": 4, "group_size": 16, "mode": "nvfp4"] as [String: Any]
        let configData = try JSONSerialization.data(withJSONObject: json)
        try configData.write(to: directory.appending(component: "config.json"))
        try Data("{}".utf8).write(to: directory.appending(component: "tokenizer.json"))
        return (configData, weights)
    }

    @Test("QuantizedSwitchLinear with per-expert nvfp4 global_scale matches dense reference and chunked decode (1, 3, 7)")
    func switchParityAndChunkedDecode() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: "basalt-677-gap2-parity-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let expertScales: [Float] = [0.75, 1.4, 2.25, 0.5, 1.8, 3.1, 0.9, 1.15]
        let (configData, weights) = try Self.makePerExpertNVFP4Checkpoint(at: root, scalesByExpert: expertScales)

        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        let resident = try TinyModel.Family.qwen3Moe.model(configData)
        try loadWeights(modelDirectory: root, model: resident, perLayerQuantization: base.perLayerQuantization)

        let reference = try TinyModel.Family.qwen3Moe.model(configData)
        var refWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasSuffix(".scales") || key.hasSuffix(".biases") || key.hasSuffix(".global_scale") { continue }
            if key.hasSuffix(".weight") {
                let stem = String(key.dropLast(".weight".count))
                if let sc = weights[stem + ".scales"], let gs = weights[stem + ".global_scale"] {
                    let deq = MLX.dequantized(value, scales: sc, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4)
                    refWeights[key] = (deq * (gs / (448 * 6)).reshaped([-1, 1, 1])).asType(deq.dtype)
                    continue
                }
            }
            refWeights[key] = value
        }
        try reference.update(parameters: ModuleParameters.unflattened(refWeights), verify: [.all])

        let prompt = MLXArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] as [Int32]).reshaped([1, 11])
        let residentLogits = resident(prompt, cache: nil)
        let refLogits = reference(prompt, cache: nil)
        eval(residentLogits, refLogits)
        let maxDiff = abs(residentLogits - refLogits).max().item(Float.self)
        #expect(maxDiff < 1e-2, "resident vs dense reference maxDiff=\(maxDiff)")
        #expect(residentLogits.argMax(axis: -1).asArray(Int32.self) == refLogits.argMax(axis: -1).asArray(Int32.self))

        for chunk in [1, 3, 7] {
            let cache = try resident.newCache(parameters: nil)
            var pieces: [MLXArray] = []
            var offset = 0
            while offset < 11 {
                let end = min(11, offset + chunk)
                let slice = prompt[0..., offset ..< end]
                pieces.append(resident(slice, cache: cache))
                offset = end
            }
            let chunked = concatenated(pieces, axis: 1)
            eval(chunked)
            let chunkDiff = abs(chunked - residentLogits).max().item(Float.self)
            #expect(chunkDiff < 1e-2, "chunk=\(chunk) maxDiff=\(chunkDiff)")
            #expect(chunked.argMax(axis: -1).asArray(Int32.self) == residentLogits.argMax(axis: -1).asArray(Int32.self))
        }
    }

    @Test("StreamedSwitchLinear with per-expert nvfp4 global_scale matches resident at 1 slot, half, and full cache")
    func streamedMatchesResident() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: "basalt-677-gap2-stream-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let expertScales: [Float] = [0.6, 1.9, 2.7, 0.85, 1.3, 2.1, 0.45, 3.0]
        let (configData, _) = try Self.makePerExpertNVFP4Checkpoint(at: root, scalesByExpert: expertScales)
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)

        let resident = try TinyModel.Family.qwen3Moe.model(configData)
        try loadWeights(modelDirectory: root, model: resident, perLayerQuantization: base.perLayerQuantization)

        let shards = [try SafetensorsShard(url: root.appending(component: "model.safetensors"))]
        let plan = try ExpertPlan(model: resident, shards: shards, quantization: base.perLayerQuantization)
        let perExpert = plan.layouts.reduce(ByteCount.zero) { $0 + $1.expertBytes }
        #expect(plan.layouts.allSatisfy { $0.globalScales != nil })

        let prompt = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped([1, 5])
        let residentOut = resident(prompt, cache: nil)
        eval(residentOut)

        for slotCount in [1, 4, 8] {
            let streamed = try TinyModel.Family.qwen3Moe.model(configData)
            try loadWeights(modelDirectory: root, model: streamed, perLayerQuantization: base.perLayerQuantization)
            let store = try ExpertStore(files: shards.map(\.url), layouts: plan.layouts, capacity: perExpert * Int64(slotCount))
            var replacements: [(String, Module)] = []
            for (index, layout) in plan.layouts.enumerated() {
                replacements.append((layout.path, StreamedSwitchLinear(store: store, module: index)))
            }
            streamed.update(modules: ModuleChildren.unflattened(replacements))
            let out = streamed(prompt, cache: nil)
            eval(out)
            let diff = abs(out - residentOut).max().item(Float.self)
            #expect(diff < 1e-3, "slots=\(slotCount) diff=\(diff)")
        }
    }

    @Test("Inkling2Checkpoint maps w13/w2 global_scale and UniversalSourceReader + WeightConverter decode nvfp4+global bit-for-bit")
    func inklingSanitizeAndConverterDecode() async throws {
        let config = try JSONDecoder.json5().decode(Inkling2Configuration.self, from: Data(TinyModel.Family.inkling2Loop.config.utf8))
        let w13gs = MLXArray([1.0, 2.0] as [Float], [2])
        let w2gs = MLXArray([0.5, 1.5] as [Float], [2])
        let mapped = Inkling2Checkpoint.sanitized(
            [
                "model.layers.0.mlp.experts.w13.global_scale": w13gs,
                "model.layers.0.mlp.experts.w2.global_scale": w2gs,
            ],
            config: config
        )
        #expect(mapped["model.language_model.layers.0.mlp.experts.gate_up_proj.global_scale"] != nil)
        #expect(mapped["model.language_model.layers.0.mlp.experts.down_proj.global_scale"] != nil)

        let root = FileManager.default.temporaryDirectory.appending(component: "basalt-677-gap2-conv-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let srcDir = root.appending(component: "src")
        let denseDir = root.appending(component: "dense")

        let expertScales: [Float] = [0.7, 1.5, 2.2, 0.9, 1.1, 2.8, 0.5, 1.9]
        let (_, weights) = try Self.makePerExpertNVFP4Checkpoint(at: srcDir, scalesByExpert: expertScales)
        let detected = try UniversalSourceReader.detect(at: srcDir)
        #expect(detected.format == "nvfp4+global")

        _ = try WeightConverter(precision: .dense(.bfloat16)).run(source: srcDir, output: denseDir)
        let decoded = try loadArrays(url: denseDir.appending(component: "model.safetensors"))
        let sampleKey = try #require(weights.keys.first { $0.hasSuffix(".global_scale") })
        let stem = String(sampleKey.dropLast(".global_scale".count))
        let actual = try #require(decoded[stem + ".weight"])
        let deq = MLX.dequantized(
            try #require(weights[stem + ".weight"]),
            scales: try #require(weights[stem + ".scales"]),
            biases: nil, groupSize: 16, bits: 4, mode: .nvfp4)
        let expected = (deq * (try #require(weights[sampleKey]) / (448 * 6)).reshaped([-1, 1, 1])).asType(actual.dtype)
        eval(actual, expected)
        #expect(abs(actual - expected).max().item(Float.self) == 0)
    }
}
