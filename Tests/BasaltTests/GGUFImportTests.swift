import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

@Suite("GGUF import end to end on a tiny qwen3_5: the reader, the tensor plan, basalt models convert (native and affine) and the native loader agree with dense weights (#408)", .serialized)
struct GGUFImportTests {
    struct Tensor {
        let name: String
        let type: GGMLType
        let dims: [Int]
        let bytes: [UInt8]
    }

    static func gguf(_ tensors: [Tensor]) -> Data {
        var data = Data()
        func put<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        func string(_ text: String) { put(UInt64(text.utf8.count)); data.append(contentsOf: Array(text.utf8)) }
        let metadata: [(String, UInt32, () -> Void)] = [
            ("general.alignment", 4, { put(UInt32(32)) }), ("general.architecture", 8, { string("qwen35") }),
            ("t.u8", 0, { put(UInt8(1)) }), ("t.i8", 1, { put(Int8(-1)) }), ("t.u16", 2, { put(UInt16(2)) }), ("t.i16", 3, { put(Int16(-2)) }),
            ("t.i32", 5, { put(Int32(-3)) }), ("t.f32", 6, { put(Float(1.5).bitPattern) }), ("t.bool", 7, { put(UInt8(1)) }),
            ("t.array", 9, { put(UInt32(5)); put(UInt64(2)); put(Int32(7)); put(Int32(8)) }),
            ("t.u64", 10, { put(UInt64(9)) }), ("t.i64", 11, { put(Int64(-9)) }), ("t.f64", 12, { put(Double(2.5).bitPattern) }),
        ]
        put(GGUFFile.magic)
        put(UInt32(3))
        put(UInt64(tensors.count))
        put(UInt64(metadata.count))
        for (key, type, value) in metadata {
            string(key)
            put(type)
            value()
        }
        var offset = 0
        for tensor in tensors {
            string(tensor.name)
            put(UInt32(tensor.dims.count))
            for dim in tensor.dims { put(UInt64(dim)) }
            put(tensor.type.rawValue)
            put(UInt64(offset))
            offset += (tensor.bytes.count + 31) / 32 * 32
        }
        data.append(contentsOf: [UInt8](repeating: 0, count: (32 - data.count % 32) % 32))
        for tensor in tensors {
            data.append(contentsOf: tensor.bytes)
            data.append(contentsOf: [UInt8](repeating: 0, count: (32 - tensor.bytes.count % 32) % 32))
        }
        return data
    }

    static func floats(_ values: [Float]) -> [UInt8] { values.withUnsafeBytes { Array($0) } }

    static func tensors(for model: Module, layout: GGUFQwen35Layout) -> [Tensor] {
        let shapes = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        var names: [(String, String)] = [("token_embd.weight", "model.embed_tokens.weight"), ("output_norm.weight", "model.norm.weight"), ("output.weight", "lm_head.weight")]
        for layer in 0 ..< 4 {
            for (gguf, (target, _)) in GGUFQwen35Layout.names where shapes["model.layers.\(layer).\(target)"] != nil {
                names.append(("blk.\(layer).\(gguf)", "model.layers.\(layer).\(target)"))
            }
        }
        var generator = SystemRandomNumberGenerator()
        return names.sorted { $0.0 < $1.0 }.enumerated().map { index, pair in
            let shape = shapes[pair.1]!
            let dims = shape.count == 1 ? shape : [shape[1], shape[0]]
            let count = dims.reduce(1, *)
            if dims.count == 2, dims[0] % 32 == 0, !pair.0.hasPrefix("blk.0.ssm_alpha") {
                return Tensor(name: pair.0, type: .q8_0, dims: dims, bytes: GGUFKernelTests.blocks(.q8_0, count: count / 32, seed: UInt64(index), scale: 0.004))
            }
            let values: [Float] = (0 ..< count).map { _ in
                pair.0.hasSuffix("ssm_a") ? -Float.random(in: 0.5 ... 2, using: &generator)
                    : pair.0.hasSuffix("norm.weight") ? 1 + Float.random(in: -0.1 ... 0.1, using: &generator) : Float.random(in: -0.05 ... 0.05, using: &generator)
            }
            return Tensor(name: pair.0, type: .f32, dims: dims, bytes: floats(values))
        }
    }

    @Test("Convert a synthetic GGUF both ways, load the native checkpoint and match the dense model's greedy tokens")
    func endToEnd() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "gguf-import-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let template = folder.appending(path: "template")
        try FileManager.default.createDirectory(at: template, withIntermediateDirectories: true)
        try Data(TinyModel.Family.qwen35.config.utf8).write(to: template.appending(path: "config.json"))
        try Data("{}".utf8).write(to: template.appending(path: "tokenizer_config.json"))
        let layout = try GGUFImporter.layout(template: template)
        #expect(layout == GGUFQwen35Layout(keyHeads: 2, valueHeads: 4, keyHeadDim: 16, valueHeadDim: 16, prefix: "model."))

        let reference = try TinyModel.make(.qwen35)
        let file = folder.appending(path: "tiny.gguf")
        try Self.gguf(Self.tensors(for: reference, layout: layout)).write(to: file)
        let gguf = try GGUFFile(url: file)
        #expect(gguf.version == 3 && gguf.metadata["general.architecture"] == .string("qwen35") && gguf.metadata["t.array"] == .array([.int(7), .int(8)]))
        #expect(gguf.metadata["t.f64"] == .float(2.5) && gguf.metadata["t.bool"] == .bool(true) && gguf.metadata["t.i8"] == .int(-1))

        var native = try ConvertSubcommand.parse([file.path, "--template", template.path, "--out", folder.appending(path: "native").path])
        try await native.run()
        var affine = try ConvertSubcommand.parse([file.path, "--template", template.path, "--out", folder.appending(path: "affine").path, "--grid", "affine", "--min-bits", "4"])
        try await affine.run()
        let affineConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appending(path: "affine/config.json"))) as? [String: Any]
        #expect((affineConfig?["quantization"] as? [String: Any])?["bits"] as? Int == 8 && affineConfig?["gguf_quantization"] == nil)
        #expect(FileManager.default.fileExists(atPath: folder.appending(path: "native/tokenizer_config.json").path))

        var entries = try #require(GGUFNativeLoader.entries(in: folder.appending(path: "native")))
        #expect(entries["model.layers.0.linear_attn.out_proj"]?.inputOrder == true && entries["model.embed_tokens"]?.type == "q8_0")
        entries["model.embed_tokens"]?.dtype = "float32"
        var weights: [String: MLXArray] = [:]
        for url in try FileManager.default.contentsOfDirectory(at: folder.appending(path: "native"), includingPropertiesForKeys: nil) where url.pathExtension == "safetensors" {
            weights.merge(try loadArrays(url: url)) { $1 }
        }
        let model = try TinyModel.make(.qwen35)
        try GGUFNativeLoader.install(entries, in: model)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys, .shapeMismatch])

        let importer = GGUFImporter(file: gguf, layout: layout)
        var dense: [String: MLXArray] = [:]
        for tensor in gguf.tensors {
            let plan = try #require(layout.plan(tensor))
            dense.merge(importer.convert(tensor, plan, bits: nil, dense: tensor.type.block.elements > 1 ? .float32 : .bfloat16)) { $1 }
        }
        let expected = try TinyModel.make(.qwen35)
        try expected.update(parameters: ModuleParameters.unflattened(dense), verify: [.noUnusedKeys, .shapeMismatch])
        eval(model, expected)
        let prompt = TinyModel.tokens("native gguf blocks")
        #expect(try CacheBlendQualityTests.greedy(prompt, count: 10, model: model) == CacheBlendQualityTests.greedy(prompt, count: 10, model: expected))
        let input = MLXArray(prompt.map(Int32.init))[.newAxis]
        #expect(GGUFKernelTests.relativeError(model(input, cache: nil), expected(input, cache: nil).asType(.float32)) < 1e-3)
    }

    @Test("A template that is not a qwen3_5 hybrid is refused before anything is written")
    func refusals() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "gguf-refuse-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try Data(TinyModel.Family.qwen3.config.utf8).write(to: folder.appending(path: "config.json"))
        #expect(throws: BasaltError.self) { try GGUFImporter.layout(template: folder) }
        var command = try ConvertSubcommand.parse([folder.appending(path: "x.gguf").path, "--out", folder.appending(path: "o").path])
        await #expect(throws: (any Error).self) { try await command.run() }
        #expect(!FileManager.default.fileExists(atPath: folder.appending(path: "o").path))
    }
}
