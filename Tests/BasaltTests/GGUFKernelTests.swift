import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("GGUF i-quant and k-quant Metal kernels against the gguf-py-exact CPU dequantizer (#408 phase 2)")
struct GGUFKernelTests {
    static let quantized = GGMLType.allCases.filter { $0.block.elements > 1 }

    static func halfOffsets(_ type: GGMLType) -> [Int] {
        switch type {
        case .q2K: [80, 82]
        case .q4K: [0, 2]
        case .q6K: [208]
        case .iq1M, .f32, .f16, .bf16: []
        default: [0]
        }
    }

    static func blocks(_ type: GGMLType, count: Int, seed: UInt64, scale: Float16 = 0.01) -> [UInt8] {
        var state = seed &+ 0x9E37_79B9_7F4A_7C15
        var bytes = (0 ..< count * type.block.bytes).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt8(truncatingIfNeeded: state >> 33)
        }
        for block in 0 ..< count {
            let base = block * type.block.bytes
            for offset in halfOffsets(type) {
                (bytes[base + offset], bytes[base + offset + 1]) = (UInt8(scale.bitPattern & 0xff), UInt8(scale.bitPattern >> 8))
            }
            guard type == .iq1M else { continue }
            for index in 0 ..< 4 { bytes[base + 49 + 2 * index] &= 0x0f }
            bytes[base + 55] |= 0x20
        }
        return bytes
    }

    static func dense(_ bytes: [UInt8], _ type: GGMLType, rows: Int, columns: Int) -> MLXArray {
        let floats = bytes.withUnsafeBytes { GGUFDequant.floats($0, type: type, count: rows * columns) }
        return MLXArray(floats, [rows, columns])
    }

    static func packed(_ bytes: [UInt8], rows: Int) -> MLXArray { MLXArray(bytes, [rows, bytes.count / rows]) }

    static func relativeError(_ actual: MLXArray, _ reference: MLXArray) -> Float {
        (abs(actual.asType(.float32) - reference).max() / abs(reference).max()).item(Float.self)
    }

    @Test("Dequantize matches the CPU dequantizer for every block type", arguments: quantized)
    func dequantizeMatchesCPU(type: GGMLType) {
        let (rows, columns) = (8, 512)
        let bytes = Self.blocks(type, count: rows * columns / type.block.elements, seed: UInt64(type.rawValue))
        let reference = Self.dense(bytes, type, rows: rows, columns: columns)
        let actual = GGUFKernels.dequantize(Self.packed(bytes, rows: rows), type: type, columns: columns, dtype: .float32)
        #expect(actual.shape == [rows, columns])
        #expect(Self.relativeError(actual, reference) < 1e-6, "\(type.name)")
    }

    @Test("Matvec for 1-8 rows and dequantize → matmul above match x · Wᵀ", arguments: quantized, [DType.float32, .bfloat16])
    func linearMatchesReference(type: GGMLType, dtype: DType) throws {
        let (outputs, inputs) = (13, 512)
        let bytes = Self.blocks(type, count: outputs * inputs / type.block.elements, seed: UInt64(type.rawValue) + 7)
        let layer = GGUFLinear(type: type, outputs: outputs, inputs: inputs)
        try layer.update(parameters: ModuleParameters.unflattened(["weight": Self.packed(bytes, rows: outputs)]), verify: .all)
        let weight = Self.dense(bytes, type, rows: outputs, columns: inputs)
        for rows in [1, 3, 8, 12] {
            let x = MLXRandom.normal([1, rows, inputs], key: MLXRandom.key(UInt64(rows))).asType(dtype)
            let actual = layer(x)
            #expect(actual.shape == [1, rows, outputs] && actual.dtype == dtype)
            let error = Self.relativeError(actual, matmul(x.asType(.float32), weight.T))
            #expect(error < (dtype == .float32 ? 1e-5 : 1e-2), "\(type.name) rows \(rows) error \(error)")
        }
    }

    @Test("A permuted input gathers the activation columns before the kernel")
    func permutedInput() throws {
        let (outputs, inputs, type) = (8, 256, GGMLType.iq4XS)
        let bytes = Self.blocks(type, count: outputs, seed: 3)
        let order = MLXArray((0 ..< inputs).reversed().map(Int32.init))
        let layer = GGUFLinear(type: type, outputs: outputs, inputs: inputs, permutedInput: true)
        try layer.update(parameters: ModuleParameters.unflattened(["weight": Self.packed(bytes, rows: outputs), "input_order": order]), verify: .all)
        let x = MLXRandom.normal([1, inputs], key: MLXRandom.key(5))
        let reference = matmul(x.take(order, axis: -1), Self.dense(bytes, type, rows: outputs, columns: inputs).T)
        #expect(Self.relativeError(layer(x), reference) < 1e-5)
    }

    @Test("Tiny qwen3_5: GGUF layers give the same greedy tokens as dense weights decoded on the CPU (#228)")
    func tinyModelParity() throws {
        let model = try TinyModel.make(.qwen35)
        var (entries, packed, dense) = ([String: GGUFNativeLoader.Entry](), [(String, MLXArray)](), [(String, MLXArray)]())
        let layers = model.leafModules().flattened().filter { $0.1 is Linear || $0.1 is Embedding }
        for (index, (path, module)) in layers.enumerated() {
            let weight = (module as? Linear)?.weight ?? (module as! Embedding).weight
            let (rows, columns) = (weight.dim(0), weight.dim(1))
            guard columns % 32 == 0 else { continue }
            let type: GGMLType = index % 2 == 0 ? .q8_0 : .iq4NL
            let bytes = Self.blocks(type, count: rows * columns / 32, seed: UInt64(index), scale: 0.002)
            entries[path] = GGUFNativeLoader.Entry(type: type.name, dtype: "\(weight.dtype)")
            packed.append((path + ".weight", Self.packed(bytes, rows: rows)))
            dense.append((path + ".weight", Self.dense(bytes, type, rows: rows, columns: columns).asType(weight.dtype)))
        }
        #expect(entries.count >= 20)
        model.update(parameters: ModuleParameters.unflattened(dense))
        eval(model)
        let prompt = TinyModel.tokens("parity of the i-quant kernels")
        let expected = try CacheBlendQualityTests.greedy(prompt, count: 12, model: model)
        let logits = model(MLXArray(prompt.map(Int32.init))[.newAxis], cache: nil).asType(.float32)
        try GGUFNativeLoader.install(entries, in: model)
        try model.update(parameters: ModuleParameters.unflattened(packed), verify: [.noUnusedKeys, .shapeMismatch])
        eval(model)
        #expect(try CacheBlendQualityTests.greedy(prompt, count: 12, model: model) == expected)
        let error = Self.relativeError(model(MLXArray(prompt.map(Int32.init))[.newAxis], cache: nil), logits)
        #expect(error < 1e-4, "logits error \(error), dtype \(logits.dtype) \(dense[0].1.dtype)")
    }
}
