import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("LoRA on microscaled layers computes in float: an adapter on mxfp4 and mxfp8 layers matches W + s·B·A on the dequantized weights (#746, #228)", .serialized)
struct MicroscaledLoRAParityTests {
    static func quantized(_ mode: QuantizationMode, bits: Int) throws -> (any LanguageModel, [PEFTAdapterTests.Pair]) {
        let model = try TinyModel.make(.qwen35)
        let pairs = PEFTAdapterTests.pairs(model, family: .qwen35)
        quantize(model: model, groupSize: 32, bits: bits, mode: mode) { path, module in
            module is Linear && pairs.contains { $0.path == path }
        }
        return (model, pairs)
    }

    static func merged(_ model: any LanguageModel, _ pairs: [PEFTAdapterTests.Pair], scale: Float) throws -> [(String, Module)] {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        return try pairs.map { pair in
            let quantized = try #require(modules[pair.path] as? QuantizedLinear)
            let dense = MLX.dequantized(
                quantized.weight, scales: quantized.scales, biases: quantized.biases,
                groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode, dtype: .float32)
            let weight = dense + scale * matmul(pair.b, pair.a)
            return (pair.path, Linear(weight: weight, bias: quantized.bias))
        }
    }

    @Test("A PEFT adapter on mxfp4 and mxfp8 layers gives the logits of the dequantized W + s·B·A model within 5e-3 (its LoRA weights are bfloat16 there; measured 2.1e-3 on logits up to 2.2) and the same greedy ids",
          arguments: [(QuantizationMode.mxfp4, 4), (.mxfp8, 8)])
    func matchesDequantizedMerge(mode: QuantizationMode, bits: Int) throws {
        let (adapted, pairs) = try Self.quantized(mode, bits: bits)
        let (reference, _) = try Self.quantized(mode, bits: bits)
        reference.update(modules: ModuleChildren.unflattened(try Self.merged(reference, pairs, scale: 2)))
        let directory = try PEFTAdapterTests.directory(pairs, config: PEFTAdapterTests.config())
        defer { try? FileManager.default.removeItem(at: directory) }
        let wrapped = try MLXInferenceEngine.attachAnyAdapterDirectory(directory, to: adapted)
        #expect(Set(wrapped) == Set(pairs.map(\.path)))
        let got = adapted(PEFTAdapterTests.tokens, cache: nil).asType(.float32)
        let expected = reference(PEFTAdapterTests.tokens, cache: nil).asType(.float32)
        #expect(abs(got - expected).max().item(Float.self) < 5e-3)
        #expect(argMax(got, axis: -1).asArray(Int32.self) == argMax(expected, axis: -1).asArray(Int32.self))
    }
}
