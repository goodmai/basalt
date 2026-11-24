import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("LoRA on microscaled layers (mxfp4) is refused with the mode named, until the fork computes it in float; affine layers still take it (#746)", .serialized)
struct MicroscaledLoRARefusalTests {
    static func quantized(_ mode: QuantizationMode) throws -> (any LanguageModel, [PEFTAdapterTests.Pair]) {
        let model = try TinyModel.make(.qwen35)
        let pairs = PEFTAdapterTests.pairs(model, family: .qwen35)
        quantize(model: model, groupSize: 32, bits: 4, mode: mode) { path, module in
            module is Linear && pairs.contains { $0.path == path }
        }
        return (model, pairs)
    }

    static func loraPaths(_ model: any LanguageModel) -> [String] {
        model.namedModules().filter { $0.1 is LoRALayer }.map(\.0)
    }

    @Test("A PEFT adapter and a rank-1 adapter_apply on mxfp4 layers are refused naming mxfp4 and #746, and leave the model without LoRA layers")
    func mxfp4IsRefused() throws {
        let (model, pairs) = try Self.quantized(.mxfp4)
        let directory = try PEFTAdapterTests.directory(pairs, config: PEFTAdapterTests.config())
        defer { try? FileManager.default.removeItem(at: directory) }
        let peft = #expect(throws: BasaltError.self) { try MLXInferenceEngine.attachAnyAdapterDirectory(directory, to: model) }
        #expect("\(String(describing: peft))".contains("mxfp4"))
        #expect("\(String(describing: peft))".contains("#746"))
        #expect(Self.loraPaths(model).isEmpty)

        let target = try #require(pairs.first)
        let (outputs, inputs) = try #require(model.namedModules().first { $0.0 == target.path }?.1 as? Linear).shape
        let rankOne = #expect(throws: BasaltError.self) {
            try MLXInferenceEngine.attachRankOne(
                target.path, input: Array(repeating: 0.1, count: inputs), output: Array(repeating: 0.1, count: outputs), scale: 1, to: model)
        }
        #expect("\(String(describing: rankOne))".contains("mxfp4"))
        #expect(Self.loraPaths(model).isEmpty)
    }

    @Test("The same PEFT adapter on affine 4-bit layers still attaches to every target")
    func affineStillAttaches() throws {
        let (model, pairs) = try Self.quantized(.affine)
        let directory = try PEFTAdapterTests.directory(pairs, config: PEFTAdapterTests.config())
        defer { try? FileManager.default.removeItem(at: directory) }
        let wrapped = try MLXInferenceEngine.attachAnyAdapterDirectory(directory, to: model)
        #expect(Set(wrapped) == Set(pairs.map(\.path)))
    }
}
