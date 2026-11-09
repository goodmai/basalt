import Foundation
import Testing
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
@testable import BasaltInference

@Suite("MTP draft head: a 4-bit copy of a wider target head drafts, the target head verifies (#209)")
struct MTPDraftHeadTests {
    static func drafter() throws -> Qwen35MTPDraftModel {
        let configuration = try JSONDecoder.json5().decode(Qwen35TextConfiguration.self, from: Data(TinyModel.Family.qwen35.config.utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 7)) { Qwen35MTPDraftModel(configuration) }
        eval(model)
        return model
    }

    static func tokens(_ target: any LanguageModel, drafter: Qwen35MTPDraftModel?) throws -> [Int] {
        let prompt = LMInput(tokens: MLXArray([5, 9, 5, 9, 5, 9, 5].map(Int32.init)))
        let parameters = GenerateParameters(maxTokens: 16, temperature: 0)
        guard let drafter else { return Array(IteratorSequence(try TokenIterator(input: prompt, model: target, parameters: parameters))) }
        var iterator = try MTPSpeculativeTokenIterator(input: prompt, mainModel: target, drafter: GreedyDraftingMTPDrafter(drafter), parameters: parameters, blockSize: 4)
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        #expect(iterator.proposedCount > 0)
        return tokens
    }

    @Test("A float target gets a 4-bit g64 draft head, and greedy output is the plain decoder's with and without it")
    func floatTarget() throws {
        let target = try TinyModel.make(.qwen35)
        let plain = try Self.tokens(target, drafter: nil)
        let baseline = try Self.drafter()
        let withHead = try Self.drafter()
        #expect(withHead.useDraftHead(of: target))
        let head = try #require(withHead.draftHead as? QuantizedLinear)
        #expect(head.bits == 4 && head.groupSize == 64)
        #expect(try Self.tokens(target, drafter: baseline) == plain)
        #expect(try Self.tokens(target, drafter: withHead) == plain)
    }

    @Test("A target whose head is already 4-bit keeps drafting through its own head")
    func quantizedTarget() throws {
        let target = try TinyModel.make(.qwen35)
        quantize(model: target, groupSize: 64, bits: 4)
        eval(target)
        let drafter = try Self.drafter()
        #expect(!drafter.useDraftHead(of: target))
        #expect(drafter.draftHead == nil)
    }
}

@Suite("MTP sidecar norms: a raw Hugging Face head gets the +1 shift, a converted one does not")
struct MTPSidecarNormTests {
    @Test("The first layer's input norm decides, even when the final norm of a raw head sits above 0.5 as in Qwen3.5-9B")
    func layerNormDecides() {
        let raw = ["mtp.layers.0.input_layernorm.weight": MLXArray.full([8], values: MLXArray(Float(0.1))), "mtp.norm.weight": MLXArray.full([8], values: MLXArray(Float(1.43)))]
        let converted = ["mtp.layers.0.input_layernorm.weight": MLXArray.full([8], values: MLXArray(Float(1.04))), "mtp.norm.weight": MLXArray.full([8], values: MLXArray(Float(2.25)))]
        #expect(MTPSidecar.preconvertedNorms(raw) == false)
        #expect(MTPSidecar.preconvertedNorms(converted) == true)
        #expect(MTPSidecar.preconvertedNorms(["mtp.norm.weight": MLXArray.full([8], values: MLXArray(Float(0.2)))]) == false)
        #expect(MTPSidecar.preconvertedNorms([:]) == nil)
    }
}
