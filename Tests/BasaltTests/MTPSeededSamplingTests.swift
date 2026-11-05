import Foundation
import Testing
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
@testable import BasaltCore
@testable import BasaltInference

final class SuccessorDrafter: Module, MTPDrafterModel {
    let successors: [Int: Int]

    init(following tokens: [Int]) {
        let links = tokens.indices.dropLast().filter { $0 % 5 != 4 }.map { (tokens[$0], tokens[$0 + 1]) }
        successors = Dictionary(links, uniquingKeysWith: { first, _ in first })
        super.init()
    }

    var requiresSharedTargetKV: Bool { false }

    func draftBlock(target: any LanguageModel, input: MTPDraftInput, sampler: any LogitSampler) -> MLXArray {
        var token = input.lastToken.item(Int.self)
        let drafts = (1 ..< input.blockSize).map { _ in
            token = successors[token] ?? (token + 1) % 256
            return Int32(token)
        }
        return MLXArray(drafts).reshaped(1, -1)
    }
}

@Suite("Qwen MTP at T > 0: a seeded sampler draws exactly the plain decoder's tokens, and acceptance is reported (#290)", .serialized)
struct MTPSeededSamplingTests {
    static var prompt: LMInput { LMInput(tokens: MLXArray(TinyModel.tokens(Conversation.firstTurn).map(Int32.init))) }

    static func parameters(repetitionPenalty: Float?, seed: UInt64 = 11) -> GenerateParameters {
        GenerateParameters(
            maxTokens: 48, temperature: 0.7, topP: 0.9, repetitionPenalty: repetitionPenalty, repetitionContextSize: 16, seed: seed)
    }

    static func plain(_ model: any LanguageModel, _ parameters: GenerateParameters) throws -> [Int] {
        Array(IteratorSequence(try TokenIterator(input: prompt, model: model, parameters: parameters)))
    }

    static func speculate(
        _ model: any LanguageModel, _ drafter: any MTPDrafterModel, _ parameters: GenerateParameters
    ) throws -> (tokens: [Int], iterator: MTPSpeculativeTokenIterator) {
        var iterator = try MTPSpeculativeTokenIterator(
            input: prompt, mainModel: model, drafter: drafter, parameters: parameters, blockSize: 4)
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        return (tokens, iterator)
    }

    @Test("At T = 0.7 with a seed, MTP emits the plain decoder's tokens on both Qwen3.5 fixtures, alone and with a repetition penalty, with the real head's drafts and with drafts that are partly accepted",
          arguments: [TinyModel.Family.qwen35, .prism], [false, true])
    func seededSamplerParity(family: TinyModel.Family, acceptedDrafts: Bool) throws {
        let model = try TinyModel.make(family)
        for penalty in [nil, 1.3] as [Float?] {
            let expected = try Self.plain(model, Self.parameters(repetitionPenalty: penalty))
            #expect(try Self.plain(model, Self.parameters(repetitionPenalty: penalty, seed: 12)) != expected)
            let drafter: any MTPDrafterModel = acceptedDrafts
                ? SuccessorDrafter(following: expected) : GreedyDraftingMTPDrafter(try MTPPromptCacheTests.head())
            let run = try Self.speculate(model, drafter, Self.parameters(repetitionPenalty: penalty))
            expectTokenSequence(run.tokens, expected)
            #expect(run.iterator.passthroughReason == nil)
            let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
            #expect(run.iterator.proposedCount > 0)
            if acceptedDrafts {
                #expect(run.iterator.acceptedCount > 0)
                #expect(telemetry.rejectedDraftTokenCount > 0)
            }
        }
    }

    @Test("Through the engine a seeded T = 0.7 request with --mtp answers like one without it and reports drafts proposed and accepted")
    func engineReportsAcceptance() async throws {
        func reply(mtp: Bool) async throws -> GenerationResponse {
            let engine = MLXInferenceEngine(seed: 11, mtp: mtp)
            let container = try TinyModel.container(.qwen35)
            await engine.attach(container)
            if mtp { await engine.useMTPDrafter(in: container) { _ in try? GreedyDraftingMTPDrafter(MTPPromptCacheTests.head()) } }
            return try await engine.generate(request: GenerationRequest(prompt: Conversation.firstTurn, maxTokens: 32, temperature: 0.7))
        }
        let plain = try await reply(mtp: false)
        let drafted = try await reply(mtp: true)
        #expect(plain.draftTokens == nil)
        #expect((drafted.draftTokens ?? 0) > 0)
        #expect(drafted.acceptedDraftTokens != nil)
        #expect(drafted.generatedText == plain.generatedText)
    }
}
