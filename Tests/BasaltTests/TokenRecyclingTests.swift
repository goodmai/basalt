import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltInference

@Suite("Token Recycling speculative decoding (#38)", .serialized)
struct TokenRecyclingTests {
    static let freshPrompt = [200, 201, 202, 203, 204]

    @Test("The recycled draft follows the latest recorded successor of each token, stopping at an unseen token or the limit")
    func chainFollowsSuccessors() {
        var table = TokenRecyclingTable(vocabularySize: 10)
        #expect(table.chain(from: 1, limit: 4).isEmpty)
        table.record(1, successor: 2)
        table.record(2, successor: 3)
        #expect(table.chain(from: 1, limit: 4) == [2, 3])
        #expect(table.chain(from: 1, limit: 1) == [2])
        table.record(2, successor: 1)
        #expect(table.chain(from: 1, limit: 5) == [2, 1, 2, 1, 2])
        #expect(table.chain(from: 3, limit: 4).isEmpty)
        #expect(table.chain(from: 1, limit: 0).isEmpty)
    }

    @Test("Tokens outside the vocabulary are neither recorded nor followed, and the table costs four bytes per vocabulary entry")
    func outOfVocabularyIsIgnored() {
        var table = TokenRecyclingTable(vocabularySize: 4)
        table.record(7, successor: 1)
        table.record(-1, successor: 1)
        table.record(1, successor: 9)
        #expect(table.chain(from: 7, limit: 3).isEmpty)
        #expect(table.chain(from: 1, limit: 3) == [9])
        #expect(TokenRecyclingTable(vocabularySize: 151_936).byteCount == 607_744)
    }

    @Test("With no n-gram in the prompt to look up, greedy decoding drafts from recycled successors, accepts them, and emits exactly the plain decoder's tokens",
          arguments: ["llama", "qwen3"])
    func greedyEquivalence(family: String) throws {
        let model = try LookupFixture.model(family)
        let expected = try LookupFixture.baseline(model, prompt: Self.freshPrompt)
        let run = try LookupFixture.lookup(model, prompt: Self.freshPrompt)
        #expect(run.tokens == expected)
        #expect(run.iterator.acceptedRecycledDrafts > 0)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.targetModelCallCount < run.tokens.count)
        #expect(telemetry.emittedTokenCount == run.tokens.count)
    }

    @Test("Recycled drafts never run past the token budget: the output is the plain decoder's prefix and the cache holds the prompt and every emitted token but the last",
          arguments: [3, 4, 9, 33])
    func budgetBoundary(maxTokens: Int) throws {
        let model = try TinyLlama.make()
        let expected = try LookupFixture.baseline(model, prompt: Self.freshPrompt, maxTokens: maxTokens)
        let run = try LookupFixture.lookup(model, prompt: Self.freshPrompt, maxTokens: maxTokens)
        #expect(run.tokens == expected)
        #expect(run.cache.allSatisfy { $0.offset == Self.freshPrompt.count + maxTokens - 1 })
    }

    @Test("Rejected recycled drafts are rolled back: the cache holds the prompt and every emitted token but the last")
    func rollback() throws {
        let run = try LookupFixture.lookup(try TinyQwen3.make(), prompt: Self.freshPrompt, maxTokens: 48)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(run.iterator.recycledRounds > 0)
        #expect(telemetry.rejectedDraftTokenCount > 0)
        #expect(run.cache.allSatisfy { $0.offset == Self.freshPrompt.count + run.tokens.count - 1 })
    }

    @Test("A seeded sampler at T > 0, alone or with a stateful processor, draws exactly the plain decoder's tokens while recycled drafts are accepted",
          arguments: ["none", "repetition", "entropy"])
    func seededSamplerParity(shaping: String) throws {
        let model = try TinyLlama.make()
        let maxTokens = EntropyDynamicTemperature.minTokens + EntropyDynamicTemperature.readbackLag + 64
        func processor() -> LogitProcessor? {
            switch shaping {
            case "repetition": RepetitionContext(repetitionPenalty: 1.1, repetitionContextSize: 16)
            case "entropy": EntropyDynamicTemperature(baseTemperature: 0.03, topP: 0.9)
            default: nil
            }
        }
        func sampler(seed: UInt64 = 11) -> LogitSampler {
            shaping == "entropy"
                ? CategoricalSampler(temperature: 0.03, seed: seed)
                : TopPSampler(temperature: 0.03, topP: 0.9, seed: seed)
        }
        let expected = try LookupFixture.baseline(
            model, prompt: Self.freshPrompt, maxTokens: maxTokens, processor: processor(), sampler: sampler())
        #expect(try LookupFixture.baseline(
            model, prompt: Self.freshPrompt, maxTokens: maxTokens, processor: processor(), sampler: sampler(seed: 12)) != expected)
        let run = try LookupFixture.lookup(
            model, prompt: Self.freshPrompt, maxTokens: maxTokens, processor: processor(), sampler: sampler())
        #expect(run.tokens == expected)
        #expect(run.iterator.acceptedRecycledDrafts > 0)
    }
}
