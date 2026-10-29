import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Token Recycling speculative decoding (#38)", .serialized)
struct TokenRecyclingTests {
    static let freshPrompt = [200, 201, 202, 203, 204]

    static func confirmed(_ links: [(Int, Int)], vocabularySize: Int = 10) -> TokenRecyclingTable {
        var table = TokenRecyclingTable(vocabularySize: vocabularySize)
        for (token, successor) in links + links { table.record(token, successor: successor) }
        return table
    }

    @Test("The recycled draft follows each token's successor once it was recorded twice in a row, stopping at an unconfirmed token or the limit")
    func chainFollowsConfirmedSuccessors() {
        var table = TokenRecyclingTable(vocabularySize: 10)
        table.record(1, successor: 2)
        #expect(table.chain(from: 1, limit: 4).isEmpty)
        table = Self.confirmed([(1, 2), (2, 3)])
        #expect(table.chain(from: 1, limit: 4) == [2, 3])
        #expect(table.chain(from: 1, limit: 1) == [2])
        #expect(table.chain(from: 1, limit: 0).isEmpty)
        #expect(table.chain(from: 3, limit: 4).isEmpty)
        table.record(2, successor: 1)
        #expect(table.chain(from: 1, limit: 5) == [2])
        table.record(2, successor: 1)
        #expect(table.chain(from: 1, limit: 5) == [2, 1, 2, 1, 2])
    }

    @Test("A token predicts a draft only when its confirmed successor has a confirmed successor of its own")
    func predictionNeedsTwoConfirmedLinks() {
        #expect(!Self.confirmed([(1, 2)]).predicts(after: 1))
        #expect(Self.confirmed([(1, 2), (2, 3)]).predicts(after: 1))
        #expect(!Self.confirmed([(1, 2), (2, 3)]).predicts(after: 2))
        #expect(!Self.confirmed([(1, 2), (2, 3)]).predicts(after: 42))
    }

    @Test("Tokens outside the vocabulary are neither recorded nor followed, and the table costs five bytes per vocabulary entry")
    func outOfVocabularyIsIgnored() {
        let table = Self.confirmed([(7, 1), (-1, 1), (1, 9)], vocabularySize: 4)
        #expect(table.chain(from: 7, limit: 3).isEmpty)
        #expect(table.chain(from: -1, limit: 3).isEmpty)
        #expect(table.chain(from: 1, limit: 3) == [9])
        #expect(TokenRecyclingTable(vocabularySize: 151_936).byteCount == 759_680)
    }

    @Test("A verify round counts down the other drafter's pause by the tokens it emitted, so accepted recycled rounds do not keep lookup paused, nor lookup rounds recycling")
    func roundsCountDownTheOtherDraftersPause() {
        var backoffs = DrafterBackoffs()
        for _ in 0 ..< 3 { backoffs.lookup.record(acceptedDrafts: 0) }
        #expect(backoffs.lookup.pause == 4)
        backoffs.recordRound(recycled: true, acceptedDrafts: 2)
        #expect(backoffs.lookup.pause == 1)
        #expect(backoffs.recycling.allowsDraft)
        backoffs.recordRound(recycled: true, acceptedDrafts: 0)
        #expect(backoffs.lookup.allowsDraft)
        #expect(backoffs.recycling.pause == 1)
        backoffs.recordRound(recycled: false, acceptedDrafts: 0)
        #expect(backoffs.recycling.allowsDraft)
        #expect(backoffs.lookup.pause == 8)
        backoffs.tick()
        #expect(backoffs.lookup.pause == 7)
    }

    @Test("With recycling off, prompt lookup keeps no table and drafts nothing from recycled successors")
    func offByDefault() throws {
        let run = try LookupFixture.lookup(try TinyLlama.make(), prompt: Self.freshPrompt)
        #expect(run.iterator.recycling.byteCount == 0)
        #expect(run.iterator.recycledRounds == 0)
    }

    @Test("When every recycled draft is rejected, recycling pauses by doubling and the output is unchanged")
    func rejectedRecycledDraftsBackOff() throws {
        let cycles = (0 ..< 26).flatMap { [1, 2, 3, 4, 10 + $0] }
        let run = try LookupFixture.lookup(
            try TinyQwen3.make(), prompt: Self.freshPrompt, maxTokens: cycles.count,
            sampler: ScriptedSampler(cycles), recycling: true)
        #expect(run.tokens == cycles)
        #expect(run.iterator.acceptedRecycledDrafts == 0)
        #expect(run.iterator.recycledRounds > 0)
        #expect(run.iterator.recycledRounds <= 8)
    }

    @Test("Through the engine, greedy output is unchanged with recycling on, and recycling drafts beyond what lookup alone drafts, with and without a cache key")
    func engineGreedyEquivalence() async throws {
        func engine(lookup: Bool, recycling: Bool) async throws -> MLXInferenceEngine {
            let engine = MLXInferenceEngine(promptCacheBytes: 1 << 26, speculativeLookup: lookup, speculativeRecycling: recycling)
            await engine.attach(try LookupFixture.byteContainer())
            return engine
        }
        let plain = try await engine(lookup: false, recycling: false)
        let lookup = try await engine(lookup: true, recycling: false)
        let recycling = try await engine(lookup: true, recycling: true)
        let request = GenerationRequest(prompt: "Hello", maxTokens: 64, temperature: 0)
        let expected = try await plain.generate(request: request)
        for key in [nil, "recycling"] {
            let lookupOnly = try await lookup.generate(request: request.withCacheKey(key))
            let actual = try await recycling.generate(request: request.withCacheKey(key))
            #expect(actual.generatedText == expected.generatedText)
            #expect(try #require(actual.draftTokens) > lookupOnly.draftTokens ?? 0)
        }
    }

    @Test("With no n-gram in the prompt to look up, greedy decoding drafts from recycled successors, accepts them, and emits exactly the plain decoder's tokens",
          arguments: ["llama", "qwen3"])
    func greedyEquivalence(family: String) throws {
        let model = try LookupFixture.model(family)
        let expected = try LookupFixture.baseline(model, prompt: Self.freshPrompt)
        let run = try LookupFixture.lookup(model, prompt: Self.freshPrompt, recycling: true)
        #expect(run.tokens == expected)
        #expect(run.iterator.acceptedRecycledDrafts > 0)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.targetModelCallCount < run.tokens.count)
        #expect(telemetry.emittedTokenCount == run.tokens.count)
    }

    @Test("Recycled drafts never run past the token budget: at every budget the output is the plain decoder's prefix and the cache holds the prompt and every emitted token but the last")
    func budgetBoundary() throws {
        let model = try TinyLlama.make()
        let expected = try LookupFixture.baseline(model, prompt: Self.freshPrompt, maxTokens: 64)
        var recycledRounds = 0
        for maxTokens in 2 ... 64 {
            let run = try LookupFixture.lookup(model, prompt: Self.freshPrompt, maxTokens: maxTokens, recycling: true)
            #expect(run.tokens == Array(expected.prefix(maxTokens)))
            #expect(run.cache.allSatisfy { $0.offset == Self.freshPrompt.count + maxTokens - 1 }, "budget \(maxTokens)")
            recycledRounds += run.iterator.recycledRounds
        }
        #expect(recycledRounds > 0)
    }

    @Test("When the last token of the budget breaks a confirmed recycled chain, it is emitted without feeding it to the model")
    func noLookAheadPastTheBudgetWhenTheChainBreaks() throws {
        let model = try TinyLlama.make()
        let greedy = try LookupFixture.baseline(model, prompt: Self.freshPrompt, maxTokens: 64)
        for maxTokens in 2 ... 64 {
            let script = Array(greedy.prefix(maxTokens - 1)) + [249]
            let run = try LookupFixture.lookup(
                model, prompt: Self.freshPrompt, maxTokens: maxTokens, sampler: ScriptedSampler(script), recycling: true)
            #expect(run.tokens == script)
            #expect(run.cache.allSatisfy { $0.offset == Self.freshPrompt.count + maxTokens - 1 }, "budget \(maxTokens)")
        }
    }

    @Test("Rejected recycled drafts are rolled back: the cache holds the prompt and every emitted token but the last")
    func rollback() throws {
        let run = try LookupFixture.lookup(try TinyLlama.make(), prompt: Self.freshPrompt, maxTokens: 64, recycling: true)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(run.iterator.recycledDraftTokens > run.iterator.acceptedRecycledDrafts)
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
            model, prompt: Self.freshPrompt, maxTokens: maxTokens, processor: processor(), sampler: sampler(), recycling: true)
        #expect(run.tokens == expected)
        #expect(run.iterator.acceptedRecycledDrafts > 0)
    }
}
