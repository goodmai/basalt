import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
@testable import BasaltCore
@testable import BasaltInference

final class ScriptedSampler: LogitSampler {
    let script: [Int]
    private var position = 0

    init(_ script: [Int]) { self.script = script }

    func sample(logits: MLXArray) -> MLXArray {
        defer { position += 1 }
        return MLXArray([Int32(script[position % script.count])])
    }
}

enum LookupFixture {
    static let unit = (0 ..< 10).map { 1 + ($0 * 37 + 265) % 250 }
    static let prompt = Array(repeating: unit, count: 6).joined() + Array(unit.prefix(4))
    static let code = String(repeating: "func add(a: Int, b: Int) -> Int { a + b }\n", count: 6) + "func add("

    static func model(_ family: String) throws -> any LanguageModel {
        family == "llama" ? try TinyLlama.make() : try TinyQwen3.make()
    }

    static func baseline(
        _ model: any LanguageModel, prompt: [Int] = prompt, maxTokens: Int = 64,
        processor: LogitProcessor? = nil, sampler: LogitSampler = ArgMaxSampler()
    ) throws -> [Int] {
        let iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), model: model,
            processor: processor, sampler: sampler, maxTokens: maxTokens)
        return Array(iterator)
    }

    static func byteContainer() throws -> ModelContainer {
        ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-llama-4"), model: try TinyLlama.make(),
            processor: ByteInputProcessor(), tokenizer: ByteTokenizer()))
    }

    static func lookup(
        _ model: any LanguageModel, prompt: [Int] = prompt, maxTokens: Int = 64,
        processor: LogitProcessor? = nil, sampler: LogitSampler = ArgMaxSampler()
    ) throws -> (tokens: [Int], iterator: PromptLookupTokenIterator, cache: [KVCache]) {
        let cache = try model.newCache(parameters: nil)
        var iterator = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), context: prompt, model: model, cache: cache,
            processor: processor, sampler: sampler, prefill: .init(), maxTokens: maxTokens)
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        return (tokens, iterator, cache)
    }
}

@Suite("Prompt lookup speculative decoding (#138)", .serialized)
struct PromptLookupDecodingTests {

    @Test("The draft continues the longest suffix n-gram, preferring the latest occurrence that fills the draft")
    func draftLookup() {
        #expect(PromptLookupTokenIterator.draft(after: [1, 2, 3, 4, 5, 6, 1, 2, 3], limit: 8) == [4, 5, 6, 1, 2, 3])
        #expect(PromptLookupTokenIterator.draft(after: [1, 2, 3, 4, 5, 6, 1, 2, 3], limit: 2) == [4, 5])
        #expect(PromptLookupTokenIterator.draft(
            after: [8, 9, 1, 2, 3, 40, 0, 7, 1, 2, 3, 50, 0, 8, 9, 1, 2, 3], limit: 1) == [40])
        #expect(PromptLookupTokenIterator.draft(after: [1, 2, 3, 40, 1, 2, 3, 50, 1, 2, 3], limit: 1) == [50])
        #expect(PromptLookupTokenIterator.draft(after: [5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7], limit: 4) == [5, 6, 7, 5])
        #expect(PromptLookupTokenIterator.draft(after: [1, 2, 3, 4, 5, 6, 7], limit: 8).isEmpty)
        #expect(PromptLookupTokenIterator.draft(after: [1, 2, 1, 2], limit: 8).isEmpty)
    }

    @Test("Greedy decoding with prompt lookup emits exactly the plain decoder's tokens, and drafts are accepted",
          arguments: ["llama", "qwen3"])
    func greedyEquivalence(family: String) throws {
        let model = try LookupFixture.model(family)
        let expected = try LookupFixture.baseline(model)
        let (tokens, iterator, _) = try LookupFixture.lookup(model)
        #expect(tokens == expected)
        let telemetry = try #require(iterator.speculativeDecodingTelemetry)
        #expect(telemetry.acceptedDraftTokenCount > 0)
        #expect(telemetry.emittedTokenCount == tokens.count)
        #expect(telemetry.targetModelCallCount < tokens.count)
    }

    @Test("When the output copies a long span of the prompt, a verify round drafts more than 8 tokens on average")
    func draftsRunPastEightTokens() throws {
        let continuation = (0 ..< 64).map { LookupFixture.unit[(4 + $0) % LookupFixture.unit.count] }
        let run = try LookupFixture.lookup(try TinyLlama.make(), sampler: ScriptedSampler(continuation))
        #expect(run.tokens == continuation)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.draftTokenCount > 8 * telemetry.roundCount)
    }

    @Test("Rejected drafts are rolled back: the cache holds the prompt and every emitted token but the last")
    func rollback() throws {
        let model = try TinyQwen3.make()
        let (tokens, iterator, cache) = try LookupFixture.lookup(model, maxTokens: 48)
        #expect(tokens.count == 48)
        #expect(try #require(iterator.speculativeDecodingTelemetry).rejectedDraftTokenCount > 0)
        #expect(cache.allSatisfy { $0.offset == LookupFixture.prompt.count + tokens.count - 1 })
    }

    @Test("Without drafts, the last token of the budget is not fed to the model: no look-ahead forward is thrown away",
          arguments: [1, 2, 17])
    func noLookAheadAtTheBudget(maxTokens: Int) throws {
        let prompt = [200, 201, 202, 203, 204]
        let run = try LookupFixture.lookup(
            try TinyQwen3.make(), prompt: prompt, maxTokens: maxTokens,
            sampler: ScriptedSampler((0 ..< maxTokens).map { 10 + $0 }))
        #expect(run.tokens == (0 ..< maxTokens).map { 10 + $0 })
        #expect(run.iterator.speculativeDecodingTelemetry == nil)
        #expect(run.cache.allSatisfy { $0.offset == prompt.count + maxTokens - 1 })
    }

    @Test("A stateful logit processor and a seeded sampler see the same calls as in plain decoding",
          arguments: ["repetition", "entropy"])
    func processorAndSamplerParity(shaping: String) throws {
        let model = try TinyLlama.make()
        func processor() -> LogitProcessor {
            shaping == "repetition"
                ? RepetitionContext(repetitionPenalty: 1.1, repetitionContextSize: 16)
                : EntropyDynamicTemperature(baseTemperature: 0.2)
        }
        func sampler() -> LogitSampler {
            shaping == "repetition" ? ArgMaxSampler() : CategoricalSampler(temperature: 0.2, seed: 7)
        }
        let expected = try LookupFixture.baseline(model, processor: processor(), sampler: sampler())
        let actual = try LookupFixture.lookup(model, processor: processor(), sampler: sampler())
        #expect(actual.tokens == expected)
        #expect(try #require(actual.iterator.speculativeDecodingTelemetry).acceptedDraftTokenCount > 0)
    }

    @Test("A seeded sampler at T > 0 draws exactly the plain decoder's tokens with lookup on, with no processor or a stateful one past its warm-up, and drafts are accepted",
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
        let expected = try LookupFixture.baseline(model, maxTokens: maxTokens, processor: processor(), sampler: sampler())
        #expect(try LookupFixture.baseline(model, maxTokens: maxTokens, processor: processor(), sampler: sampler(seed: 12)) != expected)
        let actual = try LookupFixture.lookup(model, maxTokens: maxTokens, processor: processor(), sampler: sampler())
        #expect(actual.tokens == expected)
        #expect(try #require(actual.iterator.speculativeDecodingTelemetry).acceptedDraftTokenCount > 0)
    }

    static func sampledEngines(
        seed: UInt64, _ container: () throws -> ModelContainer
    ) async throws -> (plain: MLXInferenceEngine, lookup: MLXInferenceEngine) {
        let plain = MLXInferenceEngine(seed: seed)
        await plain.attach(try container(), profile: GenericProfile())
        let lookup = MLXInferenceEngine(seed: seed, speculativeLookup: true)
        await lookup.attach(try container(), profile: GenericProfile())
        return (plain, lookup)
    }

    @Test("A JSON-constrained request sampled at T 0.7 with a seed decodes the same document with prompt lookup on, and drafts are accepted")
    func constrainedSampledEquivalence() async throws {
        let example = #"{"mode": "fast", "flags": [true, true, true, true], "inner": {"on": true, "note": null}}"#
        let request = GenerationRequest(
            prompt: "Report the mode as \(example)", maxTokens: 300, temperature: 0.7,
            responseFormat: .jsonSchema(ConstrainedModel.schema))
        let engines = try await Self.sampledEngines(seed: 1) { try ConstrainedModel.container() }
        let expected = try await engines.plain.generate(request: request)
        let actual = try await engines.lookup.generate(request: request)
        #expect(actual.generatedText == expected.generatedText)
        #expect(try ConstrainedModel.isValid(actual.generatedText), "\(actual.generatedText)")
        #expect(try #require(actual.acceptedDraftTokens) > 0)
        let reseeded = try await Self.sampledEngines(seed: 2) { try ConstrainedModel.container() }.plain.generate(request: request)
        #expect(reseeded.generatedText != expected.generatedText)
    }

    @Test("A presence penalty at T > 0 with a seed leaves the output unchanged with prompt lookup on, and drafts are accepted")
    func presencePenaltySampledEquivalence() async throws {
        let prompt = LookupFixture.code
        let request = GenerationRequest(prompt: prompt, maxTokens: 64, temperature: 0.1, presencePenalty: 0.5)
        let engines = try await Self.sampledEngines(seed: 11, LookupFixture.byteContainer)
        let expected = try await engines.plain.generate(request: request)
        let actual = try await engines.lookup.generate(request: request)
        #expect(actual.generatedText == expected.generatedText)
        #expect(try #require(actual.acceptedDraftTokens) > 0)
        let unpenalized = GenerationRequest(prompt: prompt, maxTokens: 64, temperature: 0.1)
        #expect(try await engines.plain.generate(request: unpenalized).generatedText != expected.generatedText)
    }

    @Test("Through the engine, greedy output is unchanged with prompt lookup on, and the response reports the drafts")
    func engineGreedyEquivalence() async throws {
        let prompt = LookupFixture.code
        let plain = MLXInferenceEngine(promptCacheBytes: 1 << 26)
        await plain.attach(try LookupFixture.byteContainer())
        let speculative = MLXInferenceEngine(promptCacheBytes: 1 << 26, speculativeLookup: true)
        await speculative.attach(try LookupFixture.byteContainer())
        let request = GenerationRequest(prompt: prompt, maxTokens: 64, temperature: 0)
        let expected = try await plain.generate(request: request)
        let actual = try await speculative.generate(request: request)
        #expect(actual.generatedText == expected.generatedText)
        #expect(expected.draftTokens == nil)
        #expect(try #require(actual.acceptedDraftTokens) > 0)

        for _ in 0 ..< 2 {
            let cached = try await speculative.generate(request: request.withCacheKey("lookup"))
            #expect(cached.generatedText == expected.generatedText)
            #expect(cached.acceptedDraftTokens != nil)
        }
        let optedOut = try await speculative.generate(request: request.withSpeculativeLookup(false))
        #expect(optedOut.generatedText == expected.generatedText)
        #expect(optedOut.draftTokens == nil)
    }

    @Test("A JSON-constrained request decodes the same document with prompt lookup on")
    func constrainedEquivalence() async throws {
        let request = ConstrainedModel.request(format: .jsonSchema(ConstrainedModel.schema))
        let plain = try await ConstrainedModel.engine().generate(request: request)
        let speculative = try await ConstrainedModel.engine().generate(request: request.withSpeculativeLookup(true))
        #expect(speculative.generatedText == plain.generatedText)
        #expect(speculative.draftTokens != nil)
    }

    @Test("A hybrid model with recurrent layers decodes without lookup, since its state cannot be rolled back")
    func hybridIsExcluded() async throws {
        let engine = MLXInferenceEngine(promptCacheBytes: 1 << 26, speculativeLookup: true)
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-hybrid"), model: try TinyHybrid.make(),
            processor: ByteInputProcessor(), tokenizer: ByteTokenizer())))
        let request = GenerationRequest(prompt: String(repeating: "abc ", count: 12), maxTokens: 16, temperature: 0)
        for key in [nil, "hybrid"] {
            let result = try await engine.generate(request: request.withCacheKey(key))
            #expect(result.completionTokens > 0)
            #expect(result.draftTokens == nil)
        }
    }

    @Test("Each failed verify round doubles the pause before the next draft, up to 64 tokens; an accepted draft resets it")
    func backoffDoublesAndResets() {
        var backoff = DraftBackoff()
        var pauses: [Int] = []
        for _ in 0 ..< 8 {
            backoff.record(acceptedDrafts: 0)
            pauses.append(backoff.pause)
            while !backoff.allowsDraft { backoff.tick() }
        }
        #expect(pauses == [1, 2, 4, 8, 16, 32, 64, 64])
        backoff.record(acceptedDrafts: 1)
        #expect(backoff.allowsDraft)
        backoff.record(acceptedDrafts: 0)
        #expect(backoff.pause == 1)
    }

    @Test("When every draft past its first token is rejected, lookup and recycling each back off instead of verifying each match, and the output is unchanged")
    func rejectedDraftsBackOff() throws {
        let cycles = (0 ..< 26).flatMap { [1, 2, 3, 4, 10 + $0] }
        let run = try LookupFixture.lookup(
            try TinyQwen3.make(), prompt: [200, 201, 202, 203, 204], maxTokens: cycles.count,
            sampler: ScriptedSampler(cycles))
        #expect(run.tokens == cycles)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        let lookupRounds = telemetry.roundCount - run.iterator.recycledRounds
        let acceptedLookupDrafts = telemetry.acceptedDraftTokenCount - run.iterator.acceptedRecycledDrafts
        #expect(run.iterator.acceptedRecycledDrafts == 0)
        #expect(run.iterator.recycledRounds <= 8)
        #expect(lookupRounds - acceptedLookupDrafts <= 8)
    }
}
