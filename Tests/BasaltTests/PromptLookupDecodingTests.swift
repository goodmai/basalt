import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
@testable import BasaltCore
@testable import BasaltInference

final class GreedyMarginRecorder: LogitProcessor, @unchecked Sendable {
    var margins: [Float] = []
    var rows: [MLXArray] = []

    func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let row = logits.reshaped([-1]).asType(.float32)
        let top = MLX.sorted(row)[(-2)...]
        rows.append(row)
        margins.append((top[1] - top[0]).item(Float.self))
        return logits
    }

    func didSample(token: MLXArray) {}
}

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
            configuration: ModelConfiguration(id: "tiny-qwen3-4"), model: try TinyModel.make(.qwen3),
            processor: ByteInputProcessor(), tokenizer: ByteTokenizer()))
    }

    static func lookup(
        _ model: any LanguageModel, prompt: [Int] = prompt, maxTokens: Int = 64,
        processor: LogitProcessor? = nil, sampler: LogitSampler = ArgMaxSampler(), recycling: Bool = false
    ) throws -> (tokens: [Int], iterator: PromptLookupTokenIterator, cache: [KVCache]) {
        let cache = try model.newCache(parameters: nil)
        var iterator = try PromptLookupTokenIterator(
            prompt: PromptLookupTokenIterator.Prompt(input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), history: prompt), model: model, cache: cache,
            decoding: DecodingSettings(processor: processor, sampler: sampler, prefill: .init(), maxTokens: maxTokens),
            lookup: LookupOptions(speculative: true, recycling: recycling))
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
          arguments: [TinyModel.Family.qwen3, .qwen35, .lfm2Moe, .nemotronH, .lfm2, .gemma4, .gemma4Moe])
    func greedyEquivalence(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let expected = try LookupFixture.baseline(model)
        let (tokens, iterator, _) = try LookupFixture.lookup(model)
        expectTokenSequence(tokens, expected)
        let telemetry = try #require(iterator.speculativeDecodingTelemetry)
        #expect(telemetry.acceptedDraftTokenCount > 0)
        #expect(telemetry.emittedTokenCount == tokens.count)
        #expect(telemetry.targetModelCallCount < tokens.count)
        #expect(telemetry.targetVerifiedTokenCount
            == telemetry.draftTokenCount + telemetry.targetModelCallCount - iterator.replayRounds)
    }

    @Test("A recurrent fixture's greedy path has no near-tie that the verify forward's chunked scan could flip",
          arguments: [TinyModel.Family.nemotronH, .qwen35, .lfm2], ["tokens", "code"])
    func recurrentFixturesHaveNoNearTie(family: TinyModel.Family, prompt: String) throws {
        let model = try TinyModel.make(family)
        let input = prompt == "code"
            ? ByteTokenizer().encode(text: LookupFixture.code, addSpecialTokens: false) + [ByteTokenizer.endOfUserTurn]
            : LookupFixture.prompt
        let recorder = GreedyMarginRecorder()
        let tokens = try LookupFixture.baseline(model, prompt: input, maxTokens: prompt == "code" ? 48 : 64, processor: recorder)
        let parallel = model(MLXArray((input + tokens).map(Int32.init))[.newAxis], cache: try model.newCache(parameters: nil))
            .asType(.float32)[0]
        let drift = recorder.rows.enumerated()
            .map { MLX.abs(parallel[input.count - 1 + $0.offset] - $0.element).max().item(Float.self) }.max() ?? 0
        let margin = try #require(recorder.margins.min())
        #expect(margin > 4 * drift, "\(family) \(prompt): smallest top-2 margin \(margin), step against one-pass drift \(drift)")
    }

    @Test("When the output copies a long span of the prompt, a verify round drafts more than 8 tokens on average")
    func draftsRunPastEightTokens() throws {
        let continuation = (0 ..< 64).map { LookupFixture.unit[(4 + $0) % LookupFixture.unit.count] }
        let run = try LookupFixture.lookup(try TinyModel.make(.qwen3), sampler: ScriptedSampler(continuation))
        expectTokenSequence(run.tokens, continuation)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.draftTokenCount > 8 * telemetry.roundCount)
    }

    @Test("A hybrid's first draft is a 4-token probe, and only an accepted probe opens the full draft limit; a dense model drafts the full limit at once",
          arguments: [(TinyModel.Family.qwen3, 1, 10), (TinyModel.Family.qwen35, 2, 9)])
    func hybridProbesBeforeLongDrafts(family: TinyModel.Family, rounds: Int, drafted: Int) throws {
        let continuation = (0 ..< 12).map { LookupFixture.unit[(4 + $0) % LookupFixture.unit.count] }
        let run = try LookupFixture.lookup(
            try TinyModel.make(family), maxTokens: continuation.count, sampler: ScriptedSampler(continuation))
        expectTokenSequence(run.tokens, continuation)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.roundCount == rounds)
        #expect(telemetry.draftTokenCount == drafted)
        #expect(telemetry.acceptedDraftTokenCount == drafted)
    }

    @Test("Rejected drafts are rolled back: the cache holds the prompt and every emitted token but the last")
    func rollback() throws {
        let model = try TinyModel.make(.qwen3)
        let (tokens, iterator, cache) = try LookupFixture.lookup(model, maxTokens: 48)
        #expect(tokens.count == 48)
        #expect(try #require(iterator.speculativeDecodingTelemetry).rejectedDraftTokenCount > 0)
        #expect(cache.allSatisfy { $0.offset == LookupFixture.prompt.count + tokens.count - 1 })
    }

    @Test("On a sliding-window model rejected drafts that cross the window's wrap point leave the ring as plain decoding would: the next-token logits match a plain prefill of the same tokens (#161)")
    func slidingWindowRollback() throws {
        let model = try TinyModel.make(.gemma4)
        let unit = LookupFixture.unit
        let script = (0 ..< 6).flatMap { round in Array(unit[4 ..< 8]) + [240 + round] + Array(unit[0 ..< 4]) }
        let run = try LookupFixture.lookup(model, maxTokens: script.count, sampler: ScriptedSampler(script))
        expectTokenSequence(run.tokens, script)
        #expect(run.iterator.stagesRounds)
        #expect(try #require(run.iterator.speculativeDecodingTelemetry).rejectedDraftTokenCount > 0)
        #expect(LookupFixture.prompt.count + run.tokens.count > 2 * TinyModel.slidingWindow)
        let last = MLXArray([Int32(run.tokens[run.tokens.count - 1])])[.newAxis]
        let staged = model(last, cache: run.cache)[0, -1]
        let reference = try model.newCache(parameters: nil)
        eval(model(MLXArray((LookupFixture.prompt + run.tokens.dropLast()).map(Int32.init))[.newAxis], cache: reference))
        let plain = model(last, cache: reference)[0, -1]
        #expect(allClose(staged, plain, atol: 1e-4).item(Bool.self))
    }

    static func rejectingRun(_ family: TinyModel.Family, maxTokens: Int) throws -> (model: any LanguageModel, run: (tokens: [Int], iterator: PromptLookupTokenIterator, cache: [KVCache]), reference: [KVCache]) {
        let model = try TinyModel.make(family)
        let unit = LookupFixture.unit
        let script = (0 ..< 6).flatMap { round in Array(unit[4 ..< 8]) + [240 + round] + Array(unit[0 ..< 4]) }
        let run = try LookupFixture.lookup(model, maxTokens: maxTokens, sampler: ScriptedSampler(script))
        expectTokenSequence(run.tokens, Array(script.prefix(maxTokens)))
        #expect(run.iterator.replayRounds > 0)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.targetVerifiedTokenCount
            == telemetry.draftTokenCount + telemetry.targetModelCallCount - run.iterator.replayRounds)
        let reference = try model.newCache(parameters: nil)
        eval(model(MLXArray((LookupFixture.prompt + run.tokens.dropLast()).map(Int32.init))[.newAxis], cache: reference))
        #expect(zip(run.cache, reference).allSatisfy { $0.offset == $1.offset })
        return (model, run, reference)
    }

    @Test("On a hybrid model a rejected draft restores the recurrent state and replays the accepted prefix, also when the budget clips a draft: every recurrent slot matches a plain prefill of the same tokens",
          arguments: [TinyModel.Family.qwen35, .lfm2Moe, .lfm2], [54, 7, 13])
    func hybridRejectionRestoresRecurrentState(family: TinyModel.Family, maxTokens: Int) throws {
        let (_, run, reference) = try Self.rejectingRun(family, maxTokens: maxTokens)
        for (live, plain) in zip(run.cache, reference) {
            guard let live = live as? MambaCache, let plain = plain as? MambaCache else { continue }
            for slot in 0 ..< 2 {
                #expect((live[slot] == nil) == (plain[slot] == nil))
                guard let a = live[slot], let b = plain[slot] else { continue }
                #expect(allClose(a.asType(.float32), b.asType(.float32), atol: 1e-3).item(Bool.self))
            }
        }
    }

    @Test("On Nemotron-H a rejected draft restores the Mamba state: after the replays the next-token logits match a plain prefill of the same tokens. Its chunked-prefill SSM state is not comparable slot by slot")
    func nemotronRejectionRestoresMambaState() throws {
        let (model, run, reference) = try Self.rejectingRun(.nemotronH, maxTokens: 54)
        let last = MLXArray([Int32(try #require(run.tokens.last))])[.newAxis]
        let replayed = model(last, cache: run.cache)[0, -1].asType(.float32)
        let prefilled = model(last, cache: reference)[0, -1].asType(.float32)
        #expect(abs(replayed - prefilled).max().item(Float.self) < 0.1)
    }

    @Test("Without drafts, the last token of the budget is not fed to the model: no look-ahead forward is thrown away",
          arguments: [TinyModel.Family.qwen3, .qwen35], [1, 2, 17])
    func noLookAheadAtTheBudget(family: TinyModel.Family, maxTokens: Int) throws {
        let prompt = [200, 201, 202, 203, 204]
        let run = try LookupFixture.lookup(
            try TinyModel.make(family), prompt: prompt, maxTokens: maxTokens,
            sampler: ScriptedSampler((0 ..< maxTokens).map { 10 + $0 }))
        expectTokenSequence(run.tokens, (0 ..< maxTokens).map { 10 + $0 })
        #expect(run.iterator.speculativeDecodingTelemetry == nil)
        #expect(run.cache.filter { $0 is KVCacheSimple }.allSatisfy { $0.offset == prompt.count + maxTokens - 1 })
    }

    @Test("A stateful logit processor and a seeded sampler see the same calls as in plain decoding",
          arguments: [(TinyModel.Family.qwen3, "repetition"), (TinyModel.Family.lfm2, "entropy"), (TinyModel.Family.qwen35, "repetition")])
    func processorAndSamplerParity(family: TinyModel.Family, shaping: String) throws {
        let model = try TinyModel.make(family)
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
        expectTokenSequence(actual.tokens, expected)
        #expect(try #require(actual.iterator.speculativeDecodingTelemetry).acceptedDraftTokenCount > 0)
    }

    @Test("DRY on: lookup on and off emit the same tokens, DRY changes the output, and drafts are still accepted",
          arguments: [TinyModel.Family.qwen3, .qwen35])
    func dryParity(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        func dry() -> LogitProcessor { DRYPenalty(multiplier: 0.2, allowedLength: 6, breakers: []) }
        let expected = try LookupFixture.baseline(model, processor: dry())
        #expect(expected != (try LookupFixture.baseline(model)))
        let actual = try LookupFixture.lookup(model, processor: dry())
        expectTokenSequence(actual.tokens, expected)
        #expect(try #require(actual.iterator.speculativeDecodingTelemetry).acceptedDraftTokenCount > 0)
    }

    @Test("A seeded sampler at T > 0 draws exactly the plain decoder's tokens with lookup on, with no processor or a stateful one past its warm-up, and drafts are accepted",
          arguments: [TinyModel.Family.qwen3, .qwen35], ["none", "repetition", "entropy"])
    func seededSamplerParity(family: TinyModel.Family, shaping: String) throws {
        let model = try TinyModel.make(family)
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
        expectTokenSequence(actual.tokens, expected)
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
            prompt: "Report the mode as \(example)", maxTokens: 300, temperature: 0.7).with { $0.responseFormat = .jsonSchema(ConstrainedModel.schema) }
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
        let request = GenerationRequest(prompt: prompt, maxTokens: 64, temperature: 0.1).with { $0.presencePenalty = 0.5 }
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

    @Test("Lookup admits attention-only caches, dense recurrent hybrids and sliding-window caches that stage rounds, with or without expert routing; recurrent hybrids that route tokens to experts stay on plain decoding",
          arguments: [(TinyModel.Family.qwen3, true), (TinyModel.Family.qwen35, true), (TinyModel.Family.nemotronH, true), (TinyModel.Family.lfm2, true), (TinyModel.Family.gemma4, true), (TinyModel.Family.gemma4Moe, true), (TinyModel.Family.lfm2Moe, false), (TinyModel.Family.qwen35Moe, false)])
    func guardAdmitsDenseHybridsOnly(family: TinyModel.Family, admitted: Bool) throws {
        let model = try TinyModel.make(family)
        let routesExperts = PromptLookupTokenIterator.routesExperts(model)
        #expect(routesExperts == [.lfm2Moe, .qwen35Moe, .gemma4Moe].contains(family))
        #expect(PromptLookupTokenIterator.supports(try model.newCache(parameters: nil), routesExperts: routesExperts) == admitted)
        let slidingWindow: [KVCache] = [RotatingKVCache(maxSize: 64), KVCacheSimple()]
        #expect(PromptLookupTokenIterator.supports(slidingWindow, routesExperts: false))
        #expect(PromptLookupTokenIterator.supports(slidingWindow, routesExperts: true))
        #expect(!PromptLookupTokenIterator.supports([], routesExperts: false))
    }

    @Test("Through the engine a dense hybrid decodes the plain text with drafts, with and without a prompt cache; an expert-routing hybrid decodes without drafts",
          arguments: [TinyModel.Family.qwen35, .nemotronH, .lfm2Moe])
    func engineHybridEquivalence(family: TinyModel.Family) async throws {
        func engine(lookup: Bool) async throws -> MLXInferenceEngine {
            let engine = MLXInferenceEngine(promptCacheBytes: 1 << 26, speculativeLookup: lookup)
            await engine.attach(ModelContainer(context: ModelContext(
                configuration: ModelConfiguration(id: "tiny-\(family)"), model: try TinyModel.make(family),
                processor: ByteInputProcessor(), tokenizer: ByteTokenizer())))
            return engine
        }
        let (plain, speculative) = (try await engine(lookup: false), try await engine(lookup: true))
        let request = GenerationRequest(prompt: LookupFixture.code, maxTokens: 48, temperature: 0)
        for key in [nil, "hybrid", "hybrid"] {
            let expected = try await plain.generate(request: request.withCacheKey(key))
            let actual = try await speculative.generate(request: request.withCacheKey(key))
            #expect(actual.generatedText == expected.generatedText)
            #expect((actual.draftTokens != nil) == (family != .lfm2Moe))
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

    @Test("A failed round that also paid a replay forward pauses four times longer, still doubling: 4, 8 … 256 tokens")
    func replayedFailuresPauseLonger() {
        var backoffs = DrafterBackoffs()
        var pauses: [Int] = []
        for _ in 0 ..< 8 {
            backoffs.recordRound(recycled: false, acceptedDrafts: 1, replayed: true)
            pauses.append(backoffs.lookup.pause)
            while !backoffs.lookup.allowsDraft { backoffs.tick() }
        }
        #expect(pauses == [4, 8, 16, 32, 64, 128, 256, 256])
        backoffs.recordRound(recycled: false, acceptedDrafts: 2, draftCount: 4, replayed: true)
        #expect(backoffs.lookup.allowsDraft)
        #expect(backoffs.lookup.draftLimit == 4)
    }

    @Test("A low-yield draft on a dense model backs off and shrinks the draft limit rather than resetting pause (#172)")
    func lowYieldDraftsBackOff() {
        var backoff = DraftBackoff()
        #expect(backoff.draftLimit == 16)
        backoff.record(acceptedDrafts: 1, drafted: 16)
        #expect(!backoff.allowsDraft)
        #expect(backoff.pause == 1)
        #expect(backoff.draftLimit == 2)
        while !backoff.allowsDraft { backoff.tick() }
        backoff.record(acceptedDrafts: 0, drafted: 2)
        #expect(backoff.pause == 2)
        while !backoff.allowsDraft { backoff.tick() }
        backoff.record(acceptedDrafts: 4, drafted: 4)
        #expect(backoff.allowsDraft)
        #expect(backoff.draftLimit == 6)
    }

    @Test("On a hybrid a round that accepts one draft and then replays does not pay for its extra forward, so lookup backs off; a dense model keeps drafting")
    func hybridBacksOffWhenTheReplayEatsTheGain() throws {
        let cycles = (0 ..< 20).flatMap { [1, 2, 3, 4, 5, 20 + $0] }
        func rounds(_ family: TinyModel.Family) throws -> (rounds: Int, accepted: Int) {
            let run = try LookupFixture.lookup(
                try TinyModel.make(family), prompt: [200, 201, 202, 203, 204], maxTokens: cycles.count,
                sampler: ScriptedSampler(cycles))
            expectTokenSequence(run.tokens, cycles)
            let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
            return (telemetry.roundCount, telemetry.acceptedDraftTokenCount)
        }
        let (dense, hybrid) = (try rounds(.qwen3), try rounds(.qwen35))
        #expect(dense.accepted == dense.rounds)
        #expect(hybrid.rounds * 2 <= dense.rounds)
    }

    @Test("When every draft past its first token is rejected, lookup backs off instead of verifying each match, and the output is unchanged",
          arguments: [TinyModel.Family.qwen3, .qwen35])
    func rejectedDraftsBackOff(family: TinyModel.Family) throws {
        let cycles = (0 ..< 26).flatMap { [1, 2, 3, 4, 10 + $0] }
        let run = try LookupFixture.lookup(
            try TinyModel.make(family), prompt: [200, 201, 202, 203, 204], maxTokens: cycles.count,
            sampler: ScriptedSampler(cycles))
        expectTokenSequence(run.tokens, cycles)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.acceptedDraftTokenCount == 0)
        #expect(telemetry.roundCount <= 6)
    }

    @Test("PromptLookupTokenIterator.draft produces expected matches across varied history lengths")
    func draftEquivalenceAcrossHistoryLengths() {
        let baseHistory = [42, 43, 44, 45, 46, 42, 43, 44, 45, 46]
        #expect(PromptLookupTokenIterator.draft(after: baseHistory, limit: 0).isEmpty)
        #expect(PromptLookupTokenIterator.draft(after: baseHistory, limit: -1).isEmpty)
        for length in [1, 2, 3, 4, 5, 10, 64, 256, 1024, 4096] {
            let pattern = [42, 43, 44, 45, 46]
            let history = Array(repeating: pattern, count: length / 5 + 1).flatMap { $0 }
            let draft = PromptLookupTokenIterator.draft(after: Array(history.prefix(length)), limit: 8)
            if length > 5 {
                #expect(!draft.isEmpty)
            }
        }
    }

    @Test("Micro-benchmark for Span-based draft scanning across 4k and 16k token histories")
    func draftMicroBenchmark() {
        for size in [4096, 16384] {
            let missHistory = Array(0 ..< size)
            let missDraft = PromptLookupTokenIterator.draft(after: missHistory, limit: 16)
            #expect(missDraft.isEmpty)

            let hitHistory = (0 ..< size).map { $0 % 64 }
            let hitDraft = PromptLookupTokenIterator.draft(after: hitHistory, limit: 16)
            #expect(!hitDraft.isEmpty)

            if ProcessInfo.processInfo.environment["BASALT_BENCH"] == "1" {
                let t0 = CFAbsoluteTimeGetCurrent()
                for _ in 0 ..< 100 { _ = PromptLookupTokenIterator.draft(after: missHistory, limit: 16) }
                let missUs = (CFAbsoluteTimeGetCurrent() - t0) * 10000

                let t1 = CFAbsoluteTimeGetCurrent()
                for _ in 0 ..< 100 { _ = PromptLookupTokenIterator.draft(after: hitHistory, limit: 16) }
                let hitUs = (CFAbsoluteTimeGetCurrent() - t1) * 10000

                print("[BENCH] size=\(size) miss=\(String(format: "%.1f", missUs))µs hit=\(String(format: "%.1f", hitUs))µs")
            }
        }
    }
}
