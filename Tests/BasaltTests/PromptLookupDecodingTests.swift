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
        switch family {
        case "llama": try TinyLlama.make()
        case "hybrid": try TinyHybrid.make()
        case "lfm": try TinyLFM2MoE.make()
        case "hybrid-moe": try TinyHybrid.makeMoE()
        case "nemotron": try TinyNemotronH.make()
        case "lfm-dense": try TinyLFM2.make()
        default: try TinyQwen3.make()
        }
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
        processor: LogitProcessor? = nil, sampler: LogitSampler = ArgMaxSampler(), recycling: Bool = false
    ) throws -> (tokens: [Int], iterator: PromptLookupTokenIterator, cache: [KVCache]) {
        let cache = try model.newCache(parameters: nil)
        var iterator = try PromptLookupTokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), context: prompt, model: model, cache: cache,
            processor: processor, sampler: sampler, prefill: .init(), maxTokens: maxTokens, recyclesTokens: recycling)
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        return (tokens, iterator, cache)
    }
}

extension TinyHybrid {
    static func makeMoE() throws -> Qwen35TextModel {
        let moe = config.replacingOccurrences(
            of: "\"tie_word_embeddings\": false",
            with: "\"tie_word_embeddings\": false, \"num_experts\": 4, \"num_experts_per_tok\": 2, \"moe_intermediate_size\": 32, \"shared_expert_intermediate_size\": 32")
        let model = try withRandomState(MLXRandom.RandomState(seed: 79)) {
            Qwen35TextModel(try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(moe.utf8)))
        }
        eval(model)
        return model
    }
}

enum TinyNemotronH {
    static let config = """
        {"model_type": "nemotron_h", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 4,
         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "mamba_num_heads": 4,
         "mamba_head_dim": 16, "ssm_state_size": 16, "conv_kernel": 4, "n_groups": 2, "intermediate_size": 128,
         "hybrid_override_pattern": "M*M-", "layer_norm_epsilon": 1e-5, "tie_word_embeddings": true}
        """

    static func make() throws -> NemotronHModel {
        let model = try withRandomState(MLXRandom.RandomState(seed: 11)) {
            NemotronHModel(try JSONDecoder().decode(NemotronHConfiguration.self, from: Data(config.utf8)))
        }
        eval(model)
        return model
    }
}

enum TinyLFM2 {
    static let config = """
        {"model_type": "lfm2", "vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 4,
         "num_attention_heads": 4, "num_key_value_heads": 2, "norm_eps": 1e-5, "conv_L_cache": 3,
         "block_dim": 64, "block_ff_dim": 128, "block_multiple_of": 16,
         "layer_types": ["conv", "full_attention", "conv", "full_attention"]}
        """

    static func make() throws -> LFM2Model {
        let model = try withRandomState(MLXRandom.RandomState(seed: 97)) {
            LFM2Model(try JSONDecoder().decode(LFM2Configuration.self, from: Data(config.utf8)))
        }
        eval(model)
        return model
    }
}

enum TinyLFM2MoE {
    static let config = """
        {"model_type": "lfm2_moe", "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
         "moe_intermediate_size": 32, "num_hidden_layers": 4, "num_experts": 4, "num_experts_per_tok": 2,
         "norm_topk_prob": true, "num_attention_heads": 4, "num_key_value_heads": 2,
         "max_position_embeddings": 4096, "use_expert_bias": true, "num_dense_layers": 1, "norm_eps": 1e-5,
         "conv_bias": false, "conv_L_cache": 3, "layer_types": ["conv", "full_attention", "conv", "full_attention"]}
        """

    static func make() throws -> LFM2MoEModel {
        let model = try withRandomState(MLXRandom.RandomState(seed: 83)) {
            LFM2MoEModel(try JSONDecoder().decode(LFM2MoEConfiguration.self, from: Data(config.utf8)))
        }
        eval(model)
        return model
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
          arguments: ["llama", "qwen3", "hybrid", "lfm", "nemotron", "lfm-dense"])
    func greedyEquivalence(family: String) throws {
        let model = try LookupFixture.model(family)
        let expected = try LookupFixture.baseline(model)
        let (tokens, iterator, _) = try LookupFixture.lookup(model)
        #expect(tokens == expected)
        let telemetry = try #require(iterator.speculativeDecodingTelemetry)
        #expect(telemetry.acceptedDraftTokenCount > 0)
        #expect(telemetry.emittedTokenCount == tokens.count)
        #expect(telemetry.targetModelCallCount < tokens.count)
        #expect(telemetry.targetVerifiedTokenCount
            == telemetry.draftTokenCount + telemetry.targetModelCallCount - iterator.replayRounds)
    }

    @Test("When the output copies a long span of the prompt, a verify round drafts more than 8 tokens on average")
    func draftsRunPastEightTokens() throws {
        let continuation = (0 ..< 64).map { LookupFixture.unit[(4 + $0) % LookupFixture.unit.count] }
        let run = try LookupFixture.lookup(try TinyLlama.make(), sampler: ScriptedSampler(continuation))
        #expect(run.tokens == continuation)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.draftTokenCount > 8 * telemetry.roundCount)
    }

    @Test("A hybrid's first draft is a 4-token probe, and only an accepted probe opens the full draft limit; a dense model drafts the full limit at once",
          arguments: [("qwen3", 1, 10), ("hybrid", 2, 9)])
    func hybridProbesBeforeLongDrafts(family: String, rounds: Int, drafted: Int) throws {
        let continuation = (0 ..< 12).map { LookupFixture.unit[(4 + $0) % LookupFixture.unit.count] }
        let run = try LookupFixture.lookup(
            try LookupFixture.model(family), maxTokens: continuation.count, sampler: ScriptedSampler(continuation))
        #expect(run.tokens == continuation)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.roundCount == rounds)
        #expect(telemetry.draftTokenCount == drafted)
        #expect(telemetry.acceptedDraftTokenCount == drafted)
    }

    @Test("Rejected drafts are rolled back: the cache holds the prompt and every emitted token but the last")
    func rollback() throws {
        let model = try TinyQwen3.make()
        let (tokens, iterator, cache) = try LookupFixture.lookup(model, maxTokens: 48)
        #expect(tokens.count == 48)
        #expect(try #require(iterator.speculativeDecodingTelemetry).rejectedDraftTokenCount > 0)
        #expect(cache.allSatisfy { $0.offset == LookupFixture.prompt.count + tokens.count - 1 })
    }

    static func rejectingRun(_ family: String, maxTokens: Int) throws -> (model: any LanguageModel, run: (tokens: [Int], iterator: PromptLookupTokenIterator, cache: [KVCache]), reference: [KVCache]) {
        let model = try LookupFixture.model(family)
        let unit = LookupFixture.unit
        let script = (0 ..< 6).flatMap { round in Array(unit[4 ..< 8]) + [240 + round] + Array(unit[0 ..< 4]) }
        let run = try LookupFixture.lookup(model, maxTokens: maxTokens, sampler: ScriptedSampler(script))
        #expect(run.tokens == Array(script.prefix(maxTokens)))
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
          arguments: ["hybrid", "lfm", "lfm-dense"], [54, 7, 13])
    func hybridRejectionRestoresRecurrentState(family: String, maxTokens: Int) throws {
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
        let (model, run, reference) = try Self.rejectingRun("nemotron", maxTokens: 54)
        let last = MLXArray([Int32(try #require(run.tokens.last))])[.newAxis]
        let replayed = model(last, cache: run.cache)[0, -1].asType(.float32)
        let prefilled = model(last, cache: reference)[0, -1].asType(.float32)
        #expect(abs(replayed - prefilled).max().item(Float.self) < 0.1)
    }

    @Test("Without drafts, the last token of the budget is not fed to the model: no look-ahead forward is thrown away",
          arguments: ["qwen3", "hybrid"], [1, 2, 17])
    func noLookAheadAtTheBudget(family: String, maxTokens: Int) throws {
        let prompt = [200, 201, 202, 203, 204]
        let run = try LookupFixture.lookup(
            try LookupFixture.model(family), prompt: prompt, maxTokens: maxTokens,
            sampler: ScriptedSampler((0 ..< maxTokens).map { 10 + $0 }))
        #expect(run.tokens == (0 ..< maxTokens).map { 10 + $0 })
        #expect(run.iterator.speculativeDecodingTelemetry == nil)
        #expect(run.cache.filter { $0 is KVCacheSimple }.allSatisfy { $0.offset == prompt.count + maxTokens - 1 })
    }

    @Test("A stateful logit processor and a seeded sampler see the same calls as in plain decoding",
          arguments: [("llama", "repetition"), ("llama", "entropy"), ("hybrid", "repetition")])
    func processorAndSamplerParity(family: String, shaping: String) throws {
        let model = try LookupFixture.model(family)
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
          arguments: ["llama", "hybrid"], ["none", "repetition", "entropy"])
    func seededSamplerParity(family: String, shaping: String) throws {
        let model = try LookupFixture.model(family)
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

    @Test("Lookup admits attention-only caches and dense recurrent hybrids; hybrids that route tokens to experts and sliding-window caches stay on plain decoding",
          arguments: [("qwen3", true), ("hybrid", true), ("nemotron", true), ("lfm-dense", true), ("lfm", false), ("hybrid-moe", false)])
    func guardAdmitsDenseHybridsOnly(family: String, admitted: Bool) throws {
        let model = try LookupFixture.model(family)
        let routesExperts = PromptLookupTokenIterator.routesExperts(model)
        #expect(routesExperts == ["lfm", "hybrid-moe"].contains(family))
        #expect(PromptLookupTokenIterator.supports(try model.newCache(parameters: nil), routesExperts: routesExperts) == admitted)
        let slidingWindow: [KVCache] = [RotatingKVCache(maxSize: 64), KVCacheSimple()]
        #expect(!PromptLookupTokenIterator.supports(slidingWindow, routesExperts: false))
        #expect(!PromptLookupTokenIterator.supports([], routesExperts: false))
    }

    @Test("Through the engine a dense hybrid decodes the plain text with drafts, with and without a prompt cache; an expert-routing hybrid decodes without drafts",
          arguments: ["hybrid", "lfm"])
    func engineHybridEquivalence(family: String) async throws {
        func engine(lookup: Bool) async throws -> MLXInferenceEngine {
            let engine = MLXInferenceEngine(promptCacheBytes: 1 << 26, speculativeLookup: lookup)
            await engine.attach(ModelContainer(context: ModelContext(
                configuration: ModelConfiguration(id: "tiny-\(family)"), model: try LookupFixture.model(family),
                processor: ByteInputProcessor(), tokenizer: ByteTokenizer())))
            return engine
        }
        let (plain, speculative) = (try await engine(lookup: false), try await engine(lookup: true))
        let request = GenerationRequest(prompt: LookupFixture.code, maxTokens: 48, temperature: 0)
        for key in [nil, "hybrid", "hybrid"] {
            let expected = try await plain.generate(request: request.withCacheKey(key))
            let actual = try await speculative.generate(request: request.withCacheKey(key))
            #expect(actual.generatedText == expected.generatedText)
            #expect((actual.draftTokens != nil) == (family == "hybrid"))
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
        backoffs.recordRound(recycled: false, acceptedDrafts: 2, replayed: true)
        #expect(backoffs.lookup.allowsDraft)
    }

    @Test("On a hybrid a round that accepts one draft and then replays does not pay for its extra forward, so lookup backs off; a dense model keeps drafting")
    func hybridBacksOffWhenTheReplayEatsTheGain() throws {
        let cycles = (0 ..< 20).flatMap { [1, 2, 3, 4, 5, 20 + $0] }
        func rounds(_ family: String) throws -> (rounds: Int, accepted: Int) {
            let run = try LookupFixture.lookup(
                try LookupFixture.model(family), prompt: [200, 201, 202, 203, 204], maxTokens: cycles.count,
                sampler: ScriptedSampler(cycles))
            #expect(run.tokens == cycles)
            let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
            return (telemetry.roundCount, telemetry.acceptedDraftTokenCount)
        }
        let (dense, hybrid) = (try rounds("qwen3"), try rounds("hybrid"))
        #expect(dense.accepted == dense.rounds)
        #expect(hybrid.rounds * 2 <= dense.rounds)
    }

    @Test("When every draft past its first token is rejected, lookup backs off instead of verifying each match, and the output is unchanged",
          arguments: ["qwen3", "hybrid"])
    func rejectedDraftsBackOff(family: String) throws {
        let cycles = (0 ..< 26).flatMap { [1, 2, 3, 4, 10 + $0] }
        let run = try LookupFixture.lookup(
            try LookupFixture.model(family), prompt: [200, 201, 202, 203, 204], maxTokens: cycles.count,
            sampler: ScriptedSampler(cycles))
        #expect(run.tokens == cycles)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.acceptedDraftTokenCount == 0)
        #expect(telemetry.roundCount <= 6)
    }
}
