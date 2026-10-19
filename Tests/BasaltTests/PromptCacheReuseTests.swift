import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
@testable import BasaltCore
@testable import BasaltInference

struct ByteTokenizer: MLXLMCommon.Tokenizer {
    static let endOfUserTurn = 255
    static let generatedTokenBase = 0x4E00

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.unicodeScalars.map { scalar in
            let value = Int(scalar.value)
            return value >= Self.generatedTokenBase ? value - Self.generatedTokenBase : value % Self.endOfUserTurn
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(String.UnicodeScalarView(tokenIds.compactMap { UnicodeScalar(Self.generatedTokenBase + $0) }))
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { decode(tokenIds: [id], skipSpecialTokens: false) }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { encode(text: $0["content"] as? String ?? "", addSpecialTokens: false) } + [Self.endOfUserTurn]
    }
}

struct ByteInputProcessor: UserInputProcessor {
    let tokenizer = ByteTokenizer()

    func prepare(input: UserInput) throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }
}

enum TinyModel {
    static let config = """
        {"hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
         "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
         "vocab_size": 256, "rope_theta": 10000.0, "tie_word_embeddings": false}
        """

    static let weightScale: Float = 4

    static func make() throws -> LlamaModel {
        MLXRandom.seed(56)
        let model = LlamaModel(try JSONDecoder().decode(LlamaConfiguration.self, from: Data(config.utf8)))
        model.update(parameters: model.parameters().mapValues { $0 * weightScale })
        eval(model)
        return model
    }

    static func container() throws -> ModelContainer {
        ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-llama"),
            model: try make(),
            processor: ByteInputProcessor(),
            tokenizer: ByteTokenizer()))
    }

    static let cacheBytesPerToken = 2 * 2 * 16 * 4 * 2

    static func engine(promptCacheBytes: Int? = nil, repetitionPenalty: Float? = nil) async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(promptCacheBytes: promptCacheBytes, repetitionPenalty: repetitionPenalty)
        await engine.attach(try container())
        return engine
    }

    static func tokens(_ prompt: String) -> [Int] {
        ByteTokenizer().encode(text: prompt, addSpecialTokens: false) + [ByteTokenizer.endOfUserTurn]
    }
}

enum Conversation {
    static let firstTurn = "[System]: You are a coding agent. Tools: run_command.\n\n[User]: list the files"
    static let afterToolCall = firstTurn
        + "\n\n[Assistant called run_command]: {\"cmd\":\"ls\"}\n\n[Tool result]: Package.swift README.md"
    static let otherBranch = String(firstTurn.prefix(40)) + " and then something else entirely"
}

func greedy(_ prompt: String, key: String?, maxTokens: Int = 12) -> GenerationRequest {
    GenerationRequest(prompt: prompt, maxTokens: maxTokens, temperature: 0, cacheKey: key)
}

func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
    zip(a, b).prefix { $0 == $1 }.count
}

@Suite("Prefix KV-cache reuse on a tiny random model (#56)", .serialized)
struct PromptCacheReuseTests {

    @Test("Logits after trimming a cache to a shared prefix and prefilling the tail equal a full prefill")
    func trimmedCacheLogitsMatchFullPrefill() throws {
        let model = try TinyModel.make()
        let first = TinyModel.tokens(Conversation.firstTurn)
        let second = TinyModel.tokens(Conversation.afterToolCall)
        let shared = commonPrefix(first, second)
        #expect(shared == first.count - 1)

        let full = model(MLXArray(second.map(Int32.init)).reshaped(1, -1), cache: try model.newCache(parameters: nil))

        let tailTokens = MLXArray(second[shared...].map(Int32.init)).reshaped(1, -1)
        let expected = full[0..., shared..., 0...]

        let stale = PromptCache(try model.newCache(parameters: nil))
        _ = model(MLXArray((first + [7, 8, 9]).map(Int32.init)).reshaped(1, -1), cache: stale.layers)
        let untrimmed = model(tailTokens, cache: stale.layers)
        #expect(!allClose(untrimmed, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))

        let cache = PromptCache(try model.newCache(parameters: nil))
        _ = model(MLXArray((first + [7, 8, 9]).map(Int32.init)).reshaped(1, -1), cache: cache.layers)
        #expect(cache.tokenCount == first.count + 3)
        #expect(cache.rewind(to: shared))
        let tail = model(tailTokens, cache: cache.layers)

        #expect(tail.shape == expected.shape)
        #expect(allClose(tail, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        #expect(cache.tokenCount == second.count)
    }

    @Test("A second turn reuses the first turn's prefix, prefills only the delta, and answers like a full prefill")
    func secondTurnPrefillsOnlyTheDelta() async throws {
        let engine = try await TinyModel.engine()
        let first = try await engine.generate(request: greedy(Conversation.firstTurn, key: "agent"))
        #expect(first.cachedPromptTokens == 0)
        #expect(first.promptTokens == TinyModel.tokens(Conversation.firstTurn).count)

        let reused = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "agent"))
        let full = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))

        let promptTokens = TinyModel.tokens(Conversation.afterToolCall)
        let shared = commonPrefix(TinyModel.tokens(Conversation.firstTurn), promptTokens)
        #expect(reused.promptTokens == promptTokens.count)
        #expect(reused.cachedPromptTokens == shared)
        #expect(full.cachedPromptTokens == 0)
        #expect(full.promptTokens == promptTokens.count)
        #expect(!reused.generatedText.isEmpty)
        #expect(reused.generatedText != first.generatedText)
        #expect(Set(reused.generatedText).count > 1)
        #expect(reused.generatedText == full.generatedText)

        let slot = try #require(await engine.promptCaches.slots["agent"])
        #expect(slot.tokens == promptTokens)
        #expect(slot.cache.tokenCount >= promptTokens.count)
    }

    @Test("A prompt that diverges inside the cached prefix trims to the divergence and still matches a full prefill")
    func divergingPrefixTrimsToTheCommonPart() async throws {
        let engine = try await TinyModel.engine()
        _ = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "agent"))
        let branched = try await engine.generate(request: greedy(Conversation.otherBranch, key: "agent"))
        let full = try await engine.generate(request: greedy(Conversation.otherBranch, key: nil))

        let shared = commonPrefix(TinyModel.tokens(Conversation.afterToolCall), TinyModel.tokens(Conversation.otherBranch))
        #expect(shared >= 40)
        #expect(branched.cachedPromptTokens == shared)
        #expect(branched.generatedText == full.generatedText)
        #expect(await engine.promptCaches.slots["agent"]?.tokens == TinyModel.tokens(Conversation.otherBranch))
    }

    @Test("Repeating the same prompt reuses all but its last token, which is prefilled to get logits")
    func identicalPromptKeepsOneTokenToPrefill() async throws {
        let engine = try await TinyModel.engine()
        let first = try await engine.generate(request: greedy(Conversation.firstTurn, key: "agent"))
        let again = try await engine.generate(request: greedy(Conversation.firstTurn, key: "agent"))
        #expect(again.cachedPromptTokens == TinyModel.tokens(Conversation.firstTurn).count - 1)
        #expect(again.generatedText == first.generatedText)
    }

    @Test("Different keys keep separate slots; a request without a key neither reads nor writes one")
    func slotsAreKeyed() async throws {
        let engine = try await TinyModel.engine()
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "main"))
        let other = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "subagent"))
        #expect(other.cachedPromptTokens == 0)
        _ = try await engine.generate(request: greedy(Conversation.otherBranch, key: nil))
        let slots = await engine.promptCaches.slots
        #expect(Set(slots.keys) == ["main", "subagent"])
        #expect(slots["main"]?.tokens == TinyModel.tokens(Conversation.firstTurn))
    }

    @Test("A zero budget or a quantized KV cache disables reuse")
    func reuseCanBeDisabled() async throws {
        let disabled = try await TinyModel.engine(promptCacheBytes: 0)
        _ = try await disabled.generate(request: greedy(Conversation.firstTurn, key: "agent"))
        let second = try await disabled.generate(request: greedy(Conversation.afterToolCall, key: "agent"))
        #expect(second.cachedPromptTokens == 0)
        #expect(await disabled.promptCaches.slots.isEmpty)

        let quantized = MLXInferenceEngine(kvBits: 8)
        await quantized.attach(try TinyModel.container())
        _ = try await quantized.generate(request: greedy(Conversation.firstTurn, key: "agent"))
        #expect(await quantized.promptCaches.slots.isEmpty)
    }

    @Test("A cancelled generation stops promptly and leaves a slot whose tokens match its cache; the next turn still matches a full prefill")
    func cancelledGenerationLeavesConsistentSlot() async throws {
        let longRun = 3000
        let engine = try await TinyModel.engine()
        let clock = ContinuousClock()
        let fullRunStart = clock.now
        _ = try await engine.generate(request: greedy(Conversation.otherBranch, key: "baseline", maxTokens: longRun))
        let fullRun = clock.now - fullRunStart

        let stream = try await engine.generateStream(request: greedy(Conversation.firstTurn, key: "agent", maxTokens: longRun))
        let consumer = Task {
            var chunks = 0
            for await _ in stream {
                chunks += 1
                try? await Task.sleep(for: .milliseconds(2))
            }
            return chunks
        }
        try await Task.sleep(for: .milliseconds(50))
        let cancelledAt = clock.now
        consumer.cancel()
        _ = await consumer.value

        while await engine.promptCaches.slots["agent"] == nil, clock.now - cancelledAt < fullRun * 3 {
            try await Task.sleep(for: .milliseconds(5))
        }
        let stoppedAfter = clock.now - cancelledAt
        #expect(stoppedAfter < fullRun / 4, "cancelled run took \(stoppedAfter), a full run \(fullRun)")

        let slot = try #require(await engine.promptCaches.slots["agent"])
        #expect(slot.tokens == TinyModel.tokens(Conversation.firstTurn))
        #expect(slot.cache.tokenCount >= slot.tokens.count)

        let reused = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "agent"))
        let full = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))
        #expect(reused.cachedPromptTokens > 0)
        #expect(reused.generatedText == full.generatedText)
    }

    @Test("Settling a cancelled launch stops MLX's generation instead of waiting for it to reach maxTokens")
    func settlingACancelledLaunchStopsGeneration() async throws {
        let longRun = 3000
        let container = try TinyModel.container()
        let launch = try await container.perform { context in
            let input = LMInput(tokens: MLXArray(TinyModel.tokens(Conversation.firstTurn).map(Int32.init)))
            return try MLXInferenceEngine.launch(
                input: input, lease: nil, keepsCache: true,
                params: GenerateParameters(maxTokens: longRun, temperature: 0),
                components: GenerationComponents(), tools: nil, context: context)
        }
        var iterator = launch.stream.makeAsyncIterator()
        _ = await iterator.next()

        await MLXInferenceEngine.settle(launch, cancelled: true)

        let cache = try #require(launch.cache)
        let promptCount = TinyModel.tokens(Conversation.firstTurn).count
        #expect(cache.tokenCount >= promptCount)
        #expect(cache.tokenCount < promptCount + longRun / 2, "generated \(cache.tokenCount - promptCount) tokens after cancel")
        withExtendedLifetime(iterator) {}
    }

    @Test("With a repetition penalty, a reused cache with a short tail still answers like a full prefill")
    func repetitionPenaltySeesTheWholePrompt() async throws {
        let favoured = try await TinyModel.engine().generate(request: greedy(Conversation.firstTurn, key: nil))
        let cachedTurn = Conversation.firstTurn + favoured.generatedText
        let shortDelta = cachedTurn + " ok"
        let engine = try await TinyModel.engine(repetitionPenalty: 3)
        _ = try await engine.generate(request: greedy(cachedTurn, key: "agent", maxTokens: 24))
        let reused = try await engine.generate(request: greedy(shortDelta, key: "agent", maxTokens: 24))
        let full = try await engine.generate(request: greedy(shortDelta, key: nil, maxTokens: 24))
        #expect(TinyModel.tokens(shortDelta).count - reused.cachedPromptTokens < 20)
        #expect(reused.cachedPromptTokens > 0)
        #expect(reused.generatedText == full.generatedText)
    }

    @Test("A slot parked to 1/8 of the budget is leased with its kept head and still answers like a full prefill")
    func parkedSlotIsReusedCorrectly() async throws {
        let budget = 120 * TinyModel.cacheBytesPerToken
        let engine = try await TinyModel.engine(promptCacheBytes: budget)
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "main"))
        _ = try await engine.generate(request: greedy(Conversation.otherBranch, key: "subagent"))

        let parked = try #require(await engine.promptCaches.slots["main"])
        #expect(parked.tokens.count > 0)
        #expect(parked.tokens.count < TinyModel.tokens(Conversation.firstTurn).count)
        #expect(parked.cache.byteCount <= budget / 8)

        let reused = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "main"))
        let full = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))
        #expect(reused.cachedPromptTokens == parked.tokens.count)
        #expect(reused.generatedText == full.generatedText)
    }

    @Test("A stored slot is trimmed to its ledger and compacted, so the budget counts real bytes")
    func storedSlotIsCompact() async throws {
        let engine = try await TinyModel.engine()
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "agent", maxTokens: 40))
        let slot = try #require(await engine.promptCaches.slots["agent"])
        #expect(slot.cache.tokenCount == slot.tokens.count)
        #expect(slot.cache.byteCount == slot.tokens.count * TinyModel.cacheBytesPerToken)
    }

    @Test("The automatic budget is half of what the GPU's 3/4 share of RAM leaves after the weights")
    func automaticBudget() {
        let gib = 1 << 30
        #expect(PromptCacheBudget.automatic(physicalMemory: 32 * gib, weights: 5 * gib) == 19 * gib / 2)
        #expect(PromptCacheBudget.automatic(physicalMemory: 24 * gib, weights: 16 * gib) == gib)
        #expect(PromptCacheBudget.automatic(physicalMemory: 16 * gib, weights: 14 * gib) == 0)
    }

    @Test("Concurrent generations on one key never share a cache and both answer like a full prefill")
    func concurrentGenerationsOnOneKey() async throws {
        let engine = try await TinyModel.engine()
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "agent"))
        async let a = engine.generate(request: greedy(Conversation.afterToolCall, key: "agent"))
        async let b = engine.generate(request: greedy(Conversation.afterToolCall, key: "agent"))
        let (ra, rb) = try await (a, b)
        let full = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))
        #expect(ra.generatedText == full.generatedText)
        #expect(rb.generatedText == full.generatedText)
        #expect([ra.cachedPromptTokens, rb.cachedPromptTokens].filter { $0 > 0 }.count == 1)
    }

    @Test("Loading another model drops every slot")
    func attachingAModelClearsSlots() async throws {
        let engine = try await TinyModel.engine()
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "agent"))
        #expect(await engine.promptCaches.slots.count == 1)
        await engine.attach(try TinyModel.container())
        #expect(await engine.promptCaches.slots.isEmpty)
    }
}

@Suite("Prompt cache slots: leasing, budget and parking (#56)")
struct PromptCacheSlotsTests {

    static let bytesPerToken = 2 * 2 * 8 * 4

    func filledCache(tokens: Int, layers: Int = 1) -> PromptCache {
        let cache = PromptCache((0..<layers).map { _ in KVCacheSimple() })
        for layer in cache.layers {
            _ = layer.update(keys: MLXArray.ones([1, 2, tokens, 8]), values: MLXArray.ones([1, 2, tokens, 8]))
        }
        eval(cache.layers.flatMap { $0.innerState() })
        return cache
    }

    @Test("The reusable prefix stops at the first difference and always leaves one token to prefill")
    func reusablePrefixLength() {
        #expect(PromptCacheSlots.reusablePrefixLength(cached: [1, 2, 3], prompt: [1, 2, 3, 4]) == 3)
        #expect(PromptCacheSlots.reusablePrefixLength(cached: [1, 2, 3], prompt: [1, 2, 9, 4]) == 2)
        #expect(PromptCacheSlots.reusablePrefixLength(cached: [1, 2, 3], prompt: [1, 2, 3]) == 2)
        #expect(PromptCacheSlots.reusablePrefixLength(cached: [5], prompt: [1, 2]) == 0)
        #expect(PromptCacheSlots.reusablePrefixLength(cached: [], prompt: []) == 0)
    }

    @Test("A lease takes the slot out, so a second generation on the same key gets none")
    func leaseIsExclusive() {
        var slots = PromptCacheSlots(byteBudget: 1 << 30)
        slots.store("k", tokens: Array(0..<10), cache: filledCache(tokens: 12))
        let lease = slots.lease("k", for: Array(0..<10) + [42])
        #expect(lease?.reusedTokens == 10)
        #expect(lease?.cache.tokenCount == 10)
        #expect(slots.lease("k", for: Array(0..<10)) == nil)
        #expect(slots.slots.isEmpty)
    }

    @Test("A slot with nothing in common is dropped instead of leased")
    func noCommonPrefixDropsTheSlot() {
        var slots = PromptCacheSlots(byteBudget: 1 << 30)
        slots.store("k", tokens: [1, 2, 3], cache: filledCache(tokens: 3))
        #expect(slots.lease("k", for: [9, 9, 9]) == nil)
        #expect(slots.slots.isEmpty)
    }

    @Test("Non-trimmable caches (hybrid/SSM layers) and caches shorter than their ledger are not stored")
    func onlyRewindableConsistentCachesAreStored() {
        var slots = PromptCacheSlots(byteBudget: 1 << 30)
        slots.store("ssm", tokens: [1, 2], cache: PromptCache([MambaCache()]))
        slots.store("short", tokens: Array(0..<10), cache: filledCache(tokens: 4))
        #expect(slots.slots.isEmpty)
    }

    @Test("Over budget, the least recently used inactive slot is parked at 1/8 of the budget, keeping its head")
    func parkingTrimsTheOldestSlot() throws {
        let budget = 300 * Self.bytesPerToken
        var slots = PromptCacheSlots(byteBudget: budget)
        slots.store("old", tokens: Array(0..<200), cache: filledCache(tokens: 200))
        slots.store("active", tokens: Array(0..<200), cache: filledCache(tokens: 200))
        let parked = try #require(slots.slots["old"])
        #expect(parked.tokens.count > 0)
        #expect(parked.tokens == Array(0..<parked.tokens.count))
        #expect(parked.cache.tokenCount == parked.tokens.count)
        #expect(parked.cache.byteCount <= slots.parkedByteLimit)
        #expect(slots.slots["active"]?.tokens.count == 200)
        #expect(slots.byteCount <= budget)
    }

    @Test("When parking is not enough the oldest inactive slots are evicted first")
    func evictionKeepsTheBudget() {
        let budget = 300 * Self.bytesPerToken
        var slots = PromptCacheSlots(byteBudget: budget)
        for index in 0..<25 {
            slots.store("s\(index)", tokens: Array(0..<100), cache: filledCache(tokens: 100))
            #expect(slots.byteCount <= budget)
        }
        #expect(slots.slots["s0"] == nil)
        #expect(slots.slots["s24"]?.tokens.count == 100)
        #expect((slots.slots["s23"]?.tokens.count ?? 100) < 100)
    }

    @Test("A cache larger than the whole budget is not stored and evicts nothing")
    func oversizedSlotIsRejected() {
        let budget = 300 * Self.bytesPerToken
        var slots = PromptCacheSlots(byteBudget: budget)
        slots.store("small", tokens: Array(0..<100), cache: filledCache(tokens: 100))
        slots.store("huge", tokens: Array(0..<600), cache: filledCache(tokens: 600))
        #expect(slots.slots["huge"] == nil)
        #expect(slots.slots["small"]?.tokens.count == 100)
    }

    @Test("A zero budget stores nothing")
    func zeroBudgetStoresNothing() {
        var slots = PromptCacheSlots(byteBudget: 0)
        slots.store("k", tokens: [1], cache: filledCache(tokens: 1))
        #expect(slots.slots.isEmpty)
    }
}
