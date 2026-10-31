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

enum AttachedMedia: Sendable, CaseIterable {
    case image, video, audio
}

struct VisionStyleInputProcessor: UserInputProcessor {
    let padsFirstToken: Bool
    var media: AttachedMedia? = nil

    func prepare(input: UserInput) throws -> LMInput {
        let tokens = try ByteInputProcessor().prepare(input: input).text.tokens
        let mask = (0..<tokens.dim(-1)).map { Int8(padsFirstToken && $0 == 0 ? 0 : 1) }
        let pixels = MLXArray.zeros([1, 3, 2, 2])
        return LMInput(
            text: .init(tokens: tokens, mask: MLXArray(mask)),
            image: media == .image ? .init(pixels: pixels) : nil,
            video: media == .video ? .init(pixels: pixels) : nil,
            audio: media == .audio ? .init(features: pixels) : nil)
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
        let model = try withRandomState(MLXRandom.RandomState(seed: 56)) { LlamaModel(try JSONDecoder().decode(LlamaConfiguration.self, from: Data(config.utf8))) }
        model.update(parameters: model.parameters().mapValues { $0 * weightScale })
        eval(model)
        return model
    }

    static func container(processor: any UserInputProcessor = ByteInputProcessor()) throws -> ModelContainer {
        ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-llama"),
            model: try make(),
            processor: processor,
            tokenizer: ByteTokenizer()))
    }

    static let cacheBytesPerToken = 2 * 2 * 16 * 4 * 2

    static func engine(
        promptCacheBytes: Int? = nil, repetitionPenalty: Float? = nil,
        processor: any UserInputProcessor = ByteInputProcessor()
    ) async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(promptCacheBytes: promptCacheBytes, repetitionPenalty: repetitionPenalty)
        await engine.attach(try container(processor: processor))
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
    static let otherUser = "[System]: You are a coding agent. Tools: run_command.\n\n[User]: explain the build"
}

func greedy(_ prompt: String, key: String?, maxTokens: Int = 12) -> GenerationRequest {
    GenerationRequest(prompt: prompt, maxTokens: maxTokens, temperature: 0, cacheKey: key)
}

func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
    zip(a, b).prefix { $0 == $1 }.count
}

extension PromptCacheTree {
    mutating func required(_ namespace: String, for prompt: [Int]) throws -> PromptCacheLease {
        var opt = lease(namespace, for: prompt)
        guard let l = opt.take() else {
            struct MissingLease: Error {}
            throw MissingLease()
        }
        return l
    }

    mutating func touch(_ namespace: String, _ prompt: [Int]) throws {
        _ = try required(namespace, for: prompt)
    }
}

func pinCount(_ tree: PromptCacheTree) -> Int {
    var total = 0
    var pending = Array(tree.roots.values)
    while let node = pending.popLast() {
        total += node.pins
        pending.append(contentsOf: node.children.values)
    }
    return total
}

@Suite("Prefix KV-cache tree on a tiny random model (#32, #56)", .serialized)
struct PromptCacheReuseTests {

    @Test("Logits after leasing another conversation's shared prefix and prefilling the tail equal a full prefill")
    func leasedPrefixLogitsMatchFullPrefill() throws {
        let model = try TinyModel.make()
        let first = TinyModel.tokens(Conversation.firstTurn)
        let second = TinyModel.tokens(Conversation.otherUser)
        let shared = commonPrefix(first, second)
        #expect(shared > 40)
        #expect(shared < first.count - 1)

        let full = model(MLXArray(second.map(Int32.init)).reshaped(1, -1), cache: try model.newCache(parameters: nil))
        let expected = full[0..., shared..., 0...]
        let tailTokens = MLXArray(second[shared...].map(Int32.init)).reshaped(1, -1)

        let unprimed = model(tailTokens, cache: try model.newCache(parameters: nil))
        #expect(!allClose(unprimed, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))

        let firstCache = PromptCache(try model.newCache(parameters: nil))
        _ = model(MLXArray((first + [7, 8, 9]).map(Int32.init)).reshaped(1, -1), cache: firstCache.layers)
        var tree = PromptCacheTree(byteBudget: 1 << 30)
        tree.store("ns", tokens: first, cache: firstCache)

        let lease = try tree.required("ns", for: second)
        #expect(lease.reusedTokens == shared)
        #expect(lease.cache.tokenCount == shared)
        let tail = model(tailTokens, cache: lease.cache.layers)
        #expect(tail.shape == expected.shape)
        #expect(allClose(tail, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }

    @Test("A second conversation sharing the system prompt reuses the shared tokens and answers like a full prefill")
    func secondConversationReusesTheSharedPrefix() async throws {
        let engine = try await TinyModel.engine()
        let first = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        #expect(first.cachedPromptTokens == 0)
        #expect(first.promptTokens == TinyModel.tokens(Conversation.firstTurn).count)

        let reused = try await engine.generate(request: greedy(Conversation.otherUser, key: "ns"))
        let full = try await engine.generate(request: greedy(Conversation.otherUser, key: nil))

        let promptTokens = TinyModel.tokens(Conversation.otherUser)
        let shared = commonPrefix(TinyModel.tokens(Conversation.firstTurn), promptTokens)
        #expect(reused.cachedPromptTokens == shared)
        #expect(reused.promptTokens == promptTokens.count)
        #expect(full.cachedPromptTokens == 0)
        #expect(!reused.generatedText.isEmpty)
        #expect(Set(reused.generatedText).count > 1)
        #expect(reused.generatedText == full.generatedText)
    }

    @Test("A text-only prompt a VLM processor rendered with an all-ones mask reuses the prefix and answers like a full prefill (#103)")
    func allOnesMaskReusesThePrefix() async throws {
        let engine = try await TinyModel.engine(processor: VisionStyleInputProcessor(padsFirstToken: false))
        let first = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        #expect(first.cachedPromptTokens == 0)

        let reused = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        let full = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))
        let shared = commonPrefix(TinyModel.tokens(Conversation.firstTurn), TinyModel.tokens(Conversation.afterToolCall))
        #expect(shared > 40)
        #expect(reused.cachedPromptTokens == shared)
        #expect(Set(reused.generatedText).count > 1)
        #expect(reused.generatedText == full.generatedText)
    }

    @Test("A mask that pads out a token still bypasses the prompt cache")
    func paddingMaskBypassesTheCache() async throws {
        let engine = try await TinyModel.engine(processor: VisionStyleInputProcessor(padsFirstToken: true))
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        let second = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        #expect(second.cachedPromptTokens == 0)
        #expect(await engine.promptCaches.byteCount == 0)
    }

    @Test("An all-ones mask next to an image, video or audio still bypasses the prompt cache", arguments: AttachedMedia.allCases)
    func attachedMediaBypassesTheCache(media: AttachedMedia) async throws {
        let engine = try await TinyModel.engine(processor: VisionStyleInputProcessor(padsFirstToken: false, media: media))
        let first = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        let second = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        #expect(first.cachedPromptTokens == 0)
        #expect(second.cachedPromptTokens == 0)
        #expect(await engine.promptCaches.byteCount == 0)
    }

    @Test("An assistant prefix on an all-ones-mask VLM prompt is appended and answered like on a plain text processor")
    func assistantPrefixWithAllOnesMask() async throws {
        let request = GenerationRequest(
            prompt: Conversation.firstTurn, maxTokens: 12, temperature: 0, cacheKey: "ns", assistantPrefix: "{\"")
        let vision = try await TinyModel.engine(processor: VisionStyleInputProcessor(padsFirstToken: false))
        let plain = try await TinyModel.engine()
        let prefixed = try await vision.generate(request: request)
        let expected = try await plain.generate(request: request)
        #expect(prefixed.generatedText.hasPrefix("{\""))
        #expect(prefixed.generatedText.count > 2)
        #expect(prefixed.promptTokens == TinyModel.tokens(Conversation.firstTurn).count + 2)
        #expect(prefixed.generatedText == expected.generatedText)
    }

    @Test("Branches A = P+x and B = P+y share P, and A+z then reuses the longest stored prefix, all like a full prefill")
    func branchesReuseTheirLongestPrefix() async throws {
        let engine = try await TinyModel.engine()
        let a = TinyModel.tokens(Conversation.firstTurn)
        let b = TinyModel.tokens(Conversation.otherUser)
        let az = TinyModel.tokens(Conversation.afterToolCall)
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        let branchB = try await engine.generate(request: greedy(Conversation.otherUser, key: "ns"))
        let extendA = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        let fullB = try await engine.generate(request: greedy(Conversation.otherUser, key: nil))
        let fullAZ = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))

        #expect(branchB.cachedPromptTokens == commonPrefix(a, b))
        #expect(extendA.cachedPromptTokens == commonPrefix(a, az))
        #expect(extendA.cachedPromptTokens > branchB.cachedPromptTokens)
        #expect(branchB.generatedText == fullB.generatedText)
        #expect(extendA.generatedText == fullAZ.generatedText)

        let tree = await engine.promptCaches
        let root = try #require(tree.roots["ns"])
        #expect(root.children.count == 1)
        let shared = try #require(root.children.values.first)
        #expect(shared.tokens == Array(a.prefix(commonPrefix(a, b))))
        #expect(shared.children.count == 2)
        for prompt in [a, b, az] {
            #expect(tree.storedPrefix(of: prompt, in: "ns") == prompt.count)
        }
        #expect(tree.byteCount == (az.count + 1 + b.count - commonPrefix(a, b)) * TinyModel.cacheBytesPerToken)
        #expect(pinCount(tree) == 0)
    }

    @Test("Repeating the same prompt reuses all but its last token, which is prefilled to get logits")
    func identicalPromptKeepsOneTokenToPrefill() async throws {
        let engine = try await TinyModel.engine()
        let first = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        let again = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        #expect(again.cachedPromptTokens == TinyModel.tokens(Conversation.firstTurn).count - 1)
        #expect(again.generatedText == first.generatedText)
    }

    @Test("Namespaces never share a prefix; a request without a namespace neither reads nor writes the tree")
    func namespacesAreIsolated() async throws {
        let engine = try await TinyModel.engine()
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "alice"))
        let bob = try await engine.generate(request: greedy(Conversation.firstTurn, key: "bob"))
        #expect(bob.cachedPromptTokens == 0)
        let anonymous = try await engine.generate(request: greedy(Conversation.firstTurn, key: nil))
        #expect(anonymous.cachedPromptTokens == 0)
        let tree = await engine.promptCaches
        #expect(Set(tree.roots.keys) == ["alice", "bob"])
        #expect(tree.byteCount == 2 * TinyModel.tokens(Conversation.firstTurn).count * TinyModel.cacheBytesPerToken)
    }

    @Test("A zero budget or a quantized KV cache disables reuse")
    func reuseCanBeDisabled() async throws {
        let disabled = try await TinyModel.engine(promptCacheBytes: 0)
        _ = try await disabled.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        let second = try await disabled.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        #expect(second.cachedPromptTokens == 0)
        #expect(await disabled.promptCaches.roots.isEmpty)

        let quantized = MLXInferenceEngine(kvPrecision: .q8)
        await quantized.attach(try TinyModel.container())
        _ = try await quantized.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        #expect(await quantized.promptCaches.roots.isEmpty)
    }

    @Test("A request the orchestrator quantized runs at its bits and bypasses the cache; fp16 requests still reuse it")
    func perRequestKVBits() async throws {
        let engine = try await TinyModel.engine()
        let quantized = greedy(Conversation.firstTurn, key: "q").withKVPrecision(.q8)
        #expect(await engine.sampling(for: quantized).0.kvBits == 8)
        #expect(await engine.sampling(for: greedy(Conversation.firstTurn, key: "q").withKVPrecision(.k8v4)).0.kvBits == 8)
        #expect(await engine.sampling(for: greedy(Conversation.firstTurn, key: "q")).0.kvBits == nil)
        _ = try await engine.generate(request: quantized)
        #expect(await engine.promptCaches.roots.isEmpty)
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "fp16"))
        #expect(Set(await engine.promptCaches.roots.keys) == ["fp16"])
    }

    @Test("A cancelled generation stores exactly its prompt and releases its lease; the next turn still matches a full prefill")
    func cancelledGenerationLeavesConsistentTree() async throws {
        let engine = try await TinyModel.engine()
        _ = try await engine.generate(request: greedy(Conversation.otherUser, key: "ns"))
        let stream = try await engine.generateStream(request: greedy(Conversation.firstTurn, key: "ns", maxTokens: 3000))
        let consumer = Task {
            for await _ in stream {
                try? await Task.sleep(for: .milliseconds(2))
            }
        }
        try await Task.sleep(for: .milliseconds(50))
        consumer.cancel()
        await consumer.value

        let prompt = TinyModel.tokens(Conversation.firstTurn)
        let other = TinyModel.tokens(Conversation.otherUser)
        let deadline = ContinuousClock.now + .seconds(60)
        while await engine.promptCaches.storedPrefix(of: prompt, in: "ns") < prompt.count, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        let tree = await engine.promptCaches
        #expect(tree.storedPrefix(of: prompt, in: "ns") == prompt.count)
        #expect(pinCount(tree) == 0)
        #expect(tree.byteCount == (prompt.count + other.count - commonPrefix(prompt, other)) * TinyModel.cacheBytesPerToken)

        let reused = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        let full = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))
        #expect(reused.cachedPromptTokens == prompt.count - 1)
        #expect(reused.generatedText == full.generatedText)
    }

    @Test("Settling a cancelled launch stops MLX's generation instead of waiting for it to reach maxTokens")
    func settlingACancelledLaunchStopsGeneration() async throws {
        let longRun = 3000
        let container = try TinyModel.container()
        let launch = try await container.perform { context in
            let input = LMInput(tokens: MLXArray(TinyModel.tokens(Conversation.firstTurn).map(Int32.init)))
            return try MLXInferenceEngine.launch(
                input: input, reuse: nil, keepsCache: true, boundary: input.text.tokens.size - 1,
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

    @Test("With a repetition penalty, a reused prefix with a short tail still answers like a full prefill")
    func repetitionPenaltySeesTheWholePrompt() async throws {
        let favoured = try await TinyModel.engine().generate(request: greedy(Conversation.firstTurn, key: nil))
        let cachedTurn = Conversation.firstTurn + favoured.generatedText
        let shortDelta = cachedTurn + " ok"
        let engine = try await TinyModel.engine(repetitionPenalty: 3)
        _ = try await engine.generate(request: greedy(cachedTurn, key: "ns", maxTokens: 24))
        let reused = try await engine.generate(request: greedy(shortDelta, key: "ns", maxTokens: 24))
        let full = try await engine.generate(request: greedy(shortDelta, key: nil, maxTokens: 24))
        #expect(TinyModel.tokens(shortDelta).count - reused.cachedPromptTokens < 20)
        #expect(reused.cachedPromptTokens > 0)
        #expect(reused.generatedText == full.generatedText)
    }

    @Test("A branch trimmed by eviction keeps its head and is reused correctly")
    func trimmedBranchIsReusedCorrectly() async throws {
        let firstCount = TinyModel.tokens(Conversation.firstTurn).count
        let budget = (firstCount + 20) * TinyModel.cacheBytesPerToken
        let engine = try await TinyModel.engine(promptCacheBytes: budget)
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        _ = try await engine.generate(request: greedy(Conversation.otherBranch, key: "ns"))
        let tree = await engine.promptCaches
        #expect(tree.byteCount <= budget)
        let kept = tree.storedPrefix(of: TinyModel.tokens(Conversation.afterToolCall), in: "ns")
        #expect(kept > 40)
        #expect(kept < firstCount - 1)

        let reused = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        let full = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))
        #expect(reused.cachedPromptTokens == kept)
        #expect(reused.generatedText == full.generatedText)
    }

    @Test("The tree holds exactly the prompt tokens, compacted, so the budget counts real bytes")
    func storedBranchIsCompact() async throws {
        let engine = try await TinyModel.engine()
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns", maxTokens: 40))
        let tree = await engine.promptCaches
        #expect(tree.byteCount == TinyModel.tokens(Conversation.firstTurn).count * TinyModel.cacheBytesPerToken)
        #expect(tree.byteCount == tree.leaves().reduce(0) { $0 + $1.bytes })
    }

    @Test("The automatic budget is half of what the GPU's recommended working set leaves after the weights")
    func automaticBudget() {
        let gib = 1 << 30
        #expect(PromptCacheBudget.automatic(workingSet: 24 * gib, weights: 5 * gib) == 19 * gib / 2)
        #expect(PromptCacheBudget.automatic(workingSet: 18 * gib, weights: 16 * gib) == gib)
        #expect(PromptCacheBudget.automatic(workingSet: 12 * gib, weights: 14 * gib) == 0)
        #expect(PromptCacheBudget.gpuWorkingSet() > 0)
    }

    @Test("Concurrent generations sharing a prefix both reuse it and both answer like a full prefill")
    func concurrentGenerationsShareThePrefix() async throws {
        let engine = try await TinyModel.engine()
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        async let a = engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        async let b = engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        let (ra, rb) = try await (a, b)
        let full = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))
        #expect(ra.generatedText == full.generatedText)
        #expect(rb.generatedText == full.generatedText)
        #expect(ra.cachedPromptTokens > 0)
        #expect(rb.cachedPromptTokens > 0)
        #expect(pinCount(await engine.promptCaches) == 0)
    }

    @Test("Loading another model drops the whole tree")
    func attachingAModelClearsTheTree() async throws {
        let engine = try await TinyModel.engine()
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        #expect(await engine.promptCaches.byteCount > 0)
        await engine.attach(try TinyModel.container())
        #expect(await engine.promptCaches.roots.isEmpty)
        #expect(await engine.promptCaches.byteCount == 0)
    }
}

@Suite("Prompt cache tree: matching, splitting, eviction and accounting (#32)")
struct PromptCacheTreeTests {

    static let bytesPerToken = 2 * 2 * 8 * 4

    static let layerStride = 100_000

    func cache(_ tokens: [Int], layers: Int = 1) -> PromptCache {
        let cache = PromptCache((0..<layers).map { _ in KVCacheSimple() })
        for (index, layer) in cache.layers.enumerated() {
            let ids = MLXArray(tokens.map { Float($0 + index * Self.layerStride) }).reshaped(1, 1, tokens.count, 1)
            let filled = broadcast(ids, to: [1, 2, tokens.count, 8])
            _ = layer.update(keys: filled, values: filled + 1)
        }
        eval(cache.layers.flatMap { $0.innerState() })
        return cache
    }

    func ids(_ lease: borrowing PromptCacheLease, part: Int = 0, layer: Int = 0) -> [Int] {
        lease.cache.layers[layer].state[part][0, 0, 0..., 0].asArray(Float.self).map(Int.init)
    }

    func branch(_ head: Int, _ count: Int) -> [Int] {
        [head] + (1..<count).map { 1000 + $0 }
    }

    @Test("A lease assembles the matched prefix across split nodes in order and always leaves one token to prefill")
    func leaseAssemblesThePrefix() throws {
        var tree = PromptCacheTree(byteBudget: 1 << 30)
        let a = Array(0..<10)
        let b = Array(0..<5) + [100, 101, 102]
        tree.store("ns", tokens: a, cache: cache(a))
        tree.store("ns", tokens: b, cache: cache(b))

        let lease = try tree.required("ns", for: Array(0..<5) + [100, 101, 999])
        #expect(lease.reusedTokens == 7)
        #expect(ids(lease) == Array(0..<5) + [100, 101])
        #expect(ids(lease, part: 1) == (Array(0..<5) + [100, 101]).map { $0 + 1 })
        _ = consume lease

        let whole = try tree.required("ns", for: a)
        #expect(whole.reusedTokens == a.count - 1)
        #expect(ids(whole) == Array(0..<9))
        _ = consume whole

        let noLease1 = tree.lease("ns", for: [7, 7, 7]) == nil
        #expect(noLease1)
        let noLease2 = tree.lease("other", for: a) == nil
        #expect(noLease2)
        let noLease3 = tree.lease("ns", for: [0]) == nil
        #expect(noLease3)
        #expect(pinCount(tree) == 0)
    }

    @Test("With several layers, split and assembly keep every layer's own keys and values in order")
    func multiLayerSplitAndAssembly() throws {
        var tree = PromptCacheTree(byteBudget: 1 << 30)
        let a = Array(0..<10)
        let b = Array(0..<4) + [50, 51, 52]
        let c = Array(0..<4) + [50, 60, 61]
        for prompt in [a, b, c] {
            tree.store("ns", tokens: prompt, cache: cache(prompt, layers: 3))
        }
        #expect(tree.byteCount == (10 + 3 + 2) * 3 * Self.bytesPerToken)
        let lease = try tree.required("ns", for: c + [99])
        #expect(lease.reusedTokens == c.count)
        #expect(lease.cache.layers.count == 3)
        for layer in 0..<3 {
            let expected = c.map { $0 + layer * Self.layerStride }
            #expect(ids(lease, layer: layer) == expected, "layer \(layer)")
            #expect(ids(lease, part: 1, layer: layer) == expected.map { $0 + 1 }, "layer \(layer)")
        }
        _ = consume lease
    }

    @Test("An unevaluated lease still reads the old buffers after the leased node is split under it")
    func leaseSurvivesASplitOfItsNode() throws {
        var tree = PromptCacheTree(byteBudget: 1 << 30)
        let a = Array(0..<10)
        tree.store("ns", tokens: a, cache: cache(a, layers: 2))
        let lease = try tree.required("ns", for: a + [99])
        let b = Array(0..<5) + [70, 71]
        tree.store("ns", tokens: b, cache: cache(b, layers: 2))
        #expect(tree.roots["ns"]?.children[0]?.tokens == Array(0..<5))
        for layer in 0..<2 {
            let expected = a.map { $0 + layer * Self.layerStride }
            #expect(ids(lease, layer: layer) == expected, "layer \(layer)")
            #expect(ids(lease, part: 1, layer: layer) == expected.map { $0 + 1 }, "layer \(layer)")
        }
        _ = consume lease
    }

    @Test("Removing a leaf merges its parent with the only child left, so no one-child chains build up")
    func removalMergesSingleChildParents() throws {
        var tree = PromptCacheTree(byteBudget: 1 << 30, leafLimit: 2)
        let a = Array(0..<10)
        let b = Array(0..<5) + [70, 71]
        tree.store("ns", tokens: a, cache: cache(a, layers: 2))
        tree.store("ns", tokens: b, cache: cache(b, layers: 2))
        let held = try tree.required("ns", for: b + [99])
        let c = branch(900, 6)
        tree.store("ns", tokens: c, cache: cache(c, layers: 2))

        let root = try #require(tree.roots["ns"])
        let merged = try #require(root.children[0])
        #expect(merged.tokens == b)
        #expect(merged.children.isEmpty)
        #expect(merged === held.pinned)
        #expect(merged.bytes == b.count * 2 * Self.bytesPerToken)
        #expect(tree.storedPrefix(of: a, in: "ns") == 5)
        #expect(tree.leafCount == 2)
        #expect(tree.leafCount == tree.leaves().count)
        #expect(tree.byteCount == (b.count + c.count) * 2 * Self.bytesPerToken)
        _ = consume held

        let lease = try tree.required("ns", for: b + [99])
        for layer in 0..<2 {
            #expect(ids(lease, layer: layer) == b.map { $0 + layer * Self.layerStride }, "layer \(layer)")
        }
        _ = consume lease
    }

    @Test("A branch the leaf cap would drop is refused before anything is split or copied")
    func capRefusesBeforeSplitting() throws {
        var tree = PromptCacheTree(byteBudget: 1 << 30, leafLimit: 1)
        let a = Array(0..<10)
        tree.store("ns", tokens: a, cache: cache(a))
        let held = try tree.required("ns", for: a)
        let node = try #require(tree.roots["ns"]?.children[0])
        let before = node.layers[0][0]
        tree.store("ns", tokens: Array(0..<5) + [70], cache: cache(Array(0..<5) + [70]))
        tree.store("other", tokens: [1, 2], cache: cache([1, 2]))
        #expect(tree.roots["ns"]?.children[0] === node)
        #expect(node.tokens == a)
        #expect(node.layers[0][0] === before)
        #expect(Set(tree.roots.keys) == ["ns"])
        #expect(tree.leafCount == 1)
        _ = consume held
    }

    @Test("A namespace whose last branch is evicted leaves no root behind")
    func emptyNamespacesAreRemoved() {
        var tree = PromptCacheTree(byteBudget: 100 * Self.bytesPerToken)
        let a = branch(1, 100), b = branch(2, 100)
        tree.store("first", tokens: a, cache: cache(a))
        tree.store("second", tokens: b, cache: cache(b))
        #expect(Set(tree.roots.keys) == ["second"])
        for index in 0..<20 {
            tree.store("ns\(index)", tokens: a, cache: cache(a))
        }
        #expect(Set(tree.roots.keys) == ["ns19"])
        #expect(tree.byteCount == 100 * Self.bytesPerToken)
    }

    @Test("A shared prefix is stored once; splitting keeps the byte count exact")
    func sharedPrefixIsCountedOnce() throws {
        var tree = PromptCacheTree(byteBudget: 1 << 30)
        let a = Array(0..<100)
        let b = Array(0..<60) + Array(500..<540)
        tree.store("ns", tokens: a, cache: cache(a))
        #expect(tree.byteCount == 100 * Self.bytesPerToken)
        tree.store("ns", tokens: b, cache: cache(b))
        #expect(tree.byteCount == 140 * Self.bytesPerToken)
        #expect(tree.byteCount == tree.leaves().reduce(0) { $0 + $1.bytes } + 60 * Self.bytesPerToken)
        tree.store("ns", tokens: Array(0..<30), cache: cache(Array(0..<30)))
        #expect(tree.byteCount == 140 * Self.bytesPerToken)
        let shared = try #require(tree.roots["ns"]?.children[0])
        #expect(shared.tokens == Array(0..<60))
        #expect(shared.bytes == 60 * Self.bytesPerToken)
        #expect(Set(shared.children.keys) == [60, 500])
    }

    @Test("Rotating, chunked or live recurrent layers and caches shorter than their prompt are not stored")
    func onlyPlainConsistentCachesAreStored() {
        var tree = PromptCacheTree(byteBudget: 1 << 30)
        tree.store("ns", tokens: [1, 2], cache: PromptCache([MambaCache()]))
        tree.store("ns", tokens: [1, 2], cache: PromptCache([RotatingKVCache(maxSize: 64)]))
        tree.store("ns", tokens: [1, 2], cache: PromptCache([ChunkedKVCache(chunkSize: 64)]))
        tree.store("ns", tokens: Array(0..<10), cache: cache(Array(0..<4)))
        #expect(tree.byteCount == 0)
        #expect(tree.leaves().isEmpty)
    }

    @Test("Over budget the least recently used leaf goes first")
    func evictionIsLeastRecentlyUsedFirst() throws {
        var tree = PromptCacheTree(byteBudget: 300 * Self.bytesPerToken)
        let a = branch(1, 100), b = branch(2, 100), c = branch(3, 100), d = branch(4, 100)
        for prompt in [a, b, c] {
            tree.store("ns", tokens: prompt, cache: cache(prompt))
        }
        try tree.touch("ns", a)
        tree.store("ns", tokens: d, cache: cache(d))
        #expect(tree.byteCount == 300 * Self.bytesPerToken)
        #expect(tree.storedPrefix(of: b, in: "ns") == 0)
        for prompt in [a, c, d] {
            #expect(tree.storedPrefix(of: prompt, in: "ns") == 100)
        }
    }

    @Test("Eviction trims a leaf from its end just enough, keeping its head")
    func evictionTrimsTheLeafTail() {
        var tree = PromptCacheTree(byteBudget: 250 * Self.bytesPerToken)
        let a = branch(1, 100), b = branch(2, 100), c = branch(3, 100)
        for prompt in [a, b, c] {
            tree.store("ns", tokens: prompt, cache: cache(prompt))
        }
        #expect(tree.byteCount == 250 * Self.bytesPerToken)
        #expect(tree.storedPrefix(of: a, in: "ns") == 50)
        #expect(tree.storedPrefix(of: b, in: "ns") == 100)
        #expect(tree.storedPrefix(of: c, in: "ns") == 100)
    }

    @Test("A leased node is never evicted, even when it is the least recently used")
    func leasedNodeIsProtected() throws {
        var tree = PromptCacheTree(byteBudget: 300 * Self.bytesPerToken)
        let a = branch(1, 100), b = branch(2, 100), c = branch(3, 100), d = branch(4, 100), e = branch(5, 100)
        for prompt in [a, b, c] {
            tree.store("ns", tokens: prompt, cache: cache(prompt))
        }
        let inUse = try tree.required("ns", for: a)
        try tree.touch("ns", b)
        try tree.touch("ns", c)
        tree.store("ns", tokens: d, cache: cache(d))
        tree.store("ns", tokens: e, cache: cache(e))
        #expect(tree.storedPrefix(of: a, in: "ns") == 100)
        #expect(tree.storedPrefix(of: b, in: "ns") == 0)
        #expect(tree.storedPrefix(of: c, in: "ns") == 0)
        #expect(tree.byteCount == 300 * Self.bytesPerToken)

        _ = consume inUse
        tree.store("ns", tokens: c, cache: cache(c))
        #expect(tree.storedPrefix(of: a, in: "ns") == 0)
    }

    @Test("When every leaf is in use the tree trims the new branch rather than evict a leased node")
    func everythingLeasedKeepsTheLeasedNode() throws {
        var tree = PromptCacheTree(byteBudget: 100 * Self.bytesPerToken)
        let a = branch(1, 100), b = branch(2, 100)
        tree.store("ns", tokens: a, cache: cache(a))
        let inUse = try tree.required("ns", for: a)
        tree.store("ns", tokens: b, cache: cache(b))
        #expect(tree.storedPrefix(of: a, in: "ns") == 100)
        #expect(tree.storedPrefix(of: b, in: "ns") == 0)
        #expect(tree.byteCount == 100 * Self.bytesPerToken)
        _ = consume inUse
    }

    @Test("An interior node is kept while it has children; it goes once it is a leaf again")
    func interiorNodesStayWhileTheyHaveChildren() {
        var tree = PromptCacheTree(byteBudget: 200 * Self.bytesPerToken)
        let prefix = branch(1, 50)
        let x = prefix + Array(2000..<2050), y = prefix + Array(3000..<3050)
        tree.store("ns", tokens: x, cache: cache(x))
        tree.store("ns", tokens: y, cache: cache(y))
        let q = branch(9, 100)
        tree.store("ns", tokens: q, cache: cache(q))
        #expect(tree.storedPrefix(of: x, in: "ns") == 50)
        #expect(tree.storedPrefix(of: y, in: "ns") == 100)
        #expect(tree.byteCount == 200 * Self.bytesPerToken)

        let r = branch(8, 100)
        tree.store("ns", tokens: r, cache: cache(r))
        #expect(tree.storedPrefix(of: y, in: "ns") == 0)
        #expect(tree.storedPrefix(of: q, in: "ns") == 100)
        #expect(tree.storedPrefix(of: r, in: "ns") == 100)
        #expect(tree.byteCount == 200 * Self.bytesPerToken)
    }

    @Test("A prompt larger than the whole budget keeps the longest head that fits")
    func oversizedPromptKeepsItsHead() throws {
        var tree = PromptCacheTree(byteBudget: 300 * Self.bytesPerToken)
        let small = branch(1, 100)
        tree.store("ns", tokens: small, cache: cache(small))
        let huge = branch(2, 600)
        tree.store("ns", tokens: huge, cache: cache(huge))
        #expect(tree.storedPrefix(of: huge, in: "ns") == 300)
        #expect(tree.storedPrefix(of: small, in: "ns") == 0)
        #expect(tree.byteCount == 300 * Self.bytesPerToken)
        let lease = try tree.required("ns", for: huge)
        #expect(lease.reusedTokens == 300)
        #expect(ids(lease) == Array(huge.prefix(300)))
    }

    @Test("Namespaces share the budget but never a prefix")
    func namespacesShareTheBudgetOnly() {
        var tree = PromptCacheTree(byteBudget: 150 * Self.bytesPerToken)
        let a = branch(1, 100)
        tree.store("alice", tokens: a, cache: cache(a))
        tree.store("bob", tokens: a, cache: cache(a))
        #expect(tree.storedPrefix(of: a, in: "alice") == 50)
        #expect(tree.storedPrefix(of: a, in: "bob") == 100)
        #expect(tree.storedPrefix(of: a, in: "carol") == 0)
        #expect(tree.byteCount == 150 * Self.bytesPerToken)
    }

    @Test("Beyond the leaf cap the oldest free leaves go first, whatever the byte budget")
    func leafCapEvictsTheOldestLeaves() {
        var tree = PromptCacheTree(byteBudget: 1 << 30, leafLimit: 8)
        #expect(PromptCacheTree(byteBudget: 1).leafLimit == PromptCacheTree.maxLeaves)
        #expect(PromptCacheTree.maxLeaves == 4096)
        let prompts = (1...13).map { branch($0, 10) }
        for prompt in prompts {
            tree.store("ns", tokens: prompt, cache: cache(prompt))
            #expect(tree.leafCount == tree.leaves().count)
        }
        #expect(tree.leafCount == 8)
        #expect(tree.byteCount == 8 * 10 * Self.bytesPerToken)
        for prompt in prompts.prefix(5) {
            #expect(tree.storedPrefix(of: prompt, in: "ns") == 0)
        }
        for prompt in prompts.suffix(8) {
            #expect(tree.storedPrefix(of: prompt, in: "ns") == 10)
        }
    }

    @Test("The leaf cap never evicts a leased leaf, and skips a new branch when every leaf is leased")
    func leafCapKeepsLeasedLeaves() throws {
        var tree = PromptCacheTree(byteBudget: 1 << 30, leafLimit: 3)
        let a = branch(1, 10), b = branch(2, 10), c = branch(3, 10), d = branch(4, 10), e = branch(5, 10)
        for prompt in [a, b, c] {
            tree.store("ns", tokens: prompt, cache: cache(prompt))
        }
        let heldA = try tree.required("ns", for: a)
        try tree.touch("ns", b)
        try tree.touch("ns", c)
        tree.store("ns", tokens: d, cache: cache(d))
        #expect(tree.storedPrefix(of: a, in: "ns") == 10)
        #expect(tree.storedPrefix(of: b, in: "ns") == 0)
        #expect(tree.leafCount == 3)

        let heldC = try tree.required("ns", for: c)
        let heldD = try tree.required("ns", for: d)
        tree.store("ns", tokens: e, cache: cache(e))
        #expect(tree.storedPrefix(of: e, in: "ns") == 0)
        for prompt in [a, c, d] {
            #expect(tree.storedPrefix(of: prompt, in: "ns") == 10)
        }
        #expect(tree.leafCount == 3)
        #expect(tree.leafCount == tree.leaves().count)
        _ = consume heldA
        _ = consume heldC
        _ = consume heldD
    }

    @Test("The leaf count follows splits, extensions, evictions and namespaces without walking the tree")
    func leafCountStaysExact() {
        var tree = PromptCacheTree(byteBudget: 120 * Self.bytesPerToken)
        let a = Array(0..<40)
        let steps: [(String, [Int])] = [
            ("ns", a), ("ns", Array(0..<20) + Array(500..<520)), ("ns", a + Array(900..<910)),
            ("ns", Array(0..<10)), ("other", a), ("other", branch(7, 60)), ("ns", branch(8, 90)),
        ]
        for (namespace, prompt) in steps {
            tree.store(namespace, tokens: prompt, cache: cache(prompt))
            #expect(tree.leafCount == tree.leaves().count, "\(namespace) \(prompt.count)")
            #expect(tree.byteCount <= 120 * Self.bytesPerToken)
        }
        tree.removeAll()
        #expect(tree.leafCount == 0)
    }

    @Test("A zero budget stores nothing")
    func zeroBudgetStoresNothing() {
        var tree = PromptCacheTree(byteBudget: 0)
        tree.store("ns", tokens: [1], cache: cache([1]))
        #expect(tree.roots.isEmpty)
    }

    @Test("Dropping a PromptCacheLease without manual release automatically decrements node pins so eviction works")
    func leaseAutoReleasesOnScopeExit() throws {
        var tree = PromptCacheTree(byteBudget: 100 * Self.bytesPerToken)
        let a = branch(1, 100)
        tree.store("ns", tokens: a, cache: cache(a))
        #expect(pinCount(tree) == 0)

        func throwDuringLease() throws {
            struct DummyError: Error {}
            let lease = try tree.required("ns", for: a)
            #expect(pinCount(tree) == 1)
            _ = lease
            throw DummyError()
        }

        #expect(throws: (any Error).self) {
            try throwDuringLease()
        }
        #expect(pinCount(tree) == 0)

        let b = branch(2, 100)
        tree.store("ns", tokens: b, cache: cache(b))
        #expect(tree.storedPrefix(of: a, in: "ns") == 0)
        #expect(tree.storedPrefix(of: b, in: "ns") == 100)
    }
}
