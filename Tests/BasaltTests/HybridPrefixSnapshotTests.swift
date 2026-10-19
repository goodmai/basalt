import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
import MLXVLM
@testable import BasaltCore
@testable import BasaltInference

enum TinyHybrid {
    static let config = """
        {"model_type": "qwen3_5_text", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
         "linear_num_value_heads": 4, "linear_num_key_heads": 2,
         "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
         "full_attention_interval": 2, "rms_norm_eps": 1e-5, "vocab_size": 256,
         "tie_word_embeddings": false}
        """

    static func make() throws -> Qwen35TextModel {
        MLXRandom.seed(79)
        let model = Qwen35TextModel(try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(config.utf8)))
        eval(model)
        return model
    }

    static let visionConfig = """
        {"model_type": "qwen3_5", "text_config": \(config),
         "vision_config": {"model_type": "qwen3_5", "depth": 1, "hidden_size": 16, "intermediate_size": 32,
          "out_hidden_size": 64, "num_heads": 2, "patch_size": 2, "spatial_merge_size": 2,
          "temporal_patch_size": 2, "num_position_embeddings": 16}}
        """

    static func makeVision() throws -> Qwen35 {
        MLXRandom.seed(79)
        let model = Qwen35(try JSONDecoder().decode(Qwen35Configuration.self, from: Data(visionConfig.utf8)))
        eval(model)
        return model
    }

    static func engine(promptCacheBytes: Int? = nil, vision: Bool = false) async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(promptCacheBytes: promptCacheBytes)
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-qwen35"),
            model: vision ? try makeVision() : try make(),
            processor: ByteInputProcessor(),
            tokenizer: ByteTokenizer())))
        return engine
    }

    static let attentionBytesPerToken = 2 * 2 * 2 * 16 * 4

    static func storedBytes(_ snapshot: PromptCache, tokens: Int) -> Int {
        tokens * attentionBytesPerToken + (RecurrentSnapshot(of: snapshot)?.bytes ?? 0)
    }

    static func row(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens.map(Int32.init)).reshaped(1, -1)
    }

    static func snapshotted(_ model: Qwen35TextModel, _ prompt: [Int]) throws -> PromptCache {
        let head = try HybridPrefixSnapshot.prefillHead(
            MLXArray(prompt.map(Int32.init)), into: PromptCache(try model.newCache(parameters: nil)),
            reused: 0, boundary: prompt.count - 1, model: model, prefill: PrefillParameters(stepSize: 16))
        #expect(head.start == prompt.count - 1)
        return try #require(head.kept)
    }

    static func recurrentStates(_ tree: PromptCacheTree) -> [[Float]] {
        tree.leaves().compactMap(\.snapshot).flatMap { snapshot in
            snapshot.layers.keys.sorted().flatMap { snapshot.layers[$0]!.innerState().map { $0.asType(.float32).asArray(Float.self) } }
        }
    }
}

@Suite("Prefix reuse on a hybrid linear-attention model via state snapshots (#79)", .serialized)
struct HybridPrefixSnapshotTests {

    @Test("A Qwen3.5-style cache mixes Mamba and plain attention layers and is recognised as hybrid")
    func tinyModelIsHybrid() throws {
        let cache = PromptCache(try TinyHybrid.make().newCache(parameters: nil))
        #expect(cache.layers.contains { $0 is MambaCache })
        #expect(cache.layers.contains { $0 is KVCacheSimple })
        #expect(cache.isHybrid)
        #expect(!cache.isSharable)
    }

    @Test("Restoring the snapshot and prefilling the tail gives the logits of a full prefill; skipping the SSM restore does not")
    func restoredSnapshotMatchesFullPrefill() throws {
        let model = try TinyHybrid.make()
        let first = TinyModel.tokens(Conversation.firstTurn)
        let second = TinyModel.tokens(Conversation.afterToolCall)
        let boundary = first.count - 1
        #expect(Array(second.prefix(boundary)) == Array(first.prefix(boundary)))

        var tree = PromptCacheTree(byteBudget: 1 << 30)
        let snapshot = try TinyHybrid.snapshotted(model, first)
        #expect(snapshot.tokenCount == boundary)
        tree.store("ns", tokens: Array(first.prefix(boundary)), cache: snapshot)
        #expect(tree.byteCount == TinyHybrid.storedBytes(snapshot, tokens: boundary))
        let stored = TinyHybrid.recurrentStates(tree)
        #expect(!stored.isEmpty)

        let full = model(TinyHybrid.row(second), cache: try model.newCache(parameters: nil))
        let expected = full[0..., boundary..., 0...]
        let tailTokens = TinyHybrid.row(Array(second[boundary...]))

        let lease = try tree.required("ns", for: second)
        #expect(lease.reusedTokens == boundary)
        #expect(lease.cache.layers.allSatisfy { $0.offset == boundary })
        let tail = model(tailTokens, cache: lease.cache.layers)
        eval(tail)
        tree.release(lease)
        #expect(tail.shape == expected.shape)
        #expect(allClose(tail, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        #expect(TinyHybrid.recurrentStates(tree) == stored)

        let again = try tree.required("ns", for: second)
        let unrestored = again.cache.layers.map { $0 is MambaCache ? MambaCache() : $0 }
        let control = model(tailTokens, cache: unrestored)
        tree.release(again)
        #expect(!allClose(control, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        #expect(TinyHybrid.recurrentStates(tree) == stored)
        #expect(pinCount(tree) == 0)
    }

    @Test("A prompt that does not extend the snapshot's tokens gets no lease and is prefilled in full")
    func nonPrefixFallsBackToFullPrefill() throws {
        let model = try TinyHybrid.make()
        let first = TinyModel.tokens(Conversation.firstTurn)
        var tree = PromptCacheTree(byteBudget: 1 << 30)
        tree.store("ns", tokens: Array(first.dropLast()), cache: try TinyHybrid.snapshotted(model, first))
        #expect(tree.lease("ns", for: TinyModel.tokens(Conversation.otherUser)) == nil)
        #expect(tree.lease("ns", for: TinyModel.tokens(Conversation.otherBranch)) == nil)
        #expect(tree.lease("ns", for: Array(first.prefix(first.count - 2)) + [42, 43]) == nil)
        #expect(tree.lease("ns", for: first)?.reusedTokens == first.count - 1)
    }

    @Test("A live hybrid cache, whose recurrent state sits past the stored tokens, is never stored")
    func liveHybridCacheIsRefused() throws {
        let model = try TinyHybrid.make()
        let first = TinyModel.tokens(Conversation.firstTurn)
        let cache = PromptCache(try model.newCache(parameters: nil))
        _ = model(TinyHybrid.row(first), cache: cache.layers)
        eval(cache.layers.flatMap { $0.innerState() })
        var tree = PromptCacheTree(byteBudget: 1 << 30)
        tree.store("ns", tokens: Array(first.dropLast()), cache: cache)
        #expect(tree.byteCount == 0)
        #expect(tree.leaves().isEmpty)
    }

    @Test("Eviction removes a snapshot leaf whole instead of trimming it past its recurrent state")
    func evictionDropsSnapshotLeavesWhole() throws {
        let model = try TinyHybrid.make()
        let first = TinyModel.tokens(Conversation.firstTurn)
        let snapshot = try TinyHybrid.snapshotted(model, first)
        var tree = PromptCacheTree(byteBudget: TinyHybrid.storedBytes(snapshot, tokens: first.count - 1) - 1)
        tree.store("ns", tokens: Array(first.dropLast()), cache: snapshot)
        #expect(tree.byteCount == 0)
        #expect(tree.leaves().isEmpty)
    }

    @Test("On the engine, a follow-up turn reuses all but the last token of the previous prompt and answers like a full prefill")
    func engineReusesTheSnapshot() async throws {
        try await reuseOnEngine(vision: false)
    }

    @Test("Through MLXVLM's Qwen35, whose warm continuation needs the carried rope delta, a restored snapshot answers like a full prefill")
    func visionWrapperCarriesItsState() async throws {
        try await reuseOnEngine(vision: true)
        let model = try TinyHybrid.makeVision()
        let first = TinyModel.tokens(Conversation.firstTurn)
        let head = try HybridPrefixSnapshot.prefillHead(
            MLXArray(first.map(Int32.init)), into: PromptCache(try model.newCache(parameters: nil)),
            reused: 0, boundary: first.count - 1, model: model, prefill: PrefillParameters(stepSize: 16))
        #expect(head.state != nil)
        var tree = PromptCacheTree(byteBudget: 1 << 30)
        tree.store("ns", tokens: Array(first.dropLast()), cache: try #require(head.kept))
        let lease = try tree.required("ns", for: TinyModel.tokens(Conversation.afterToolCall))
        #expect(lease.cache.state != nil)
        tree.release(lease)
    }

    func reuseOnEngine(vision: Bool) async throws {
        let engine = try await TinyHybrid.engine(promptCacheBytes: 1 << 30, vision: vision)
        let first = TinyModel.tokens(Conversation.firstTurn)
        let turn1 = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        #expect(turn1.cachedPromptTokens == 0)
        #expect(turn1.promptTokens == first.count)
        let stored = await engine.promptCaches
        #expect(stored.storedPrefix(of: first, in: "ns") == first.count - 1)
        #expect(stored.byteCount == stored.leaves().reduce(0) { $0 + $1.bytes })
        #expect(stored.leaves().allSatisfy { $0.snapshot != nil })

        let turn2 = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        let full2 = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))
        #expect(turn2.cachedPromptTokens == first.count - 1)
        #expect(turn2.promptTokens == TinyModel.tokens(Conversation.afterToolCall).count)
        #expect(!turn2.generatedText.isEmpty)
        #expect(Set(turn2.generatedText).count > 1)
        #expect(turn2.generatedText == full2.generatedText)

        let turn3 = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        #expect(turn3.cachedPromptTokens == TinyModel.tokens(Conversation.afterToolCall).count - 1)
        #expect(turn3.generatedText == full2.generatedText)

        let other = try await engine.generate(request: greedy(Conversation.otherUser, key: "ns"))
        let fullOther = try await engine.generate(request: greedy(Conversation.otherUser, key: nil))
        #expect(other.cachedPromptTokens == 0)
        #expect(other.generatedText == fullOther.generatedText)
        #expect(pinCount(await engine.promptCaches) == 0)
    }

    @Test("A snapshot node left with one child after an eviction is not merged away, so its restore point survives")
    func snapshotNodeIsNotMerged() throws {
        let model = try TinyHybrid.make()
        let p = TinyModel.tokens(Conversation.firstTurn)
        let head = Array(p.dropLast())
        var tree = PromptCacheTree(byteBudget: 1 << 30, leafLimit: 2)
        for prompt in [p, head + [10, 11, 12], head + [20, 21, 22], Array(200..<220)] {
            tree.store("ns", tokens: Array(prompt.dropLast()), cache: try TinyHybrid.snapshotted(model, prompt))
        }
        #expect(tree.leaves().count == 2)
        #expect(tree.byteCount == tree.leaves().reduce(0) { $0 + $1.bytes } + (tree.roots["ns"]?.children[p[0]]?.bytes ?? 0))
        let lease = try tree.required("ns", for: head + [30, 31])
        #expect(lease.reusedTokens == p.count - 1)
        tree.release(lease)
    }

    @Test("A split head without a snapshot never outlives its branches as an unrestorable leaf")
    func noStrandedHeadAfterEviction() throws {
        let model = try TinyHybrid.make()
        let head = Array(TinyModel.tokens(Conversation.firstTurn).dropLast())
        var tree = PromptCacheTree(byteBudget: 1 << 30, leafLimit: 1)
        for prompt in [head + [10, 11, 12], head + [20, 21, 22], Array(200..<220)] {
            tree.store("ns", tokens: Array(prompt.dropLast()), cache: try TinyHybrid.snapshotted(model, prompt))
            #expect(tree.leaves().count == 1)
            #expect(tree.leaves().allSatisfy { $0.snapshot != nil && $0.parent?.tokens.isEmpty == true })
            #expect(tree.byteCount == tree.leaves().reduce(0) { $0 + $1.bytes })
        }
    }

    @Test("The snapshot copies only the recurrent layers; attention layers stay the live caches, sliced at store time")
    func snapshotSharesLiveAttention() throws {
        let model = try TinyHybrid.make()
        let prompt = TinyModel.tokens(Conversation.firstTurn)
        let live = PromptCache(try model.newCache(parameters: nil))
        let head = try HybridPrefixSnapshot.prefillHead(
            MLXArray(prompt.map(Int32.init)), into: live, reused: 0, boundary: prompt.count - 1,
            model: model, prefill: PrefillParameters(stepSize: 16))
        let kept = try #require(head.kept)
        for (original, snapshot) in zip(live.layers, kept.layers) {
            #expect(((original as AnyObject) === (snapshot as AnyObject)) == !(original is ArraysCache))
        }
        _ = model(TinyHybrid.row(Array(prompt[(prompt.count - 1)...]) + [1, 2, 3]), cache: live.layers)
        #expect(kept.isSnapshot(of: prompt.count - 1))
        #expect(kept.tokenCount == prompt.count - 1)
    }

    @Test("The snapshot boundary is the end of the request text: the template's suffix after it is measured, not assumed")
    func templateSuffixIsMeasured() async throws {
        let container = ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-qwen35"),
            model: try TinyHybrid.make(),
            processor: SuffixedInputProcessor(),
            tokenizer: ByteTokenizer()))
        let suffix = await HybridPrefixSnapshot.templateSuffixLength(container) {
            UserInput(chat: [.user($0)])
        }
        #expect(suffix == SuffixedInputProcessor.suffix.count)
    }
}

struct SuffixedInputProcessor: UserInputProcessor {
    static let prefix = [250, 251]
    static let suffix = [252, 253, 254, 0, 1]

    func prepare(input: UserInput) throws -> LMInput {
        let text = DefaultMessageGenerator().generate(from: input).compactMap { $0["content"] as? String }.joined()
        let tokens = Self.prefix + ByteTokenizer().encode(text: text, addSpecialTokens: false) + Self.suffix
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }
}
