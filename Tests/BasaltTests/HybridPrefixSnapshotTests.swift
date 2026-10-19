import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
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

    static func engine(promptCacheBytes: Int? = nil) async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(promptCacheBytes: promptCacheBytes)
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-qwen35"),
            model: try make(),
            processor: ByteInputProcessor(),
            tokenizer: ByteTokenizer())))
        return engine
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
        #expect(tree.byteCount == snapshot.byteCount)
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
        var tree = PromptCacheTree(byteBudget: snapshot.byteCount - 1)
        tree.store("ns", tokens: Array(first.dropLast()), cache: snapshot)
        #expect(tree.byteCount == 0)
        #expect(tree.leaves().isEmpty)
    }

    @Test("On the engine, a follow-up turn reuses all but the last token of the previous prompt and answers like a full prefill")
    func engineReusesTheSnapshot() async throws {
        let engine = try await TinyHybrid.engine(promptCacheBytes: 1 << 30)
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
