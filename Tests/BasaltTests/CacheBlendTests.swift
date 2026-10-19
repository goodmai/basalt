import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
@testable import BasaltCore
@testable import BasaltInference

struct ParagraphTokenizer: MLXLMCommon.Tokenizer {
    static let paragraph = 0
    var newline = "\n\n"

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { $0 == Self.paragraph ? newline : "x" }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

enum TinyQwen3 {
    static let config = """
        {"hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-5,
         "vocab_size": 256, "rope_theta": 10000.0, "tie_word_embeddings": false}
        """

    static func make() throws -> Qwen3Model {
        MLXRandom.seed(36)
        let model = Qwen3Model(try JSONDecoder().decode(Qwen3Configuration.self, from: Data(config.utf8)))
        eval(model)
        return model
    }
}

enum Files {
    static func chunk(_ seed: Int, length: Int = 24) -> [Int] {
        (0 ..< length - 1).map { 1 + ($0 * 37 + seed * 101) % 250 } + [ParagraphTokenizer.paragraph]
    }

    static let header = [3, 9, 27, ParagraphTokenizer.paragraph]
    static let files = (1 ... 4).map { chunk($0) }
    static let question = [11, 13, 17, 19]
    static let original = header + files.joined() + question
    static let reordered = header + [files[2], files[0], files[3], files[1]].joined() + question
    static let reusable = files.joined().count
}

enum TinyLlama {
    static func make() throws -> LlamaModel {
        MLXRandom.seed(36)
        let config = TinyModel.config.replacingOccurrences(of: "\"num_hidden_layers\": 2", with: "\"num_hidden_layers\": 4")
        let model = LlamaModel(try JSONDecoder().decode(LlamaConfiguration.self, from: Data(config.utf8)))
        eval(model)
        return model
    }
}

func blendModels() throws -> [any LanguageModel] {
    [try TinyLlama.make(), try TinyQwen3.make()]
}

func fullLogits(_ model: any LanguageModel, _ prompt: [Int]) throws -> MLXArray {
    let logits = model(MLXArray(prompt.map(Int32.init)).reshaped(1, -1), cache: try model.newCache(parameters: nil))
    return logits[0, -1].asType(.float32)
}

func blendedLogits(
    _ model: any LanguageModel, store: CacheBlendStore, _ prompt: [Int], after prefix: Int = 0,
    tokenizer: ParagraphTokenizer = ParagraphTokenizer()
) throws -> (MLXArray, CacheBlendOutcome) {
    let cache = PromptCache(try model.newCache(parameters: nil))
    if prefix > 0 {
        _ = model(MLXArray(prompt[..<prefix].map(Int32.init)).reshaped(1, -1), cache: cache.layers)
    }
    let outcome = try #require(store.extend(
        cache, prompt: prompt, from: prefix, namespace: "ns",
        model: model, tokenizer: tokenizer))
    let logits = model(MLXArray([Int32(prompt.last!)]).reshaped(1, 1), cache: outcome.cache.layers)
    return (logits[0, -1].asType(.float32), outcome)
}

func primed(_ model: any LanguageModel, ratio: Float) throws -> CacheBlendStore {
    let store = CacheBlendStore(ratio: ratio, byteBudget: 1 << 30)
    let (logits, first) = try blendedLogits(model, store: store, Files.original)
    #expect(first.reused == 0)
    #expect(first.recomputed == 0)
    #expect(first.exactPrefix == Files.original.count - 1)
    #expect(allClose(logits, try fullLogits(model, Files.original), rtol: 1e-3, atol: 1e-3).item(Bool.self))
    return store
}

func distance(_ a: MLXArray, _ b: MLXArray) -> Float {
    (a - b).square().sum().sqrt().item(Float.self)
}

@Suite("CacheBlend non-prefix KV reuse on tiny random Llama and Qwen3 (#36)", .serialized)
struct CacheBlendTests {

    @Test("Chunks end after a blank line, in one token or across two newline tokens, so a reordered file keeps its span")
    func chunksSplitAfterBlankLines() throws {
        #expect(CacheBlendStore.chunks(["a", "b", "\n\n", "c", "}\n\n", "d"]) == [0 ..< 3, 3 ..< 5, 5 ..< 6])
        #expect(CacheBlendStore.chunks(["a", "\n", "\n", "b", "}\n", "\n", "c", "\n", "d"]) == [0 ..< 3, 3 ..< 6, 6 ..< 9])
        let store = CacheBlendStore(ratio: 0, byteBudget: 0)
        let reordered = Array(Files.reordered.dropLast())
        let spans = Set(store.chunks(reordered, tokenizer: ParagraphTokenizer()).map { Array(reordered[$0]) })
        #expect(Files.files.allSatisfy(spans.contains))
    }

    @Test("A SentencePiece-style tokenizer that emits one newline per token still finds the reordered files")
    func singleNewlineTokensStillChunk() throws {
        let model = try TinyLlama.make()
        let pieces = ParagraphTokenizer(newline: "\n")
        let doubled = { (tokens: [Int]) in tokens.flatMap { $0 == ParagraphTokenizer.paragraph ? [0, 0] : [$0] } }
        let store = CacheBlendStore(ratio: 0.5, byteBudget: 1 << 30)
        _ = try blendedLogits(model, store: store, doubled(Files.original), tokenizer: pieces)
        let (_, outcome) = try blendedLogits(model, store: store, doubled(Files.reordered), tokenizer: pieces)
        #expect(outcome.reused == Files.reusable + Files.files.count)
    }

    @Test("Only Llama and Qwen3 are blended; other families fall back to the prefix path")
    func supportsDenseLlamaAndQwen3() throws {
        for model in try blendModels() {
            let blender = try #require(CacheBlendModel(model))
            #expect(blender.layers.count == (model as? KVCacheDimensionProvider)?.kvHeads.count)
            #expect(blender.headDim == 16)
        }
        let qwen2 = Qwen2Model(try JSONDecoder().decode(Qwen2Configuration.self, from: Data(TinyModel.config.utf8)))
        #expect(CacheBlendModel(qwen2) == nil)
    }

    @Test("With ratio 1 the reordered files blend to the logits of a full prefill, RoPE at the new positions included")
    func fullRatioMatchesFullPrefill() throws {
        for model in try blendModels() {
            let store = try primed(model, ratio: 1)
            let (logits, outcome) = try blendedLogits(model, store: store, Files.reordered)
            #expect(outcome.reused == Files.reusable)
            #expect(outcome.recomputed == outcome.reused)
            #expect(outcome.exactPrefix == Files.reordered.count - 1)
            let expected = try fullLogits(model, Files.reordered)
            #expect(allClose(logits, expected, rtol: 1e-3, atol: 1e-3).item(Bool.self))
            let (leased, tail) = try blendedLogits(model, store: store, Files.reordered, after: Files.header.count)
            #expect(tail.reused == Files.reusable)
            #expect(allClose(leased, expected, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        }
    }

    @Test("Chunks reused at their own positions after their own context blend to a full prefill at any ratio")
    func unchangedContextBlendsExactly() throws {
        for model in try blendModels() {
            for ratio: Float in [0, 0.15] {
                let (logits, outcome) = try blendedLogits(model, store: try primed(model, ratio: ratio), Files.original)
                #expect(outcome.reused == Files.reusable)
                #expect(outcome.recomputed == Int(Float(outcome.reused) * ratio))
                #expect(allClose(logits, try fullLogits(model, Files.original), rtol: 1e-3, atol: 1e-3).item(Bool.self))
            }
        }
    }

    @Test("With ratio 0.15 the blend lands closer to a full prefill than reusing every cached chunk as is")
    func partialRatioBeatsNaiveConcat() throws {
        for model in try blendModels() {
            let expected = try fullLogits(model, Files.reordered)
            let (blend, outcome) = try blendedLogits(model, store: try primed(model, ratio: 0.15), Files.reordered)
            let (naive, _) = try blendedLogits(model, store: try primed(model, ratio: 0), Files.reordered)
            #expect(outcome.recomputed == Int(Float(outcome.reused) * 0.15))
            #expect(outcome.exactPrefix < Files.reordered.count - 1)
            #expect(distance(blend, expected) < distance(naive, expected))
        }
    }

    @Test("Uncached and short chunks are recomputed, only long misses are stored, and the byte budget evicts and refuses")
    func missesAreRecomputedAndStoredWithinBudget() throws {
        let model = try TinyLlama.make()
        let store = try primed(model, ratio: 0)
        #expect(store.chunkCount == Files.files.count)
        let changed = Files.header + Files.chunk(9) + Files.files[0] + Files.question
        let (logits, outcome) = try blendedLogits(model, store: store, changed)
        #expect(outcome.reused == Files.files[0].count)
        #expect(outcome.recomputed == 0)
        #expect(store.chunkCount == Files.files.count + 1)
        #expect(logits.shape == [256])

        let bytes = store.byteCount
        let tight = CacheBlendStore(ratio: 0, byteBudget: bytes / 2)
        _ = try blendedLogits(model, store: tight, Files.original)
        #expect(tight.byteCount <= bytes / 2)
        #expect(tight.chunkCount < Files.files.count)

        let tiny = CacheBlendStore(ratio: 0, byteBudget: bytes / store.chunkCount - 1)
        _ = try blendedLogits(model, store: tiny, Files.original)
        #expect(tiny.chunkCount == 0)
        #expect(tiny.byteCount == 0)
    }

    @Test("The engine answers like a full prefill, reports only the exact prefix as cached and keeps the tree exact")
    func engineReportsExactCachedTokensOnly() async throws {
        let engine = MLXInferenceEngine(promptCacheBytes: 1 << 26, cacheBlendRatio: 0.15)
        await engine.attach(try TinyModel.container())
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        let blended = try await engine.generate(request: greedy(Conversation.otherUser, key: "ns"))
        let full = try await engine.generate(request: greedy(Conversation.otherUser, key: nil))

        let promptTokens = TinyModel.tokens(Conversation.otherUser)
        #expect(blended.cachedPromptTokens == commonPrefix(TinyModel.tokens(Conversation.firstTurn), promptTokens))
        #expect(blended.promptTokens == promptTokens.count)
        #expect(blended.generatedText == full.generatedText)
        let store = try #require(await engine.cacheBlend)
        #expect(store.chunkCount == 1)
        #expect(await engine.promptCaches.byteBudget == 1 << 25)
    }

    @Test("A hybrid SSM model never blends: the snapshot path serves it and the chunk store stays empty")
    func hybridModelsNeverBlend() async throws {
        #expect(CacheBlendModel(try TinyHybrid.make()) == nil)
        let engine = MLXInferenceEngine(promptCacheBytes: 1 << 26, cacheBlendRatio: 0.15)
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-qwen35"), model: try TinyHybrid.make(),
            processor: ByteInputProcessor(), tokenizer: ByteTokenizer())))
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        let reused = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        let full = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))
        #expect(reused.cachedPromptTokens > 0)
        #expect(reused.generatedText == full.generatedText)
        #expect(try #require(await engine.cacheBlend).chunkCount == 0)
    }
}
