import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Inkling 2 P3: prefix anchors across agent turns on the inkling2Loop TinyModel (#650)", .serialized)
struct InklingPromptCacheTests {
    static let first: [Int32] = (0 ..< 23).map { Int32(5 + ($0 * 37) % 240) }
    static let second: [Int32] = first + [17, 200, 41, 9, 133, 61, 2, 99, 250, 12, 77]

    static func logits(_ tokens: MLXArray, from start: Int, cache: PromptCache, state: LMOutput.State?, model: any LanguageModel) throws -> MLXArray {
        try HybridPrefixDriftTests.lastLogits(tokens, from: start, cache: cache, state: state, model: model)
    }

    @Test("The cache is hybrid: window and conv states are snapshot-only, global KV is plain")
    func hybridLayout() throws {
        let model = try TinyModel.make(.inkling2Loop)
        let cache = PromptCache(try model.newCache(parameters: nil))
        #expect(cache.isHybrid)
        #expect(cache.layers.filter(\.isSnapshotOnly).count == 15)
    }

    @Test("A restored anchor past the 8-token window, with all 10 visits' conv states, gives the logits of a from-scratch prefill cut at the same points, bit for bit",
          arguments: [11, 16, 22])
    func restoredAnchorIsBitwise(cut: Int) throws {
        let model = try TinyModel.make(.inkling2Loop)
        let prefill = GenerateParameters().prefill
        let tokens1 = MLXArray(Self.first)
        let tokens2 = MLXArray(Self.second)
        let end = Self.second.count - 1

        let turn1 = try HybridPrefixSnapshot.prefillHead(
            tokens1, into: PromptCache(try model.newCache(parameters: nil)), plan: CachedPrompt(boundary: cut), model: model, prefill: prefill)
        var tree = PromptCacheTree(byteBudget: ByteCount(1 << 30))
        tree.store("agent", tokens: Self.first.prefix(cut).map(Int.init), cache: try #require(turn1.kept))
        guard let lease = tree.lease("agent", for: Self.second.map(Int.init)) else {
            Issue.record("no lease for the shared prefix")
            return
        }
        #expect(lease.reusedTokens == cut)
        let turn2 = try HybridPrefixSnapshot.prefillHead(
            tokens2, into: lease.cache, plan: CachedPrompt(reuse: (lease.cache, cut), boundary: end), model: model, prefill: prefill)
        let restored = try Self.logits(tokens2, from: end, cache: lease.cache, state: turn2.state, model: model)

        let scratch = PromptCache(try model.newCache(parameters: nil))
        let head = try HybridPrefixSnapshot.prefillHead(tokens2, into: scratch, plan: CachedPrompt(boundary: cut), model: model, prefill: prefill)
        scratch.state = head.state
        let rest = try HybridPrefixSnapshot.prefillHead(
            tokens2, into: scratch, plan: CachedPrompt(reuse: (scratch, cut), boundary: end), model: model, prefill: prefill)
        let expected = try Self.logits(tokens2, from: end, cache: scratch, state: rest.state, model: model)
        #expect(abs(restored - expected).max().item(Float.self) == 0)
    }

    @Test("On the engine, the second agent turn reuses the first turn's prompt and answers like a fresh engine")
    func secondTurnReusesAndMatches() async throws {
        let cached = try await TinyModel.engine(.inkling2Loop, promptCacheBytes: 1 << 26)
        let fresh = try await TinyModel.engine(.inkling2Loop)
        _ = try await cached.generate(request: greedy(Conversation.firstTurn, key: "agent"))
        let reused = try await cached.generate(request: greedy(Conversation.afterToolCall, key: "agent"))
        let plain = try await fresh.generate(request: greedy(Conversation.afterToolCall, key: nil))
        #expect(reused.cachedPromptTokens > 0)
        #expect(reused.generatedText == plain.generatedText)
    }
}
