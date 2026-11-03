import Testing
import MLX
@testable import BasaltCore
@testable import BasaltInference

@Suite("Shared tiny-model fixtures decode greedily and reuse the prompt cache on every family (#228)", .serialized)
struct TinyModelFixtureTests {

    @Test("Greedy decoding through the cache picks the argmax of one uncached forward pass, and a fixture rebuilt from its seed decodes the same tokens",
          arguments: TinyModel.Family.allCases)
    func greedySmoke(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let prompt = TinyModel.tokens(Conversation.firstTurn)
        let tokens = try LookupFixture.baseline(model, prompt: prompt, maxTokens: 16)
        #expect(tokens.count == 16)
        let sequence = prompt + tokens
        let logits = model(MLXArray(sequence.map(Int32.init)).reshaped(1, -1), cache: try model.newCache(parameters: nil))
        let uncached = argMax(logits[0, (prompt.count - 1) ..< (sequence.count - 1)], axis: -1).asArray(Int32.self).map(Int.init)
        expectTokenSequence(uncached, tokens)
        expectTokenSequence(try LookupFixture.baseline(try TinyModel.make(family), prompt: prompt, maxTokens: 16), tokens)
    }

    @Test("The sigmoid-bias router fixture has a random gate and correction bias on both MoE layers, not the all-zero init that ties every expert")
    func sigmoidRouterIsRandom() throws {
        let router = try TinyModel.make(.sigmoidMoe).parameters().flattened()
            .filter { $0.0.hasSuffix(".gate.weight") || $0.0.hasSuffix(".e_score_correction_bias") }
        #expect(router.count == 4)
        #expect(router.allSatisfy { abs($0.1).max().item(Float.self) > 0 })
    }

    @Test("The next turn reuses the cached first turn through the engine and answers like a full prefill",
          arguments: TinyModel.Family.allCases)
    func promptCacheReuse(family: TinyModel.Family) async throws {
        let engine = try await TinyModel.engine(family, promptCacheBytes: 1 << 30)
        let first = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        let reused = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        let full = try await engine.generate(request: greedy(Conversation.afterToolCall, key: nil))
        #expect(first.cachedPromptTokens == 0)
        #expect(reused.cachedPromptTokens == TinyModel.tokens(Conversation.firstTurn).count - 1)
        #expect(full.cachedPromptTokens == 0)
        #expect(reused.generatedText == full.generatedText)
    }
}
