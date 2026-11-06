import Foundation
import Testing
import MLX
import MLXLLM
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("MTP drafting reuses the prompt cache: the drafter continues from a stored prefix exactly as a full prefill would (#207)", .serialized)
struct MTPPromptCacheTests {
    static func head() throws -> Qwen35MTPDraftModel {
        let config = try JSONDecoder.json5().decode(Qwen35TextConfiguration.self, from: Data(TinyModel.Family.qwen35.config.utf8))
        MLXRandom.seed(5)
        let head = Qwen35MTPDraftModel(config, preconvertedNorms: false)
        eval(head)
        return head
    }

    static func hidden(_ tokens: [Int], _ model: any LanguageModel, _ cache: [KVCache]) throws -> MLXArray {
        var emit = LMOutput.State()
        emit[mtpEmitFlagKey] = true
        let output = model(LMInput.Text(tokens: MLXArray(tokens.map(Int32.init)))[text: .newAxis], cache: cache, state: emit)
        return try #require(output.state?[mtpLastHiddenStatesKey])
    }

    @Test("A prefix prefilled without its next token, then the tail with the bonus, leaves the drafter cache, position and seed of one full prefill")
    func splitPrefillMatchesFull() throws {
        let target = try TinyModel.make(.qwen35)
        let head = try Self.head()
        let prompt = TinyModel.tokens(Conversation.afterToolCall)
        let cut = TinyModel.tokens(Conversation.firstTurn).count - 1
        let bonus = MLXArray([Int32(7)])
        let sampler = ArgMaxSampler()

        var full = head.makeState(parameters: nil)
        head.prepareDrafterState(
            target: target,
            input: MTPPrefillInput(
                promptTokens: MLXArray(prompt.map(Int32.init)),
                targetHidden: try Self.hidden(prompt, target, try target.newCache(parameters: nil)), firstBonus: bonus),
            state: &full, sampler: sampler)

        var split = head.makeState(parameters: nil)
        let cache = try target.newCache(parameters: nil)
        let prefix = Array(prompt[..<cut]), tail = Array(prompt[cut...])
        head.prepareDrafterState(
            target: target,
            input: MTPPrefillInput(
                promptTokens: MLXArray(prefix.map(Int32.init)), targetHidden: try Self.hidden(prefix, target, cache), firstBonus: nil),
            state: &split, sampler: sampler)
        #expect(split.nextPosition == cut - 1)
        #expect(split.carryHidden != nil)
        head.prepareDrafterState(
            target: target,
            input: MTPPrefillInput(
                promptTokens: MLXArray(tail.map(Int32.init)), targetHidden: try Self.hidden(tail, target, cache), firstBonus: bonus),
            state: &split, sampler: sampler)

        #expect(split.nextPosition == full.nextPosition)
        #expect(split.carryHidden == nil)
        #expect(try #require(split.seedToken).item(Int.self) == (try #require(full.seedToken)).item(Int.self))
        #expect(allClose(try #require(split.seedHidden), try #require(full.seedHidden), atol: 1e-4).item(Bool.self))
        for (a, b) in zip(split.cache.flatMap { $0.innerState() }, full.cache.flatMap { $0.innerState() }) {
            #expect(a.shape == b.shape)
            #expect(allClose(a, b, atol: 1e-4).item(Bool.self))
        }
    }

    @Test("Through the engine a warm follow-up turn with --mtp leases the stored prefix, still drafts, and answers like a cold full prefill")
    func warmTurnReusesPrefix() async throws {
        func engine() async throws -> MLXInferenceEngine {
            let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 1 << 30, mtp: true))
            let container = try TinyModel.container(.qwen35)
            await engine.attach(container)
            await engine.useMTPDrafter(in: container) { _ in try? GreedyDraftingMTPDrafter(Self.head()) }
            return engine
        }
        let warm = try await engine()
        let first = try await warm.generate(request: greedy(Conversation.firstTurn, key: "ns", maxTokens: 24))
        let reused = try await warm.generate(request: greedy(Conversation.afterToolCall, key: "ns", maxTokens: 24))
        let cold = try await engine().generate(request: greedy(Conversation.afterToolCall, key: nil, maxTokens: 24))
        #expect(first.cachedPromptTokens == 0)
        #expect(reused.cachedPromptTokens > 0)
        #expect((reused.draftTokens ?? 0) > 0)
        #expect(cold.cachedPromptTokens == 0)
        #expect(reused.generatedText == cold.generatedText)
        #expect(reused.acceptedDraftTokens == cold.acceptedDraftTokens)
        let stored = try #require(await warm.mtpPrefixes.last)
        #expect(stored.namespace == "ns")
        #expect(stored.tokens.count > reused.cachedPromptTokens)
        #expect(stored.prefix.nextPosition == stored.tokens.count - 1)
    }

    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_MTP_MODEL_PATH"]
    }

    @Test("On a real MTP model a warm follow-up turn prefills only its tail: lower TTFT than a cold full prefill, the same greedy text",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_MTP_MODEL_PATH"))
    func warmFollowUpOnRealModel() async throws {
        let engine = MLXInferenceEngine(EngineConfiguration(mtp: true))
        try await engine.load(modelPath: try #require(Self.modelPath))
        let code = try String(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "benchmarks/speculative-lookup/code.txt"), encoding: .utf8)
        let second = code + "\n\n[Assistant]: Applied the rename.\n\n[Tool result]: build ok, 12 tests passed. Now add a doc line for limit."
        func request(_ prompt: String, key: String?) -> GenerationRequest {
            GenerationRequest(prompt: prompt, maxTokens: 64, temperature: 0).with { $0.reasoningEffort = "none"; $0.cacheKey = key }
        }
        _ = try await engine.generate(request: request(code, key: "e2e"))
        let warm = try await engine.generate(request: request(second, key: "e2e"))
        let cold = try await engine.generate(request: request(second, key: nil))
        print("MTPPromptCache prompt=\(warm.promptTokens) warm: cached=\(warm.cachedPromptTokens) ttft=\(warm.timeToFirstToken) drafts=\(warm.acceptedDraftTokens ?? 0)/\(warm.draftTokens ?? 0) tps=\(warm.tokensPerSecond) | cold: cached=\(cold.cachedPromptTokens) ttft=\(cold.timeToFirstToken) drafts=\(cold.acceptedDraftTokens ?? 0)/\(cold.draftTokens ?? 0) tps=\(cold.tokensPerSecond)")
        #expect(warm.cachedPromptTokens > 0)
        #expect(cold.cachedPromptTokens == 0)
        #expect((warm.draftTokens ?? 0) > 0)
        #expect(warm.timeToFirstToken < cold.timeToFirstToken)
        #expect(warm.generatedText == cold.generatedText)
    }
}
