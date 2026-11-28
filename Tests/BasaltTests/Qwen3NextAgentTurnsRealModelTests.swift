import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Qwen3-Coder-Next with streamed experts on a real checkpoint: the second agent turn prefills only the appended tokens, and 256 greedy tokens are identical across expert arenas (#642 P2, P3)", .serialized)
struct Qwen3NextAgentTurnsRealModelTests {
    static var env: [String: String] { ExpertStreamingRealModelTests.env }

    static var modelPath: String? {
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_QWEN3NEXT_MODEL_PATH"].map(GoldenLogitsRealModelTests.path)
    }

    static func number(_ key: String, _ fallback: Double) -> Double { Double(env[key] ?? "") ?? fallback }

    static func bytes(_ gigabytes: Double) -> ByteCount { ByteCount(Int64(gigabytes * 1_073_741_824)) }

    static var streaming: ExpertStreaming {
        ExpertStreaming(cacheBytes: env["BASALT_E2E_EXPERT_CACHE_GB"].flatMap { Double($0) }.map(bytes))
    }

    static var arenas: [Double] {
        (env["BASALT_E2E_QWEN3NEXT_ARENAS_GB"] ?? "6,10").split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
    }

    static let longPrompt = "Write a complete Swift implementation of an LRU cache backed by a dictionary and a doubly linked list, with get, set, remove and a capacity limit. Then write at least ten unit tests for it, one per function, each with a short explanation. Do not stop before every test is written."

    static func request(_ messages: [ChatMessageInput], key: String?) -> GenerationRequest {
        GenerationRequest(
            prompt: "", maxTokens: Int(number("BASALT_E2E_QWEN3NEXT_TOKENS", 64)), temperature: 0,
            tools: [GoldenPrompt.readFile], messages: messages).withCacheKey(key)
    }

    static func describe(_ response: GenerationResponse, peak: Int, swap: UInt64, stats: ExpertCacheStats?) -> String {
        String(
            format: "prompt %d, from prefix cache %d, prefilled %d, completion %d, ttft %.2f s, decode %.2f tok/s, peak %d MB, swap %llu, %@",
            response.promptTokens, response.cachedPromptTokens, response.promptTokens - response.cachedPromptTokens,
            response.completionTokens, response.timeToFirstToken, response.tokensPerSecond, peak >> 20, swap,
            stats?.summary ?? "no expert cache")
    }

    static func recurrentState(_ tree: PromptCacheTree) -> String {
        guard let snapshot = tree.leaves().compactMap(\.snapshot).first else { return "no recurrent snapshot held" }
        let layers = snapshot.layers.keys.sorted().compactMap { snapshot.layers[$0] as? MambaCache }
        let parts = [("conv", layers.first.flatMap { $0[0] }), ("recurrent", layers.first.flatMap { $0[1] })].map { name, array in
            "\(name) \(array.map { "\($0.dtype) \($0.shape)" } ?? "missing")"
        }
        return "GDN state of \(layers.count) layers, \(parts.joined(separator: ", ")) per layer, \(snapshot.bytes.rawValue) bytes per sequence"
    }

    static func turn(_ engine: MLXInferenceEngine, _ request: GenerationRequest, step: String, log: ModelRunLog) async throws -> GenerationResponse {
        Memory.peakMemory = 0
        let swapBefore = MachMemory.swapOuts()
        if let container = await engine.container {
            await container.perform {
                guard let store = ExpertStreamingLoader.store(in: $0.model) else { return }
                store.resetStats()
            }
        }
        let response = try await log.step(step) { try await engine.generate(request: request) }
        let swapAfter = MachMemory.swapOuts()
        let stats = await engine.expertCacheStats()
        log.record(step, describe(
            response, peak: Memory.peakMemory, swap: swapAfter >= swapBefore ? swapAfter - swapBefore : 0, stats: stats?.stats))
        return response
    }

    @Test("Turn 2 of an agent trace with a tool reuses turn 1's prefix: time to first token at most 10% of turn 1's, under 25% of its prompt prefilled, and the text of a fresh engine without a cache",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_QWEN3NEXT_MODEL_PATH"))
    func secondAgentTurnPrefillsOnlyTheAppendedTokens() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("qwen3next-agent-turns", modelPath: path)
        defer { log.releaseGPU() }
        let first = GoldenPrompt.agentTrace(turns: Int(Self.number("BASALT_E2E_QWEN3NEXT_TURNS", 16)))
        Memory.clearCache()
        let engine = MLXInferenceEngine(EngineConfiguration(expertStreaming: Self.streaming))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let turn1 = try await Self.turn(engine, Self.request(first, key: "agent"), step: "turn 1", log: log)
        let anchors = await engine.hybridAnchorUsage
        let state = await engine.readPromptCaches(Self.recurrentState)
        log.record("turn 1 state", "prefix anchors \(anchors?.count ?? 0) holding \(anchors?.bytes ?? 0) bytes, \(state)")

        let second = first + [
            ChatMessageInput(role: .assistant, text: turn1.generatedText, toolCalls: turn1.toolCalls),
            GoldenPrompt.user("Now name the function in that file that applies the budget. Answer in one sentence."),
        ]
        let turn2 = try await Self.turn(engine, Self.request(second, key: "agent"), step: "turn 2", log: log)
        await engine.unloadCurrentModel()
        let cold = try await ExpertStreamingRealModelTests.run(
            path, spec: .init(streaming: Self.streaming, step: "turn 2 cold", request: Self.request(second, key: nil), computeLogits: false), log: log)
        log.record("turn 2 cold", Self.describe(cold.response, peak: cold.peakBytes, swap: cold.swapDelta, stats: cold.stats?.stats))

        let ratio = Self.number("BASALT_E2E_QWEN3NEXT_TTFT_RATIO", 0.10)
        let prefilled = turn2.promptTokens - turn2.cachedPromptTokens
        let identical = cold.response.generatedText == turn2.generatedText
        log.record(identical ? "gate" : "FAIL gate", String(
            format: "ttft turn 2 / turn 1 %.3f (limit %.3f), prefilled %d of %d prompt tokens, cached text %@ the cold text",
            turn2.timeToFirstToken / max(turn1.timeToFirstToken, 0.001), ratio, prefilled, turn2.promptTokens, identical ? "equals" : "differs from"))
        if !identical {
            log.record("diff", "cached: \(turn2.generatedText.debugDescription) cold: \(cold.response.generatedText.debugDescription)")
        }
        #expect(turn2.timeToFirstToken <= ratio * turn1.timeToFirstToken, "ttft \(turn2.timeToFirstToken) s against \(turn1.timeToFirstToken) s")
        #expect(Double(prefilled) < 0.25 * Double(turn2.promptTokens), "prefilled \(prefilled) of \(turn2.promptTokens)")
        #expect(identical)
    }

    @Test("A plain coding prompt decodes 256 greedy tokens that are identical with expert arenas of 6 and 10 GiB",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_QWEN3NEXT_MODEL_PATH"))
    func longDecodeIsIdenticalAcrossArenas() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("qwen3next-arenas", modelPath: path)
        defer { log.releaseGPU() }
        let request = GenerationRequest(prompt: "", maxTokens: 256, temperature: 0, messages: [GoldenPrompt.user(Self.longPrompt)])
        var texts: [String] = []
        for gigabytes in Self.arenas {
            let run = try await ExpertStreamingRealModelTests.run(
                path, spec: .init(streaming: ExpertStreaming(cacheBytes: Self.bytes(gigabytes)), step: "arena \(gigabytes) GiB", request: request, computeLogits: false),
                log: log)
            log.record("arena \(gigabytes) GiB", Self.describe(run.response, peak: run.peakBytes, swap: run.swapDelta, stats: run.stats?.stats))
            #expect(run.response.completionTokens == 256, "arena \(gigabytes) GiB wrote \(run.response.completionTokens) tokens")
            texts.append(run.response.generatedText)
        }
        #expect(texts.count >= 2)
        #expect(texts.allSatisfy { $0 == texts.first })
    }
}
