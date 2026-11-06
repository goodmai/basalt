import Foundation
import Testing
import MLX
@testable import BasaltCore
@testable import BasaltInference

@Suite("Expert streaming on a real MoE that fits resident, forced to a small expert cache (#225)", .serialized)
struct ExpertStreamingRealModelTests {
    static var env: [String: String] { ProcessInfo.processInfo.environment }

    static var models: [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return ["BASALT_E2E_LFM25_MODEL_PATH", "BASALT_E2E_GEMMA4_MODEL_PATH"].compactMap { env[$0] }.map(GoldenLogitsRealModelTests.path)
    }

    static var cacheSizes: [Double] {
        (env["BASALT_E2E_EXPERT_CACHE_GB"] ?? "1,4").split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
    }

    static var tokens: Int { Int(env["BASALT_E2E_EXPERT_STREAMING_TOKENS"] ?? "") ?? 128 }

    static let prompt = "Explain step by step how a mixture-of-experts transformer routes each token to its experts, then write a short Swift function that picks the top-k entries of an array."

    struct Run {
        let response: GenerationResponse
        let peakBytes: Int
        let stats: (stats: ExpertCacheStats, capacity: ByteCount)?
        let logits: [Float]
    }

    static func lastLogits(_ engine: MLXInferenceEngine) async throws -> [Float] {
        let container = try #require(await engine.container)
        return try await container.performEvaluating(nonSendable: ()) { context, _ in
            let tokens = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
            let logits = context.model(MLXArray(tokens.map(Int32.init)).reshaped(1, -1), cache: try context.model.newCache(parameters: nil))
            return logits[0, -1].asType(.float32).asArray(Float.self)
        }
    }

    static func comparison(_ streamed: [Float], _ resident: [Float]) -> String {
        func top(_ logits: [Float]) -> [Int] { Array(logits.indices.sorted { logits[$0] > logits[$1] }.prefix(5)) }
        let drift = zip(streamed, resident).map { abs($0 - $1) }.max() ?? .infinity
        let shared = Set(top(streamed)).intersection(top(resident)).count
        return String(format: "prefill last-position logits max |diff| %g, top-5 overlap %d/5, argmax %@", drift, shared, top(streamed).first == top(resident).first ? "equal" : "differs")
    }

    static func run(_ path: String, streaming: ExpertStreaming?, log: ModelRunLog, step: String) async throws -> Run {
        Memory.clearCache()
        Memory.peakMemory = 0
        let engine = MLXInferenceEngine()
        await engine.streamExperts(streaming)
        try await log.step("\(step) load") { try await engine.load(modelPath: path) }
        let response = try await log.step("\(step) generate") {
            try await engine.generate(request: GenerationRequest(prompt: prompt, maxTokens: tokens, temperature: 0))
        }
        let logits = try await log.step("\(step) logits") { try await lastLogits(engine) }
        let run = Run(response: response, peakBytes: Memory.peakMemory, stats: await engine.expertCacheStats(), logits: logits)
        await engine.unloadCurrentModel()
        return run
    }

    @Test("Greedy identity against resident, decode tok/s, expert hit rate and peak memory per forced expert-cache size",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_LFM25_MODEL_PATH and/or BASALT_E2E_GEMMA4_MODEL_PATH (optional BASALT_E2E_EXPERT_CACHE_GB=1,4 and BASALT_E2E_EXPERT_STREAMING_TOKENS=128)"),
          arguments: models)
    func streamingAgainstResident(path: String) async throws {
        let log = try await ModelRunLog("expert-streaming", modelPath: path)
        defer { log.releaseGPU() }
        let resident = try await Self.run(path, streaming: nil, log: log, step: "resident")
        log.record("resident", String(format: "decode %.2f tok/s, %d tokens, peak %d MB", resident.response.tokensPerSecond,
                                      resident.response.completionTokens, resident.peakBytes >> 20))
        var mismatches = 0
        for gigabytes in Self.cacheSizes {
            let streaming = ExpertStreaming(cacheBytes: ByteCount(Int64(gigabytes * 1_073_741_824)))
            let streamed = try await Self.run(path, streaming: streaming, log: log, step: "cache \(gigabytes) GB")
            let identical = streamed.response.generatedText == resident.response.generatedText
            if !identical { mismatches += 1 }
            log.record(identical ? "cache \(gigabytes) GB" : "FAIL cache \(gigabytes) GB", String(
                format: "greedy %@, decode %.2f tok/s (resident %.2f), ttft %.2f s, peak %d MB (resident %d MB), %@",
                identical ? "identical" : "differs", streamed.response.tokensPerSecond, resident.response.tokensPerSecond,
                streamed.response.timeToFirstToken, streamed.peakBytes >> 20, resident.peakBytes >> 20, streamed.stats?.stats.summary ?? "no expert cache"))
            log.record("cache \(gigabytes) GB logits", Self.comparison(streamed.logits, resident.logits))
            if !identical {
                log.record("diff", "resident: \(resident.response.generatedText.debugDescription) streamed: \(streamed.response.generatedText.debugDescription)")
            }
            #expect(streamed.stats != nil)
        }
        #expect(mismatches == 0)
    }
}
