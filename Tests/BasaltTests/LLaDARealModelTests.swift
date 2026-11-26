import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("LLaDA-8B under --experimental-dllm on real weights: wall-clock tokens per second and the answers (#819)", .serialized)
struct LLaDARealModelTests {
    static var env: [String: String] { ProcessInfo.processInfo.environment }
    static var path: String? { env["BASALT_E2E"] == "1" ? env["BASALT_E2E_LLADA_MODEL_PATH"] : nil }
    static let prompts = [
        "What is 17 times 23? Answer with the number.",
        "Write a Python function that returns the n-th Fibonacci number.",
        "Name three planets of the solar system.",
        "Translate to French: The library opens at nine.",
        "Explain in two sentences why the sky is blue.",
    ]

    @Test("Each prompt generates through Fast-dLLM: completion tokens, wall seconds, wall tokens per second and the text, logged per prompt",
          .enabled(if: path != nil, "set BASALT_E2E=1 and BASALT_E2E_LLADA_MODEL_PATH=<repo or path>"))
    func generates() async throws {
        let model = GoldenLogitsRealModelTests.path(try #require(Self.path))
        let log = try await ModelRunLog("llada-dllm", modelPath: model)
        defer { log.releaseGPU() }
        var configuration = EngineConfiguration(gpuCacheLimit: 0)
        configuration.dllm = true
        let engine = MLXInferenceEngine(configuration)
        try await log.step("load") { try await engine.load(modelPath: model) }
        var totalTokens = 0
        var totalSeconds = 0.0
        for prompt in Self.prompts {
            let request = GenerationRequest(prompt: prompt, maxTokens: 128, temperature: 0)
            let start = ContinuousClock.now
            let reply = try await log.step("generate") { try await engine.generate(request: request) }
            let seconds = (ContinuousClock.now - start).inSeconds
            totalTokens += reply.completionTokens
            totalSeconds += seconds
            log.record("result", "tokens=\(reply.completionTokens) seconds=\(String(format: "%.2f", seconds)) tok/s=\(String(format: "%.2f", Double(reply.completionTokens) / max(seconds, 1e-9))) prompt=\(prompt.prefix(40)) text=\(reply.generatedText.replacingOccurrences(of: "\n", with: " ").prefix(240))")
            #expect(reply.completionTokens > 0)
        }
        log.record("summary", "prompts=\(Self.prompts.count) tokens=\(totalTokens) wall tok/s=\(String(format: "%.2f", Double(totalTokens) / max(totalSeconds, 1e-9)))")
    }
}
