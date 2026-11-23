import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Long prompts in a row on one engine: which pool keeps memory between requests (#741 P0)", .serialized)
struct LongPromptMemoryRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static var modelPath: String? { env["BASALT_E2E"] == "1" ? env["BASALT_E2E_LONG_MEMORY_MODEL_PATH"] : nil }
    static var tokens: Int { Int(env["BASALT_E2E_LONG_MEMORY_TOKENS"] ?? "") ?? 32_768 }
    static let filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

    static func prompt(key: Int, depth: Double) -> String {
        var lines = Array(repeating: filler, count: tokens / 22)
        lines.insert("The pass key is \(key). Remember it. \(key) is the pass key.", at: Int(Double(lines.count) * depth))
        return lines.joined(separator: "\n") + "\nWhat is the pass key? Reply with the number only."
    }

    static func megabytes(_ bytes: Int) -> Int { bytes / 1_048_576 }

    @Test("Three long prompts under the shared cache key, with the default prompt cache and with none: active, cache and peak memory before and after each",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_LONG_MEMORY_MODEL_PATH"), arguments: ["default", "no-prompt-cache"])
    func longPromptsInARow(variant: String) async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("long-prompt-memory-\(variant)", modelPath: path)
        var configuration = EngineConfiguration()
        if variant == "no-prompt-cache" { configuration.promptCacheBytes = 0 }
        let engine = MLXInferenceEngine(configuration)
        try await engine.load(modelPath: path)
        log.record("loaded", "active_mb=\(Self.megabytes(Memory.activeMemory)) cache_mb=\(Self.megabytes(Memory.cacheMemory)) tokens=\(Self.tokens)")
        var found = 0
        for (round, depth) in [0.1, 0.5, 0.9].enumerated() {
            let key = 100_000 + round * 271_828
            let before = (Self.megabytes(Memory.activeMemory), Self.megabytes(Memory.cacheMemory))
            Memory.peakMemory = 0
            let response = try await log.step("\(variant) round \(round + 1)") {
                try await engine.generate(request: GenerationRequest(prompt: Self.prompt(key: key, depth: depth), maxTokens: 32, temperature: 0).withCacheKey("shared"))
            }
            let ok = response.generatedText.contains(String(key))
            found += ok ? 1 : 0
            log.record("result", "\(variant) round=\(round + 1) prompt_tokens=\(response.promptTokens) cached=\(response.cachedPromptTokens) ok=\(ok) before_active_mb=\(before.0) before_cache_mb=\(before.1) after_active_mb=\(Self.megabytes(Memory.activeMemory)) after_cache_mb=\(Self.megabytes(Memory.cacheMemory)) peak_mb=\(Self.megabytes(Memory.peakMemory))")
        }
        #expect(found == 3)
    }
}
