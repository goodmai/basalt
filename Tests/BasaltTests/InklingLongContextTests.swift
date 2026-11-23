import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Inkling 2 P2c: the prefill chunk shrinks with the key count so the relative-bias mask fits, and 32K passkey runs (#650)", .serialized)
struct InklingLongContextTests {
    static let env = ProcessInfo.processInfo.environment
    static var modelPath: String? { env["BASALT_E2E"] == "1" ? env["BASALT_E2E_INKLING_MODEL_PATH"] : nil }
    static let filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

    @Test("The cap leaves 4K prompts on the default 512-token chunk, cuts 32K to 85 rows, never goes under 64, and only applies to Inkling 2")
    func stepCap() throws {
        let model = try #require(try TinyModel.make(.inkling2Loop) as? Inkling2Model)
        let heads = model.config.heads
        #expect(model.prefillStepCap(keys: 4096 * 24 / heads) == nil)
        #expect(model.prefillStepCap(keys: 1 << 30) == Inkling2Model.minimumPrefillStep)
        let keys = 32_768 * 24 / heads
        #expect(model.prefillStepCap(keys: keys) == Inkling2Model.maskBudgetBytes / (heads * keys * 4))
        let parameters = GenerateParameters()
        #expect(parameters.capped(for: model, promptTokens: keys).prefill.stepSize == model.prefillStepCap(keys: keys))
        #expect(parameters.capped(for: model, promptTokens: 1024).prefill.stepSize == parameters.prefill.stepSize)
        #expect(parameters.capped(for: try TinyModel.make(.qwen35), promptTokens: keys).prefill.stepSize == parameters.prefill.stepSize)
    }

    @Test("A 300-token prompt prefilled in 64-row chunks gives the last logits of a single forward within 1e-3")
    func cappedChunksMatch() throws {
        let model = try TinyModel.make(.inkling2Loop)
        let tokens = MLXArray((0 ..< 300).map { Int32(($0 * 37 + 11) % 250) })[.newAxis]
        let whole = model(tokens, cache: try model.newCache(parameters: nil))[0..., (-1)..., 0...]
        let cache = try model.newCache(parameters: nil)
        var last = whole
        for start in stride(from: 0, to: 300, by: Inkling2Model.minimumPrefillStep) {
            last = model(tokens[0..., start ..< min(start + Inkling2Model.minimumPrefillStep, 300)], cache: cache)[0..., (-1)..., 0...]
        }
        #expect(abs(last - whole).max().item(Float.self) < 1e-3)
    }

    @Test("32K passkey at five depths on the real model, with the peak memory recorded (acceptance for P2c)",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_INKLING_MODEL_PATH"))
    func passkey32K() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("inkling-passkey", modelPath: path)
        let engine = MLXInferenceEngine()
        try await engine.load(modelPath: path)
        var found = 0
        for (index, depth) in [0.1, 0.3, 0.5, 0.7, 0.9].enumerated() {
            let key = 100_000 + (index * 137_911 + 32_768) % 900_000
            var lines = Array(repeating: Self.filler, count: 1_500)
            lines.insert("The pass key is \(key). Remember it. \(key) is the pass key.", at: Int(Double(lines.count) * depth))
            let prompt = lines.joined(separator: "\n") + "\nWhat is the pass key? Reply with the number only."
            Memory.peakMemory = 0
            let response = try await log.step("passkey depth=\(depth)") {
                try await engine.generate(request: GenerationRequest(prompt: prompt, maxTokens: 32, temperature: 0))
            }
            let ok = response.generatedText.contains(String(key))
            found += ok ? 1 : 0
            log.record("result", "depth=\(depth) prompt_tokens=\(response.promptTokens) ok=\(ok) peak_mb=\(Memory.peakMemory / 1_048_576) answer=\(response.generatedText.prefix(40).debugDescription)")
        }
        log.record("result", "passkey 32K \(found)/5")
        #expect(found == 5)
    }
}
