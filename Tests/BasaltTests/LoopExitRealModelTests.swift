import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("--experimental-loop-exit on real looped models: perplexity, passes per token and decode speed against the threshold (#820)", .serialized)
struct LoopExitRealModelTests {
    struct Measurement {
        let perplexity: Double
        let passesPerToken: Double
        let tokensPerSecond: Double
        let prefillActiveBytes: Int
        let peakBytes: Int
    }

    static var env: [String: String] { ProcessInfo.processInfo.environment }
    static var models: [String] { (env["BASALT_E2E_LOOP_MODELS"] ?? "").split(separator: ",").map(String.init) }
    static var enabled: Bool { env["BASALT_E2E"] == "1" && !models.isEmpty && env["BASALT_E2E_LOOP_CORPUS"] != nil }
    static var thresholds: [Float?] { [nil] + (env["BASALT_E2E_LOOP_THRESHOLDS"] ?? "0.02,0.05,0.1,0.3,0.6,0.9").split(separator: ",").compactMap { Float($0) } }
    static let scored = 512
    static let prefix = 1536

    static func setThreshold(_ threshold: Float?, on model: any LanguageModel) -> () -> (steps: Int, passes: Int) {
        if let ouro = model as? OuroModel {
            ouro.model.loopExitThreshold = threshold
            let base = (ouro.model.exitSteps, ouro.model.exitPasses)
            return { (ouro.model.exitSteps - base.0, ouro.model.exitPasses - base.1) }
        }
        if let inkling = model as? Inkling2Model {
            let text = inkling.model.languageModel
            text.loopExitThreshold = threshold
            let base = (text.exitSteps, text.exitPasses)
            return { (text.exitSteps - base.0, text.exitPasses - base.1) }
        }
        return { (0, 0) }
    }

    static func measure(_ tokens: [Int], threshold: Float?, context: ModelContext) throws -> Measurement {
        let counters = setThreshold(threshold, on: context.model)
        Memory.peakMemory = 0
        let cache = try context.model.newCache(parameters: nil)
        let prefix = tokens.count - scored - 1
        for start in stride(from: 0, to: prefix, by: 256) {
            _ = context.model(MLXArray(Array(tokens[start ..< min(start + 256, prefix)]))[.newAxis], cache: cache)
            eval(cache)
        }
        let prefillActive = Memory.activeMemory
        var nll = 0.0
        let start = ContinuousClock.now
        for index in prefix ..< prefix + scored {
            let logits = context.model(MLXArray([Int32(tokens[index])])[.newAxis], cache: cache)[0, 0].asType(.float32)
            let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
            nll -= Double(logp[tokens[index + 1]].item(Float.self))
        }
        let seconds = (ContinuousClock.now - start).inSeconds
        let (steps, passes) = counters()
        _ = setThreshold(nil, on: context.model)
        Memory.clearCache()
        return Measurement(perplexity: exp(nll / Double(scored)), passesPerToken: steps > 0 ? Double(passes) / Double(steps) : .nan, tokensPerSecond: Double(scored) / max(seconds, 1e-9), prefillActiveBytes: prefillActive, peakBytes: Memory.peakMemory)
    }

    @Test("Each threshold against the fixed-pass model on the same text: teacher-forced one-token perplexity over 512 tokens, mean passes per token and tokens per second",
          .enabled(if: enabled, "set BASALT_E2E=1, BASALT_E2E_LOOP_MODELS=<repo,…> and BASALT_E2E_LOOP_CORPUS=<text file>"), arguments: models)
    func thresholdSweep(entry: String) async throws {
        let path = GoldenLogitsRealModelTests.path(entry)
        let text = try String(contentsOfFile: try #require(Self.env["BASALT_E2E_LOOP_CORPUS"]), encoding: .utf8)
        let log = try await ModelRunLog("loop-exit", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(gpuCacheLimit: 0))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let lines = try await container.perform { context -> [String] in
            let tokens = Array(context.tokenizer.encode(text: text).prefix(Self.prefix + Self.scored + 1))
            #expect(tokens.count > Self.scored + 1)
            return try Self.thresholds.map { threshold in
                let m = try Self.measure(tokens, threshold: threshold, context: context)
                return "threshold=\(threshold.map { String($0) } ?? "off") ppl=\(String(format: "%.4f", m.perplexity)) passes/token=\(String(format: "%.3f", m.passesPerToken)) tok/s=\(String(format: "%.2f", m.tokensPerSecond)) prefillActiveGB=\(String(format: "%.2f", Double(m.prefillActiveBytes) / 1e9)) peakGB=\(String(format: "%.2f", Double(m.peakBytes) / 1e9))"
            }
        }
        for line in lines { log.record("result", line) }
    }
}
