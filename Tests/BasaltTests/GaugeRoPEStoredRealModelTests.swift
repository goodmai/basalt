import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("GaugeRoPE through the shipped cache on real weights: perplexity with the stored codec, the per-request codec and exact K/V (#647)", .serialized)
struct GaugeRoPEStoredRealModelTests {
    static var env: [String: String] { ProcessInfo.processInfo.environment }
    static var enabled: Bool { env["BASALT_E2E"] == "1" && env["BASALT_E2E_GAUGEROPE_STORED_PATH"] != nil && env["BASALT_E2E_KV_CORPUS"] != nil }
    static var context: Int { Int(env["BASALT_E2E_GAUGEROPE_CONTEXT"] ?? "") ?? 16384 }

    static func perplexity(_ tokens: [Int], model: any LanguageModel, cache: [KVCache]) -> Double {
        var (nll, scored) = (0.0, 0)
        let half = tokens.count / 2
        for start in stride(from: 0, to: tokens.count - 1, by: 512) {
            let end = min(tokens.count - 1, start + 512)
            let logits = model(MLXArray(tokens[start ..< end].map(Int32.init))[.newAxis], cache: cache)
            let first = max(start, half)
            if first < end {
                let rows = logits[0, (first - start) ..< (end - start)].asType(.float32)
                let logp = rows - logSumExp(rows, axis: -1, keepDims: true)
                let targets = MLXArray(tokens[(first + 1) ... end].map(Int32.init))[0..., .newAxis]
                nll -= Double(takeAlong(logp, targets, axis: -1).sum().item(Float.self))
                scored += end - first
            }
            eval(cache)
        }
        return exp(nll / Double(max(scored, 1)))
    }

    @Test("Second half of a long window: exact K/V, the stored codec from token 0, and the codec calibrated on the request's first 512 tokens",
          .enabled(if: enabled, "set BASALT_E2E=1, BASALT_E2E_GAUGEROPE_STORED_PATH=<dir converted with --to gaugerope> and BASALT_E2E_KV_CORPUS"))
    func storedAgainstOnline() async throws {
        let directory = URL(fileURLWithPath: try #require(Self.env["BASALT_E2E_GAUGEROPE_STORED_PATH"]))
        let text = try String(contentsOfFile: try #require(Self.env["BASALT_E2E_KV_CORPUS"]), encoding: .utf8)
        let log = try await ModelRunLog("gaugerope-stored", modelPath: directory.path)
        defer { log.releaseGPU() }
        let config = try #require(ModelCache.config(at: directory))
        let layout = try #require(GaugeRoPELayout(config: config))
        let loaded = try GaugeRoPECodecStore.load(
            from: directory, layout: layout, fingerprint: GaugeRoPECodecStore.fingerprint(modelDirectory: directory))
        let stored = try #require(loaded)
        await MLXInferenceEngine.registerAliases()
        let container = try await loadModelContainer(from: directory, using: DefaultTokenizerLoader())
        let lines = try await container.perform { context -> [String] in
            let tokens = Array(context.tokenizer.encode(text: text, addSpecialTokens: false).prefix(Self.context))
            #expect(tokens.count == Self.context)
            let arms: [(String, () throws -> [KVCache])] = [
                ("fp16", { try context.model.newCache(parameters: nil) }),
                ("gaugerope-stored", { GaugeRoPEKVCache.replacingFullAttention(try context.model.newCache(parameters: nil), layout: layout, codecs: stored) }),
                ("gaugerope-online", { GaugeRoPEKVCache.replacingFullAttention(try context.model.newCache(parameters: nil), layout: layout, codecs: [:]) }),
            ]
            return try arms.map { name, make in
                Memory.clearCache()
                Memory.peakMemory = 0
                let ppl = Self.perplexity(tokens, model: context.model, cache: try make())
                return "arm=\(name) ppl=\(String(format: "%.4f", ppl)) context=\(tokens.count) peakGB=\(String(format: "%.2f", Double(Memory.peakMemory) / 1e9))"
            }
        }
        for line in lines { log.record("result", line) }
    }
}
