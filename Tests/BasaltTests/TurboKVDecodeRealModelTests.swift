import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltInference

@Suite("TurboQuant KV on real models (#471, experimental): decode perplexity over 384 tokens after 8k, where the compressed keys are read; chunked prefill reads raw keys") struct TurboKVDecodeRealModelTests {
    @Test("fp16, affine q4, turbo3 and turbo4: decode perplexity, decode speed and the cache kinds", .enabled(if: ProcessInfo.processInfo.environment["BASALT_E2E"] == "1" && ProcessInfo.processInfo.environment["BASALT_E2E_TURBO_MODEL"] != nil, "set BASALT_E2E=1, BASALT_E2E_TURBO_MODEL and BASALT_ZONED_CORPUS; run a release build"))
    func decodePerplexity() async throws {
        let corpus = try String(contentsOfFile: ProcessInfo.processInfo.environment["BASALT_ZONED_CORPUS"]!, encoding: .utf8)
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        try await engine.load(modelPath: NF4Run.path(ProcessInfo.processInfo.environment["BASALT_E2E_TURBO_MODEL"]!))
        let container = try #require(await engine.container)
        let lines = try await container.perform { context -> [String] in
            let tokens = Array(context.tokenizer.encode(text: corpus).prefix(8192 + 385)).map(Int32.init)
            var lines: [String] = []
            for (label, bits, scheme) in [("fp16", nil, nil), ("affine-q4", 4, nil), ("turbo3", nil, "turbo3"), ("turbo4", nil, "turbo4")] as [(String, Int?, String?)] {
                var cache = try context.model.newCache(parameters: nil)
                for start in stride(from: 0, to: 8192, by: 512) {
                    _ = context.model(MLXArray(Array(tokens[start ..< start + 512]))[.newAxis], cache: cache)
                    eval(cache)
                }
                maybeQuantizeKVCache(cache: &cache, kvBits: bits, quantizedKVStart: 0, kvScheme: scheme)
                var nll = 0.0
                let started = Date()
                for position in 8192 ..< 8192 + 384 {
                    let logits = context.model(MLXArray([tokens[position]])[.newAxis], cache: cache)[0, -1].asType(.float32)
                    let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
                    nll -= Double(logp[Int(tokens[position + 1])].item(Float.self))
                    maybeQuantizeKVCache(cache: &cache, kvBits: bits, quantizedKVStart: 0, kvScheme: scheme)
                }
                var kinds: [String: Int] = [:]
                for layer in cache { kinds[String(describing: type(of: layer)), default: 0] += 1 }
                lines.append(String(format: "%@: decode perplexity over 384 tokens after 8192 = %.4f, %.1f tok/s, caches %@", label, exp(nll / 384), 384 / Date().timeIntervalSince(started), kinds.description))
            }
            return lines
        }
        for line in lines { print("RESULT \(line)") }
    }
}
