import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Polar keys on real models (#351, experimental): decode perplexity, speed and KV bytes against zoned affine, and a 16k passkey through serve's path", .serialized)
struct PolarKeysRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static let models = env["BASALT_E2E"] == "1" ? (env["BASALT_E2E_POLAR_SERVE_MODELS"] ?? "").split(separator: ",").map(String.init) : []

    static func decode(_ context: ModelContext, tokens: [Int32], precision: KVPrecision) throws -> (perplexity: Double, tokensPerSecond: Double, bytes: Int) {
        let cache = ZonedKVCache.replacingFullAttention(try context.model.newCache(parameters: nil), zones: .standard, precision: precision)
        for start in stride(from: 0, to: 8192, by: 512) {
            _ = context.model(MLXArray(Array(tokens[start ..< start + 512]))[.newAxis], cache: cache)
            eval(cache)
        }
        var nll = 0.0
        let started = Date()
        for position in 8192 ..< 8448 {
            let logits = context.model(MLXArray([tokens[position]])[.newAxis], cache: cache)[0, -1].asType(.float32)
            nll -= Double((logits - logSumExp(logits, axis: -1, keepDims: true))[Int(tokens[position + 1])].item(Float.self))
        }
        let seconds = Date().timeIntervalSince(started)
        let bytes = cache.compactMap { $0 as? ZonedKVCache }.reduce(0) { $0 + $1.innerState().reduce(0) { $0 + $1.nbytes } }
        return (exp(nll / 256), 256 / seconds, bytes)
    }

    static func passkey(_ engine: MLXInferenceEngine, corpus: String, depth: Double, key: Int) async throws -> Bool {
        let container = try #require(await engine.container)
        let haystack = await container.perform { context in
            context.tokenizer.decode(tokenIds: Array(context.tokenizer.encode(text: corpus).prefix(15_800)))
        }
        let cut = haystack.index(haystack.startIndex, offsetBy: Int(Double(haystack.count) * depth))
        let prompt = String(haystack[..<cut]) + "\nThe pass key is \(key). Remember it.\n" + String(haystack[cut...]) + "\n\nWhat is the pass key? Answer with the number only."
        let reply = try await engine.generate(request: GenerationRequest(prompt: prompt, maxTokens: 600, temperature: 0))
        return (reply.generatedText + (reply.reasoningText ?? "")).contains(String(key))
    }

    @Test("Zoned polar4 against zoned q4 and k8v4: decode perplexity over 256 tokens after 8k, decode speed and KV bytes; passkey at 16k (depths 0.2 and 0.8) through serve's configuration",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1, BASALT_E2E_POLAR_SERVE_MODELS and BASALT_ZONED_CORPUS; run a release build"), arguments: models)
    func measure(model: String) async throws {
        let path = NF4Run.path(model)
        let corpus = try String(contentsOfFile: try #require(Self.env["BASALT_ZONED_CORPUS"]), encoding: .utf8)
        let log = try await ModelRunLog("polar-keys-serve", modelPath: path)
        defer { log.releaseGPU() }
        var configuration = EngineConfiguration(promptCacheBytes: 0)
        (configuration.kvPrecision, configuration.kvZones) = (.polar4, .standard)
        let engine = MLXInferenceEngine(configuration)
        try await log.step("load") { try await engine.load(modelPath: path) }
        let polar = try #require(await engine.effectiveKVPrecision)
        log.record("geometry", "polar rotated dims \(polar.polarRotated ?? 0), zones apply \(await engine.modelAttendsThroughZones)")
        let container = try #require(await engine.container)
        let lines = try await container.perform { context -> [String] in
            let tokens = Array(context.tokenizer.encode(text: corpus).prefix(8449)).map(Int32.init)
            return try [("zoned-q4", KVPrecision.q4), ("zoned-k8v4", .k8v4), ("zoned-polar4", polar)].map { label, precision in
                let result = try Self.decode(context, tokens: tokens, precision: precision)
                return String(format: "%@: decode perplexity %.4f, %.2f tok/s at 8k, zoned KV %.1f MB", label, result.perplexity, result.tokensPerSecond, Double(result.bytes) / 1_048_576)
            }
        }
        for line in lines { log.record("RESULT", line) }
        for (depth, key) in [(0.2, 17448), (0.8, 39648)] {
            let found = try await Self.passkey(engine, corpus: corpus, depth: depth, key: key)
            log.record("RESULT", "passkey 16k depth \(depth) with --kv-bits polar4 --kv-zones: \(found ? "found" : "missed")")
        }
        await engine.unload()
    }
}
