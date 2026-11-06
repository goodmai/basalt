import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("fp16 KV eviction on Gemma 4 full-attention layers at 16k (#171)", .serialized)
struct KVEvictGemmaRealModelTests {
    struct StepFailure: Error, CustomStringConvertible {
        let description: String
    }

    struct Run {
        let tokens: [Int]
        let bytes: Int
        let stored: Int
        let evicted: Int
    }

    static var env: [String: String] { ProcessInfo.processInfo.environment }
    static var enabled: Bool { env["BASALT_E2E"] == "1" && env["BASALT_E2E_GEMMA4_MODEL_PATH"] != nil && env["BASALT_E2E_KV_EVICT_CORPUS"] != nil }
    static var length: Int { Int(env["BASALT_E2E_KV_EVICT_LENGTH"] ?? "") ?? 16_000 }
    static let fractions = [0.0, 0.25, 0.5]
    static let depths = [0.1, 0.3, 0.5, 0.7, 0.9]

    static func zones(_ fraction: Double) -> KVZones {
        KVZones.standard.evicting(fraction).keepingMiddleFullPrecision()
    }

    static func cache(_ fraction: Double, _ model: any LanguageModel) throws -> [any KVCache] {
        let fresh = try model.newCache(parameters: nil)
        return fraction > 0 ? EvictingKVCache.replacingFullAttention(fresh, zones: zones(fraction)) : fresh
    }

    static func greedy(_ fraction: Double, tokens: [Int], maxTokens: Int, context: ModelContext) throws -> Run {
        let cache = try cache(fraction, context.model)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(tokens)), model: context.model, cache: cache, parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))
        var out = [Int]()
        while let token = iterator.next() { out.append(token) }
        let evicting = cache.compactMap { $0 as? EvictingKVCache }
        return Run(
            tokens: out, bytes: cache.flatMap(\.state).reduce(0) { $0 + $1.nbytes },
            stored: evicting.first?.storedCount ?? cache.first?.offset ?? 0, evicted: evicting.first?.evictedCount ?? 0)
    }

    static func passkeyPrompt(_ filler: [Int], key: Int, depth: Double, context: ModelContext) throws -> [Int] {
        let needle = context.tokenizer.encode(text: "\nThe pass key is \(key). Remember it. \(key) is the pass key.\n")
        let question = context.tokenizer.encode(text: "\nWhat is the pass key? The pass key is")
        let end = length - needle.count - question.count
        guard filler.count >= end else { throw StepFailure(description: "corpus has \(filler.count) tokens, needs \(end)") }
        let at = Int(Double(end) * depth)
        return Array(filler[0..<at]) + needle + Array(filler[at..<end]) + question
    }

    static func perplexity(_ fraction: Double, tokens: [Int], context: ModelContext) throws -> Double {
        let total = tokens.count - 1
        let prefix = total / 2
        let cache = try cache(fraction, context.model)
        var (start, nll) = (0, 0.0)
        while start < total {
            let end = min(start < prefix ? prefix : total, start + 512)
            let logits = context.model(MLXArray(Array(tokens[start..<end]))[.newAxis], cache: cache)[0].asType(.float32)
            if start >= prefix {
                let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
                let targets = MLXArray(Array(tokens[(start + 1)...end]))
                nll -= Double(takeAlong(logp, targets[0..., .newAxis], axis: -1).sum().item(Float.self))
            }
            eval(cache)
            start = end
        }
        return exp(nll / Double(total - prefix))
    }

    @Test("Gemma 4 with --kv-evict 0.25 and 0.5 against off: KV bytes against KVCacheGeometry, passkey at five depths, perplexity and greedy identity",
          .enabled(if: enabled, "set BASALT_E2E=1, BASALT_E2E_GEMMA4_MODEL_PATH=<Gemma 4 26B-A4B repo or path> and BASALT_E2E_KV_EVICT_CORPUS=<text file>"))
    func evictionAgainstOff() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.env["BASALT_E2E_GEMMA4_MODEL_PATH"]))
        let corpus = try String(contentsOfFile: try #require(Self.env["BASALT_E2E_KV_EVICT_CORPUS"]), encoding: .utf8)
        let log = try await ModelRunLog("kv-evict-gemma", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(gpuCacheLimit: 0, kvPrecision: .q8, kvZones: KVZones.standard.evicting(0.5)))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let geometry = try #require(KVCacheGeometry.load(modelDirectory: URL(fileURLWithPath: path)))
        let wired = await (zones: engine.supportsKVZones, fp16Eviction: engine.evictsKVInFullPrecision)
        log.record("probe", "zonesApply=\(wired.zones) evictsInFP16=\(wired.fp16Eviction) length=\(Self.length) corpusBytes=\(corpus.utf8.count)")
        #expect(wired.fp16Eviction && !wired.zones, "the engine must pick the fp16 evicting cache for Gemma 4")
        let runs = try await kvBytesAndGreedy(container: container, corpus: corpus, geometry: geometry, log: log)
        #expect(runs[0.5].map { $0.bytes < runs[0]?.bytes ?? 0 } == true)
        try await passkeys(container: container, corpus: corpus, log: log)
        try await perplexities(container: container, corpus: corpus, log: log)
    }

    func kvBytesAndGreedy(container: ModelContainer, corpus: String, geometry: KVCacheGeometry, log: ModelRunLog) async throws -> [Double: Run] {
        var runs = [Double: Run]()
        let length = Self.length
        for fraction in Self.fractions {
            let run = try await log.step("kv-bytes evict=\(fraction) prompt=\(length) generated=64") {
                try await container.perform { (context: ModelContext) -> Run in
                    let tokens = Array(context.tokenizer.encode(text: corpus).prefix(length))
                    guard tokens.count == length else { throw StepFailure(description: "corpus has \(tokens.count) tokens") }
                    return try Self.greedy(fraction, tokens: tokens, maxTokens: 64, context: context)
                }
            }
            runs[fraction] = run
            let held = TokenCount(length + run.tokens.count)
            let estimate = fraction > 0 ? geometry.bytes(tokens: held, kvPrecision: .q8, zones: Self.zones(fraction)) : geometry.bytes(tokens: held)
            let reference = runs[0]?.tokens ?? run.tokens
            let shared = zip(run.tokens, reference).prefix { $0 == $1 }.count
            log.record("result", String(
                format: "KV-EVICT bytes evict=%.2f held=%d geometry=%lld ratio=%.4f stored=%d evicted=%d greedyIdentical=%@ sharedPrefix=%d/%d",
                fraction, run.bytes, estimate.rawValue, Double(run.bytes) / Double(estimate.rawValue), run.stored, run.evicted,
                String(run.tokens == reference), shared, reference.count))
        }
        return runs
    }

    func passkeys(container: ModelContainer, corpus: String, log: ModelRunLog) async throws {
        var found = [Double: Int]()
        for depth in Self.depths {
            let key = 10_000 + Int(depth * 1_000) * 37 + Self.length % 997
            for fraction in Self.fractions {
                let answer = try await log.step("passkey evict=\(fraction) length=\(Self.length) depth=\(depth) key=\(key)") {
                    try await container.perform { (context: ModelContext) -> String in
                        let prompt = try Self.passkeyPrompt(context.tokenizer.encode(text: corpus), key: key, depth: depth, context: context)
                        return context.tokenizer.decode(tokenIds: try Self.greedy(fraction, tokens: prompt, maxTokens: 8, context: context).tokens)
                    }
                }
                let ok = answer.contains(String(key))
                found[fraction, default: 0] += ok ? 1 : 0
                log.record("result", "KV-EVICT passkey evict=\(fraction) depth=\(depth) ok=\(ok) answer=\(answer.debugDescription)")
            }
        }
        for fraction in Self.fractions {
            log.record("result", "KV-EVICT passkey evict=\(fraction) found=\(found[fraction] ?? 0)/\(Self.depths.count) at length=\(Self.length)")
        }
    }

    func perplexities(container: ModelContainer, corpus: String, log: ModelRunLog) async throws {
        let total = Self.length
        for fraction in Self.fractions {
            let ppl = try await log.step("ppl evict=\(fraction) context=\(total / 2) scored=\(total - total / 2)") {
                try await container.perform { (context: ModelContext) -> Double in
                    let tokens = Array(context.tokenizer.encode(text: corpus).prefix(total + 1))
                    guard tokens.count == total + 1 else { throw StepFailure(description: "corpus has \(tokens.count) tokens") }
                    return try Self.perplexity(fraction, tokens: tokens, context: context)
                }
            }
            log.record("result", String(format: "KV-EVICT ppl evict=%.2f ppl=%.4f", fraction, ppl))
        }
    }
}
