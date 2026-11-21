import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Gemma 4 K=V shared-row KV cache on real models (#645)", .serialized)
struct KEqVRowRealModelTests {
    struct StepFailure: Error, CustomStringConvertible {
        let description: String
    }

    struct Run {
        let tokens: [Int]
        let bytes: Int
        let peak: Int
        let prefillSeconds: Double
        let decodeTokensPerSecond: Double
    }

    static var env: [String: String] { ProcessInfo.processInfo.environment }
    static var enabled: Bool { env["BASALT_E2E"] == "1" && env["BASALT_E2E_GEMMA4_MODEL_PATH"] != nil && env["BASALT_E2E_KV_CORPUS"] != nil }
    static var lengths: [Int] { (env["BASALT_E2E_KV_ROW_LENGTHS"] ?? "8192,32768").split(separator: ",").compactMap { Int($0) } }
    static var passkeyLengths: [Int] { (env["BASALT_E2E_KV_ROW_PASSKEY"] ?? "").split(separator: ",").compactMap { Int($0) } }
    static let depths = [0.1, 0.3, 0.5, 0.7, 0.9]
    static let prompts = [
        "Explain in five sentences why the sky is blue.",
        "Write a Python function that merges two sorted lists, then explain it briefly.",
        "List the planets of the solar system with one fact each.",
        "Translate to French: The library opens at nine and closes at six on weekdays.",
        "What are three differences between TCP and UDP?",
    ]

    static func cache(row: Bool, layout: KEqVRowLayout, model: any LanguageModel) throws -> [any KVCache] {
        let fresh = try model.newCache(parameters: nil)
        return row ? KEqVRowKVCache.replacingFullAttention(fresh, model: model, layout: layout) : fresh
    }

    static func greedy(row: Bool, layout: KEqVRowLayout, tokens: [Int], maxTokens: Int, context: ModelContext) throws -> Run {
        let cache = try cache(row: row, layout: layout, model: context.model)
        Memory.peakMemory = 0
        let start = ContinuousClock.now
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(tokens)), model: context.model, cache: cache, parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))
        var out = [Int]()
        guard let first = iterator.next() else { throw StepFailure(description: "no token") }
        out.append(first)
        let prefilled = ContinuousClock.now
        while let token = iterator.next() { out.append(token) }
        let done = ContinuousClock.now
        let seconds = { (d: Duration) in Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }
        return Run(
            tokens: out, bytes: cache.flatMap(\.state).reduce(0) { $0 + $1.nbytes }, peak: Memory.peakMemory,
            prefillSeconds: seconds(prefilled - start), decodeTokensPerSecond: Double(out.count - 1) / max(seconds(done - prefilled), 1e-9))
    }

    static func passkeyPrompt(_ filler: [Int], key: Int, depth: Double, length: Int, context: ModelContext) throws -> [Int] {
        let needle = context.tokenizer.encode(text: "\nThe pass key is \(key). Remember it. \(key) is the pass key.\n")
        let question = context.tokenizer.encode(text: "\nWhat is the pass key? The pass key is")
        let end = length - needle.count - question.count
        guard filler.count >= end else { throw StepFailure(description: "corpus has \(filler.count) tokens, needs \(end)") }
        let at = Int(Double(end) * depth)
        return Array(filler[0 ..< at]) + needle + Array(filler[at ..< end]) + question
    }

    static func perplexity(row: Bool, layout: KEqVRowLayout, tokens: [Int], context: ModelContext) throws -> Double {
        let total = tokens.count - 1
        let prefix = total / 2
        let cache = try cache(row: row, layout: layout, model: context.model)
        var (start, nll) = (0, 0.0)
        while start < total {
            let end = min(start < prefix ? prefix : total, start + 512)
            let logits = context.model(MLXArray(Array(tokens[start ..< end]))[.newAxis], cache: cache)[0].asType(.float32)
            if start >= prefix {
                let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
                let targets = MLXArray(Array(tokens[(start + 1) ... end]))
                nll -= Double(takeAlong(logp, targets[0..., .newAxis], axis: -1).sum().item(Float.self))
            }
            eval(cache)
            start = end
        }
        return exp(nll / Double(total - prefix))
    }

    @Test("Shared row against stock KVCacheSimple: whole-model KV bytes, peak, prefill and decode speed, greedy identity, passkey and perplexity",
          .enabled(if: enabled, "set BASALT_E2E=1, BASALT_E2E_GEMMA4_MODEL_PATH=<Gemma 4 repo or path> and BASALT_E2E_KV_CORPUS=<text file>"))
    func rowAgainstStock() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.env["BASALT_E2E_GEMMA4_MODEL_PATH"]))
        let corpus = try String(contentsOfFile: try #require(Self.env["BASALT_E2E_KV_CORPUS"]), encoding: .utf8)
        let log = try await ModelRunLog("kv-shared-row", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(gpuCacheLimit: 0))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        guard let layout = ModelCache.config(at: URL(fileURLWithPath: path)).flatMap(KEqVRowLayout.init(config:)) else {
            return log.record("result", "KV-ROW refused: no attention_k_eq_v layout in config.json")
        }
        let attends = try await container.perform { try KEqVRowKVCache.modelAttendsThroughCache($0.model, layout: layout) }
        log.record("probe", "layout headDim=\(layout.headDim) rotated=\(2 * layout.rotatedHalf) kvHeads=\(layout.kvHeads) globalLayers=\(layout.globalLayers) attendsThroughCache=\(attends)")
        #expect(attends)
        try await lengthsRuns(container: container, corpus: corpus, layout: layout, log: log)
        try await promptIdentity(container: container, layout: layout, log: log)
        try await passkeys(container: container, corpus: corpus, layout: layout, log: log)
        if Self.env["BASALT_E2E_KV_ROW_PPL"] != "0" { try await perplexities(container: container, corpus: corpus, layout: layout, log: log) }
    }

    func lengthsRuns(container: ModelContainer, corpus: String, layout: KEqVRowLayout, log: ModelRunLog) async throws {
        for length in Self.lengths {
            var runs = [Bool: Run]()
            for row in [false, true] {
                runs[row] = try await log.step("context row=\(row) prompt=\(length) generated=64") {
                    try await container.perform { (context: ModelContext) -> Run in
                        let tokens = Array(context.tokenizer.encode(text: corpus).prefix(length))
                        guard tokens.count == length else { throw StepFailure(description: "corpus has \(tokens.count) tokens") }
                        return try Self.greedy(row: row, layout: layout, tokens: tokens, maxTokens: 64, context: context)
                    }
                }
            }
            guard let stock = runs[false], let row = runs[true] else { continue }
            let shared = zip(row.tokens, stock.tokens).prefix { $0 == $1 }.count
            log.record("result", String(
                format: "KV-ROW length=%ld kvBytes stock=%ld row=%ld ratio=%.3fx peak stock=%ld row=%ld prefill stock=%.2fs row=%.2fs decode stock=%.2f row=%.2f tok/s greedyShared=%ld/%ld",
                length, stock.bytes, row.bytes, Double(stock.bytes) / Double(max(row.bytes, 1)), stock.peak, row.peak,
                stock.prefillSeconds, row.prefillSeconds, stock.decodeTokensPerSecond, row.decodeTokensPerSecond, shared, stock.tokens.count))
            #expect(row.bytes < stock.bytes)
        }
    }

    func promptIdentity(container: ModelContainer, layout: KEqVRowLayout, log: ModelRunLog) async throws {
        var identical = 0
        for (index, prompt) in Self.prompts.enumerated() {
            let pair = try await log.step("greedy prompt=\(index) generated=256") {
                try await container.perform { (context: ModelContext) -> ([Int], [Int]) in
                    let tokens = try context.tokenizer.applyChatTemplate(messages: [["role": "user", "content": prompt]])
                    let stock = try Self.greedy(row: false, layout: layout, tokens: tokens, maxTokens: 256, context: context).tokens
                    let row = try Self.greedy(row: true, layout: layout, tokens: tokens, maxTokens: 256, context: context).tokens
                    return (stock, row)
                }
            }
            let shared = zip(pair.0, pair.1).prefix { $0 == $1 }.count
            identical += pair.0 == pair.1 ? 1 : 0
            log.record("result", "KV-ROW greedy prompt=\(index) identical=\(pair.0 == pair.1) sharedPrefix=\(shared)/\(pair.0.count)")
        }
        log.record("result", "KV-ROW greedy identical=\(identical)/\(Self.prompts.count)")
    }

    func passkeys(container: ModelContainer, corpus: String, layout: KEqVRowLayout, log: ModelRunLog) async throws {
        for length in Self.passkeyLengths {
            var found = [Bool: Int]()
            for depth in Self.depths {
                let key = 10_000 + Int(depth * 1_000) * 37 + length % 997
                for row in [false, true] {
                    let answer = try await log.step("passkey row=\(row) length=\(length) depth=\(depth) key=\(key)") {
                        try await container.perform { (context: ModelContext) -> String in
                            let prompt = try Self.passkeyPrompt(context.tokenizer.encode(text: corpus), key: key, depth: depth, length: length, context: context)
                            return context.tokenizer.decode(tokenIds: try Self.greedy(row: row, layout: layout, tokens: prompt, maxTokens: 8, context: context).tokens)
                        }
                    }
                    let ok = answer.contains(String(key))
                    found[row, default: 0] += ok ? 1 : 0
                    log.record("result", "KV-ROW passkey row=\(row) length=\(length) depth=\(depth) ok=\(ok) answer=\(answer.debugDescription)")
                }
            }
            log.record("result", "KV-ROW passkey length=\(length) stock=\(found[false] ?? 0)/\(Self.depths.count) row=\(found[true] ?? 0)/\(Self.depths.count)")
        }
    }

    func perplexities(container: ModelContainer, corpus: String, layout: KEqVRowLayout, log: ModelRunLog) async throws {
        let total = 8192
        var values = [Bool: Double]()
        for row in [false, true] {
            values[row] = try await log.step("ppl row=\(row) context=\(total / 2) scored=\(total - total / 2)") {
                try await container.perform { (context: ModelContext) -> Double in
                    let tokens = Array(context.tokenizer.encode(text: corpus).prefix(total + 1))
                    guard tokens.count == total + 1 else { throw StepFailure(description: "corpus has \(tokens.count) tokens") }
                    return try Self.perplexity(row: row, layout: layout, tokens: tokens, context: context)
                }
            }
        }
        let (stock, row) = (values[false] ?? 0, values[true] ?? 0)
        log.record("result", String(format: "KV-ROW ppl stock=%.4f row=%.4f delta=%.3f%%", stock, row, 100 * (row - stock) / max(stock, 1e-9)))
    }
}
