import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("YaRN P2 real-model gate: short-context PPL/passkey/decode cost, past-native --kv-zones passkey, and Gemma 4 p-RoPE refusal (#729)", .serialized)
struct RopeYarnRealModelTests {
    static var env: [String: String] { ProcessInfo.processInfo.environment }
    static var modelPath: String? { env["BASALT_E2E"] == "1" ? env["BASALT_E2E_YARN_MODEL_PATH"] : nil }
    static var gemma4Path: String? { env["BASALT_E2E"] == "1" ? env["BASALT_E2E_YARN_GEMMA4_MODEL_PATH"] : nil }
    static var factors: [Double] {
        parseDoubles(env["BASALT_E2E_YARN_FACTORS"] ?? "1,2,4", fallback: [1, 2, 4])
    }
    static var pplLengths: [Int] {
        parseInts(env["BASALT_E2E_YARN_PPL_LENGTHS"], fallback: [4096, 32768])
    }
    static var decodeLengths: [Int] {
        parseInts(env["BASALT_E2E_YARN_DECODE_LENGTHS"], fallback: [4096])
    }
    static var passkeyLengths: [Int] {
        parseInts(env["BASALT_E2E_YARN_PASSKEY_LENGTHS"], fallback: [32768])
    }
    static var longLengths: [Int] {
        parseInts(env["BASALT_E2E_YARN_LONG_LENGTHS"], fallback: [])
    }
    static var tableBFactors: [Double] {
        parseDoubles(env["BASALT_E2E_YARN_TABLE_B_FACTORS"] ?? "1,2", fallback: [1, 2])
    }
    static var tableBKVPrecision: KVPrecision {
        env["BASALT_E2E_YARN_TABLE_B_KV"] == "k8v4" ? .k8v4 : .q4
    }
    static var depths: [Double] {
        let raw = env["BASALT_E2E_YARN_DEPTHS"] ?? "0.1,0.3,0.5,0.7,0.9"
        let parsed = raw.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }.filter { $0 > 0 && $0 < 1 }
        return parsed.isEmpty ? [0.1, 0.3, 0.5, 0.7, 0.9] : parsed
    }
    static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    static func parseInts(_ raw: String?, fallback: [Int]) -> [Int] {
        guard let raw else { return fallback }
        if raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "none" { return [] }
        let parsed = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }.filter { $0 > 0 }
        return parsed.isEmpty ? fallback : parsed
    }

    static func parseDoubles(_ raw: String, fallback: [Double]) -> [Double] {
        let parsed = raw.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }.filter { $0 >= 1 }
        return parsed.isEmpty ? fallback : parsed
    }

    static func fullAttentionRoPETypeNames(in model: any LanguageModel) -> [String] {
        var names: [String] = []
        for (path, module) in model.namedModules() where path.hasSuffix(".self_attn") {
            for child in Mirror(reflecting: module).children where child.label == "rope" || child.label == "rotaryEmbedding" {
                names.append(String(describing: type(of: child.value)))
            }
        }
        return names
    }

    static func noThinkRequest(prompt: String, maxTokens: Int) -> GenerationRequest {
        var req = GenerationRequest(prompt: prompt, maxTokens: maxTokens, temperature: 0)
        req.enableThinking = false
        req.reasoningEffort = "none"
        return req
    }

    static func passkeyRequest(prompt: String, maxTokens: Int) -> GenerationRequest {
        var req = GenerationRequest(
            prompt: "",
            maxTokens: maxTokens,
            temperature: 0,
            messages: [
                ChatMessageInput(role: .user, text: prompt),
                ChatMessageInput(role: .assistant, text: "The pass key is "),
            ]
        )
        req.enableThinking = false
        req.reasoningEffort = "none"
        return req
    }

    static func loadCorpusText() -> String {
        if let custom = env["BASALT_E2E_KV_CORPUS"],
           let text = try? String(contentsOfFile: custom, encoding: .utf8),
           !text.isEmpty {
            return text
        }
        let files = [
            "README.md", "CONTRIBUTING.md", "CHANGELOG.md",
            "docs/features.md", "docs/compose.md", "docs/commands.md",
            "docs/research-tools.md", "docs/research-foundations.md",
            "docs/rest-api.md", "docs/models.md", "docs/benchmarks.md",
        ]
        var text = files.compactMap {
            try? String(contentsOf: repoRoot.appending(path: $0), encoding: .utf8)
        }.joined(separator: "\n\n")
        for marker in ["<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|turn|>", "<|channel|>", "<|channel>"] {
            text = text.replacingOccurrences(of: marker, with: "")
        }
        return text
    }

    static func repeatedSlice(_ tokens: [Int], count: Int) -> [Int] {
        guard !tokens.isEmpty, count > 0 else { return [] }
        if tokens.count >= count { return Array(tokens.prefix(count)) }
        var out: [Int] = []
        out.reserveCapacity(count)
        while out.count < count {
            let take = min(tokens.count, count - out.count)
            out.append(contentsOf: tokens.prefix(take))
        }
        return out
    }

    static func perplexity(tokens: [Int], prefix: Int, context: ModelContext) throws -> Double {
        let total = tokens.count - 1
        let cache = try context.model.newCache(parameters: nil)
        defer { Memory.clearCache() }
        var start = 0
        var nll = 0.0
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

    static func decodeSpeed(tokens: [Int], maxTokens: Int, context: ModelContext) throws -> (tps: Double, prefillSec: Double, peakBytes: Int) {
        let cache = try context.model.newCache(parameters: nil)
        defer { Memory.clearCache() }
        Memory.peakMemory = 0
        let start = ContinuousClock.now
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(tokens)),
            model: context.model,
            cache: cache,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0)
        )
        var count = 0
        guard iterator.next() != nil else { return (0, 0, Memory.peakMemory) }
        count += 1
        let prefilled = ContinuousClock.now
        while iterator.next() != nil { count += 1 }
        let done = ContinuousClock.now
        let secs = { (d: Duration) in Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }
        return (Double(max(count - 1, 0)) / max(secs(done - prefilled), 1e-9), secs(prefilled - start), Memory.peakMemory)
    }

    static func passkeyPrompt(fillerTokens: [Int], targetTokens: Int, key: Int, depth: Double, tokenizer: any Tokenizer) -> String {
        let needleText = "\nThe pass key is \(key). Remember it. \(key) is the pass key.\n"
        let questionText = "\nWhat is the pass key? Reply with the number only."
        let overhead = tokenizer.encode(text: needleText).count + tokenizer.encode(text: questionText).count + 32
        let fillerCount = max(64, targetTokens - overhead)
        let body = repeatedSlice(fillerTokens, count: fillerCount)
        let splitAt = min(body.count, max(0, Int(Double(body.count) * depth)))
        let before = tokenizer.decode(tokenIds: Array(body[0 ..< splitAt]), skipSpecialTokens: true)
        let after = tokenizer.decode(tokenIds: Array(body[splitAt ..< body.count]), skipSpecialTokens: true)
        return before + needleText + after + questionText
    }

    static func evaluatePasskeys(
        engine: MLXInferenceEngine,
        container: ModelContainer,
        fillerTokens: [Int],
        length: Int,
        label: String,
        log: ModelRunLog
    ) async throws -> Int {
        var hits = 0
        for depth in Self.depths {
            let key = 100_000 + Int(depth * 1_000) * 37 + length % 997
            let prompt = await container.perform { context in
                Self.passkeyPrompt(fillerTokens: fillerTokens, targetTokens: length, key: key, depth: depth, tokenizer: context.tokenizer)
            }
            Memory.peakMemory = 0
            let req = Self.passkeyRequest(prompt: prompt, maxTokens: 16)
            let resp = try await log.step("passkey \(label) len=\(length) depth=\(depth)") {
                try await engine.generate(request: req)
            }
            let ok = resp.generatedText.contains(String(key))
            if ok { hits += 1 }
            let peakGB = Double(resp.memory.peakBytes) / 1_073_741_824.0
            let reasoning = resp.reasoningText ?? ""
            log.record(
                "yarn-passkey",
                String(
                    format: "rope=%@ target=%d actualTokens=%d depth=%.1f ok=%@ prefill=%.2fs decode=%.2f tok/s peak=%.2fGB answer=%@ reasoning=%@",
                    label, length, resp.promptTokens, depth, ok ? "true" : "false",
                    resp.timeToFirstToken, resp.tokensPerSecond, peakGB, resp.generatedText.debugDescription, reasoning.debugDescription))
        }
        log.record("yarn-passkey-summary", "rope=\(label) length=\(length) hits=\(hits)/\(Self.depths.count)")
        return hits
    }
}

extension RopeYarnRealModelTests {
    static func measureFactor(
        path: String,
        factor: Double,
        corpus: String,
        log: ModelRunLog
    ) async throws -> [Int: Double] {
        let label = factor > 1 ? "yarn-x\(ModelRoPEStatus.formatFactor(factor))" : "stock"
        var config = EngineConfiguration(gpuCacheLimit: 0, promptCacheBytes: 0)
        config.ropeYarn = factor > 1 ? factor : nil
        let engine = MLXInferenceEngine(config)
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await log.step("load \(label)") { try await engine.load(modelPath: path) }
        let status = await engine.ropeStatus
        let container = try #require(await engine.container)
        let (modelTypeName, ropeTypes, fillerTokens) = await container.perform { context in
            (String(describing: type(of: context.model)),
             Self.fullAttentionRoPETypeNames(in: context.model),
             context.tokenizer.encode(text: corpus))
        }
        let uniqueRopeTypes = Set(ropeTypes).sorted().joined(separator: ",")
        log.record(
            "yarn-load",
            "rope=\(label) modelClass=\(modelTypeName) ropeTypes=\(uniqueRopeTypes) count=\(ropeTypes.count) statusLabel=\(status.ropeLabel ?? "stock") native=\(status.nativeContext ?? 0) effective=\(status.effectiveContext ?? 0) warnings=\(status.loadWarnings.count)")
        #expect(!ropeTypes.isEmpty)
        if factor > 1 {
            #expect(ropeTypes.allSatisfy { $0.contains("Yarn") || $0.contains("RotaryEmbedding") })
        }

        var ppls: [Int: Double] = [:]
        for length in Self.pplLengths {
            let scored = min(length, 4096)
            let ppl = try await log.step("ppl \(label) ctx=\(length) scored=\(scored)") {
                try await container.perform { context -> Double in
                    let tokens = Self.repeatedSlice(fillerTokens, count: length + scored + 1)
                    return try Self.perplexity(tokens: tokens, prefix: length, context: context)
                }
            }
            ppls[length] = ppl
            log.record("yarn-ppl", String(format: "rope=%@ context=%d scored=%d ppl=%.4f", label, length, scored, ppl))
        }

        for length in Self.decodeLengths {
            let speed = try await log.step("decode \(label) len=\(length)") {
                try await container.perform { context -> (tps: Double, prefillSec: Double, peakBytes: Int) in
                    try Self.decodeSpeed(tokens: Self.repeatedSlice(fillerTokens, count: length), maxTokens: 64, context: context)
                }
            }
            log.record(
                "yarn-decode",
                String(format: "rope=%@ len=%d prefill=%.2fs decode=%.2f tok/s peak=%.2fGB",
                       label, length, speed.prefillSec, speed.tps, Double(speed.peakBytes) / 1_073_741_824.0))
        }

        for length in Self.passkeyLengths {
            _ = try await Self.evaluatePasskeys(engine: engine, container: container, fillerTokens: fillerTokens, length: length, label: label, log: log)
        }
        return ppls
    }

    @Test("Table A: short-context YaRN PPL, passkey, and decode speed vs stock on Qwen 3.8 (#729 P2)",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_YARN_MODEL_PATH"))
    func tableAShortContextCost() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.modelPath))
        let log = try await ModelRunLog("yarn-p2-table-a", modelPath: path)
        defer { log.releaseGPU() }
        let corpus = Self.loadCorpusText()
        var stockPPL: [Int: Double] = [:]
        for factor in Self.factors {
            let ppls = try await Self.measureFactor(path: path, factor: factor, corpus: corpus, log: log)
            if factor <= 1 {
                stockPPL = ppls
            } else {
                for (len, val) in ppls.sorted(by: { $0.key < $1.key }) {
                    if let base = stockPPL[len], base > 0 {
                        log.record("yarn-ppl-delta", String(format: "factor=%g context=%d stock=%.4f yarn=%.4f delta=%.3f%%",
                                                            factor, len, base, val, 100.0 * (val - base) / base))
                    }
                }
            }
        }
    }

    @Test("Table B: past-native passkey with --kv-zones for stock vs yarn x2 (#729 P2)",
          .enabled(if: modelPath != nil && !longLengths.isEmpty, "set BASALT_E2E=1, BASALT_E2E_YARN_MODEL_PATH, and BASALT_E2E_YARN_LONG_LENGTHS"))
    func tableBPastNativePasskey() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.modelPath))
        let log = try await ModelRunLog("yarn-p2-table-b", modelPath: path)
        defer { log.releaseGPU() }
        let corpus = Self.loadCorpusText()
        let kvPrecision = Self.tableBKVPrecision
        for factor in Self.tableBFactors {
            let label = factor > 1 ? "yarn-x\(ModelRoPEStatus.formatFactor(factor))-kvzones-\(kvPrecision)" : "stock-kvzones-\(kvPrecision)"
            var config = EngineConfiguration(gpuCacheLimit: 0, promptCacheBytes: 0)
            config.kvPrecision = kvPrecision
            config.kvZones = .standard
            config.ropeYarn = factor > 1 ? factor : nil
            let engine = MLXInferenceEngine(config)
            defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
            try await log.step("load \(label)") { try await engine.load(modelPath: path) }
            let container = try #require(await engine.container)
            let (ropeTypes, fillerTokens) = await container.perform { context in
                (Self.fullAttentionRoPETypeNames(in: context.model), context.tokenizer.encode(text: corpus))
            }
            log.record("yarn-load-b", "rope=\(label) kv=\(kvPrecision) ropeTypes=\(Set(ropeTypes).sorted().joined(separator: ","))")
            if factor > 1 { #expect(ropeTypes.allSatisfy { $0.contains("Yarn") || $0.contains("RotaryEmbedding") }) }
            for length in Self.longLengths {
                _ = try await Self.evaluatePasskeys(engine: engine, container: container, fillerTokens: fillerTokens, length: length, label: label, log: log)
            }
        }
    }

    @Test("Gemma 4 control: --rope-yarn 2 warns, keeps stock RoPE, and generates normally (#729 P2)",
          .enabled(if: gemma4Path != nil, "set BASALT_E2E=1 and BASALT_E2E_YARN_GEMMA4_MODEL_PATH"))
    func gemma4RefusesYarnKeepsStockRoPE() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.gemma4Path))
        let log = try await ModelRunLog("yarn-p2-gemma4-control", modelPath: path)
        defer { log.releaseGPU() }
        var config = EngineConfiguration(gpuCacheLimit: 0, promptCacheBytes: 0)
        config.ropeYarn = 2.0
        let engine = MLXInferenceEngine(config)
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await log.step("load gemma4 with --rope-yarn 2") { try await engine.load(modelPath: path) }
        let status = await engine.ropeStatus
        let factorText = status.factor.map { "\($0)" } ?? "nil"
        log.record("yarn-gemma4", "factor=\(factorText) label=\(status.ropeLabel ?? "stock") warnings=\(status.loadWarnings)")
        #expect(status.factor == nil)
        #expect(status.ropeLabel == nil)
        #expect(status.loadWarnings.contains { $0.contains("p-RoPE") })
        let resp = try await log.step("sanity generate") {
            try await engine.generate(request: Self.noThinkRequest(prompt: "What is 17 * 23? Reply with the number only.", maxTokens: 16))
        }
        log.record("yarn-gemma4-gen", "text=\(resp.generatedText.debugDescription) tps=\(resp.tokensPerSecond)")
        #expect(resp.generatedText.contains("391"))
    }
}
