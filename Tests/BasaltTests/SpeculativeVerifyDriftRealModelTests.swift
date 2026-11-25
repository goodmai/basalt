import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Speculative verify drift S1 real-model probe: per-slot M=1 vs K-row logits, slopes, kernel thresholds, and end-to-end divergence (#646)", .serialized)
struct SpeculativeVerifyDriftRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static var modelPath: String? { env["BASALT_E2E"] == "1" ? env["BASALT_E2E_VERIFY_DRIFT_MODEL_PATH"] : nil }
    static var useMTP: Bool { env["BASALT_E2E_VERIFY_DRIFT_MTP"] == "1" }
    static var prefixLengths: [Int] {
        guard let raw = env["BASALT_E2E_VERIFY_DRIFT_LENGTHS"] else { return [128, 512, 1024] }
        let parsed = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }.filter { $0 >= 32 }
        return parsed.isEmpty ? [128, 512, 1024] : parsed
    }

    static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    static func sampleTexts() -> [(name: String, text: String)] {
        let codeURL = repoRoot.appending(path: "benchmarks/ladder/run.py")
        let proseURL = repoRoot.appending(path: "benchmarks/arc-agi/ARC_AGI.md")
        let code = (try? String(contentsOf: codeURL, encoding: .utf8)) ?? String(repeating: "def solve(x):\n    return x * x + 1\n", count: 160)
        let prose = (try? String(contentsOf: proseURL, encoding: .utf8)) ?? String(repeating: "Speculative decoding verifies K draft tokens in one forward pass. ", count: 160)
        let math = String(
            repeating: "Prove that for every integer n >= 1, the sum 1^3 + 2^3 + ... + n^3 equals (n(n+1)/2)^2. Step 1: check n = 1. Step 2: assume true for k and add (k+1)^3. ",
            count: 60)
        return [("code", code), ("prose", prose), ("math", math)]
    }

    static func prefixes(tokenizer: any Tokenizer) -> [[Int32]] {
        var result: [[Int32]] = []
        for sample in sampleTexts() {
            let encoded = tokenizer.encode(text: sample.text).map(Int32.init)
            for length in prefixLengths where encoded.count >= length {
                result.append(Array(encoded.prefix(length)))
            }
        }
        return result
    }

    static func recordSummary(mode: String, runs: [[Int: [VerifySlotDrift]]], log: ModelRunLog) {
        var allEntries: [VerifySlotDrift] = []
        for size in VerifyDriftProbe.blockSizes {
            let points = runs.compactMap { $0[size] }.flatMap { $0 }
            allEntries.append(contentsOf: points)
            let count = Double(max(points.count, 1))
            let maxDiff = points.map(\.maxAbsLogitDiff).max() ?? 0
            let meanDiff = points.reduce(0.0) { $0 + Double($1.maxAbsLogitDiff) } / count
            let meanTop4 = points.reduce(0.0) { $0 + Double($1.top4LogProbDelta) } / count
            let meanEmit = points.reduce(0.0) { $0 + Double($1.emittedLogProbDelta) } / count
            let meanKL = points.reduce(0.0) { $0 + Double($1.klPlainVerify) } / count
            let greedy = points.filter(\.greedyFlipped).count
            let sampled = points.filter(\.sampleFlipped).count
            let driftFit = VerifyDriftProbe.olsSlope(points.map { (x: Double($0.slot), y: Double($0.maxAbsLogitDiff)) })
            let lpFit = VerifyDriftProbe.olsSlope(points.map { (x: Double($0.slot), y: Double($0.top4LogProbDelta)) })
            log.record(
                "verify-drift",
                String(
                    format: "mode=%@ K=%d n=%d meanAbs=%.3e maxAbs=%.3e top4dLP=%.3e emitdLP=%.3e kl=%.3e greedyFlips=%d/%d sampleFlips=%d/%d absSlope=%.3e±%.3e lpSlope=%.3e±%.3e",
                    mode, size, points.count, meanDiff, maxDiff, meanTop4, meanEmit, meanKL,
                    greedy, points.count, sampled, points.count,
                    driftFit.slope, 1.96 * driftFit.standardError,
                    lpFit.slope, 1.96 * lpFit.standardError))
        }
        recordK16Slots(mode: mode, blocks: runs.compactMap { $0[16] }, log: log)
        recordDivergences(mode: mode, entries: allEntries, log: log)
    }

    static func recordK16Slots(mode: String, blocks: [[VerifySlotDrift]], log: ModelRunLog) {
        guard !blocks.isEmpty else { return }
        for slot in 0 ..< 16 {
            let items = blocks.map { $0[slot] }
            let count = Double(items.count)
            let meanDiff = items.reduce(0.0) { $0 + Double($1.maxAbsLogitDiff) } / count
            let maxDiff = items.map(\.maxAbsLogitDiff).max() ?? 0
            let meanTop4 = items.reduce(0.0) { $0 + Double($1.top4LogProbDelta) } / count
            let greedy = items.filter(\.greedyFlipped).count
            let sampled = items.filter(\.sampleFlipped).count
            log.record(
                "verify-slot",
                String(
                    format: "mode=%@ K=16 slot=%d meanAbs=%.3e maxAbs=%.3e top4dLP=%.3e greedy=%d/%d sample=%d/%d",
                    mode, slot, meanDiff, maxDiff, meanTop4, greedy, items.count, sampled, items.count))
        }
    }

    static func recordDivergences(mode: String, entries: [VerifySlotDrift], log: ModelRunLog) {
        let greedyFlips = entries.filter(\.greedyFlipped)
        let sampleFlips = entries.filter(\.sampleFlipped)
        let greedyNearTies = greedyFlips.filter { $0.top2Margin <= 2 * $0.maxAbsLogitDiff + 1e-6 }.count
        let sampleNearTies = sampleFlips.filter { $0.top2Margin <= 2 * $0.maxAbsLogitDiff + 1e-6 }.count
        let greedyMargins = greedyFlips.map { String(format: "%.3e", $0.top2Margin) }.prefix(8).joined(separator: ",")
        let sampleMargins = sampleFlips.map { String(format: "%.3e", $0.top2Margin) }.prefix(8).joined(separator: ",")
        log.record(
            "verify-divergences",
            "mode=\(mode) totalSlots=\(entries.count) greedyFlips=\(greedyFlips.count) greedyNearTies=\(greedyNearTies) sampleFlips=\(sampleFlips.count) sampleNearTies=\(sampleNearTies) greedyMargins=[\(greedyMargins)] sampleMargins=[\(sampleMargins)]")
    }

    @Test("Real-model S1: measures per-slot verify vs M=1 logit drift, OLS slope, kernel jumps, and flip margins for stock and SmallM (#646)",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_VERIFY_DRIFT_MODEL_PATH"))
    func measureRealModelVerifyDrift() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.modelPath))
        let log = try await ModelRunLog("verify-drift-s1", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(gpuCacheLimit: 0, promptCacheBytes: 0))
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let runsStock = try await log.step("measure-stock") {
            try await container.perform { context -> [[Int: [VerifySlotDrift]]] in
                let prompts = Self.prefixes(tokenizer: context.tokenizer)
                return try prompts.enumerated().map { idx, prefix in
                    defer { Memory.clearCache() }
                    return try VerifyDriftProbe.measure(model: context.model, prefix: prefix, seed: UInt64(6460 + idx))
                }
            }
        }
        Self.recordSummary(mode: "stock", runs: runsStock, log: log)
        let stockEntries = runsStock.flatMap { $0.values.flatMap { $0 } }
        #expect(!stockEntries.isEmpty)
        #expect(stockEntries.allSatisfy { $0.maxAbsLogitDiff.isFinite && $0.top4LogProbDelta.isFinite && $0.klPlainVerify.isFinite })

        let installed = await container.perform { SmallMQuantizedLinear.install(in: $0.model) }
        guard installed > 0 else {
            return log.record("verify-smallm", "skipped (0 eligible 4-bit affine layers)")
        }
        log.record("verify-smallm", "installed on \(installed) layers")
        let runsSmallM = try await log.step("measure-smallm") {
            try await container.perform { context -> [[Int: [VerifySlotDrift]]] in
                let prompts = Self.prefixes(tokenizer: context.tokenizer)
                return try prompts.enumerated().map { idx, prefix in
                    defer { Memory.clearCache() }
                    return try VerifyDriftProbe.measure(model: context.model, prefix: prefix, seed: UInt64(6460 + idx))
                }
            }
        }
        Self.recordSummary(mode: "smallm", runs: runsSmallM, log: log)
        let smallMEntries = runsSmallM.flatMap { $0.values.flatMap { $0 } }
        #expect(smallMEntries.allSatisfy { $0.maxAbsLogitDiff.isFinite && $0.top4LogProbDelta.isFinite && $0.klPlainVerify.isFinite })
    }
}

extension SpeculativeVerifyDriftRealModelTests {
    static func e2ePrompts() -> [String] {
        let lookupCode = (try? String(contentsOf: repoRoot.appending(path: "benchmarks/speculative-lookup/code.txt"), encoding: .utf8)) ?? ""
        let lookupProse = (try? String(contentsOf: repoRoot.appending(path: "benchmarks/speculative-lookup/prose.txt"), encoding: .utf8)) ?? ""
        let lookupReason = (try? String(contentsOf: repoRoot.appending(path: "benchmarks/speculative-lookup/reasoning.txt"), encoding: .utf8)) ?? ""
        let basePrompts = [lookupCode, lookupProse, lookupReason].filter { !$0.isEmpty }
        let refactorSnippets: [String] = (1 ... 9).map { idx in
            """
            Rename `Buffer\(idx)` to `Ring\(idx)` everywhere and return only the updated Swift code:
            struct Buffer\(idx): Sendable {
                var items: [Int]
                var capacity: Int
                mutating func push(_ value: Int) {
                    if items.count >= capacity { items.removeFirst() }
                    items.append(value)
                }
                func snapshot() -> [Int] { items }
                func total() -> Int { items.reduce(0, +) }
            }
            """
        }
        let structuredTasks: [String] = (1 ... 8).map { idx in
            """
            Convert this JSON record #\(idx) to YAML preserving every key and value verbatim:
            {"service":"basalt-node-\(idx)","port":\(8080 + idx),"retries":3,"timeout_ms":\(500 * idx),"flags":["kv_cache","speculative","deterministic"],"limits":{"max_tokens":4096,"max_batch":8}}
            """
        }
        return Array((basePrompts + refactorSnippets + structuredTasks).prefix(20))
    }

    static func logitMargin(for item: OutputTokenTelemetry) -> Float {
        if item.topLogprobs.count >= 2 {
            return item.topLogprobs[0].logprob - item.topLogprobs[1].logprob
        }
        return item.margin
    }

    static func firstDivergence(plain: GenerationResponse, spec: GenerationResponse) -> (index: Int, margin: Float?, plainToken: String)? {
        let plainText = (plain.reasoningText ?? "") + plain.generatedText
        let specText = (spec.reasoningText ?? "") + spec.generatedText
        guard plainText != specText else { return nil }
        guard let telemetry = plain.tokenTelemetry, !telemetry.isEmpty else {
            let shared = zip(plainText, specText).prefix { $0 == $1 }.count
            return (shared, nil, "")
        }
        let filtered = telemetry.filter { $0.token != "<think>" && $0.token != "</think>" }
        var prefix = ""
        for item in filtered {
            let next = prefix + item.token
            if !specText.hasPrefix(next) {
                return (item.index, logitMargin(for: item), item.token)
            }
            prefix = next
        }
        return (filtered.count, nil, "<eof>")
    }

    static func runE2ESweep(path: String, temperature: Double, mtp: Bool, prompts: [String], log: ModelRunLog) async throws {
        let mode = mtp ? "mtp" : "lookup"
        var plainConfig = EngineConfiguration(gpuCacheLimit: 0, promptCacheBytes: 0, seed: 7)
        plainConfig.recordConfidence = true
        let plainEngine = MLXInferenceEngine(plainConfig)
        try await plainEngine.load(modelPath: path)
        var plainReplies: [GenerationResponse] = []
        for prompt in prompts {
            let req = GenerationRequest(prompt: prompt, maxTokens: 256, temperature: temperature).with {
                $0.reasoningEffort = "none"
                $0.logprobs = true
                $0.topLogprobs = 2
            }
            plainReplies.append(try await plainEngine.generate(request: req))
        }
        await plainEngine.unload()
        Memory.clearCache()

        var specConfig = EngineConfiguration(gpuCacheLimit: 0, promptCacheBytes: 0, seed: 7)
        specConfig.speculativeLookup = !mtp
        specConfig.mtp = mtp
        let specEngine = MLXInferenceEngine(specConfig)
        defer { await withTaskCancellationShield { await specEngine.unload(); Memory.clearCache() } }
        try await specEngine.load(modelPath: path)

        var diverged = 0
        var withDrafts = 0
        var totalDrafts = 0
        var totalAccepted = 0
        var margins: [Float] = []
        for (idx, prompt) in prompts.enumerated() {
            let req = GenerationRequest(prompt: prompt, maxTokens: 256, temperature: temperature).with { $0.reasoningEffort = "none" }
            let reply = try await specEngine.generate(request: req)
            let drafted = reply.draftTokens ?? 0
            let accepted = reply.acceptedDraftTokens ?? 0
            totalDrafts += drafted
            totalAccepted += accepted
            if accepted > 0 { withDrafts += 1 }
            if let div = firstDivergence(plain: plainReplies[idx], spec: reply) {
                diverged += 1
                if let m = div.margin { margins.append(m) }
                log.record("verify-e2e-div", String(format: "mode=%@ T=%.1f prompt=%d tokIdx=%d accepted=%d/%d margin=%@",
                                                    mode, temperature, idx, div.index, accepted, drafted,
                                                    div.margin.map { String(format: "%.3e", $0) } ?? "nil"))
            }
        }
        let marginList = margins.map { String(format: "%.3e", $0) }.joined(separator: ",")
        log.record("verify-e2e-summary", String(format: "mode=%@ T=%.1f diverged=%d/%d (%.1f%%) withDrafts=%d/%d accepted=%d/%d margins=[%@]",
                                                mode, temperature, diverged, prompts.count, 100.0 * Double(diverged) / Double(max(prompts.count, 1)),
                                                withDrafts, prompts.count, totalAccepted, totalDrafts, marginList))
    }

    @Test("Real-model S1 end-to-end: 20 x 256-token plain vs speculative (lookup/MTP) divergence at T=0 and T=0.7 with plain top-2 margins (#646)",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_VERIFY_DRIFT_MODEL_PATH"))
    func measureEndToEndSpeculativeDivergence() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.modelPath))
        let log = try await ModelRunLog("verify-drift-e2e", modelPath: path)
        defer { log.releaseGPU() }
        let prompts = Self.e2ePrompts()
        for temp in [0.0, 0.7] {
            try await log.step("e2e-T\(temp)") {
                try await Self.runE2ESweep(path: path, temperature: temp, mtp: Self.useMTP, prompts: prompts, log: log)
            }
        }
    }
}
