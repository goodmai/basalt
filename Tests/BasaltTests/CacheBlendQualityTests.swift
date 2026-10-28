import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

struct ModelRunLog: Sendable {
    let url: URL

    init(_ name: String, modelPath: String) async throws {
        let directory = ProcessInfo.processInfo.environment["BASALT_E2E_LOG_DIR"] ?? NSTemporaryDirectory()
        url = URL(fileURLWithPath: directory).appendingPathComponent("basalt-\(name).log")
        let snapshot = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
        record("start", "commit \(Self.commit()) model \(snapshot.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent) revision \(snapshot.lastPathComponent) machine \(SystemProfiler.detectChipModel()) \(ProcessInfo.processInfo.physicalMemory >> 30) GB")
        let report = await ModelInspector().inspect(modelIdOrPath: snapshot.path)
        guard report.intact else {
            record("error", "incomplete snapshot: \(report.errors.joined(separator: "; "))")
            throw BasaltError.modelNotFound(identifier: modelPath)
        }
        record("snapshot", "\(report.totalShards) shards, \(report.totalSizeFormatted), \(report.bitWidthDescription)")
    }

    func record(_ step: String, _ detail: String) {
        let line = "\(Date().ISO8601Format()) \(step) \(detail)\n"
        print(line, terminator: "")
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            try? handle.close()
        } else {
            try? Data(line.utf8).write(to: url)
        }
    }

    func step<T>(_ name: String, _ body: () async throws -> T) async throws -> T {
        do {
            return try await body()
        } catch {
            record("error", "\(name): \(error)")
            throw error
        }
    }

    static func commit() -> String {
        let git = Process()
        let output = Pipe()
        git.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        git.arguments = ["-C", CacheBlendQualityTests.root.path, "rev-parse", "HEAD"]
        git.standardOutput = output
        guard (try? git.run()) != nil else { return "unknown" }
        git.waitUntilExit()
        return String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

@Suite("CacheBlend output quality on a real model, teacher-forced against a full prefill (#36)", .serialized)
struct CacheBlendQualityTests {
    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_BLEND_MODEL_PATH"]
    }

    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()

    static let fixtures = root.appendingPathComponent("benchmarks/cacheblend")

    static let questions = [
        "in two sentences, what does the file that defines PromptCacheTree do?",
        "which command-line options does ServeCommand declare? List five.",
        "what does RepetitionGuard detect, and when does it stop generation?",
        "explain the formula used by EntropyDynamicTemperature.",
        "what fields does GenerationResponse carry?",
    ]

    static let orders = [[3, 5, 0, 2, 4, 1], [5, 4, 3, 2, 1, 0], [1, 3, 5, 0, 4, 2], [2, 0, 4, 5, 1, 3]]

    static func prompt(_ order: [Int], question: Int) throws -> String {
        let docs = try (0 ..< 6).map {
            try String(contentsOf: fixtures.appendingPathComponent("file\($0).txt"), encoding: .utf8)
        }
        return "You are reviewing a Swift codebase. Here are the files.\n\n" + order.map { docs[$0] }.joined()
            + "Question: " + questions[question]
    }

    static func greedy(_ prompt: [Int], count: Int, model: any LanguageModel) throws -> [Int] {
        let cache = try model.newCache(parameters: nil)
        var logits = model(MLXArray(prompt.map(Int32.init)).reshaped(1, -1), cache: cache)
        var answer: [Int] = []
        for _ in 0 ..< count {
            let next = argMax(logits[0, -1]).item(Int.self)
            answer.append(next)
            logits = model(MLXArray([Int32(next)]).reshaped(1, 1), cache: cache)
        }
        return answer
    }

    static func teacherForced(
        _ prompt: [Int], answer: [Int], cache: [KVCache], model: any LanguageModel
    ) -> (logProb: Float, top1: Float) {
        let fed = [prompt.last!] + answer.dropLast()
        let logits = model(MLXArray(fed.map(Int32.init)).reshaped(1, -1), cache: cache)[0].asType(.float32)
        let target = MLXArray(answer.map(Int32.init))
        let logProbs = logits - logSumExp(logits, axis: -1, keepDims: true)
        let picked = takeAlong(logProbs, target.expandedDimensions(axis: 1), axis: 1)
        let agree = argMax(logits, axis: -1) .== target
        return (picked.mean().item(Float.self), agree.asType(.float32).mean().item(Float.self))
    }

    @Test("Teacher-forced log-prob, top-1 agreement and perplexity of the full-prefill answer under blended caches; at the default ratio 0.3, with chunk heads recomputed first, the perplexity stays within 0.05 of a full prefill",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_BLEND_MODEL_PATH (a Llama or Qwen3 model)"))
    func blendQualityAgainstFullPrefill() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("cacheblend-quality", modelPath: path)
        let engine = MLXInferenceEngine()
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let input: @Sendable (String) -> UserInput = {
            UserInput(chat: [.user($0)], additionalContext: ["enable_thinking": false])
        }
        let primer = try await container.prepare(input: input(try Self.prompt([0, 1, 2, 3, 4, 5], question: 0)))
            .text.tokens.asArray(Int.self)
        var collected: [[Int]] = []
        for (index, order) in Self.orders.enumerated() {
            collected.append(try await container.prepare(input: input(try Self.prompt(order, question: index + 1)))
                .text.tokens.asArray(Int.self))
        }
        let prompts = collected
        let ratios: [Float] = [0, 0.15, 0.3, 0.5, 1]
        log.record("prompts", "\(prompts.count) reordered prompts, \(prompts.map(\.count)) tokens, ratios \(ratios)")
        let rows = try await log.step("measure") { try await container.perform { context in
            let model = context.model
            let store = CacheBlendStore(ratio: 1, byteBudget: 1 << 34)
            _ = try #require(store.extend(
                PromptCache(try model.newCache(parameters: nil)), prompt: primer, from: 0, namespace: "ns",
                model: model, tokenizer: context.tokenizer))
            func timedExtend(_ prompt: [Int], ratio: Float) throws -> (CacheBlendOutcome, Double) {
                let started = ContinuousClock.now
                let outcome = try #require(store.extend(
                    PromptCache(try model.newCache(parameters: nil)), prompt: prompt, from: 0, namespace: "ns",
                    model: model, tokenizer: context.tokenizer, ratio: ratio, storing: false))
                eval(outcome.cache.layers.flatMap { $0.innerState() })
                return (outcome, (ContinuousClock.now - started) / .seconds(1))
            }
            _ = try timedExtend(prompts[0], ratio: 1)
            var sums = [String: (logProb: Float, top1: Float, reused: Int, seconds: Double)]()
            for prompt in prompts {
                let answer = try Self.greedy(prompt, count: 50, model: model)
                let full = try model.newCache(parameters: nil)
                let started = ContinuousClock.now
                _ = model(MLXArray(prompt.dropLast().map(Int32.init)).reshaped(1, -1), cache: full)
                eval(full.flatMap { $0.innerState() })
                let fullSeconds = (ContinuousClock.now - started) / .seconds(1)
                let exact = Self.teacherForced(prompt, answer: answer, cache: full, model: model)
                let old = sums["full", default: (0, 0, 0, 0)]
                sums["full"] = (old.logProb + exact.logProb, old.top1 + exact.top1, 0, old.seconds + fullSeconds)
                for ratio in ratios {
                    let (outcome, seconds) = try timedExtend(prompt, ratio: ratio)
                    let blended = Self.teacherForced(prompt, answer: answer, cache: outcome.cache.layers, model: model)
                    let key = String(format: "%.2f", ratio)
                    let before = sums[key, default: (0, 0, 0, 0)]
                    sums[key] = (
                        before.logProb + blended.logProb, before.top1 + blended.top1, before.reused + outcome.reused,
                        before.seconds + seconds)
                }
            }
            let coldSeconds = sums["1.00"]!.seconds
            return (["full"] + ratios.map { String(format: "%.2f", $0) }).map { key -> (String, Float, Float, Float, Float) in
                let sum = sums[key]!
                let n = Float(prompts.count)
                let perplexity = exp(-sum.logProb / n)
                let line = String(
                    format: "%@ meanLogProb %.3f ppl %.4f top1 %.3f reused %d/%d prefill %.2fs speedup %.2fx", key,
                    sum.logProb / n, perplexity, sum.top1 / n, sum.reused / prompts.count,
                    prompts.map(\.count).reduce(0, +) / prompts.count, sum.seconds / Double(prompts.count),
                    coldSeconds / sum.seconds)
                let share = Float(sum.reused) / Float(prompts.map { $0.count - 1 }.reduce(0, +))
                return (line, sum.logProb / n, sum.top1 / n, share, perplexity)
            }
        } }
        for row in rows { log.record("result", "BLEND-QUALITY \(row.0)") }
        let byKey = Dictionary(uniqueKeysWithValues: zip(["full", "0.00", "0.15", "0.30", "0.50", "1.00"], rows))
        let full = try #require(byKey["full"])
        let exact = try #require(byKey["1.00"])
        let none = try #require(byKey["0.00"])
        let standard = try #require(byKey["0.30"])
        #expect(exact.2 >= full.2 - 0.01)
        #expect(full.2 >= 0.95)
        for key in ["0.00", "0.15", "0.30", "0.50"] {
            #expect(try #require(byKey[key]).3 >= 0.9)
        }
        #expect(none.1 < full.1 - 0.02)
        #expect(standard.1 > none.1)
        #expect(standard.2 >= 0.9)
        #expect(standard.4 - full.4 < 0.05)
    }
}

@Suite("CacheBlend refuses models it cannot splice, on real checkpoints (#137)", .serialized)
struct CacheBlendExclusionTests {
    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_BLEND_EXCLUDED_MODEL_PATH"]
    }

    static func answers(
        cacheBlendRatio: Float?, path: String, log: ModelRunLog
    ) async throws -> (texts: [String], cached: [Int], chunks: Int?) {
        let mode = cacheBlendRatio.map { "blend \($0)" } ?? "plain"
        let engine = MLXInferenceEngine(promptCacheBytes: 1 << 30, cacheBlendRatio: cacheBlendRatio)
        try await log.step("load \(mode)") { try await engine.load(modelPath: path) }
        var texts: [String] = []
        var cached: [Int] = []
        for (index, order) in ([[0, 1, 2, 3, 4, 5]] + CacheBlendQualityTests.orders.prefix(2)).enumerated() {
            let prompt = try CacheBlendQualityTests.prompt(order, question: index)
            let response = try await log.step("generate \(mode) \(index)") {
                try await engine.generate(request: greedy(prompt, key: "ns", maxTokens: 24))
            }
            log.record("generate", "\(mode) prompt \(index) order \(order) prompt \(response.promptTokens) cached \(response.cachedPromptTokens) answer \(response.generatedText.debugDescription)")
            texts.append(response.generatedText)
            cached.append(response.cachedPromptTokens)
        }
        return (texts, cached, await engine.cacheBlend?.chunkCount)
    }

    @Test("A sliding-window or hybrid model under --cache-blend stores no chunks and answers reordered files exactly like the plain prompt-cache path",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_BLEND_EXCLUDED_MODEL_PATH (a Gemma 4, Qwen3.5 or LFM2.5 model)"))
    func excludedModelMatchesPlainPath() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("cacheblend-excluded", modelPath: path)
        let blended = try await Self.answers(cacheBlendRatio: 0.3, path: path, log: log)
        Memory.clearCache()
        let plain = try await Self.answers(cacheBlendRatio: nil, path: path, log: log)
        log.record("result", "BLEND-EXCLUDED chunks \(blended.chunks ?? -1) cached \(blended.cached) plain \(plain.cached) same \(blended.texts == plain.texts)")
        #expect(blended.chunks == 0)
        #expect(plain.chunks == nil)
        #expect(blended.cached == plain.cached)
        #expect(blended.texts == plain.texts)
        #expect(blended.texts.allSatisfy { !$0.isEmpty })
    }
}
