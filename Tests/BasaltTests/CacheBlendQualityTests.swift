import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

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

    @Test("Teacher-forced log-prob and top-1 agreement of the full-prefill answer under blended caches",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_BLEND_MODEL_PATH (a Llama or Qwen3 model)"))
    func blendQualityAgainstFullPrefill() async throws {
        let engine = MLXInferenceEngine()
        try await engine.load(modelPath: try #require(Self.modelPath))
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
        let rows = try await container.perform { context in
            let model = context.model
            let store = CacheBlendStore(ratio: 1, byteBudget: 1 << 34)
            _ = try #require(store.extend(
                PromptCache(try model.newCache(parameters: nil)), prompt: primer, from: 0, namespace: "ns",
                model: model, tokenizer: context.tokenizer))
            var sums = [String: (logProb: Float, top1: Float, reused: Int)]()
            for prompt in prompts {
                let answer = try Self.greedy(prompt, count: 50, model: model)
                let full = try model.newCache(parameters: nil)
                _ = model(MLXArray(prompt.dropLast().map(Int32.init)).reshaped(1, -1), cache: full)
                let exact = Self.teacherForced(prompt, answer: answer, cache: full, model: model)
                let old = sums["full", default: (0, 0, 0)]
                sums["full"] = (old.logProb + exact.logProb, old.top1 + exact.top1, 0)
                for ratio in ratios {
                    let outcome = try #require(store.extend(
                        PromptCache(try model.newCache(parameters: nil)), prompt: prompt, from: 0, namespace: "ns",
                        model: model, tokenizer: context.tokenizer, ratio: ratio, storing: false))
                    let blended = Self.teacherForced(prompt, answer: answer, cache: outcome.cache.layers, model: model)
                    let key = String(format: "%.2f", ratio)
                    let before = sums[key, default: (0, 0, 0)]
                    sums[key] = (before.logProb + blended.logProb, before.top1 + blended.top1, before.reused + outcome.reused)
                }
            }
            return (["full"] + ratios.map { String(format: "%.2f", $0) }).map { key -> (String, Float, Float) in
                let sum = sums[key]!
                let n = Float(prompts.count)
                let line = String(format: "%@ meanLogProb %.3f top1 %.3f reused %d/%d", key, sum.logProb / n, sum.top1 / n,
                                  sum.reused / prompts.count, prompts.map(\.count).reduce(0, +) / prompts.count)
                return (line, sum.logProb / n, sum.top1 / n)
            }
        }
        for row in rows { print("BLEND-QUALITY", row.0) }
        let byKey = Dictionary(uniqueKeysWithValues: zip(["full", "0.00", "0.15", "0.30", "0.50", "1.00"], rows))
        let full = try #require(byKey["full"])
        let exact = try #require(byKey["1.00"])
        let none = try #require(byKey["0.00"])
        let half = try #require(byKey["0.50"])
        #expect(exact.2 >= full.2 - 0.01)
        #expect(half.1 >= none.1)
        #expect(full.2 >= 0.95)
    }
}
