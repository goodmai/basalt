import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Prefill FFN split on a real model against GPU-only: cold-prompt TTFT at 8k and 16k, decode, process memory, perplexity and greedy agreement (#274 3b-4)", .serialized)
struct FFNSplitRealModelTests {
    struct Arm {
        var ttft: [Int: Double] = [:]
        var decode = 0.0
        var footprintMB = 0.0
        var perplexity = 0.0
        var greedy = ""
    }

    static let env = ProcessInfo.processInfo.environment
    static let model = env["BASALT_E2E"] == "1" ? env["BASALT_E2E_FFN_SPLIT_MODEL"] : nil
    static let bundle = env["BASALT_E2E_FFN_SPLIT_BUNDLE"]
    static let lengths = [8192, 16384]

    static var corpus: String {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let files = ["README.md", "CONTRIBUTING.md"] + ((try? FileManager.default.contentsOfDirectory(atPath: root.appending(path: "docs").path)) ?? [])
            .filter { $0.hasSuffix(".md") }.sorted().map { "docs/" + $0 }
        return files.compactMap { try? String(contentsOf: root.appending(path: $0), encoding: .utf8) }.joined(separator: "\n\n")
    }

    static func text(_ container: ModelContainer, tokens: Int) async -> String {
        let corpus = corpus
        return await container.perform { context in context.tokenizer.decode(tokenIds: Array(context.tokenizer.encode(text: corpus).prefix(tokens))) }
    }

    static func perplexity(_ container: ModelContainer, text: String) async throws -> Double {
        try await container.performSending(nonSendable: text) { context, text in
            let model = try #require(context.model as? any SplitFeedForwardPrefill & Module)
            let box = context.feedForwardSplit as? SplitFeedForwardBox
            let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
            let tokens = Array(context.tokenizer.encode(text: text).prefix(4097))
            let cache = try context.model.newCache(parameters: nil)
            var (nll, start) = (0.0, 0)
            while start < tokens.count - 1 {
                let end = min(tokens.count - 1, start + 2048)
                let logits = try await model.prefill(text: MLXArray(tokens[start ..< end].map(Int32.init)), cache: cache) { layer, x in
                    if let box { return try await box.split(layer, x) }
                    let mlp = try #require(modules[model.feedForwardPath(layer: layer)] as? any UnaryLayer)
                    return mlp(x)
                }[0].asType(.float32)
                let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
                nll -= Double(takeAlong(logp, MLXArray(tokens[(start + 1) ... end].map(Int32.init))[0..., .newAxis], axis: -1).sum().item(Float.self))
                eval(cache)
                start = end
            }
            return exp(nll / Double(tokens.count - 1))
        }
    }

    static func measure(split: Bool, path: String, log: ModelRunLog) async throws -> Arm {
        var configuration = EngineConfiguration(promptCacheBytes: 0)
        configuration.feedForwardSplitBundle = split ? bundle : nil
        let engine = MLXInferenceEngine(configuration)
        let label = split ? "split" : "gpu"
        try await log.step("load \(label)") { try await engine.load(modelPath: path) }
        if split { try #require(await engine.modelFacts.feedForwardSplitLengths != nil, "the split did not attach") }
        let container = try #require(await engine.container)
        var arm = Arm(footprintMB: Double(await AppBudgetTests.footprint()) / 1_048_576)
        _ = try await engine.generate(request: GenerationRequest(prompt: await text(container, tokens: 2048), maxTokens: 1, temperature: 0))
        for length in lengths {
            let prompt = await text(container, tokens: length)
            arm.ttft[length] = try await log.step("ttft \(label) \(length)") {
                try await engine.generate(request: GenerationRequest(prompt: prompt, maxTokens: 1, temperature: 0)).timeToFirstToken
            }
        }
        arm.decode = try await engine.generate(request: GenerationRequest(prompt: "Write a long story about a lighthouse keeper.", maxTokens: 256, temperature: 0)).tokensPerSecond
        arm.greedy = try await engine.generate(request: GenerationRequest(prompt: await text(container, tokens: 4096) + "\n\nSummarize the text above.", maxTokens: 64, temperature: 0)).generatedText
        arm.perplexity = try await perplexity(container, text: corpus)
        log.record("arm", String(format: "%@ ttft 8k %.2f s 16k %.2f s, decode %.2f tok/s, footprint %.0f MB, perplexity %.4f", label, arm.ttft[8192] ?? 0, arm.ttft[16384] ?? 0, arm.decode, arm.footprintMB, arm.perplexity))
        await engine.unload()
        return arm
    }

    @Test("TTFT at 8k and 16k is at least 1.3x GPU-only, decode stays within 2%, perplexity within 0.5%; memory and greedy agreement are logged",
          .enabled(if: model != nil && bundle != nil, "set BASALT_E2E=1, BASALT_E2E_FFN_SPLIT_MODEL and BASALT_E2E_FFN_SPLIT_BUNDLE (swift scripts/ffn_split_export.swift); run a release build"))
    func againstGPU() async throws {
        let path = NF4Run.path(try #require(Self.model))
        let log = try await ModelRunLog("ffn-split-e2e", modelPath: path)
        defer { log.releaseGPU() }
        let gpu = try await Self.measure(split: false, path: path, log: log)
        let split = try await Self.measure(split: true, path: path, log: log)
        let agreement = zip(gpu.greedy, split.greedy).prefix { $0 == $1 }.count
        for length in Self.lengths {
            let speedup = (gpu.ttft[length] ?? 0) / max(split.ttft[length] ?? 1, 1e-9)
            log.record("RESULT", String(format: "%d tokens: TTFT %.2f → %.2f s (%.2fx)", length, gpu.ttft[length] ?? 0, split.ttft[length] ?? 0, speedup))
            #expect(speedup >= 1.3, "TTFT at \(length): \(speedup)x")
        }
        log.record("RESULT", String(format: "decode %.2f → %.2f tok/s, footprint %.0f → %.0f MB (+%.0f), perplexity %.4f → %.4f (%+.2f%%), greedy agrees on the first %d of %d characters",
            gpu.decode, split.decode, gpu.footprintMB, split.footprintMB, split.footprintMB - gpu.footprintMB, gpu.perplexity, split.perplexity,
            (split.perplexity / gpu.perplexity - 1) * 100, agreement, gpu.greedy.count))
        #expect(abs(split.decode / gpu.decode - 1) <= 0.02, "decode \(split.decode) against \(gpu.decode)")
        #expect(split.perplexity <= gpu.perplexity * 1.005, "perplexity \(split.perplexity) against \(gpu.perplexity)")
    }
}
