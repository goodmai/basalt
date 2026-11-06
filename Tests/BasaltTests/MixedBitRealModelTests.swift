import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("A noise-variance mixed-bit checkpoint next to uniform affine of the same total bytes on real models (#292)", .serialized)
struct MixedBitRealModelTests {
    struct Measurement {
        var perplexity: Double?
        var decodeTokensPerSecond: Double
    }

    static let env = ProcessInfo.processInfo.environment

    static var sources: [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env["BASALT_E2E_MIXED_SOURCES"] ?? "").split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    static var corpus: String? { env["BASALT_E2E_MIXED_CORPUS"].flatMap { try? String(contentsOfFile: $0, encoding: .utf8) } }
    static var perplexityTokens: Int { env["BASALT_E2E_MIXED_PPL_TOKENS"].flatMap { Int($0) } ?? 4096 }
    static var decodeTokens: Int { env["BASALT_E2E_MIXED_DECODE_TOKENS"].flatMap { Int($0) } ?? 256 }

    @Test("Size on disk, perplexity and decode tok/s of uniform affine and of mixed bits planned to the uniform size",
          .enabled(if: !sources.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_MIXED_SOURCES"), arguments: sources)
    func compare(source: String) async throws {
        let log = try await ModelRunLog("mixed-bit", modelPath: source)
        defer { log.releaseGPU() }
        let root = env["BASALT_E2E_MIXED_OUT"].map { URL(fileURLWithPath: $0) } ?? FileManager.default.temporaryDirectory
        let stem = root.appending(component: "basalt-mixed-\(UUID().uuidString.prefix(8))")
        defer { if env["BASALT_E2E_MIXED_KEEP"] != "1" { try? FileManager.default.removeItem(at: stem) } }
        let uniformSpec = env["BASALT_E2E_MIXED_UNIFORM"] ?? "affine:3:64"
        let mixedSpec = env["BASALT_E2E_MIXED"] ?? "mixed:2:4:64"
        log.record("setup", "uniform \(uniformSpec) mixed \(mixedSpec) out \(stem.path) corpus \(env["BASALT_E2E_MIXED_CORPUS"] ?? "unset") pplTokens \(Self.perplexityTokens) decodeTokens \(Self.decodeTokens)")
        let uniform = try await log.step("convert \(uniformSpec)") {
            try Self.convert(source, spec: uniformSpec, budget: nil, to: stem.appending(component: "uniform"), log: log)
        }
        _ = try await log.step("convert \(mixedSpec) budget \(uniform.bytesAfter)") {
            try Self.convert(source, spec: mixedSpec, budget: uniform.bytesAfter, to: stem.appending(component: "mixed"), log: log)
        }
        var results: [String: Measurement] = [:]
        for label in ["uniform", "mixed"] {
            results[label] = try? await log.step("measure \(label)") { try await Self.measure(stem.appending(component: label), label: label, log: log) }
        }
        if let base = results["uniform"], let mixed = results["mixed"] {
            log.record("compare", String(
                format: "ppl uniform %.4f mixed %.4f, decode tok/s uniform %.2f mixed %.2f (ratio %.3f)",
                base.perplexity ?? .nan, mixed.perplexity ?? .nan, base.decodeTokensPerSecond, mixed.decodeTokensPerSecond,
                mixed.decodeTokensPerSecond / base.decodeTokensPerSecond))
        }
        #expect(results.count == 2, "a measurement failed; see the FAIL lines in \(log.url.path)")
    }

    static func convert(_ source: String, spec: String, budget: Int?, to output: URL, log: ModelRunLog) throws -> WeightConverter.Summary {
        var precision = try WeightConverter.precision(quantize: spec, dequantize: nil, format: .mlx, budgetGB: budget.map { _ in 1 })
        if let budget, case .mixed(var mixed) = precision {
            mixed.budgetBytes = budget
            precision = .mixed(mixed)
        }
        let start = ContinuousClock.now
        let summary = try WeightConverter(precision: precision).run(source: URL(fileURLWithPath: source), output: output)
        let seconds = (ContinuousClock.now - start) / .seconds(1)
        let plan = summary.plan.map { " high \($0.highLayers.count)/\($0.layers) layers, unscored \($0.unscored)" } ?? ""
        log.record("converted", String(format: "%@ bytes %d → %d, %d shards%@, %.0f s", spec, summary.bytesBefore, summary.bytesAfter, summary.shards, plan, seconds))
        return summary
    }

    static func measure(_ directory: URL, label: String, log: ModelRunLog) async throws -> Measurement {
        let engine = MLXInferenceEngine()
        try await engine.load(modelPath: directory.path)
        var perplexity: Double?
        if let corpus, let container = await engine.container {
            let count = perplexityTokens
            perplexity = try await container.perform { (context: ModelContext) throws -> Double in
                let tokens = Array(context.tokenizer.encode(text: corpus).prefix(count + 1))
                return try Self.perplexity(tokens: tokens, context: context)
            }
        } else {
            log.record("SKIP ppl", "\(label): BASALT_E2E_MIXED_CORPUS is unset or unreadable")
        }
        var request = GenerationRequest(prompt: "Explain how basalt columns form as lava cools.", maxTokens: decodeTokens, temperature: 0)
        request.enableThinking = false
        let response = try await engine.generate(request: request)
        log.record("measured", String(
            format: "%@ ppl %.4f over %d tokens, decode %.2f tok/s over %d tokens, ttft %.3f s",
            label, perplexity ?? .nan, perplexityTokens, response.tokensPerSecond, response.completionTokens, response.timeToFirstToken))
        return Measurement(perplexity: perplexity, decodeTokensPerSecond: response.tokensPerSecond)
    }

    static func perplexity(tokens: [Int], context: ModelContext) throws -> Double {
        guard tokens.count > 1 else { throw BasaltError.invalidRequestStructure(details: "the corpus has \(tokens.count) tokens") }
        let cache = try context.model.newCache(parameters: nil)
        var nll = 0.0
        var start = 0
        while start < tokens.count - 1 {
            let end = min(tokens.count - 1, start + 512)
            let logits = context.model(MLXArray(tokens[start ..< end].map(Int32.init))[.newAxis], cache: cache)[0].asType(.float32)
            let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
            let targets = MLXArray(tokens[(start + 1) ... end].map(Int32.init))
            nll -= Double(takeAlong(logp, targets[0..., .newAxis], axis: -1).sum().item(Float.self))
            eval(cache)
            start = end
        }
        return exp(nll / Double(tokens.count - 1))
    }
}
