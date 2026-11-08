import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Abliterated and low-bit checkpoints: non-finite logits, layer activation norms and perplexity at 4k and 16k tokens (#44)", .serialized)
struct AbliterationRealModelTests {
    static let models = NF4Run.models("BASALT_E2E_ABLITERATION_MODELS")
    static let lengths = [4096, 16_384]

    static var corpus: String {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let files = ["README.md", "CONTRIBUTING.md", "docs/features.md", "docs/research-tools.md", "docs/ARCHITECTURE_PATTERNS_AND_MODULES.md", "CHANGELOG.md"]
        return files.compactMap { try? String(contentsOf: root.appendingPathComponent($0), encoding: .utf8) }.joined(separator: "\n\n")
    }

    struct Pass {
        var tokens: Int
        var nonFinite: Int
        var perplexity: Double
    }

    static func pass(_ container: ModelContainer, length: Int) async throws -> Pass {
        try await container.perform { context in
            let tokens = Array(context.tokenizer.encode(text: corpus).prefix(length + 1))
            let cache = try context.model.newCache(parameters: nil)
            var (nll, counted, nonFinite, start) = (0.0, 0, 0, 0)
            while start < tokens.count - 1 {
                let end = min(tokens.count - 1, start + 512)
                let logits = context.model(MLXArray(tokens[start ..< end].map(Int32.init))[.newAxis], cache: cache)[0].asType(.float32)
                nonFinite += logits.size - (abs(logits) .< Float.infinity).sum().item(Int.self)
                let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
                let picked = takeAlong(logp, MLXArray(tokens[(start + 1) ... end].map(Int32.init))[0..., .newAxis], axis: -1)
                let usable = abs(picked) .< Float.infinity
                nll -= Double(which(usable, picked, 0).sum().item(Float.self))
                counted += usable.sum().item(Int.self)
                eval(cache)
                start = end
            }
            return Pass(tokens: tokens.count - 1, nonFinite: nonFinite, perplexity: exp(nll / Double(max(counted, 1))))
        }
    }

    @Test("Every non-finite logit is counted, perplexity is taken over the finite ones, and the hidden-state norm of each layer is traced at the last position of a 2k prompt (trace caps at 2048 tokens)",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_ABLITERATION_MODELS"), arguments: models)
    func measure(entry: String) async throws {
        let path = NF4Run.path(entry)
        let log = try await ModelRunLog("abliteration", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 1 << 30))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let speed = try await log.step("decode") { try await NF4Run.decodeSpeed(engine) }
        log.record("RESULT", "decode \(speed) tok/s")
        var total = 0
        for length in Self.lengths {
            let result = try await log.step("prefill \(length)") { try await Self.pass(container, length: length) }
            log.record("RESULT", "\(result.tokens) tokens, non-finite logits \(result.nonFinite), perplexity \(result.perplexity)")
            total += result.nonFinite
        }
        let prompt = await container.perform { context in
            context.tokenizer.decode(tokenIds: Array(context.tokenizer.encode(text: Self.corpus).prefix(2000)), skipSpecialTokens: true)
        }
        let trace = try await log.step("trace") { try await engine.trace(TraceRequest(prompt: prompt, topK: 1)) }
        let norms = trace.layers.map(\.hiddenNorm)
        let peak = norms.enumerated().max { $0.element < $1.element }
        log.record("RESULT", "layer hidden norms: max \(peak?.element ?? .nan) at layer \(peak?.offset ?? -1), last \(norms.last ?? .nan), non-finite layers \(norms.filter { !$0.isFinite }.count), all \(norms.map { String(format: "%.0f", $0) }.joined(separator: " "))")
        #expect(total == 0 && norms.allSatisfy(\.isFinite), "non-finite logits \(total)")
    }
}
