import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("GSQ-RCO allocation transfer: perplexity the way llama-perplexity counts it, on the same corpus file, for each MLX checkpoint (#408 phase 0)", .serialized)
struct GGUFTransferRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static let models = NF4Run.models("BASALT_E2E_GGUF_TRANSFER_MODELS")
    static let corpus = env["BASALT_E2E_GGUF_CORPUS"]
    static let context = Int(env["BASALT_E2E_GGUF_CONTEXT"] ?? "") ?? 4096
    static let chunks = Int(env["BASALT_E2E_GGUF_CHUNKS"] ?? "") ?? 8

    static func load(_ url: URL) async throws -> ModelContainer {
        if let native = try await GGUFNativeLoader.loadContainer(from: url, tokenizerLoader: DefaultTokenizerLoader()) { return native }
        return try await loadModelContainer(from: url, using: DefaultTokenizerLoader())
    }

    static func perplexity(_ container: ModelContainer, text: String) async throws -> (ppl: Double, scored: Int) {
        try await container.perform { model in
            let tokens = model.tokenizer.encode(text: text, addSpecialTokens: false)
            var (nll, scored) = (0.0, 0)
            for chunk in 0 ..< min(chunks, tokens.count / context) {
                let window = Array(tokens[chunk * context ..< (chunk + 1) * context])
                let cache = try model.model.newCache(parameters: nil)
                var state: LMOutput.State?
                for start in stride(from: 0, to: context, by: 512) {
                    let end = min(context, start + 512)
                    let output = model.model(LMInput.Text(tokens: MLXArray(window[start ..< end].map(Int32.init))[.newAxis]), cache: cache, state: state)
                    state = output.state
                    let (first, last) = (max(start, context / 2), min(end, context - 1))
                    if first < last {
                        let logits = output.logits[0, (first - start) ..< (last - start)].asType(.float32)
                        let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
                        let targets = MLXArray(window[(first + 1) ... last].map(Int32.init))[0..., .newAxis]
                        nll -= Double(takeAlong(logp, targets, axis: -1).sum().item(Float.self))
                        scored += last - first
                    }
                    eval(cache)
                }
            }
            return (exp(nll / Double(max(scored, 1))), scored)
        }
    }

    @Test("Perplexity over chunks of the context, scoring each chunk's second half, plus peak memory",
          .enabled(if: !models.isEmpty && corpus != nil, "set BASALT_E2E=1, BASALT_E2E_GGUF_TRANSFER_MODELS and BASALT_E2E_GGUF_CORPUS"))
    func transfer() async throws {
        let text = try String(contentsOfFile: try #require(Self.corpus), encoding: .utf8)
        for entry in Self.models {
            let path = NF4Run.path(entry)
            let log = try await ModelRunLog("gguf-transfer", modelPath: path)
            defer { log.releaseGPU() }
            Memory.peakMemory = 0
            let container = try await log.step("load") { try await Self.load(URL(fileURLWithPath: path)) }
            let (ppl, scored) = try await log.step("perplexity") { try await Self.perplexity(container, text: text) }
            log.record("RESULT", String(format: "%@: ppl %.4f over %d scored tokens (%d chunks of %d), peak %.2f GB, %@", entry, ppl, scored, Self.chunks, Self.context,
                                        Double(Memory.peakMemory) / 1e9, log.host()))
            #expect(ppl.isFinite && ppl < 50)
        }
    }
}
