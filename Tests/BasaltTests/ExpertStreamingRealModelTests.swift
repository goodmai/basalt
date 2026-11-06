import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Expert streaming on a real MoE that fits resident, forced to a small expert cache (#225)", .serialized)
struct ExpertStreamingRealModelTests {
    static var env: [String: String] { ProcessInfo.processInfo.environment }

    static var models: [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return ["BASALT_E2E_LFM25_MODEL_PATH", "BASALT_E2E_GEMMA4_MODEL_PATH"].compactMap { env[$0] }.map(GoldenLogitsRealModelTests.path)
    }

    static var cacheSizes: [Double] {
        (env["BASALT_E2E_EXPERT_CACHE_GB"] ?? "1,4").split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
    }

    static var tokens: Int { Int(env["BASALT_E2E_EXPERT_STREAMING_TOKENS"] ?? "") ?? 128 }

    static let prompt = "Explain step by step how a mixture-of-experts transformer routes each token to its experts, then write a short Swift function that picks the top-k entries of an array."

    struct Run {
        let response: GenerationResponse
        let peakBytes: Int
        let stats: (stats: ExpertCacheStats, capacity: ByteCount)?
        let logits: [Float]
    }

    static func lastLogits(_ engine: MLXInferenceEngine) async throws -> [Float] {
        let container = try #require(await engine.container)
        return try await container.performEvaluating(nonSendable: ()) { context, _ in
            let tokens = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
            let logits = context.model(MLXArray(tokens.map(Int32.init)).reshaped(1, -1), cache: try context.model.newCache(parameters: nil))
            return logits[0, -1].asType(.float32).asArray(Float.self)
        }
    }

    static func comparison(_ streamed: [Float], _ resident: [Float]) -> String {
        func top(_ logits: [Float]) -> [Int] { Array(logits.indices.sorted { logits[$0] > logits[$1] }.prefix(5)) }
        let drift = zip(streamed, resident).map { abs($0 - $1) }.max() ?? .infinity
        let shared = Set(top(streamed)).intersection(top(resident)).count
        return String(format: "prefill last-position logits max |diff| %g, top-5 overlap %d/5, argmax %@", drift, shared, top(streamed).first == top(resident).first ? "equal" : "differs")
    }

    static func run(_ path: String, streaming: ExpertStreaming?, log: ModelRunLog, step: String) async throws -> Run {
        Memory.clearCache()
        Memory.peakMemory = 0
        let engine = MLXInferenceEngine(EngineConfiguration(expertStreaming: streaming))
        try await log.step("\(step) load") { try await engine.load(modelPath: path) }
        let response = try await log.step("\(step) generate") {
            try await engine.generate(request: GenerationRequest(prompt: prompt, maxTokens: tokens, temperature: 0))
        }
        let logits = try await log.step("\(step) logits") { try await lastLogits(engine) }
        let run = Run(response: response, peakBytes: Memory.peakMemory, stats: await engine.expertCacheStats(), logits: logits)
        await engine.unloadCurrentModel()
        return run
    }

    static func fileTensors(_ directory: URL, model: any LanguageModel) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        var metadata: [String: String] = [:]
        for url in try ExpertStreamingLoader.shardURLs(in: directory) {
            let (arrays, meta) = try loadArraysAndMetadata(url: url)
            weights.merge(arrays) { _, new in new }
            if metadata.isEmpty { metadata = meta }
        }
        return model.sanitize(weights: weights, metadata: metadata)
    }

    static func audit(_ context: ModelContext, directory: URL) throws -> [String] {
        let files = try fileTensors(directory, model: context.model)
        let streamed = context.model.namedModules().compactMap { $0.1 as? StreamedSwitchLinear }
        guard let store = streamed.first?.store else { return ["no streamed expert modules"] }
        var findings: [String] = []
        let resident = context.model.parameters().flattened().filter { $0.1.shape != [1] }
        let differing = resident.filter { key, value in files[key].map { !MLX.arrayEqual($0, value).item(Bool.self) } ?? true }
        findings.append("non-expert parameters: \(resident.count), differing from the file or missing: \(differing.count) \(differing.prefix(5).map(\.0))")
        var expertMismatches: [String] = []
        for (index, layout) in store.layouts.enumerated() {
            for expert in [0, layout.numExperts / 2, layout.numExperts - 1] {
                let loaded = store.experts(module: index, [expert])[0]
                let parts: [(String, MLXArray?)] = [("weight", loaded.weight), ("scales", loaded.scales), ("biases", loaded.biases)]
                for (suffix, array) in parts {
                    guard let array, let file = files["\(layout.path).\(suffix)"] else { continue }
                    if !MLX.arrayEqual(file[expert], array).item(Bool.self) { expertMismatches.append("\(layout.path).\(suffix)[\(expert)]") }
                }
            }
        }
        findings.append("expert slices checked against the file: \(store.layouts.count) modules x 3 experts, differing: \(expertMismatches.count) \(expertMismatches.prefix(5))")
        for index in [0, store.layouts.count - 1] {
            let layout = store.layouts[index]
            guard let q = layout.quantization, let w = files["\(layout.path).weight"], let s = files["\(layout.path).scales"] else { continue }
            let assignments = 280
            let inputs = layout.dimensions.input
            let x = MLXRandom.normal([assignments, 1, inputs]).asType(.bfloat16)
            let indices = MLXArray((0 ..< assignments).map { UInt32($0 * layout.numExperts / assignments) })
            let reference = MLX.gatherQuantizedMM(
                x, w, scales: s, biases: files["\(layout.path).biases"], rhsIndices: indices, transpose: true,
                groupSize: q.groupSize, bits: q.bits, mode: q.mode, sortedIndices: true)
            let module = streamed.first { $0.module == index }!
            let output = module(x, indices, sortedIndices: true)
            findings.append("\(layout.path) forward on \(assignments) sorted assignments: max |diff| \(MLX.abs(output.asType(.float32) - reference.asType(.float32)).max().item(Float.self))")
        }
        return findings
    }

    @Test("Streamed weights equal the file tensors: non-expert parameters, expert slices read from disk, and one projection's forward against the full expert tensor",
          .enabled(if: env["BASALT_E2E"] == "1" && env["BASALT_E2E_EXPERT_AUDIT_MODEL_PATH"] != nil, "set BASALT_E2E=1 and BASALT_E2E_EXPERT_AUDIT_MODEL_PATH"))
    func streamedWeightsMatchFiles() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.env["BASALT_E2E_EXPERT_AUDIT_MODEL_PATH"]))
        let log = try await ModelRunLog("expert-streaming-audit", modelPath: path)
        defer { log.releaseGPU() }
        let directory = URL(fileURLWithPath: path)
        let container = try await log.step("load") {
            try await ExpertStreamingLoader.loadContainer(
                from: directory, streaming: ExpertStreaming(cacheBytes: ByteCount(1 << 30)), tokenizerLoader: DefaultTokenizerLoader())
        }
        let findings = try await container.performEvaluating(nonSendable: ()) { context, _ in try Self.audit(context, directory: directory) }
        for finding in findings { log.record("audit", finding) }
    }

    @Test("Greedy identity against resident, decode tok/s, expert hit rate and peak memory per forced expert-cache size",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_LFM25_MODEL_PATH and/or BASALT_E2E_GEMMA4_MODEL_PATH (optional BASALT_E2E_EXPERT_CACHE_GB=1,4 and BASALT_E2E_EXPERT_STREAMING_TOKENS=128)"),
          arguments: models)
    func streamingAgainstResident(path: String) async throws {
        let log = try await ModelRunLog("expert-streaming", modelPath: path)
        defer { log.releaseGPU() }
        let resident = try await Self.run(path, streaming: nil, log: log, step: "resident")
        log.record("resident", String(format: "decode %.2f tok/s, %d tokens, peak %d MB", resident.response.tokensPerSecond,
                                      resident.response.completionTokens, resident.peakBytes >> 20))
        var mismatches = 0
        for gigabytes in Self.cacheSizes {
            let streaming = ExpertStreaming(cacheBytes: ByteCount(Int64(gigabytes * 1_073_741_824)))
            let streamed = try await Self.run(path, streaming: streaming, log: log, step: "cache \(gigabytes) GB")
            let identical = streamed.response.generatedText == resident.response.generatedText
            if !identical { mismatches += 1 }
            log.record(identical ? "cache \(gigabytes) GB" : "FAIL cache \(gigabytes) GB", String(
                format: "greedy %@, decode %.2f tok/s (resident %.2f), ttft %.2f s, peak %d MB (resident %d MB), %@",
                identical ? "identical" : "differs", streamed.response.tokensPerSecond, resident.response.tokensPerSecond,
                streamed.response.timeToFirstToken, streamed.peakBytes >> 20, resident.peakBytes >> 20, streamed.stats?.stats.summary ?? "no expert cache"))
            log.record("cache \(gigabytes) GB logits", Self.comparison(streamed.logits, resident.logits))
            if !identical {
                log.record("diff", "resident: \(resident.response.generatedText.debugDescription) streamed: \(streamed.response.generatedText.debugDescription)")
            }
            #expect(streamed.stats != nil)
        }
        #expect(mismatches == 0)
    }
}
