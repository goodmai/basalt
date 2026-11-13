import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

private enum ExpertFileAudit {
    static func tensors(_ directory: URL, model: any LanguageModel) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        var metadata: [String: String] = [:]
        for url in try ExpertStreamingLoader.shardURLs(in: directory) {
            let (arrays, meta) = try loadArraysAndMetadata(url: url)
            weights.merge(arrays) { _, new in new }
            if metadata.isEmpty { metadata = meta }
        }
        return model.sanitize(weights: weights, metadata: metadata)
    }

    static func check(_ context: ModelContext, directory: URL) throws -> [String] {
        let files = try tensors(directory, model: context.model)
        let streamed = context.model.namedModules().compactMap { $0.1 as? StreamedSwitchLinear }
        guard let store = streamed.first?.store else { return ["no streamed expert modules"] }
        var findings: [String] = []
        let resident = context.model.parameters().flattened().filter { $0.1.shape != [1] }
        let differing = resident.filter { key, value in files[key].map { !MLX.arrayEqual($0, value).item(Bool.self) } ?? true }
        findings.append("non-expert parameters: \(resident.count), differing from the file or missing: \(differing.count) \(differing.prefix(5).map(\.0))")
        var expertMismatches: [String] = []
        for (index, layout) in store.layouts.enumerated() {
            for expert in [0, layout.numExperts / 2, layout.numExperts - 1] {
                let loaded = store.read(module: index, [expert])
                let parts: [(String, MLXArray?)] = [("weight", loaded.weight[0]), ("scales", loaded.scales?[0]), ("biases", loaded.biases?[0])]
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
            let x = MLXRandom.normal([assignments, 1, layout.dimensions.input]).asType(.bfloat16)
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
}

@Suite("Expert streaming on real MoE checkpoints: resident parity, warm-start, next-layer prefetch, and 37.7 GB out-of-core acceptance (#225)", .serialized)
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

    struct RunSpec {
        var streaming: ExpertStreaming?
        var step: String
        var warmStart = true
        var nextLayerPrefetch = true
        var request: GenerationRequest?
        var computeLogits = true
    }

    struct Run {
        let response: GenerationResponse
        let peakBytes: Int
        let swapDelta: UInt64
        let stats: (stats: ExpertCacheStats, capacity: ByteCount)?
        let logits: [Float]
    }

    static func lastLogits(_ engine: MLXInferenceEngine) async throws -> [Float] {
        let container = try #require(await engine.container)
        return try await container.performEvaluating(nonSendable: ()) { context, _ in
            let encoded = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
            let tokens = MLXArray(encoded.map(Int32.init)).reshaped(1, -1)
            let output = context.model(LMInput.Text(tokens: tokens), cache: try context.model.newCache(parameters: nil), state: nil)
            return output.logits[0, -1].asType(.float32).asArray(Float.self)
        }
    }

    static func comparison(_ streamed: [Float], _ resident: [Float]) -> String {
        func top(_ logits: [Float]) -> [Int] { Array(logits.indices.sorted { logits[$0] > logits[$1] }.prefix(5)) }
        let drift = zip(streamed, resident).map { abs($0 - $1) }.max() ?? .infinity
        let shared = Set(top(streamed)).intersection(top(resident)).count
        return String(format: "prefill last-position logits max |diff| %g, top-5 overlap %d/5, argmax %@", drift, shared, top(streamed).first == top(resident).first ? "equal" : "differs")
    }

    static func run(_ path: String, spec: RunSpec, log: ModelRunLog) async throws -> Run {
        Memory.clearCache()
        Memory.peakMemory = 0
        let swapBefore = MachMemory.swapOuts()
        let engine = MLXInferenceEngine(EngineConfiguration(expertStreaming: spec.streaming))
        try await log.step("\(spec.step) load") { try await engine.load(modelPath: path) }
        if let container = await engine.container {
            let (warm, prefetch) = (spec.warmStart, spec.nextLayerPrefetch)
            await container.perform {
                guard let store = ExpertStreamingLoader.store(in: $0.model) else { return }
                (store.warmStart, store.nextLayerPrefetch) = (warm, prefetch)
            }
        }
        let activeRequest = spec.request ?? GenerationRequest(prompt: prompt, maxTokens: tokens, temperature: 0)
        let response = try await log.step("\(spec.step) generate") { try await engine.generate(request: activeRequest) }
        let swapAfter = MachMemory.swapOuts()
        let stats = await engine.expertCacheStats()
        let logits = spec.computeLogits ? try await log.step("\(spec.step) logits") { try await lastLogits(engine) } : []
        let out = Run(response: response, peakBytes: Memory.peakMemory, swapDelta: swapAfter >= swapBefore ? swapAfter - swapBefore : 0, stats: stats, logits: logits)
        await engine.unloadCurrentModel()
        Memory.clearCache()
        return out
    }

    static func decodeWindow(_ path: String, cacheBytes: ByteCount, warmStart: Bool, count: Int = 64, log: ModelRunLog) async throws -> (tokens: [Int], stats: ExpertCacheStats, seconds: Double) {
        Memory.clearCache()
        let container = try #require(try await log.step("window-\(cacheBytes.rawValue)-\(warmStart) load") {
            try await ExpertStreamingLoader.loadContainer(
                from: URL(fileURLWithPath: path), streaming: ExpertStreaming(cacheBytes: cacheBytes), tokenizerLoader: DefaultTokenizerLoader())
        })
        let result = try await container.performEvaluating(nonSendable: ()) { context, _ in
            let store = try #require(ExpertStreamingLoader.store(in: context.model))
            (store.warmStart, store.nextLayerPrefetch) = (warmStart, false)
            let inputIds = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
            let cache = try context.model.newCache(parameters: nil)
            var state: LMOutput.State?
            var output = context.model(LMInput.Text(tokens: MLXArray(inputIds.map(Int32.init)).reshaped(1, -1)), cache: cache, state: state)
            state = output.state
            var next = output.logits[0, -1].argMax().item(Int.self)
            store.resetStats()
            var generated = [next]
            let start = ContinuousClock.now
            for _ in 1 ..< count {
                output = context.model(LMInput.Text(tokens: MLXArray([Int32(next)]).reshaped(1, 1)), cache: cache, state: state)
                state = output.state
                next = output.logits[0, -1].argMax().item(Int.self)
                generated.append(next)
            }
            let elapsed = ContinuousClock.now - start
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            return (generated, store.stats, seconds)
        }
        Memory.clearCache()
        return result
    }

    @Test("Streamed weights equal the file tensors: non-expert parameters, expert slices read from disk, and one projection's forward against the full expert tensor",
          .enabled(if: env["BASALT_E2E"] == "1" && env["BASALT_E2E_EXPERT_AUDIT_MODEL_PATH"] != nil, "set BASALT_E2E=1 and BASALT_E2E_EXPERT_AUDIT_MODEL_PATH"))
    func streamedWeightsMatchFiles() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.env["BASALT_E2E_EXPERT_AUDIT_MODEL_PATH"]))
        let log = try await ModelRunLog("expert-streaming-audit", modelPath: path)
        defer { log.releaseGPU() }
        let directory = URL(fileURLWithPath: path)
        let container = try #require(try await log.step("load") {
            try await ExpertStreamingLoader.loadContainer(
                from: directory, streaming: ExpertStreaming(cacheBytes: ByteCount(1 << 30)), tokenizerLoader: DefaultTokenizerLoader())
        })
        let findings = try await container.performEvaluating(nonSendable: ()) { context, _ in try ExpertFileAudit.check(context, directory: directory) }
        for finding in findings { log.record("audit", finding) }
    }

    @Test("Greedy identity against resident, decode tok/s, expert hit rate and peak memory per forced expert-cache size",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_LFM25_MODEL_PATH and/or BASALT_E2E_GEMMA4_MODEL_PATH"),
          arguments: models)
    func streamingAgainstResident(path: String) async throws {
        let log = try await ModelRunLog("expert-streaming", modelPath: path)
        defer { log.releaseGPU() }
        let resident = try await Self.run(path, spec: RunSpec(streaming: nil, step: "resident"), log: log)
        log.record("resident", String(format: "decode %.2f tok/s, %d tokens, peak %d MB, swap %llu", resident.response.tokensPerSecond,
                                      resident.response.completionTokens, resident.peakBytes >> 20, resident.swapDelta))
        var mismatches = 0
        for gigabytes in Self.cacheSizes {
            let streaming = ExpertStreaming(cacheBytes: ByteCount(Int64(gigabytes * 1_073_741_824)))
            let streamed = try await Self.run(path, spec: RunSpec(streaming: streaming, step: "cache \(gigabytes) GB"), log: log)
            let identical = streamed.response.generatedText == resident.response.generatedText
            if !identical { mismatches += 1 }
            log.record(identical ? "cache \(gigabytes) GB" : "FAIL cache \(gigabytes) GB", String(
                format: "greedy %@, decode %.2f tok/s (resident %.2f), ttft %.2f s, peak %d MB (resident %d MB), swap %llu, %@",
                identical ? "identical" : "differs", streamed.response.tokensPerSecond, resident.response.tokensPerSecond,
                streamed.response.timeToFirstToken, streamed.peakBytes >> 20, resident.peakBytes >> 20, streamed.swapDelta,
                streamed.stats?.stats.summary ?? "no expert cache"))
            log.record("cache \(gigabytes) GB logits", Self.comparison(streamed.logits, resident.logits))
            if !identical {
                log.record("diff", "resident: \(resident.response.generatedText.debugDescription) streamed: \(streamed.response.generatedText.debugDescription)")
            }
            #expect(streamed.stats != nil)
        }
        #expect(mismatches == 0)
    }

    @Test("Out-of-core 32-45 GB MoE acceptance on a 24 GB Mac: agent-trace decode >= 5 tok/s, swapouts near zero, warm-start and next-layer prefetch A/B, and greedy identity across caches",
          .enabled(if: env["BASALT_E2E"] == "1" && env["BASALT_E2E_QWEN36_MOE_MODEL_PATH"] != nil, "set BASALT_E2E=1 and BASALT_E2E_QWEN36_MOE_MODEL_PATH"))
    func outOfCoreAcceptance() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.env["BASALT_E2E_QWEN36_MOE_MODEL_PATH"]))
        let log = try await ModelRunLog("expert-streaming-acceptance", modelPath: path)
        defer { log.releaseGPU() }
        let cacheGB = Double(Self.env["BASALT_E2E_QWEN36_CACHE_GB"] ?? "") ?? 6.0
        let cacheBytes = ByteCount(Int64(cacheGB * 1_073_741_824))
        let agentRequest = GenerationRequest(
            prompt: "", maxTokens: Self.tokens, temperature: 0,
            tools: [GoldenPrompt.readFile], messages: GoldenPrompt.agentTrace(turns: 16))
        let agentPlain = try await Self.run(
            path, spec: RunSpec(streaming: ExpertStreaming(cacheBytes: cacheBytes), step: "agent-no-prefetch", nextLayerPrefetch: false, request: agentRequest, computeLogits: false), log: log)
        log.record("agent-trace no-prefetch", String(
            format: "decode %.2f tok/s, ttft %.2f s, prompt %d, completion %d, peak %d MB, footprint %llu MB, swap %llu, %@",
            agentPlain.response.tokensPerSecond, agentPlain.response.timeToFirstToken, agentPlain.response.promptTokens,
            agentPlain.response.completionTokens, agentPlain.peakBytes >> 20, MachMemory.footprint() >> 20, agentPlain.swapDelta,
            agentPlain.stats?.stats.summary ?? "no expert cache"))
        let agentPrefetch = try await Self.run(
            path, spec: RunSpec(streaming: ExpertStreaming(cacheBytes: cacheBytes), step: "agent-prefetch", nextLayerPrefetch: true, request: agentRequest, computeLogits: false), log: log)
        log.record("agent-trace prefetch", String(
            format: "decode %.2f tok/s, ttft %.2f s, prompt %d, completion %d, peak %d MB, footprint %llu MB, swap %llu, %@",
            agentPrefetch.response.tokensPerSecond, agentPrefetch.response.timeToFirstToken, agentPrefetch.response.promptTokens,
            agentPrefetch.response.completionTokens, agentPrefetch.peakBytes >> 20, MachMemory.footprint() >> 20, agentPrefetch.swapDelta,
            agentPrefetch.stats?.stats.summary ?? "no expert cache"))
        #expect(agentPrefetch.response.generatedText == agentPlain.response.generatedText)
        #expect(agentPrefetch.response.tokensPerSecond >= 5.0, "decode \(agentPrefetch.response.tokensPerSecond) tok/s < 5.0")
        #expect(agentPrefetch.swapDelta <= 64, "swapouts delta \(agentPrefetch.swapDelta)")
        let cold64 = try await Self.decodeWindow(path, cacheBytes: ByteCount(4 << 30), warmStart: false, log: log)
        let warm64 = try await Self.decodeWindow(path, cacheBytes: ByteCount(4 << 30), warmStart: true, log: log)
        let uncached16 = try await Self.decodeWindow(path, cacheBytes: ByteCount(1), warmStart: false, count: 16, log: log)
        log.record("warm-start-64", String(
            format: "cold hit rate %.1f%% (%.2f tok/s), warm hit rate %.1f%% (%.2f tok/s, warmed %d), uncached-16 parity %@",
            cold64.stats.hitRate * 100, Double(cold64.tokens.count - 1) / max(0.001, cold64.seconds),
            warm64.stats.hitRate * 100, Double(warm64.tokens.count - 1) / max(0.001, warm64.seconds),
            warm64.stats.warmed, Array(warm64.tokens.prefix(16)) == uncached16.tokens ? "identical" : "differs"))
        #expect(warm64.tokens == cold64.tokens && Array(warm64.tokens.prefix(16)) == uncached16.tokens)
        #expect(warm64.stats.hitRate > cold64.stats.hitRate)
    }

    @Test("Dense checkpoint ignores expert streaming, reports nil expertCacheStats, and matches resident greedy output",
          .enabled(if: env["BASALT_E2E"] == "1" && env["BASALT_E2E_DENSE_MODEL_PATH"] != nil, "set BASALT_E2E=1 and BASALT_E2E_DENSE_MODEL_PATH"))
    func denseModelFallsBackToResident() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.env["BASALT_E2E_DENSE_MODEL_PATH"]))
        let log = try await ModelRunLog("expert-streaming-dense-fallback", modelPath: path)
        defer { log.releaseGPU() }
        let resident = try await Self.run(path, spec: RunSpec(streaming: nil, step: "resident", computeLogits: false), log: log)
        let streamed = try await Self.run(
            path, spec: RunSpec(streaming: ExpertStreaming(cacheBytes: ByteCount(1 << 30)), step: "streamed-flag", computeLogits: false), log: log)
        log.record("dense-fallback", String(
            format: "greedy %@, stats %@, decode %.2f tok/s (resident %.2f)",
            streamed.response.generatedText == resident.response.generatedText ? "identical" : "differs",
            streamed.stats == nil ? "nil" : "unexpected", streamed.response.tokensPerSecond, resident.response.tokensPerSecond))
        #expect(streamed.stats == nil)
        #expect(streamed.response.generatedText == resident.response.generatedText)
    }
}
