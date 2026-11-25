import Foundation
import MLX
import MLXLMCommon
import Synchronization
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Long-prefill GPU transient attribution and gate on real Gemma 4 models (#769)", .serialized)
struct PrefillTransientRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static var modelPath: String? { env["BASALT_E2E"] == "1" ? env["BASALT_E2E_PREFILL_TRANSIENT_MODEL_PATH"] : nil }
    static var targetTokens: Int { Int(env["BASALT_E2E_PREFILL_TRANSIENT_TOKENS"] ?? "") ?? 32_768 }
    static var decodeTokens: Int { Int(env["BASALT_E2E_PREFILL_TRANSIENT_DECODE"] ?? "") ?? 32 }
    static var skipStock: Bool { env["BASALT_E2E_PREFILL_TRANSIENT_SKIP_STOCK"] == "1" }
    static var cacheKeyMode: Bool { env["BASALT_E2E_PREFILL_TRANSIENT_CACHE_KEY"] == "1" }
    static let filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

    struct ArmConfig: Sendable {
        let arm: String
        let usePlan: Bool
        let scratch: PrefillAttentionScratch?
        let promptText: String
    }

    struct ArmOutcome: Sendable {
        let arm: String
        let promptTokens: Int
        let tokens: [Int]
        let answer: String
        let prefillSeconds: Double
        let decodeSeconds: Double
        let kvMB: Int
        let activeAfterPrefillMB: Int
        let peakPrefillMB: Int
        let peakDecodeMB: Int
        let peakStoreMB: Int
    }

    static func mb(_ bytes: Int) -> Int { bytes / 1_048_576 }

    static var syntheticEnabled: Bool { env["BASALT_E2E_SYNTHETIC_DONATION"] == "1" }

    @Test("Synthetic 8-layer D=512 prefill, decode, and PromptCacheTree.store measure in-place donation and draining store peak",
          .enabled(if: syntheticEnabled, "set BASALT_E2E_SYNTHETIC_DONATION=1"))
    func syntheticDonationProbe() {
        let layerCount = 8
        let totalTokens = 8_192
        let kvBytes = layerCount * 2 * totalTokens * 512 * 2
        let stock = Self.measureSyntheticPass(preallocate: false, syncPerChunk: false, totalTokens: totalTokens, layerCount: layerCount)
        let preallocOnly = Self.measureSyntheticPass(preallocate: true, syncPerChunk: false, totalTokens: totalTokens, layerCount: layerCount)
        let preallocSync = Self.measureSyntheticPass(preallocate: true, syncPerChunk: true, totalTokens: totalTokens, layerCount: layerCount)
        let storeStockMB = Self.measureSyntheticStorePeak(drainSource: false, totalTokens: totalTokens, layerCount: layerCount)
        let storeDrainMB = Self.measureSyntheticStorePeak(drainSource: true, totalTokens: totalTokens, layerCount: layerCount)
        print("SYNTHETIC_DONATION kv_mb=\(Self.mb(kvBytes)) stock_transient_mb=\(Self.mb(stock.transient)) prealloc_only_transient_mb=\(Self.mb(preallocOnly.transient)) prealloc_sync_transient_mb=\(Self.mb(preallocSync.transient)) store_stock_transient_mb=\(storeStockMB) store_drain_transient_mb=\(storeDrainMB)")
        #expect(abs(stock.checksum - preallocSync.checksum) < 1e-2)
        #expect(preallocSync.transient * 3 < stock.transient)
        #expect(storeDrainMB * 3 < storeStockMB)
    }

    @Test("Attributes GPU peak across prefill chunks, final logits and decode on stock vs PrefillKVPlan",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_PREFILL_TRANSIENT_MODEL_PATH"))
    func attributionAndGate() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("prefill-transient", modelPath: path)
        var configuration = EngineConfiguration()
        configuration.promptCacheBytes = 0
        configuration.reasoningEffort = .none
        let engine = MLXInferenceEngine(configuration)
        try await engine.load(modelPath: path)
        let container = try #require(await engine.container)
        let scratch = await engine.modelFacts.prefillScratch
        let key = 739_184
        let prompt = Self.buildPrompt(key: key, targetTokens: Self.targetTokens)
        log.record("loaded", "active_mb=\(Self.mb(Memory.activeMemory)) cache_mb=\(Self.mb(Memory.cacheMemory)) target_tokens=\(Self.targetTokens)")

        let planResult = try await log.step("plan") {
            try await Self.runArm(ArmConfig(arm: "plan", usePlan: true, scratch: scratch, promptText: prompt), container: container, log: log)
        }

        if !Self.skipStock {
            let stockResult = try await log.step("stock") {
                try await Self.runArm(ArmConfig(arm: "stock", usePlan: false, scratch: scratch, promptText: prompt), container: container, log: log)
            }
            let savedMB = stockResult.peakDecodeMB - planResult.peakDecodeMB
            let matching = zip(stockResult.tokens, planResult.tokens).prefix { $0 == $1 }.count
            log.record(
                "compare",
                "prompt=\(planResult.promptTokens) saved_peak_mb=\(savedMB) stock_transient_mb=\(stockResult.peakDecodeMB - stockResult.activeAfterPrefillMB) plan_transient_mb=\(planResult.peakDecodeMB - planResult.activeAfterPrefillMB) greedy_match=\(matching)/\(planResult.tokens.count)")
            #expect(planResult.tokens == stockResult.tokens)
            #expect(savedMB >= 0)
        } else {
            #expect(planResult.answer.contains(String(key)))
        }
    }

    @Test("Full MLXInferenceEngine generate path honors the low-transient budget on long prompts",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_PREFILL_TRANSIENT_MODEL_PATH"))
    func engineEndToEnd() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("prefill-transient-engine", modelPath: path)
        var configuration = EngineConfiguration()
        if !Self.cacheKeyMode { configuration.promptCacheBytes = 0 }
        configuration.reasoningEffort = .none
        let engine = MLXInferenceEngine(configuration)
        try await engine.load(modelPath: path)
        let weightsMB = Self.mb(Memory.activeMemory)
        let key = 482_915
        let prompt = Self.buildPrompt(key: key, targetTokens: Self.targetTokens)
        Memory.clearCache()
        Memory.peakMemory = 0
        var request = GenerationRequest(prompt: prompt, maxTokens: Self.decodeTokens, temperature: 0)
        if Self.cacheKeyMode { request = request.withCacheKey("shared") }
        let response = try await log.step("engine-generate") {
            try await engine.generate(request: request)
        }
        let peakMB = Self.mb(Memory.peakMemory)
        let activeAfterMB = Self.mb(Memory.activeMemory)
        let transientMB = peakMB - activeAfterMB
        log.record(
            "engine-result",
            String(
                format: "prompt=%d cache_key=%@ weights_mb=%d active_after_mb=%d peak_mb=%d transient_mb=%d ttft_s=%.2f decode_tok_s=%.1f ok=%@ answer=%@",
                response.promptTokens, Self.cacheKeyMode ? "shared" : "none",
                weightsMB, activeAfterMB, peakMB, transientMB,
                response.timeToFirstToken, response.tokensPerSecond,
                response.generatedText.contains(String(key)) ? "true" : "false",
                response.generatedText.replacingOccurrences(of: "\n", with: " ")))
        #expect(response.generatedText.contains(String(key)))
    }
}

private extension PrefillTransientRealModelTests {
    static func measureSyntheticPass(
        preallocate: Bool, syncPerChunk: Bool, totalTokens: Int, layerCount: Int
    ) -> (transient: Int, checksum: Float) {
        let chunkSize = 64
        let headDim = 512
        let q = MLXArray.ones([1, 2, chunkSize, headDim], dtype: .bfloat16) * Float(0.01)
        let kChunk = MLXArray.ones([1, 1, chunkSize, headDim], dtype: .bfloat16) * Float(0.01)
        let vChunk = MLXArray.ones([1, 1, chunkSize, headDim], dtype: .bfloat16) * Float(0.01)
        let qOne = MLXArray.ones([1, 2, 1, headDim], dtype: .bfloat16) * Float(0.01)
        let kOne = MLXArray.ones([1, 1, 1, headDim], dtype: .bfloat16) * Float(0.01)
        let vOne = MLXArray.ones([1, 1, 1, headDim], dtype: .bfloat16) * Float(0.01)
        eval(q, kChunk, vChunk, qOne, kOne, vOne)
        Stream.defaultStream.synchronize()
        Memory.clearCache()
        Memory.peakMemory = 0
        let layers = (0 ..< layerCount).map { _ in KVCacheSimple() }
        let plan = PrefillKVPlan.make(promptTokens: totalTokens, simpleLayerCount: layerCount, maxTokens: 8)
        if preallocate { plan.apply(to: layers) }
        var lastOut = q
        for _ in stride(from: 0, to: totalTokens, by: chunkSize) {
            autoreleasepool {
                for layer in layers {
                    let (k, v) = layer.update(keys: kChunk, values: vChunk)
                    lastOut = MLXFast.scaledDotProductAttention(
                        queries: q, keys: k, values: v, scale: 1.0 / Float(headDim).squareRoot(), mask: .causal)
                }
                asyncEval([lastOut] + layers.flatMap { $0.innerState() })
                if syncPerChunk { Stream.defaultStream.synchronize() }
            }
        }
        eval([lastOut] + layers.flatMap { $0.innerState() })
        if preallocate { plan.restore(layers) }
        for _ in 0 ..< 4 {
            autoreleasepool {
                for layer in layers {
                    let (k, v) = layer.update(keys: kOne, values: vOne)
                    lastOut = MLXFast.scaledDotProductAttention(
                        queries: qOne, keys: k, values: v, scale: 1.0 / Float(headDim).squareRoot(), mask: .none)
                }
                asyncEval([lastOut] + layers.flatMap { $0.innerState() })
                if syncPerChunk { Stream.defaultStream.synchronize() }
            }
        }
        eval([lastOut] + layers.flatMap { $0.innerState() })
        Stream.defaultStream.synchronize()
        return (Memory.peakMemory - Memory.activeMemory, lastOut.asType(.float32).sum().item(Float.self))
    }

    static func measureSyntheticStorePeak(drainSource: Bool, totalTokens: Int, layerCount: Int) -> Int {
        Stream.defaultStream.synchronize()
        Memory.clearCache()
        let layers = (0 ..< layerCount).map { _ -> KVCacheSimple in
            let cache = KVCacheSimple()
            cache.step = totalTokens + 32
            let k = MLXArray.ones([1, 1, totalTokens, 512], dtype: .bfloat16)
            let v = MLXArray.ones([1, 1, totalTokens, 512], dtype: .bfloat16)
            _ = cache.update(keys: k, values: v)
            eval(cache.innerState())
            return cache
        }
        Stream.defaultStream.synchronize()
        let beforeActive = Memory.activeMemory
        Memory.peakMemory = 0
        var tree = PromptCacheTree(byteBudget: ByteCount(1 << 30))
        tree.store("shared", tokens: Array(0 ..< totalTokens), cache: PromptCache(layers), drainSource: drainSource)
        Stream.defaultStream.synchronize()
        return Self.mb(Memory.peakMemory - beforeActive)
    }
}

private extension PrefillTransientRealModelTests {
    static func buildPrompt(key: Int, targetTokens: Int) -> String {
        let count = max(16, targetTokens / 22)
        var lines = Array(repeating: filler, count: count)
        lines.insert("The pass key is \(key). Remember it. \(key) is the pass key.", at: count / 2)
        return lines.joined(separator: "\n") + "\nWhat is the pass key? Reply with the number only."
    }

    static func cacheBytes(_ layers: [any KVCache]) -> Int {
        layers.reduce(0) { total, layer in
            total + layer.innerState().reduce(0) { $0 + $1.nbytes }
        }
    }

    static func progressLogger(arm: String, promptCount: Int, log: ModelRunLog) -> @Sendable (Int, Int) -> Void {
        let stepMark = max(1, promptCount / 8)
        let checkpoints = Mutex(0)
        return { processed, total in
            let shouldLog = checkpoints.withLock { count -> Bool in
                guard processed >= count + stepMark || processed >= total - 128 || processed == total else { return false }
                count = processed
                return true
            }
            guard shouldLog else { return }
            let active = Self.mb(Memory.activeMemory)
            let peak = Self.mb(Memory.peakMemory)
            log.record(
                "\(arm)-chunk",
                "processed=\(processed)/\(total) active_mb=\(active) cache_mb=\(Self.mb(Memory.cacheMemory)) peak_mb=\(peak) transient_mb=\(peak - active)")
        }
    }

    static func decodeTokens<T: TokenIteratorProtocol>(_ iterator: inout T, arm: String, log: ModelRunLog) -> [Int] {
        var generated: [Int] = []
        while let tok = iterator.next() {
            generated.append(tok)
            if generated.count == 1 || generated.count == 2 || generated.count == Self.decodeTokens {
                log.record(
                    "\(arm)-decode",
                    "step=\(generated.count) active_mb=\(Self.mb(Memory.activeMemory)) peak_mb=\(Self.mb(Memory.peakMemory))")
            }
        }
        return generated
    }

    static func measureStore(cache: [any KVCache], tokens: MLXArray, drainSource: Bool, arm: String, log: ModelRunLog) -> Int {
        let promptCache = PromptCache(cache)
        let storedCount = promptCache.tokenCount
        let snap = promptCache.isHybrid ? promptCache.snapshot(at: storedCount, state: nil) : promptCache
        var tree = PromptCacheTree(byteBudget: ByteCount(4 << 30))
        var ids = tokens.asArray(Int.self)
        if ids.count < storedCount {
            ids.append(contentsOf: Array(repeating: 1, count: storedCount - ids.count))
        }
        tree.store("bench", tokens: Array(ids.prefix(storedCount)), cache: snap, drainSource: drainSource)
        Stream.defaultStream.synchronize()
        let peakStore = Self.mb(Memory.peakMemory)
        log.record(
            "\(arm)-store",
            "active_mb=\(Self.mb(Memory.activeMemory)) peak_mb=\(peakStore) tree_mb=\(Self.mb(Int(tree.byteCount.rawValue)))")
        return peakStore
    }

    static func evaluateArm(_ config: ArmConfig, prepared: LMInput, context: ModelContext, log: ModelRunLog) throws -> ArmOutcome {
        let promptCount = prepared.text.tokens.dim(-1)
        let started = ContinuousClock.now
        var params = GenerateParameters(maxTokens: Self.decodeTokens, temperature: 0)
            .capped(for: context.model, promptTokens: promptCount, scratch: config.scratch)
        let cache = try context.model.newCache(parameters: params)
        let plan = PrefillKVPlan.make(
            promptTokens: promptCount,
            simpleLayerCount: PrefillKVPlan.simpleLayers(in: cache).count,
            maxTokens: params.maxTokens)
        if config.usePlan { plan.apply(to: cache) }
        defer { if config.usePlan { plan.restore(cache) } }

        params.prefill.progress = Self.progressLogger(arm: config.arm, promptCount: promptCount, log: log)
        if config.usePlan { params.prefill = plan.synchronizing(params.prefill) }

        var rawIterator = try TokenIterator(input: prepared, model: context.model, cache: cache, parameters: params)
        Stream.defaultStream.synchronize()
        let prefillSeconds = (ContinuousClock.now - started).inSeconds
        let activeAfterPrefill = Self.mb(Memory.activeMemory)
        let peakPrefill = Self.mb(Memory.peakMemory)
        let kvMB = Self.mb(Self.cacheBytes(cache))
        log.record(
            "\(config.arm)-prefill-done",
            String(
                format: "prompt=%d step=%d kv_mb=%d active_mb=%d peak_mb=%d transient_mb=%d prefill_s=%.2f tok_s=%.1f",
                promptCount, params.prefill.stepSize ?? PrefillParameters.defaultStepSize, kvMB,
                activeAfterPrefill, peakPrefill, peakPrefill - activeAfterPrefill,
                prefillSeconds, Double(promptCount) / max(prefillSeconds, 1e-6)))

        let decodeStart = ContinuousClock.now
        let generated = Self.decodeTokens(&rawIterator, arm: config.arm, log: log)
        Stream.defaultStream.synchronize()
        let decodeSeconds = (ContinuousClock.now - decodeStart).inSeconds
        let peakDecode = Self.mb(Memory.peakMemory)
        let answer = context.tokenizer.decode(tokenIds: generated, skipSpecialTokens: true)
        let peakStore = Self.cacheKeyMode
            ? Self.measureStore(cache: cache, tokens: prepared.text.tokens, drainSource: config.usePlan, arm: config.arm, log: log)
            : peakDecode

        return ArmOutcome(
            arm: config.arm, promptTokens: promptCount, tokens: generated, answer: answer,
            prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds, kvMB: kvMB,
            activeAfterPrefillMB: activeAfterPrefill, peakPrefillMB: peakPrefill,
            peakDecodeMB: peakDecode, peakStoreMB: peakStore)
    }

    static func runArm(_ config: ArmConfig, container: ModelContainer, log: ModelRunLog) async throws -> ArmOutcome {
        Stream.defaultStream.synchronize()
        Memory.clearCache()
        log.record(config.arm, "begin base_active_mb=\(mb(Memory.activeMemory)) base_cache_mb=\(mb(Memory.cacheMemory))")
        let input = try await container.prepare(input: UserInput(prompt: config.promptText))
        Memory.peakMemory = 0
        let result = try await container.performEvaluating(nonSendable: input) { context, prepared in
            try Self.evaluateArm(config, prepared: prepared, context: context, log: log)
        }
        Stream.defaultStream.synchronize()
        Memory.clearCache()
        log.record(
            "\(config.arm)-summary",
            String(
                format: "prompt=%d kv_mb=%d active_mb=%d peak_prefill_mb=%d peak_decode_mb=%d transient_mb=%d prefill_s=%.2f decode_tok_s=%.1f answer=%@",
                result.promptTokens, result.kvMB, result.activeAfterPrefillMB,
                result.peakPrefillMB, result.peakDecodeMB,
                result.peakDecodeMB - result.activeAfterPrefillMB,
                result.prefillSeconds, Double(result.tokens.count) / max(result.decodeSeconds, 1e-6),
                result.answer.replacingOccurrences(of: "\n", with: " ")))
        return result
    }
}
