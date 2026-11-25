import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

final class GaugeRoPEReconstructCache: KVCacheAttentionProtocol {
    typealias Transform = (_ keys: MLXArray, _ values: MLXArray, _ offset: Int) -> GaugeRoPEKV

    private let transform: Transform
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private(set) var offset = 0

    init(_ transform: @escaping Transform) {
        self.transform = transform
    }

    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    func innerState() -> [MLXArray] { [keys, values].compactMap { $0 } }

    var state: [MLXArray] {
        get { innerState() }
        set { preconditionFailure("GaugeRoPEReconstructCache is never restored") }
    }

    var metaState: [String] {
        get { [] }
        set {}
    }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func copy() -> any KVCache { preconditionFailure("GaugeRoPEReconstructCache is never copied") }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard n > 1 else { return .none }
        if returnArray || (windowSize.map { n > $0 } ?? false) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let hat = transform(
            newKeys[0].transposed(1, 0, 2).asType(.float32), newValues[0].transposed(1, 0, 2).asType(.float32), offset)
        let k = hat.keys.transposed(1, 0, 2).expandedDimensions(axis: 0).asType(newKeys.dtype)
        let v = hat.values.transposed(1, 0, 2).expandedDimensions(axis: 0).asType(newValues.dtype)
        keys = keys.map { concatenated([$0, k], axis: 2) } ?? k
        values = values.map { concatenated([$0, v], axis: 2) } ?? v
        offset += newKeys.dim(2)
        return (keys!, values!)
    }

    func updateAndAttend(
        queries: MLXArray, keys newKeys: MLXArray, values newValues: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (k, v) = update(keys: newKeys, values: newValues)
        return MLXFast.scaledDotProductAttention(queries: queries, keys: k, values: v, scale: scale, mask: mask)
    }
}

struct GaugeRoPEPPLPlan {
    let name: String
    let bytesVsK8v4: Double
    let transform: (GaugeRoPELayerKV) -> GaugeRoPEReconstructCache.Transform

    static func positional(
        _ layer: GaugeRoPELayerKV, rope: GaugeRoPERealModelTests.RopeConfig, _ reconstruct: @escaping (MLXArray, MLXArray) -> GaugeRoPEKV
    ) -> GaugeRoPEReconstructCache.Transform {
        positional(headDim: layer.keysPost.dim(2), rope: rope, reconstruct)
    }

    static func positional(
        headDim: Int, rope: GaugeRoPERealModelTests.RopeConfig, _ reconstruct: @escaping (MLXArray, MLXArray) -> GaugeRoPEKV
    ) -> GaugeRoPEReconstructCache.Transform {
        let order = rope.order(headDim: headDim)
        let inverse = order.map { argSort($0) }
        let (rotary, base) = (rope.rotaryDims(headDim: headDim), rope.effectiveBase(headDim: headDim))
        return { keys, values, offset in
            let k = order.map { take(keys, $0, axis: -1) } ?? keys
            let pre = GaugeRoPEArms.rope(k, rotaryDims: rotary, base: base, offset: offset, inverse: true)
            let hat = reconstruct(pre, values)
            let post = GaugeRoPEArms.rope(hat.keys, rotaryDims: rotary, base: base, offset: offset)
            return GaugeRoPEKV(keys: inverse.map { take(post, $0, axis: -1) } ?? post, values: hat.values)
        }
    }

    static func plans(rope: GaugeRoPERealModelTests.RopeConfig, ratios: [Double]) -> [GaugeRoPEPPLPlan] {
        var plans = [
            GaugeRoPEPPLPlan(name: "identity", bytesVsK8v4: 2.15) { _ in
                { keys, values, _ in GaugeRoPEKV(keys: keys, values: values) }
            },
            GaugeRoPEPPLPlan(name: "rope-roundtrip", bytesVsK8v4: 2.15) { cal in
                positional(cal, rope: rope) { GaugeRoPEKV(keys: $0, values: $1) }
            },
            GaugeRoPEPPLPlan(name: "k8v4", bytesVsK8v4: 1) { _ in
                { keys, values, _ in
                    GaugeRoPEKV(keys: GaugeRoPEArms.fakeQuantized(keys, bits: 8), values: GaugeRoPEArms.fakeQuantized(values, bits: 4))
                }
            },
            GaugeRoPEPPLPlan(name: "kvarn", bytesVsK8v4: 0.40) { _ in
                { keys, values, _ in
                    GaugeRoPEKV(keys: GaugeRoPEArms.kvarnRoundTrip(keys).decoded, values: GaugeRoPEArms.kvarnRoundTrip(values).decoded)
                }
            },
        ]
        for ratio in ratios {
            plans.append(GaugeRoPEPPLPlan(name: "mla-split", bytesVsK8v4: ratio) { cal in
                let (heads, r) = (cal.keysPost.dim(1), cal.rotaryDims)
                let arm = MLASplitArm.fit(keys: cal.keysPre, values: cal.values, rotaryDims: r, metric: metrics(cal)?.mla)
                let rank = codeRank(cal, ratio: ratio, exactFloats: heads * r)
                return positional(cal, rope: rope) { arm.reconstruct(keys: $0, values: $1, rank: rank, quantized: true) }
            })
            plans.append(GaugeRoPEPPLPlan(name: "gauge-c2", bytesVsK8v4: ratio) { cal in
                let arm = GaugeRoPEArm.fit(
                    keys: cal.keysPre, values: cal.values, rotaryDims: cal.rotaryDims, carrierRank: 2,
                    metric: metrics(cal)?.gauge, quantized: true)
                let rank = codeRank(cal, ratio: ratio, exactFloats: arm.carrier.carrierHeads * cal.rotaryDims)
                return positional(cal, rope: rope) { arm.reconstruct(keys: $0, values: $1, rank: rank, quantized: true) }
            })
        }
        return plans
    }

    static func metrics(_ cal: GaugeRoPELayerKV) -> (mla: MLXArray, gauge: MLXArray)? {
        guard cal.queries.dim(1) > 0 else { return nil }
        return GaugeRoPEMetric.metrics(
            queries: cal.queries, kvHeads: cal.keysPost.dim(1), rotaryDims: cal.rotaryDims, valueDims: cal.values.dim(2))
    }

    static func codeRank(_ cal: GaugeRoPELayerKV, ratio: Double, exactFloats: Int) -> Int {
        let budget = ratio * GaugeRoPEByteBudget.k8v4BytesPerToken(kvHeads: cal.keysPost.dim(1), headDim: cal.keysPost.dim(2))
        return GaugeRoPEByteBudget.codeRank(budget: budget, exactFloats: exactFloats)
    }
}

@Suite("GaugeRoPE S0 gate: perplexity with each arm's reconstructed K/V in the full-attention caches (#647)", .serialized)
struct GaugeRoPEPerplexityTests {
    static let env = ProcessInfo.processInfo.environment
    static let enabled = env["BASALT_E2E"] == "1" && env["BASALT_E2E_GAUGEROPE_PPL"] == "1"
        && env["BASALT_E2E_GAUGEROPE_MODEL_PATH"] != nil && env["BASALT_E2E_KV_CORPUS"] != nil
    static let calibrationTokens = Int(env["BASALT_E2E_GAUGEROPE_CALIBRATION"] ?? "") ?? 4096
    static let context = Int(env["BASALT_E2E_GAUGEROPE_CONTEXT"] ?? "") ?? 16384
    static let chunk = 512

    static func perplexity(_ ids: [Int], context: ModelContext, caches: [any KVCache], scoreFrom: Int) -> (ppl: Double, scored: Int) {
        var (nll, scored, start) = (0.0, 0, 0)
        while start < ids.count - 1 {
            let end = min(ids.count, start + chunk)
            let logits = context.model(MLXArray(Array(ids[start ..< end]))[.newAxis], cache: caches)
            let (lo, hi) = (max(start, scoreFrom - 1), min(end, ids.count - 1))
            if lo < hi {
                let l = logits[0, (lo - start) ..< (hi - start)].asType(.float32)
                let logp = l - logSumExp(l, axis: -1, keepDims: true)
                let targets = MLXArray(ids[(lo + 1) ... hi].map(Int32.init))[0..., .newAxis]
                nll -= Double(takeAlong(logp, targets, axis: -1).sum().item(Float.self))
                scored += hi - lo
            }
            eval(caches.flatMap { $0.innerState() })
            start = end
        }
        return (exp(nll / Double(max(scored, 1))), scored)
    }

    static func shortContextPerplexity(
        _ ids: [Int], context: ModelContext, transforms: [Int: GaugeRoPEReconstructCache.Transform], scoreFrom: Int, window: Int
    ) throws -> (ppl: Double, scored: Int) {
        var (nll, scored) = (0.0, 0)
        for start in stride(from: scoreFrom, to: ids.count, by: chunk) {
            let end = min(ids.count, start + chunk)
            let run = perplexity(Array(ids[(start - window) ..< end]), context: context, caches: try caches(context, transforms: transforms), scoreFrom: window)
            nll += log(run.ppl) * Double(run.scored)
            scored += run.scored
        }
        return (exp(nll / Double(max(scored, 1))), scored)
    }

    static func caches(_ context: ModelContext, transforms: [Int: GaugeRoPEReconstructCache.Transform]) throws -> [any KVCache] {
        try context.model.newCache(parameters: nil).enumerated().map { index, cache in
            guard type(of: cache) == KVCacheSimple.self, let transform = transforms[index] else { return cache }
            return GaugeRoPEReconstructCache(transform)
        }
    }

    @Test("PPL over the second half of a 16K window: fp16, k8v4, KVarN, MLA-split and GaugeRoPE at equal k8v4-priced bytes",
          .enabled(if: enabled, "set BASALT_E2E=1, BASALT_E2E_GAUGEROPE_PPL=1, BASALT_E2E_GAUGEROPE_MODEL_PATH and BASALT_E2E_KV_CORPUS"))
    func perplexityPerArm() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.env["BASALT_E2E_GAUGEROPE_MODEL_PATH"]))
        let corpus = try String(contentsOfFile: try #require(Self.env["BASALT_E2E_KV_CORPUS"]), encoding: .utf8)
        let log = try await ModelRunLog("gaugerope-ppl", modelPath: path)
        defer { log.releaseGPU() }
        let rope = try #require(GaugeRoPERealModelTests.ropeConfig(at: path))
        let engine = MLXInferenceEngine(EngineConfiguration(gpuCacheLimit: 0))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let (cal, ctx) = (Self.calibrationTokens, Self.context)
        try await container.perform { (context: ModelContext) in
            let ids = context.tokenizer.encode(text: corpus)
            guard ids.count >= cal + ctx else { throw GaugeRoPERealModelTests.StepFailure(description: "corpus has \(ids.count) tokens, needs \(cal + ctx)") }
            let layers = try GaugeRoPERealModelTests.capture(Array(ids[0 ..< cal]), context: context, rope: rope, needsQueries: false)
            let window = Array(ids[cal ..< (cal + ctx)])
            let queryAware = layers.allSatisfy { $0.queries.dim(1) > 0 }
            log.record("result", "GAUGEROPE-PPL layers=\(layers.count) calibration=\(cal) context=\(ctx) scored=\(ctx / 2) metric=\(queryAware ? "query-whitened" : "none (queries not captured)")")
            let short = Self.env["BASALT_E2E_GAUGEROPE_SHORT"].flatMap { Int($0) }
            let measure = { (transforms: [Int: GaugeRoPEReconstructCache.Transform]) throws -> (ppl: Double, scored: Int) in
                guard let short else { return Self.perplexity(window, context: context, caches: try Self.caches(context, transforms: transforms), scoreFrom: ctx / 2) }
                return try Self.shortContextPerplexity(window, context: context, transforms: transforms, scoreFrom: ctx / 2, window: short)
            }
            if let short { log.record("result", "GAUGEROPE-PPL short-context window=\(short) (each 512-token chunk of the scored half sees only the \(short) tokens before it)") }
            let reference = try measure([:])
            log.record("result", String(format: "GAUGEROPE-PPL arm=fp16 bytes=- ppl=%.4f scored=%d", reference.ppl, reference.scored))
            let only = Self.env["BASALT_E2E_GAUGEROPE_ARMS"].map { Set($0.split(separator: ",").map(String.init)) }
            for plan in GaugeRoPEPPLPlan.plans(rope: rope, ratios: [0.6, 0.4]) where only?.contains(plan.name) ?? true {
                let transforms = Dictionary(uniqueKeysWithValues: layers.map { ($0.layer, plan.transform($0)) })
                let run = try measure(transforms)
                log.record("result", String(format: "GAUGEROPE-PPL arm=%@ bytes=%.2f ppl=%.4f vs-fp16=%+.2f%%",
                                            plan.name, plan.bytesVsK8v4, run.ppl, (run.ppl / reference.ppl - 1) * 100))
                #expect(run.ppl.isFinite)
            }
        }
    }
}

@Suite("GaugeRoPE reconstruct cache: an identity or RoPE round-trip transform leaves chunked logits unchanged (#647, #228)")
struct GaugeRoPEReconstructCacheParityTests {
    static let tokens = (0 ..< 96).map { ($0 * 37 + 11) % 200 }

    static func chunkedLogits(_ model: any LanguageModel, transform: GaugeRoPEReconstructCache.Transform?) throws -> MLXArray {
        let caches = try model.newCache(parameters: nil).map { cache -> any KVCache in
            guard let transform, type(of: cache) == KVCacheSimple.self else { return cache }
            return GaugeRoPEReconstructCache(transform)
        }
        var parts: [MLXArray] = []
        for start in stride(from: 0, to: tokens.count, by: 32) {
            parts.append(model(MLXArray(Array(tokens[start ..< (start + 32)]))[.newAxis], cache: caches).asType(.float32))
        }
        return concatenated(parts, axis: 1)
    }

    @Test("Identity and RoPE round trip match the stock caches",
          arguments: [
              (TinyModel.Family.qwen35, GaugeRoPERealModelTests.RopeConfig(base: 10_000, partialFactor: 0.25), 16),
              (.gemma4KEqV, GaugeRoPERealModelTests.RopeConfig(base: 1_000_000, partialFactor: 0.25, proportional: true), 64),
          ])
    func identityMatchesStock(family: TinyModel.Family, rope: GaugeRoPERealModelTests.RopeConfig, headDim: Int) throws {
        let model = try TinyModel.make(family)
        let stock = try Self.chunkedLogits(model, transform: nil)
        let identity = try Self.chunkedLogits(model) { keys, values, _ in GaugeRoPEKV(keys: keys, values: values) }
        #expect(abs(identity - stock).max().item(Float.self) < 1e-4)
        let roundTrip = try Self.chunkedLogits(model, transform: GaugeRoPEPPLPlan.positional(headDim: headDim, rope: rope) { GaugeRoPEKV(keys: $0, values: $1) })
        #expect(abs(roundTrip - stock).max().item(Float.self) < 1e-2, "\(family) RoPE round trip")
    }
}

extension GaugeRoPEReconstructCacheParityTests {
    @Test("Identity matches the stock caches on the model classes the engine loads (MLXLLM Qwen35Model, multimodal Gemma 4)",
          arguments: [TinyModel.Family.prism, .gemma4Unified])
    func identityMatchesStockOnEngineClasses(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let stock = try Self.chunkedLogits(model, transform: nil)
        let identity = try Self.chunkedLogits(model) { keys, values, _ in GaugeRoPEKV(keys: keys, values: values) }
        #expect(abs(identity - stock).max().item(Float.self) < 1e-4, "\(family)")
    }
}

@Suite("GaugeRoPE S0 gate: passkey at 16K and 32K with each arm's reconstructed K/V (#647)", .serialized)
struct GaugeRoPEPasskeyTests {
    static let env = ProcessInfo.processInfo.environment
    static let enabled = env["BASALT_E2E"] == "1" && env["BASALT_E2E_GAUGEROPE_PASSKEY"] == "1"
        && env["BASALT_E2E_GAUGEROPE_MODEL_PATH"] != nil && env["BASALT_E2E_KV_CORPUS"] != nil
    static let lengths = (env["BASALT_E2E_GAUGEROPE_PASSKEY_LENGTHS"] ?? "16384,32768").split(separator: ",").compactMap { Int($0) }
    static let depths = [0.2, 0.8]

    static func greedy(_ prompt: [Int], context: ModelContext, caches: [any KVCache], maxTokens: Int) -> [Int] {
        var (start, logits) = (0, MLXArray(0))
        while start < prompt.count {
            let end = min(prompt.count, start + GaugeRoPEPerplexityTests.chunk)
            logits = context.model(MLXArray(Array(prompt[start ..< end]))[.newAxis], cache: caches)
            eval(caches.flatMap { $0.innerState() })
            start = end
        }
        var out: [Int] = []
        var next = argMax(logits[0, -1], axis: -1).item(Int.self)
        for _ in 0 ..< maxTokens {
            out.append(next)
            let step = context.model(MLXArray([Int32(next)])[.newAxis], cache: caches)
            next = argMax(step[0, -1], axis: -1).item(Int.self)
        }
        return out
    }

    @Test("Passkey found at depth 0.2 and 0.8, 16K and 32K: fp16, k8v4, MLA-split and GaugeRoPE at 0.6x k8v4 bytes",
          .enabled(if: enabled, "set BASALT_E2E=1, BASALT_E2E_GAUGEROPE_PASSKEY=1, BASALT_E2E_GAUGEROPE_MODEL_PATH and BASALT_E2E_KV_CORPUS"))
    func passkeyPerArm() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.env["BASALT_E2E_GAUGEROPE_MODEL_PATH"]))
        let corpus = try String(contentsOfFile: try #require(Self.env["BASALT_E2E_KV_CORPUS"]), encoding: .utf8)
        let log = try await ModelRunLog("gaugerope-passkey", modelPath: path)
        defer { log.releaseGPU() }
        let rope = try #require(GaugeRoPERealModelTests.ropeConfig(at: path))
        let engine = MLXInferenceEngine(EngineConfiguration(gpuCacheLimit: 0))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let cal = GaugeRoPEPerplexityTests.calibrationTokens
        let only = Set((Self.env["BASALT_E2E_GAUGEROPE_ARMS"] ?? "k8v4,mla-split,gauge-c2").split(separator: ",").map(String.init))
        try await container.perform { (context: ModelContext) in
            let ids = context.tokenizer.encode(text: corpus)
            let layers = try GaugeRoPERealModelTests.capture(Array(ids[0 ..< cal]), context: context, rope: rope, needsQueries: false)
            let filler = Array(ids[cal...])
            let plans = GaugeRoPEPPLPlan.plans(rope: rope, ratios: [0.6]).filter { only.contains($0.name) }
            let arms: [(String, [Int: GaugeRoPEReconstructCache.Transform])] = [("fp16", [:])]
                + plans.map { plan in (plan.name, Dictionary(uniqueKeysWithValues: layers.map { ($0.layer, plan.transform($0)) })) }
            for length in Self.lengths {
                for depth in Self.depths {
                    let key = 10_000 + Int(depth * 1_000) * 37 + length % 997
                    let prompt = try KEqVRowRealModelTests.passkeyPrompt(filler, key: key, depth: depth, length: length, context: context)
                    for (name, transforms) in arms {
                        let caches = try GaugeRoPEPerplexityTests.caches(context, transforms: transforms)
                        let answer = context.tokenizer.decode(tokenIds: Self.greedy(prompt, context: context, caches: caches, maxTokens: 8))
                        log.record("result", "GAUGEROPE-PASSKEY arm=\(name) length=\(length) depth=\(depth) ok=\(answer.contains(String(key))) answer=\(answer.debugDescription)")
                    }
                }
            }
        }
    }
}
