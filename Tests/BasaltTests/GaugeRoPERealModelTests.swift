import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

final class GaugeRoPECaptureCache: KVCacheAttentionProtocol {
    private(set) var queries: [MLXArray] = []
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private(set) var scale: Float = 1
    private(set) var offset = 0

    var captured: [MLXArray] { queries + [keys, values].compactMap { $0 } }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    func innerState() -> [MLXArray] { captured }

    var state: [MLXArray] {
        get { [keys, values].compactMap { $0 } }
        set { preconditionFailure("GaugeRoPECaptureCache is never restored") }
    }

    var metaState: [String] {
        get { [] }
        set {}
    }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func copy() -> any KVCache { preconditionFailure("GaugeRoPECaptureCache is never copied") }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard n > 1 else { return .none }
        if returnArray || (windowSize.map { n > $0 } ?? false) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let k = self.keys.map { concatenated([$0, keys], axis: 2) } ?? keys
        let v = self.values.map { concatenated([$0, values], axis: 2) } ?? values
        self.keys = k
        self.values = v
        offset += keys.dim(2)
        return (k, v)
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (k, v) = update(keys: keys, values: values)
        self.queries.append(queries)
        self.scale = scale
        return MLXFast.scaledDotProductAttention(queries: queries, keys: k, values: v, scale: scale, mask: mask)
    }
}

struct GaugeRoPELayerKV {
    let layer: Int
    let queries: MLXArray
    let keysPost: MLXArray
    let keysPre: MLXArray
    let values: MLXArray
    let scale: Float
    let rotaryDims: Int
    let ropeBase: Float
}

struct GaugeRoPERow {
    let layer: Int
    let arm: String
    let rank: Int
    let bytes: Double
    let ratio: Double
    let error: Double

    var line: String {
        "GAUGEROPE layer=\(layer) arm=\(arm) rank=\(rank) "
            + String(format: "bytesPerLayerToken=%.1f ratio=%.3f attnRelErr=%.5f", bytes, ratio, error)
    }

    static func summary(_ rows: [GaugeRoPERow]) -> [String] {
        let groups = Dictionary(grouping: rows) { "arm=\($0.arm) " + String(format: "ratio=%.3f", $0.ratio) }
        return groups.keys.sorted().map { key in
            let errors = groups[key, default: []].map(\.error)
            let mean = errors.reduce(0, +) / Double(max(errors.count, 1))
            return "GAUGEROPE mean \(key) layers=\(errors.count) "
                + String(format: "attnRelErr=%.5f max=%.5f", mean, errors.max() ?? 0)
        }
    }
}

struct GaugeRoPEProbe {
    let kv: GaugeRoPELayerKV
    let exact: MLXArray
    let k8v4Bytes: Double

    init(_ kv: GaugeRoPELayerKV) {
        let output = GaugeRoPEArms.attend(queries: kv.queries, keys: kv.keysPost, values: kv.values, scale: kv.scale)
        eval(output)
        self.kv = kv
        exact = output
        k8v4Bytes = GaugeRoPEByteBudget.k8v4BytesPerToken(kvHeads: kv.keysPost.dim(1), headDim: kv.keysPost.dim(2))
    }

    func error(keysPost: MLXArray, values: MLXArray) -> Double {
        let approx = GaugeRoPEArms.attend(queries: kv.queries, keys: keysPost, values: values, scale: kv.scale)
        return GaugeRoPEArms.relativeError([approx], [exact])
    }

    func error(pre: GaugeRoPEKV) -> Double {
        error(keysPost: GaugeRoPERotary.apply(pre.keys, rotaryDims: kv.rotaryDims, base: kv.ropeBase), values: pre.values)
    }

    func row(_ arm: String, rank: Int, bytes: Double, error: Double) -> GaugeRoPERow {
        GaugeRoPERow(layer: kv.layer, arm: arm, rank: rank, bytes: bytes, ratio: bytes / k8v4Bytes, error: error)
    }

    func baselineRows() -> [GaugeRoPERow] {
        let k8 = error(
            keysPost: GaugeRoPEArms.fakeQuantized(kv.keysPost, bits: 8), values: GaugeRoPEArms.fakeQuantized(kv.values, bits: 4))
        let kvarnK = GaugeRoPEArms.kvarnRoundTrip(kv.keysPost)
        let kvarnV = GaugeRoPEArms.kvarnRoundTrip(kv.values)
        let kvarnBytes = Double(kvarnK.bytes + kvarnV.bytes) / Double(kv.keysPost.dim(0))
        return [
            row("k8v4", rank: 0, bytes: k8v4Bytes, error: k8),
            row("kvarn", rank: 0, bytes: kvarnBytes, error: error(keysPost: kvarnK.decoded, values: kvarnV.decoded)),
        ]
    }

    func latentRows(calibration: GaugeRoPELayerKV, ratios: [Double], carrierRanks: [Int]) -> [GaugeRoPERow] {
        let (heads, r) = (kv.keysPost.dim(1), kv.rotaryDims)
        let metric = GaugeRoPEMetric.metrics(
            queries: calibration.queries, kvHeads: heads, rotaryDims: r, valueDims: kv.values.dim(2))
        let mla = MLASplitArm.fit(keys: calibration.keysPre, values: calibration.values, rotaryDims: r, metric: metric.mla)
        var rows = ratios.map { ratio in
            let rank = GaugeRoPEByteBudget.codeRank(budget: ratio * k8v4Bytes, exactFloats: heads * r)
            let hat = mla.reconstruct(keys: kv.keysPre, values: kv.values, rank: rank, quantized: true)
            let bytes = GaugeRoPEByteBudget.latentBytesPerToken(exactFloats: heads * r, codeRank: rank)
            return row("mla-split", rank: rank, bytes: bytes, error: error(pre: hat))
        }
        for carrierRank in carrierRanks {
            let gauge = GaugeRoPEArm.fit(
                keys: calibration.keysPre, values: calibration.values, rotaryDims: r, carrierRank: carrierRank,
                metric: metric.gauge, quantized: true)
            let exactFloats = gauge.carrier.carrierHeads * r
            rows += ratios.map { ratio in
                let rank = GaugeRoPEByteBudget.codeRank(budget: ratio * k8v4Bytes, exactFloats: exactFloats)
                let hat = gauge.reconstruct(keys: kv.keysPre, values: kv.values, rank: rank, quantized: true)
                let bytes = GaugeRoPEByteBudget.latentBytesPerToken(exactFloats: exactFloats, codeRank: rank)
                return row("gauge-c\(gauge.carrier.carrierHeads)", rank: rank, bytes: bytes, error: error(pre: hat))
            }
        }
        return rows
    }
}

@Suite("GaugeRoPE S0 on captured Qwen3.8 full-attention K/V: attention-output error per arm at equal bytes (#647)", .serialized)
struct GaugeRoPERealModelTests {
    struct StepFailure: Error, CustomStringConvertible {
        let description: String
    }

    struct RopeConfig {
        let base: Float
        let partialFactor: Double
        var proportional = false

        func rotaryDims(headDim: Int) -> Int {
            proportional ? 2 * Int(partialFactor * Double(headDim) / 2) : Int(Double(headDim) * partialFactor)
        }

        func effectiveBase(headDim: Int) -> Float {
            proportional ? pow(base, Float(rotaryDims(headDim: headDim)) / Float(headDim)) : base
        }

        func order(headDim: Int) -> MLXArray? {
            guard proportional else { return nil }
            let (half, rotated) = (headDim / 2, rotaryDims(headDim: headDim) / 2)
            let front = Array(0 ..< rotated) + Array(half ..< half + rotated)
            let rest = (0 ..< headDim).filter { !(0 ..< rotated).contains($0) && !(half ..< half + rotated).contains($0) }
            return MLXArray((front + rest).map(Int32.init))
        }
    }

    static var env: [String: String] { ProcessInfo.processInfo.environment }
    static var enabled: Bool {
        env["BASALT_E2E"] == "1" && env["BASALT_E2E_GAUGEROPE_MODEL_PATH"] != nil && env["BASALT_E2E_KV_CORPUS"] != nil
    }
    static var tokens: Int { env["BASALT_E2E_GAUGEROPE_TOKENS"].flatMap { Int($0) } ?? 2048 }
    static let ratios = [0.6, 0.45, 0.3]
    static let carrierRanks = [1, 2]
    static let chunk = 512

    static func ropeConfig(at path: String) -> RopeConfig? {
        guard let config = ModelCache.config(at: URL(fileURLWithPath: path)) else { return nil }
        let text = config["text_config"] as? [String: Any] ?? config
        let nested = text["rope_parameters"] as? [String: Any]
        let params = nested?["full_attention"] as? [String: Any] ?? nested ?? text
        let kind = params["rope_type"] as? String ?? params["type"] as? String ?? "default"
        guard kind == "default" || kind == "proportional", let theta = params["rope_theta"] as? NSNumber else { return nil }
        let factor = (params["partial_rotary_factor"] as? NSNumber ?? text["partial_rotary_factor"] as? NSNumber)?.doubleValue
        return RopeConfig(base: theta.floatValue, partialFactor: factor ?? 1, proportional: kind == "proportional")
    }

    static func layerKV(_ cache: GaugeRoPECaptureCache, layer: Int, rope: RopeConfig, needsQueries: Bool = true) throws -> GaugeRoPELayerKV {
        guard let keys = cache.keys, let values = cache.values, !(needsQueries && cache.queries.isEmpty) else {
            throw StepFailure(description: "layer \(layer) captured no attention call: the model attends outside the cache after update(keys:values:), as Gemma 4 does, so its queries need a forward probe")
        }
        var k = keys[0].transposed(1, 0, 2).asType(.float32)
        var q = cache.queries.isEmpty
            ? MLXArray.zeros([k.dim(1), 0, k.dim(2)], dtype: .float32)
            : concatenated(cache.queries, axis: 2)[0].asType(.float32)
        let headDim = k.dim(2)
        if let order = rope.order(headDim: headDim) {
            (k, q) = (take(k, order, axis: -1), take(q, order, axis: -1))
        }
        let (rotary, base) = (rope.rotaryDims(headDim: headDim), rope.effectiveBase(headDim: headDim))
        return GaugeRoPELayerKV(
            layer: layer, queries: q, keysPost: k,
            keysPre: GaugeRoPERotary.apply(k, rotaryDims: rotary, base: base, inverse: true),
            values: values[0].transposed(1, 0, 2).asType(.float32), scale: cache.scale, rotaryDims: rotary, ropeBase: base)
    }

    static func capture(_ tokens: [Int], context: ModelContext, rope: RopeConfig, needsQueries: Bool = true) throws -> [GaugeRoPELayerKV] {
        let caches = try context.model.newCache(parameters: nil).map { cache -> any KVCache in
            type(of: cache) == KVCacheSimple.self ? GaugeRoPECaptureCache() : cache
        }
        var start = 0
        while start < tokens.count {
            let end = min(tokens.count, start + Self.chunk)
            eval(context.model(MLXArray(Array(tokens[start ..< end]))[.newAxis], cache: caches))
            eval(caches.compactMap { ($0 as? GaugeRoPECaptureCache)?.captured }.flatMap { $0 })
            start = end
        }
        return try caches.enumerated().compactMap { index, cache in
            try (cache as? GaugeRoPECaptureCache).map { try Self.layerKV($0, layer: index, rope: rope, needsQueries: needsQueries) }
        }
    }

    static func measure(calibration: [GaugeRoPELayerKV], evaluation: [GaugeRoPELayerKV]) throws -> [GaugeRoPERow] {
        guard let first = calibration.first, calibration.count == evaluation.count else {
            throw StepFailure(description: "captured \(calibration.count) calibration and \(evaluation.count) evaluation layers")
        }
        let jointDims = first.keysPost.dim(1) * (first.keysPost.dim(2) + first.values.dim(2))
        guard first.keysPost.dim(0) >= jointDims else {
            throw StepFailure(description: "calibration has \(first.keysPost.dim(0)) tokens, the joint code needs \(jointDims)")
        }
        return zip(calibration, evaluation).flatMap { cal, ev in
            let probe = GaugeRoPEProbe(ev)
            return probe.baselineRows() + probe.latentRows(calibration: cal, ratios: Self.ratios, carrierRanks: Self.carrierRanks)
        }
    }

    @Test("Captured full-attention K/V: attention-output error of k8v4, KVarN, MLA-split and GaugeRoPE at 0.6x/0.45x/0.3x k8v4 bytes",
          .enabled(if: enabled, "set BASALT_E2E=1, BASALT_E2E_GAUGEROPE_MODEL_PATH=<Qwen3.8 repo or path> and BASALT_E2E_KV_CORPUS=<text file>"))
    func attentionErrorPerArm() async throws {
        let path = GoldenLogitsRealModelTests.path(try #require(Self.env["BASALT_E2E_GAUGEROPE_MODEL_PATH"]))
        let corpus = try String(contentsOfFile: try #require(Self.env["BASALT_E2E_KV_CORPUS"]), encoding: .utf8)
        let log = try await ModelRunLog("gaugerope-s0", modelPath: path)
        defer { log.releaseGPU() }
        guard let rope = Self.ropeConfig(at: path) else {
            return log.record("result", "GAUGEROPE refused: config.json has no default or proportional rope_type with rope_theta")
        }
        let engine = MLXInferenceEngine(EngineConfiguration(gpuCacheLimit: 0))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let count = Self.tokens
        let rows = try await log.step("capture and measure calibration=\(count) evaluation=\(count)") {
            try await container.perform { (context: ModelContext) -> [GaugeRoPERow] in
                let ids = context.tokenizer.encode(text: corpus)
                guard ids.count >= 2 * count else {
                    throw StepFailure(description: "corpus has \(ids.count) tokens, needs \(2 * count)")
                }
                let calibration = try Self.capture(Array(ids[0 ..< count]), context: context, rope: rope)
                let evaluation = try Self.capture(Array(ids[count ..< (2 * count)]), context: context, rope: rope)
                return try Self.measure(calibration: calibration, evaluation: evaluation)
            }
        }
        for row in rows { log.record("result", row.line) }
        for line in GaugeRoPERow.summary(rows) { log.record("result", line) }
        #expect(!rows.isEmpty)
        #expect(rows.allSatisfy { $0.error.isFinite })
    }
}
