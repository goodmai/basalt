import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("GaugeRoPE S0 building blocks: RoPE indexing, exact full rank, arm ordering on known structure, byte budgets (#647)", .serialized)
struct GaugeRoPESpikeTests {
    static let heads = 4
    static let headDim = 64
    static let rotaryDims = 16

    static func syntheticKV(tokens: Int, seed: UInt64) -> GaugeRoPEKV {
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 2)
        let nopeWidth = heads * (headDim - rotaryDims)
        let width = nopeWidth + heads * headDim
        let spectrum = MLX.reciprocal(MLXArray(0 ..< width).asType(.float32) / Float(32) + Float(1))
        let joint = MLXRandom.normal([tokens, width], dtype: .float32, key: keys[0]) * spectrum
        let shared = MLXRandom.normal([tokens, 1, rotaryDims], dtype: .float32, key: keys[1]) * Float(1.5)
        let mix = MLXArray([1.0, 0.8, -0.6, 0.5] as [Float]).reshaped([1, heads, 1])
        let nope = joint[0..., 0 ..< nopeWidth].reshaped([tokens, heads, headDim - rotaryDims])
        let values = joint[0..., nopeWidth...].reshaped([tokens, heads, headDim])
        return GaugeRoPEKV(keys: concatenated([shared * mix, nope], axis: -1), values: values)
    }

    static func error(_ approx: GaugeRoPEKV, _ exact: GaugeRoPEKV) -> Double {
        GaugeRoPEArms.relativeError([approx.keys, approx.values], [exact.keys, exact.values])
    }

    static func attentionError(_ queries: MLXArray, _ approx: GaugeRoPEKV, _ exact: GaugeRoPEKV) -> Double {
        let scale = 1 / Float(Self.headDim).squareRoot()
        let output = { (kv: GaugeRoPEKV) in
            GaugeRoPEArms.attend(
                queries: queries, keys: GaugeRoPEArms.rope(kv.keys, rotaryDims: Self.rotaryDims, base: 10_000),
                values: kv.values, scale: scale)
        }
        return GaugeRoPEArms.relativeError([output(approx)], [output(exact)])
    }

    @Test("Carrier indexing check: per-frequency encode commutes with RoPE at offsets 0...4096 (true for any per-frequency matrix, so it checks indexing, not quality)")
    func carrierIndexingCommutesWithRoPE() {
        let keys = MLXRandom.normal([64, 4, 64], dtype: .float32, key: MLXRandom.key(6470))
        let carrier = GaugeRoPECarrier.calibrate(preRopeKeys: keys, rotaryDims: 64, carrierHeads: 2)
        for offset in [0, 17, 512, 4096] {
            let rotThenEncode = carrier.encode(GaugeRoPEArms.rope(keys, rotaryDims: 64, base: 10_000, offset: offset))
            let encodeThenRot = GaugeRoPEArms.rope(carrier.encode(keys), rotaryDims: 64, base: 10_000, offset: offset)
            eval(rotThenEncode, encodeThenRot)
            let maxDiff = abs(rotThenEncode - encodeThenRot).max().item(Float.self)
            #expect(maxDiff < 1e-4, "offset=\(offset) maxDiff=\(maxDiff)")
        }
    }

    @Test("Spike RoPE helper matches MLXNN.RoPE (traditional false, partial dims, base 1e7) and its inverse restores the input")
    func ropeHelperMatchesStockRoPE() {
        let x = MLXRandom.normal([32, 4, 64], dtype: .float32, key: MLXRandom.key(6473))
        let ours = GaugeRoPEArms.rope(x, rotaryDims: 16, base: 10_000_000, offset: 5)
        let stock = MLXNN.RoPE(dimensions: 16, traditional: false, base: 10_000_000)(x.transposed(1, 0, 2), offset: 5)
            .transposed(1, 0, 2)
        let back = GaugeRoPEArms.rope(ours, rotaryDims: 16, base: 10_000_000, offset: 5, inverse: true)
        eval(ours, stock, back)
        #expect(abs(ours - stock).max().item(Float.self) < 1e-4)
        #expect(abs(back - x).max().item(Float.self) < 1e-4)
    }

    @Test("Gemma 4 proportional RoPE: reordering its two rotated quarters to the front and rescaling the base gives the spike helper the stock rotation")
    func proportionalRopeMapsToHelper() throws {
        let x = MLXRandom.normal([32, 2, 64], dtype: .float32, key: MLXRandom.key(6474))
        let stock = initializeRope(
            dims: 64, base: 1_000_000, traditional: false,
            scalingConfig: ["type": .string("proportional"), "partial_rotary_factor": .float(0.25)],
            maxPositionEmbeddings: nil)
        let rope = GaugeRoPERealModelTests.RopeConfig(base: 1_000_000, partialFactor: 0.25, proportional: true)
        let order = try #require(rope.order(headDim: 64))
        let expected = take(stock(x.transposed(1, 0, 2), offset: 7).transposed(1, 0, 2), order, axis: -1)
        let ours = GaugeRoPEArms.rope(
            take(x, order, axis: -1), rotaryDims: rope.rotaryDims(headDim: 64), base: rope.effectiveBase(headDim: 64), offset: 7)
        eval(expected, ours)
        #expect(rope.rotaryDims(headDim: 64) == 16)
        #expect(abs(ours - expected).max().item(Float.self) < 1e-4)
    }

    @Test("Full rank: MLA-split and GaugeRoPE (carrier rank = KV heads, full joint code) reconstruct pre-RoPE K and V exactly")
    func fullRankArmsAreExact() {
        let kv = Self.syntheticKV(tokens: 640, seed: 6471)
        let mla = MLASplitArm.fit(keys: kv.keys, values: kv.values, rotaryDims: Self.rotaryDims)
        let gauge = GaugeRoPEArm.fit(keys: kv.keys, values: kv.values, rotaryDims: Self.rotaryDims, carrierRank: Self.heads)
        let mlaHat = mla.reconstruct(keys: kv.keys, values: kv.values, rank: mla.codec.maxRank, quantized: false)
        let gaugeHat = gauge.reconstruct(keys: kv.keys, values: kv.values, rank: gauge.codec.maxRank, quantized: false)
        #expect(Self.error(mlaHat, kv) < 1e-4)
        #expect(Self.error(gaugeHat, kv) < 1e-4)
    }

    @Test("Known structure (rotated dims rank 1 across heads): GaugeRoPE beats MLA-split at 0.45x k8v4 bytes, and with a full carrier it equals MLA-split")
    func armOrderingOnKnownStructure() {
        let calibration = Self.syntheticKV(tokens: 1024, seed: 6472)
        let evaluation = Self.syntheticKV(tokens: 256, seed: 6474)
        let budget = 0.45 * GaugeRoPEByteBudget.k8v4BytesPerToken(kvHeads: Self.heads, headDim: Self.headDim)
        let mlaRank = GaugeRoPEByteBudget.codeRank(budget: budget, exactFloats: Self.heads * Self.rotaryDims, multiple: 1)
        let gaugeRank = GaugeRoPEByteBudget.codeRank(budget: budget, exactFloats: Self.rotaryDims, multiple: 1)
        let fit = { (rank: Int) in
            GaugeRoPEArm.fit(keys: calibration.keys, values: calibration.values, rotaryDims: Self.rotaryDims, carrierRank: rank)
        }
        let mla = MLASplitArm.fit(keys: calibration.keys, values: calibration.values, rotaryDims: Self.rotaryDims)
        let mlaHat = mla.reconstruct(keys: evaluation.keys, values: evaluation.values, rank: mlaRank, quantized: false)
        let gaugeHat = fit(1).reconstruct(keys: evaluation.keys, values: evaluation.values, rank: gaugeRank, quantized: false)
        let fullHat = fit(Self.heads).reconstruct(keys: evaluation.keys, values: evaluation.values, rank: mlaRank, quantized: false)
        let queries = MLXRandom.normal([2 * Self.heads, 256, Self.headDim], dtype: .float32, key: MLXRandom.key(6475))
        let (mlaErr, gaugeErr, fullErr) = (Self.error(mlaHat, evaluation), Self.error(gaugeHat, evaluation), Self.error(fullHat, evaluation))
        let mlaAttn = Self.attentionError(queries, mlaHat, evaluation)
        let gaugeAttn = Self.attentionError(queries, gaugeHat, evaluation)
        print(String(
            format: "GAUGEROPE synthetic budget=%.1fB mla-split rank=%ld kvErr=%.5f attnErr=%.5f gauge-c1 rank=%ld kvErr=%.5f attnErr=%.5f gauge-c4 kvErr=%.5f",
            budget, mlaRank, mlaErr, mlaAttn, gaugeRank, gaugeErr, gaugeAttn, fullErr))
        #expect(mlaRank == 211)
        #expect(gaugeRank == 302)
        #expect(gaugeErr < 0.85 * mlaErr)
        #expect(abs(fullErr - mlaErr) < 1e-3)
    }

    @Test("Byte budget: latent arms pay 8.5 bits per exact float and 4.5 per code float like k8v4; ranks are the largest multiple of 64 that fits 0.6x/0.45x/0.3x on Qwen3.8")
    func byteBudgetRanks() {
        let k8v4 = GaugeRoPEByteBudget.k8v4BytesPerToken(kvHeads: 4, headDim: 256)
        #expect(k8v4 * 16 == 26_624)
        let expected: [Int: [Int]] = [256: [1280, 832, 384], 64: [1600, 1152, 704], 128: [1472, 1088, 640]]
        for (exactFloats, ranks) in expected {
            for (ratio, rank) in zip([0.6, 0.45, 0.3], ranks) {
                let budget = ratio * k8v4
                let picked = GaugeRoPEByteBudget.codeRank(budget: budget, exactFloats: exactFloats)
                #expect(picked == rank, "exact=\(exactFloats) ratio=\(ratio)")
                #expect(GaugeRoPEByteBudget.latentBytesPerToken(exactFloats: exactFloats, codeRank: picked) <= budget)
                #expect(GaugeRoPEByteBudget.latentBytesPerToken(exactFloats: exactFloats, codeRank: picked + 64) > budget)
            }
        }
    }
}
