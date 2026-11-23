import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

struct VerifySlotDrift: Sendable {
    let slot: Int
    let maxAbsLogitDiff: Float
    let meanAbsLogitDiff: Float
    let top2Margin: Float
    let greedyFlipped: Bool
}

enum SpeculativeDriftProbe {
    static func measure(
        model: any LanguageModel,
        prefix: [Int32],
        block: [Int32]
    ) throws -> [VerifySlotDrift] {
        let batchedCache = try model.newCache(parameters: nil)
        let stepCache = try model.newCache(parameters: nil)
        let prefixArray = MLXArray(prefix)[.newAxis]
        let preA = model(prefixArray, cache: batchedCache)
        let preB = model(prefixArray, cache: stepCache)
        eval(preA, preB)

        let batchedLogits = model(MLXArray(block)[.newAxis], cache: batchedCache).asType(.float32)
        eval(batchedLogits)

        var results: [VerifySlotDrift] = []
        results.reserveCapacity(block.count)
        for (slot, token) in block.enumerated() {
            let single = model(MLXArray([token])[.newAxis], cache: stepCache)[0, 0].asType(.float32)
            let verify = batchedLogits[0, slot]
            let diff = abs(verify - single)
            let maxDiff = diff.max().item(Float.self)
            let meanDiff = diff.mean().item(Float.self)
            let top2 = sorted(single, axis: -1)
            let vocab = single.dim(-1)
            let margin = (top2[vocab - 1] - top2[vocab - 2]).item(Float.self)
            let singleTop = argMax(single, axis: -1).item(Int32.self)
            let verifyTop = argMax(verify, axis: -1).item(Int32.self)
            results.append(VerifySlotDrift(
                slot: slot,
                maxAbsLogitDiff: maxDiff,
                meanAbsLogitDiff: meanDiff,
                top2Margin: margin,
                greedyFlipped: singleTop != verifyTop
            ))
        }
        return results
    }

    static func linearSlope(_ values: [Float]) -> Float {
        guard values.count >= 2 else { return 0 }
        let n = Float(values.count)
        let meanX = (n - 1) / 2
        let meanY = values.reduce(0, +) / n
        var num: Float = 0
        var den: Float = 0
        for (i, y) in values.enumerated() {
            let dx = Float(i) - meanX
            num += dx * (y - meanY)
            den += dx * dx
        }
        return den > 0 ? num / den : 0
    }
}

@Suite("Speculative verify drift S0 spike: per-slot M=1 vs K-token verify logits and kernel thresholds (#646)")
struct SpeculativeVerifyDriftSpikeTests {
    private static let prefix = (0 ..< 16).map { Int32(($0 * 17 + 3) % 240) }
    private static func block(count: Int) -> [Int32] {
        (0 ..< count).map { Int32(($0 * 29 + 7) % 240) }
    }

    @Test("S0 harness sanity: tiny fp32 Qwen3 has near-zero per-slot verify drift (<= 1e-4) and zero greedy flips")
    func tinyFP32HarnessSanity() throws {
        let model = try TinyModel.make(.qwen3Small)
        let fp32Params = model.parameters().mapValues { $0.dtype.isFloatingPoint ? $0.asType(.float32) : $0 }
        model.update(parameters: fp32Params)

        let slots = try SpeculativeDriftProbe.measure(model: model, prefix: Self.prefix, block: Self.block(count: 8))
        #expect(slots.count == 8)
        for entry in slots {
            #expect(entry.maxAbsLogitDiff <= 1e-4, "slot \(entry.slot) maxDiff=\(entry.maxAbsLogitDiff)")
            #expect(!entry.greedyFlipped, "slot \(entry.slot) flipped greedy token with margin=\(entry.top2Margin)")
        }
    }

    @Test("S0 kernel threshold probe: tiny bf16 reports per-slot drift across K = 8, 9 (SDPA) and K = 12, 13 (qmv_wide/qmm)")
    func tinyBF16KernelThresholds() throws {
        let model = try TinyModel.make(.qwen3Small)
        let bf16Params = model.parameters().mapValues { $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0 }
        model.update(parameters: bf16Params)

        var maxByK: [Int: Float] = [:]
        for k in [8, 9, 12, 13] {
            let slots = try SpeculativeDriftProbe.measure(model: model, prefix: Self.prefix, block: Self.block(count: k))
            #expect(slots.count == k)
            let peak = slots.map(\.maxAbsLogitDiff).max() ?? 0
            let slope = SpeculativeDriftProbe.linearSlope(slots.map(\.maxAbsLogitDiff))
            maxByK[k] = peak
            #expect(peak.isFinite && slope.isFinite)
            #expect(peak < 0.25, "K=\(k) peak drift unexpectedly large: \(peak)")
        }
        #expect(maxByK.count == 4)
    }

    @Test("Linear slope helper accurately measures per-slot drift growth")
    func slopeMeasurement() {
        let flat = SpeculativeDriftProbe.linearSlope([0.01, 0.01, 0.01, 0.01])
        let rising = SpeculativeDriftProbe.linearSlope([0.0, 0.02, 0.04, 0.06])
        #expect(abs(flat) < 1e-6)
        #expect(abs(rising - 0.02) < 1e-6)
    }
}
