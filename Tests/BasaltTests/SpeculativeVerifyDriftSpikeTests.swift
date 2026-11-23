import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

struct VerifySlotDrift: Sendable {
    let slot: Int
    let maxAbsLogitDiff: Float
    let logitScale: Float
    let top4LogProbDelta: Float
    let emittedLogProbDelta: Float
    let klPlainVerify: Float
    let top2Margin: Float
    let greedyFlipped: Bool
    let sampleFlipped: Bool

    init(slot: Int, row: ArraySlice<Float>) {
        let values = Array(row)
        self.slot = slot
        maxAbsLogitDiff = values[0]
        logitScale = values[1]
        top4LogProbDelta = values[2]
        emittedLogProbDelta = values[3]
        klPlainVerify = values[4]
        top2Margin = values[5]
        greedyFlipped = values[6] > 0
        sampleFlipped = values[7] > 0
    }
}

enum VerifyDriftProbe {
    static let blockSizes = [2, 4, 8, 9, 12, 13, 16]
    static let temperature: Float = 0.7
    private static let columnCount = 8

    static func fixture(_ family: TinyModel.Family, dtype: DType) throws -> any LanguageModel {
        let shapes = family == .gemma4Moe
            ? "\"hidden_size\": 256, \"intermediate_size\": 512, \"head_dim\": 64, \"global_head_dim\": 64, \"moe_intermediate_size\": 64, "
            : "\"hidden_size\": 256, \"intermediate_size\": 512, \"head_dim\": 64, "
        let model = try TinyModel.make(family, adding: shapes)
        model.update(parameters: model.parameters().mapValues { $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0 })
        quantize(model: model, groupSize: 64, bits: 4) { _, module in
            if let linear = module as? Linear { return linear.weight.dim(-1).isMultiple(of: 64) }
            if let embedding = module as? Embedding { return embedding.weight.dim(-1).isMultiple(of: 64) }
            return module is SwitchLinear
        }
        eval(model)
        return model
    }

    static func measure(model: any LanguageModel, prefix: [Int32], seed: UInt64) throws -> [Int: [VerifySlotDrift]] {
        let longest = blockSizes.max() ?? 0
        let reference = try reference(model: model, prefix: prefix, steps: longest)
        let noise = MLXRandom.gumbel([longest, reference.rows.dim(-1)], key: MLXRandom.key(seed))
        var bySize: [Int: [VerifySlotDrift]] = [:]
        for size in blockSizes {
            let verify = try verifyRows(model: model, prefix: prefix, block: Array(reference.block.prefix(size)))
            bySize[size] = slots(plain: reference.rows[..<size], verify: verify, noise: noise[..<size])
        }
        return bySize
    }

    private static func reference(model: any LanguageModel, prefix: [Int32], steps: Int) throws -> (block: [Int32], rows: MLXArray) {
        let cache = try model.newCache(parameters: nil)
        var logits = model(MLXArray(prefix)[.newAxis], cache: cache)[0, -1]
        var block: [Int32] = []
        var rows: [MLXArray] = []
        for _ in 0 ..< steps {
            let token = argMax(logits, axis: -1).item(Int32.self)
            block.append(token)
            logits = model(MLXArray([token])[.newAxis], cache: cache)[0, 0]
            eval(logits)
            rows.append(logits.asType(.float32))
        }
        return (block, stacked(rows))
    }

    private static func verifyRows(model: any LanguageModel, prefix: [Int32], block: [Int32]) throws -> MLXArray {
        let cache = try model.newCache(parameters: nil)
        eval(model(MLXArray(prefix)[.newAxis], cache: cache))
        let rows = model(MLXArray(block)[.newAxis], cache: cache)[0].asType(.float32)
        eval(rows)
        return rows
    }

    private static func slots(plain: MLXArray, verify: MLXArray, noise: MLXArray) -> [VerifySlotDrift] {
        let plainLogProbs = logSoftmax(plain, axis: -1)
        let verifyLogProbs = logSoftmax(verify, axis: -1)
        let logProbDelta = MLX.abs(plainLogProbs - verifyLogProbs)
        let top4 = argSort(-plain, axis: -1)[.ellipsis, ..<4]
        let plainSample = argMax(plain / temperature + noise, axis: -1)
        let verifySample = argMax(verify / temperature + noise, axis: -1)
        let ranked = MLX.sorted(plain, axis: -1)
        let columns: [MLXArray] = [
            MLX.abs(verify - plain).max(axis: -1),
            MLX.abs(plain).max(axis: -1),
            takeAlong(logProbDelta, top4, axis: -1).max(axis: -1),
            takeAlong(logProbDelta, plainSample[.ellipsis, .newAxis], axis: -1).reshaped(-1),
            (MLX.exp(plainLogProbs) * (plainLogProbs - verifyLogProbs)).sum(axis: -1),
            ranked[.ellipsis, -1] - ranked[.ellipsis, -2],
            (argMax(plain, axis: -1) .!= argMax(verify, axis: -1)).asType(.float32),
            (plainSample .!= verifySample).asType(.float32),
        ]
        let table = stacked(columns, axis: 1)
        eval(table)
        let values = table.asArray(Float.self)
        return (0 ..< plain.dim(0)).map { slot in
            VerifySlotDrift(slot: slot, row: values[(slot * columnCount) ..< ((slot + 1) * columnCount)])
        }
    }

    static func olsSlope(_ points: [(x: Double, y: Double)]) -> (slope: Double, standardError: Double) {
        let count = Double(points.count)
        guard count > 2 else { return (0, .infinity) }
        let meanX = points.reduce(0) { $0 + $1.x } / count
        let meanY = points.reduce(0) { $0 + $1.y } / count
        let sxx = points.reduce(0) { $0 + ($1.x - meanX) * ($1.x - meanX) }
        guard sxx > 0 else { return (0, .infinity) }
        let slope = points.reduce(0) { $0 + ($1.x - meanX) * ($1.y - meanY) } / sxx
        let intercept = meanY - slope * meanX
        let residuals = points.reduce(0) { sum, point in
            let residual = point.y - intercept - slope * point.x
            return sum + residual * residual
        }
        return (slope, (residuals / (count - 2) / sxx).squareRoot())
    }

    static func report(_ label: String, runs: [[Int: [VerifySlotDrift]]]) {
        for size in blockSizes {
            let blocks = runs.compactMap { $0[size] }
            for slot in 0 ..< size {
                print(slotLine(label, size: size, slot: slot, entries: blocks.map { $0[slot] }))
            }
            let points = blocks.flatMap { $0 }
            let drift = olsSlope(points.map { (x: Double($0.slot), y: Double($0.maxAbsLogitDiff)) })
            let logProb = olsSlope(points.map { (x: Double($0.slot), y: Double($0.top4LogProbDelta)) })
            print("VERIFYDRIFT \(label) K=\(size) fit n=\(points.count) maxAbs \(fitText(drift)) top4dLP \(fitText(logProb))")
        }
    }

    private static func fitText(_ fit: (slope: Double, standardError: Double)) -> String {
        let half = 1.96 * fit.standardError
        return String(format: "slope=%.3e se=%.3e ci95=[%.3e, %.3e]", fit.slope, fit.standardError, fit.slope - half, fit.slope + half)
    }

    private static func slotLine(_ label: String, size: Int, slot: Int, entries: [VerifySlotDrift]) -> String {
        let count = Float(max(entries.count, 1))
        let numbers = String(
            format: "maxAbs=%.3e top4dLP=%.3e emitdLP=%.3e kl=%.3e minMargin=%.3e",
            entries.map(\.maxAbsLogitDiff).max() ?? 0,
            entries.reduce(0) { $0 + $1.top4LogProbDelta } / count,
            entries.reduce(0) { $0 + $1.emittedLogProbDelta } / count,
            entries.reduce(0) { $0 + $1.klPlainVerify } / count,
            entries.map(\.top2Margin).min() ?? 0)
        let greedy = entries.filter(\.greedyFlipped).count
        let sampled = entries.filter(\.sampleFlipped).count
        return "VERIFYDRIFT \(label) K=\(size) slot=\(slot) \(numbers) greedyFlips=\(greedy)/\(entries.count) sampleFlips=\(sampled)/\(entries.count)"
    }
}

@Suite("Speculative verify drift S0 spike: per-slot M = 1 vs K-row verify logits on quantized tiny fixtures (#646)", .serialized)
struct SpeculativeVerifyDriftSpikeTests {
    private static let tokens = (0 ..< 32).map { Int32(($0 * 17 + 3) % 240) }
    private static let positions = [8, 19, 32]

    private static func run(_ family: TinyModel.Family, dtype: DType) throws -> [VerifySlotDrift] {
        let model = try VerifyDriftProbe.fixture(family, dtype: dtype)
        #expect(model.leafModules().flattened().contains { $0.1 is QuantizedLinear })
        let runs = try positions.map { position in
            try VerifyDriftProbe.measure(model: model, prefix: Array(tokens.prefix(position)), seed: UInt64(646 + position))
        }
        VerifyDriftProbe.report("\(family.rawValue) \(dtype) q4", runs: runs)
        return runs.flatMap { $0.values.flatMap { $0 } }
    }

    @Test("S0 harness sanity: tiny fp32 4-bit fixtures keep every verify slot within 1e-5 of max(1, max |logit|), about 84 fp32 ulps, with no greedy flip",
          arguments: [TinyModel.Family.qwen3, .qwen35, .gemma4Moe])
    func fp32HarnessSanity(family: TinyModel.Family) throws {
        for entry in try Self.run(family, dtype: .float32) {
            #expect(entry.maxAbsLogitDiff <= 1e-5 * max(1, entry.logitScale),
                    "\(family) slot \(entry.slot): drift \(entry.maxAbsLogitDiff), logit scale \(entry.logitScale)")
            #expect(!entry.greedyFlipped, "\(family) slot \(entry.slot) flipped the greedy token at margin \(entry.top2Margin)")
        }
    }

    @Test("S0 kernel thresholds: tiny bf16 4-bit fixtures with head dim 64 report per-slot drift for K in {2, 4, 8, 9, 12, 13, 16}; a greedy flip needs a top-2 margin within twice the slot drift",
          arguments: [TinyModel.Family.qwen3, .qwen35, .gemma4Moe])
    func bf16KernelThresholds(family: TinyModel.Family) throws {
        for entry in try Self.run(family, dtype: .bfloat16) {
            #expect(entry.maxAbsLogitDiff.isFinite && entry.top4LogProbDelta.isFinite && entry.klPlainVerify.isFinite)
            #expect(!entry.greedyFlipped || entry.top2Margin <= 2 * entry.maxAbsLogitDiff + 1e-6,
                    "\(family) slot \(entry.slot): flip at margin \(entry.top2Margin) above twice the drift \(entry.maxAbsLogitDiff)")
        }
    }

    @Test("OLS slope and standard error match hand-computed values for flat, linear and noisy per-slot drift")
    func olsSlopeAndStandardError() {
        let flat = VerifyDriftProbe.olsSlope([0.01, 0.01, 0.01, 0.01].enumerated().map { (x: Double($0.offset), y: $0.element) })
        let rising = VerifyDriftProbe.olsSlope([0.0, 0.02, 0.04, 0.06].enumerated().map { (x: Double($0.offset), y: $0.element) })
        let noisy = VerifyDriftProbe.olsSlope([0.0, 1.0, 0.0, 1.0].enumerated().map { (x: Double($0.offset), y: $0.element) })
        #expect(abs(flat.slope) < 1e-12 && flat.standardError < 1e-12)
        #expect(abs(rising.slope - 0.02) < 1e-12 && rising.standardError < 1e-9)
        #expect(abs(noisy.slope - 0.2) < 1e-12)
        #expect(abs(noisy.standardError - 0.08.squareRoot()) < 1e-12)
    }
}
