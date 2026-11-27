import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
@testable import BasaltCore
@testable import BasaltInference

private enum StabilityTestSupport {
    static func deterministicFloats(count: Int, seed: UInt64, scale: Double = 1.0, offset: Double = 0.0) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0 ..< count).map { idx in
            let u = Double(rng.next() >> 11) / Double(1 << 53)
            return Float(((u - 0.5) * 2.0 + 0.1 * Foundation.sin(Double(idx) * 0.37)) * scale + offset)
        }
    }

    static func maxParamAbsDiff(_ lhs: Module, _ rhs: Module) -> Float {
        let leftMap = Dictionary(uniqueKeysWithValues: lhs.trainableParameters().flattened())
        let rightMap = Dictionary(uniqueKeysWithValues: rhs.trainableParameters().flattened())
        var worst: Float = 0
        for (key, leftArr) in leftMap {
            guard let rightArr = rightMap[key] else { return .infinity }
            let diff = MLX.abs(leftArr.asType(.float32) - rightArr.asType(.float32)).max()
            eval(diff)
            worst = Swift.max(worst, diff.item(Float.self))
        }
        return worst
    }

    static func makeLoRAQwen3(seed: UInt64) throws -> any LanguageModel {
        let model = try TinyModel.make(.qwen3)
        _ = try withRandomState(MLXRandom.RandomState(seed: seed)) {
            try MLXInferenceEngine.attachLoRA(to: model, layers: [2, 3], rank: 4)
        }
        eval(model as Module)
        return model
    }

    static func sftBatches() -> [(MLXArray, MLXArray)] {
        [
            (
                MLXArray([Int32(1), 5, 9, 13, 21, 34, 55, 89]).reshaped([1, 8]),
                MLXArray([Int32(5), 9, 13, 21, 34, 55, 89, 12]).reshaped([1, 8])
            ),
            (
                MLXArray([Int32(7), 14, 28, 42, 63, 77, 91, 105]).reshaped([1, 8]),
                MLXArray([Int32(14), 28, 42, 63, 77, 91, 105, 19]).reshaped([1, 8])
            ),
        ]
    }
}

@Suite("Training stability telemetry value core: math oracles, GRPO metrics, accumulator, and forward-compatible Codable (#861)")
struct TrainingStabilityCoreTests {
    @Test("TrainingStabilityMath.updateToWeightRatio and gradNorm match closed-form Frobenius norms, step RMS, and edge cases")
    func updateToWeightRatioAndGradNormOracles() {
        let same = TrainingStabilityMath.updateToWeightRatio(before: [3.0, 4.0], after: [3.0, 4.0])
        #expect(same.updateNorm == 0.0)
        #expect(same.weightNorm == 5.0)
        #expect(same.ratio == 0.0)
        #expect(same.stepRMS == 0.0)

        let doubled = TrainingStabilityMath.updateToWeightRatio(before: [3.0, 4.0], after: [6.0, 8.0])
        #expect(Swift.abs(doubled.updateNorm - 5.0) < 1e-12)
        #expect(Swift.abs(doubled.weightNorm - 5.0) < 1e-12)
        #expect(Swift.abs(doubled.ratio - 1.0) < 1e-11)
        #expect(Swift.abs(doubled.stepRMS - (25.0 / 2.0).squareRoot()) < 1e-12)

        let zeroInit = TrainingStabilityMath.updateToWeightRatio(before: [0.0, 0.0], after: [1e-3, 0.0], eps: 1e-12)
        #expect(zeroInit.weightNorm == 0.0)
        #expect(zeroInit.ratio.isFinite)
        #expect(Swift.abs(zeroInit.ratio - 1e9) < 1e-3)

        let empty = TrainingStabilityMath.updateToWeightRatio(before: [], after: [])
        let mismatch = TrainingStabilityMath.updateToWeightRatio(before: [1.0], after: [1.0, 2.0])
        #expect(empty.ratio == 0.0)
        #expect(empty.stepRMS == 0.0)
        #expect(mismatch.ratio == 0.0)
        #expect(mismatch.updateNorm == 0.0)

        #expect(Swift.abs(TrainingStabilityMath.gradNorm(tensors: [[3.0, 4.0], [12.0]]) - 13.0) < 1e-12)
        #expect(TrainingStabilityMath.gradNorm(tensors: []) == 0.0)
    }

    @Test("TrainingStabilityMath.logitZLossStats is numerically stable for large and negative logits and propagates NaN")
    func logitZLossStatsStabilityAndEdgeCases() {
        let uniform = TrainingStabilityMath.logitZLossStats(logitsRows: [[0.0, 0.0]])
        let log2 = Foundation.log(2.0)
        #expect(Swift.abs(uniform.zLoss - log2 * log2) < 1e-12)
        #expect(uniform.maxLogit == 0.0)

        let shifted = TrainingStabilityMath.logitZLossStats(logitsRows: [[1000.0, 1000.0]])
        let expectedLSE = 1000.0 + log2
        #expect(shifted.zLoss.isFinite)
        #expect(Swift.abs(shifted.zLoss - expectedLSE * expectedLSE) < 1e-6)
        #expect(shifted.maxLogit == 1000.0)

        let negative = TrainingStabilityMath.logitZLossStats(logitsRows: [[-5.0, -2.0], [-4.0, -3.0]])
        #expect(negative.zLoss > 0.0)
        #expect(negative.zLoss.isFinite)
        #expect(negative.maxLogit == -2.0)
        #expect(TrainingStabilityMath.logitZLossStats(logitsRows: []) == LogitZLossStats(zLoss: 0.0, maxLogit: 0.0))

        let nanStats = TrainingStabilityMath.logitZLossStats(logitsRows: [[1.0, .nan]])
        #expect(nanStats.zLoss.isNaN)
        #expect(nanStats.maxLogit.isNaN)
    }

    @Test("TrainingStabilityMath.grpoMetrics computes zero-advantage group rate, clip fraction, KL, and reward variance")
    func grpoMetricsAndClipCountFormulas() {
        let metrics = TrainingStabilityMath.grpoMetrics(
            updatesApplied: 6, skippedGroups: 2, clippedTokenCount: 15,
            totalTokenCount: 100, meanKL: 0.035, rewards: [0.0, 1.0, 0.0, 1.0]
        )
        #expect(metrics.totalGroups == 8)
        #expect(metrics.skippedZeroAdvantageGroups == 2)
        #expect(Swift.abs(metrics.zeroAdvantageGroupRate - 0.25) < 1e-12)
        #expect(Swift.abs(metrics.clipFraction - 0.15) < 1e-12)
        #expect(Swift.abs(metrics.meanKL - 0.035) < 1e-12)
        #expect(Swift.abs(metrics.rewardVariance - 0.25) < 1e-12)

        let empty = TrainingStabilityMath.grpoMetrics(
            updatesApplied: 0, skippedGroups: 0, clippedTokenCount: 5,
            totalTokenCount: 0, meanKL: .nan, rewards: [1.0, 1.0]
        )
        #expect(empty.totalGroups == 0)
        #expect(empty.zeroAdvantageGroupRate == 0.0)
        #expect(empty.clipFraction == 0.0)
        #expect(empty.meanKL == 0.0)
        #expect(empty.rewardVariance == 0.0)

        let posClip = TrainingStabilityMath.grpoClippedTokenCount(logPolicy: [0.5, 0.0, -0.5], logOld: [0.0, 0.0, 0.0], advantage: 1.0, clipEpsilon: 0.2)
        let negClip = TrainingStabilityMath.grpoClippedTokenCount(logPolicy: [0.5, 0.0, -0.5], logOld: [0.0, 0.0, 0.0], advantage: -1.0, clipEpsilon: 0.2)
        let zeroAdv = TrainingStabilityMath.grpoClippedTokenCount(logPolicy: [0.5, -0.5], logOld: [0.0, 0.0], advantage: 0.0, clipEpsilon: 0.2)
        #expect(posClip == 1)
        #expect(negClip == 1)
        #expect(zeroAdv == 0)
    }

    @Test("TrainingStabilityAccumulator bounds recentSteps and isolates non-finite steps without poisoning aggregates")
    func accumulatorBoundsAndNonFiniteIsolation() throws {
        var accumulator = TrainingStabilityAccumulator(maxRecentSteps: 3)
        for step in 0 ..< 5 {
            let ratio = 0.01 * Double(step + 1)
            let layer = LayerUpdateRatio(layerName: "layer.\(step)", updateNorm: ratio, weightNorm: 1.0, ratio: ratio, stepRMS: ratio * 0.5)
            let stats = LogitZLossStats(zLoss: Double(step + 1) * 4.0, maxLogit: Double(step) - 3.0)
            accumulator.record(TrainingStepTelemetry.from(step: step, gradNorm: Double(step + 1), layerRatios: [layer], logitStats: stats, topK: 2))
        }
        let nanStep = TrainingStepTelemetry(
            step: 5, gradNorm: .nan, meanUpdateToWeightRatio: .infinity,
            maxUpdateToWeightRatio: .nan, meanStepRMS: .nan,
            logitStats: LogitZLossStats(zLoss: .nan, maxLogit: .infinity)
        )
        accumulator.record(nanStep)
        accumulator.setGRPO(TrainingStabilityMath.grpoMetrics(updatesApplied: 3, skippedGroups: 1, clippedTokenCount: 2, totalTokenCount: 10, meanKL: 0.02, rewards: [0.0, 1.0]))

        let summary = accumulator.finalize()
        #expect(summary.sampledSteps == 6)
        #expect(summary.nonFiniteStepCount == 1)
        #expect(summary.recentSteps.map(\.step) == [3, 4, 5])
        #expect(Swift.abs(summary.maxGradNorm - 5.0) < 1e-12)
        #expect(Swift.abs(summary.meanGradNorm - 3.0) < 1e-12)
        #expect(Swift.abs(summary.maxUpdateToWeightRatio - 0.05) < 1e-12)
        #expect(Swift.abs(summary.meanUpdateToWeightRatio - 0.03) < 1e-12)
        #expect(Swift.abs(summary.meanStepRMS - 0.015) < 1e-12)
        #expect(summary.maxZLoss == 20.0)
        #expect(summary.maxLogit == 1.0)

        let line = summary.formattedSummaryLine()
        #expect(line.contains("stability: steps=6"))
        #expect(line.contains("zLoss=20.00"))
        #expect(line.contains("grpo"))
        #expect(line.contains("nonFinite=1"))
        #expect(try JSONDecoder().decode(TrainingStabilitySummary.self, from: JSONEncoder().encode(summary)) == summary)
        #expect(try JSONDecoder().decode(TrainingStabilityAccumulator.self, from: JSONEncoder().encode(accumulator)) == accumulator)
    }

    @Test("Telemetry types decode forward-compatibly from partial JSON payloads with safe defaults")
    func forwardCompatibleCodableDecoding() throws {
        let emptyJSON = Data("{}".utf8)
        #expect(try JSONDecoder().decode(TrainingStabilitySummary.self, from: emptyJSON) == TrainingStabilitySummary())
        #expect(try JSONDecoder().decode(TrainingStepTelemetry.self, from: emptyJSON) == TrainingStepTelemetry())
        #expect(try JSONDecoder().decode(LayerUpdateRatio.self, from: emptyJSON) == LayerUpdateRatio(layerName: "", updateNorm: 0, weightNorm: 0, ratio: 0, stepRMS: 0))
        #expect(try JSONDecoder().decode(LogitZLossStats.self, from: emptyJSON) == LogitZLossStats(zLoss: 0, maxLogit: 0))
        #expect(try JSONDecoder().decode(GRPOStabilityMetrics.self, from: emptyJSON) == GRPOStabilityMetrics(totalGroups: 0, skippedZeroAdvantageGroups: 0, zeroAdvantageGroupRate: 0, clipFraction: 0, meanKL: 0, rewardVariance: 0))
        #expect(try JSONDecoder().decode(TrainingStabilityAccumulator.self, from: emptyJSON) == TrainingStabilityAccumulator())
    }
}

@Suite("Training stability monitor MLX parity and live TinyModel Qwen3 SFT verification (#861)", .serialized, .timeLimit(.minutes(2)))
struct TrainingStabilityMonitorTests {
    private static func makeSyntheticFixtures() -> (
        before: ModuleParameters,
        after: ModuleParameters,
        grads: ModuleParameters,
        logits: MLXArray,
        expected: TrainingStepTelemetry
    ) {
        let specs: [(String, [Int], UInt64, Double)] = [
            ("attn.q_proj.lora_a", [12, 8], 101, 0.03),
            ("attn.q_proj.lora_b", [8, 16], 102, 0.08),
            ("mlp.bias", [16], 103, 0.01),
            ("conv.weight", [2, 4, 6], 104, 0.05),
        ]
        var beforePairs: [(String, MLXArray)] = [("quant.packed", MLXArray.zeros([4], dtype: .uint32))]
        var afterPairs: [(String, MLXArray)] = [("quant.packed", MLXArray.ones([4], dtype: .uint32))]
        var gradPairs: [(String, MLXArray)] = []
        var expectedLayers: [LayerUpdateRatio] = []
        var expectedGrads: [[Double]] = []
        for (name, shape, seed, stepScale) in specs {
            let count = shape.reduce(1, *)
            let bVals = StabilityTestSupport.deterministicFloats(count: count, seed: seed, scale: 0.7)
            let dVals = StabilityTestSupport.deterministicFloats(count: count, seed: seed + 50, scale: stepScale)
            let aVals = zip(bVals, dVals).map { $0 + $1 }
            let gVals = StabilityTestSupport.deterministicFloats(count: count, seed: seed + 100, scale: 0.4)
            beforePairs.append((name, MLXArray(bVals, shape)))
            afterPairs.append((name, MLXArray(aVals, shape)))
            gradPairs.append((name, MLXArray(gVals, shape)))
            let oracle = TrainingStabilityMath.updateToWeightRatio(before: bVals.map(Double.init), after: aVals.map(Double.init))
            expectedLayers.append(LayerUpdateRatio(layerName: name, updateNorm: oracle.updateNorm, weightNorm: oracle.weightNorm, ratio: oracle.ratio, stepRMS: oracle.stepRMS))
            expectedGrads.append(gVals.map(Double.init))
        }
        let logitVals = StabilityTestSupport.deterministicFloats(count: 2 * 5 * 32, seed: 861, scale: 3.5, offset: -0.4)
        let rows: [[Double]] = stride(from: 0, to: logitVals.count, by: 32).map { logitVals[$0 ..< ($0 + 32)].map(Double.init) }
        let expected = TrainingStepTelemetry.from(
            step: 7,
            gradNorm: TrainingStabilityMath.gradNorm(tensors: expectedGrads),
            layerRatios: expectedLayers,
            logitStats: TrainingStabilityMath.logitZLossStats(logitsRows: rows),
            topK: 3
        )
        return (
            ModuleParameters.unflattened(beforePairs),
            ModuleParameters.unflattened(afterPairs),
            ModuleParameters.unflattened(gradPairs),
            MLXArray(logitVals, [2, 5, 32]),
            expected
        )
    }

    @Test("TrainingStabilityMonitor MLX path matches TrainingStabilityMath CPU Double oracle to < 1e-4")
    func mlxParityAgainstCPUDoubleOracle() throws {
        let fixture = Self.makeSyntheticFixtures()
        let snapshot = TrainingStabilityMonitor.snapshotWeights(fixture.before)
        #expect(snapshot["quant.packed"] == nil)
        #expect(snapshot.count == 4)

        let actual = TrainingStabilityMonitor.computeStepTelemetry(
            step: 7, before: snapshot, after: fixture.after,
            gradients: fixture.grads, logits: fixture.logits, topK: 3
        )
        let expected = fixture.expected
        #expect(actual.step == expected.step)
        #expect(Swift.abs(actual.gradNorm - expected.gradNorm) < 1e-4)
        #expect(Swift.abs(actual.meanUpdateToWeightRatio - expected.meanUpdateToWeightRatio) < 1e-4)
        #expect(Swift.abs(actual.maxUpdateToWeightRatio - expected.maxUpdateToWeightRatio) < 1e-4)
        #expect(Swift.abs(actual.meanStepRMS - expected.meanStepRMS) < 1e-4)

        let actualLogits = try #require(actual.logitStats)
        let expectedLogits = try #require(expected.logitStats)
        #expect(Swift.abs(actualLogits.zLoss - expectedLogits.zLoss) < 1e-4)
        #expect(Swift.abs(actualLogits.maxLogit - expectedLogits.maxLogit) < 1e-4)

        let standalone = TrainingStabilityMonitor.computeLogitZLossStats(fixture.logits)
        #expect(Swift.abs(standalone.zLoss - expectedLogits.zLoss) < 1e-4)
        #expect(Swift.abs(standalone.maxLogit - expectedLogits.maxLogit) < 1e-4)
        #expect(TrainingStabilityMonitor.computeLogitZLossStats(MLXArray.zeros([0], dtype: .float32)) == LogitZLossStats(zLoss: 0, maxLogit: 0))

        #expect(actual.topLayerRatios.count == 3)
        for (got, want) in zip(actual.topLayerRatios, expected.topLayerRatios) {
            #expect(got.layerName == want.layerName)
            #expect(Swift.abs(got.updateNorm - want.updateNorm) < 1e-4)
            #expect(Swift.abs(got.weightNorm - want.weightNorm) < 1e-4)
            #expect(Swift.abs(got.ratio - want.ratio) < 1e-4)
            #expect(Swift.abs(got.stepRMS - want.stepRMS) < 1e-4)
        }
    }

    @Test("TrainingStabilityMonitor GRPO clip count matches CPU oracle and disabled or cadenced sampling skips work")
    func grpoClipParityAndSamplingCadence() {
        let logPolicyVals: [Float] = [-1.0, -1.5, -2.0, -0.5, -3.0]
        let logOldVals: [Float] = [-1.5, -1.5, -1.4, -0.6, -2.1]
        let (logPolicy, logOld) = (MLXArray(logPolicyVals), MLXArray(logOldVals))
        for adv in [Float(1.0), Float(-1.0), Float(0.0)] {
            let (clippedArr, total) = TrainingStabilityMonitor.grpoClippedTokenCount(
                logPolicy: logPolicy, logOld: logOld, advantage: adv, clipEpsilon: 0.2
            )
            eval(clippedArr)
            let expected = TrainingStabilityMath.grpoClippedTokenCount(
                logPolicy: logPolicyVals.map(Double.init), logOld: logOldVals.map(Double.init),
                advantage: Double(adv), clipEpsilon: 0.2
            )
            #expect(total == logPolicyVals.count)
            #expect(Int(clippedArr.item(Int32.self)) == expected)
        }

        let params = ModuleParameters.unflattened([("w", MLXArray([Float(1), 2, 3, 4], [2, 2]))])
        let after = ModuleParameters.unflattened([("w", MLXArray([Float(1.1), 2.1, 3.1, 4.1], [2, 2]))])
        let disabled = TrainingStabilityMonitor(sampleEvery: 0)
        #expect(!disabled.isEnabled)
        #expect(!disabled.shouldSample(step: 0))
        #expect(disabled.snapshotWeights(params, step: 0).isEmpty)
        #expect(TrainingStabilityMonitor.snapshotWeights(params, enabled: false).isEmpty)
        var disabledAcc = TrainingStabilityAccumulator()
        disabled.recordStep(step: 0, before: [:], after: after, into: &disabledAcc)
        #expect(disabledAcc.sampledSteps == 0)

        let cadenced = TrainingStabilityMonitor(sampleEvery: 2, topK: 2)
        var cadencedAcc = TrainingStabilityAccumulator()
        for step in 0 ..< 5 {
            let snap = cadenced.snapshotWeights(params, step: step)
            #expect(snap.isEmpty == (step % 2 != 0))
            cadenced.recordStep(step: step, before: snap, after: after, into: &cadencedAcc)
        }
        #expect(cadencedAcc.sampledSteps == 3)
        #expect(cadencedAcc.recentSteps.map(\.step) == [0, 2, 4])
    }
}

extension TrainingStabilityMonitorTests {
    @Test("Live TinyModel Qwen3 2-step SFT update produces finite positive telemetry without corrupting autograd or weights")
    func liveTinyModelQwen3TwoStepSFTTelemetry() throws {
        let observed = try StabilityTestSupport.makeLoRAQwen3(seed: 861)
        let reference = try StabilityTestSupport.makeLoRAQwen3(seed: 861)
        try reference.update(parameters: observed.parameters(), verify: [.all])
        eval(observed, reference)
        defer {
            MLXInferenceEngine.detachLoRA(from: observed)
            MLXInferenceEngine.detachLoRA(from: reference)
        }

        let optObserved = AdamW(learningRate: 1e-3, weightDecay: 0.0)
        let optReference = AdamW(learningRate: 1e-3, weightDecay: 0.0)
        let monitor = TrainingStabilityMonitor(sampleEvery: 1, topK: 5)
        var accumulator = TrainingStabilityAccumulator()

        let stepObserved = valueAndGrad(model: observed as Module) { _, arrays in
            let logits = observed(arrays[0], cache: nil).asType(.float32)
            return [crossEntropy(logits: logits, targets: arrays[1], reduction: .mean), stopGradient(logits)]
        }
        let stepReference = valueAndGrad(model: reference as Module) { _, arrays in
            [crossEntropy(logits: reference(arrays[0], cache: nil).asType(.float32), targets: arrays[1], reduction: .mean)]
        }

        for (step, batch) in StabilityTestSupport.sftBatches().enumerated() {
            let before = monitor.snapshotWeights(observed.trainableParameters(), step: step)
            let (obsValues, obsGrads) = stepObserved(observed as Module, [batch.0, batch.1])
            optObserved.update(model: observed as Module, gradients: obsGrads)
            eval(observed as Module, optObserved, obsValues)
            monitor.recordStep(
                step: step, before: before, after: observed.trainableParameters(),
                gradients: obsGrads, logits: obsValues[1], into: &accumulator
            )

            let (refValues, refGrads) = stepReference(reference as Module, [batch.0, batch.1])
            optReference.update(model: reference as Module, gradients: refGrads)
            eval(reference as Module, optReference, refValues)
            #expect(Swift.abs(obsValues[0].item(Float.self) - refValues[0].item(Float.self)) < 1e-6)
        }

        #expect(StabilityTestSupport.maxParamAbsDiff(observed as Module, reference as Module) < 1e-6)
        try Self.expectValidTwoStepSummary(accumulator.finalize())
    }

    private static func expectValidTwoStepSummary(_ summary: TrainingStabilitySummary) throws {
        #expect(summary.sampledSteps == 2)
        #expect(summary.nonFiniteStepCount == 0)
        #expect(summary.recentSteps.count == 2)
        #expect(summary.maxGradNorm.isFinite)
        #expect(summary.maxGradNorm > 0)
        #expect(summary.maxUpdateToWeightRatio.isFinite)
        #expect(summary.maxUpdateToWeightRatio > 0)
        #expect(summary.maxUpdateToWeightRatio < 100.0)
        #expect(summary.meanStepRMS.isFinite)
        #expect(summary.meanStepRMS > 0)
        let maxZLoss = try #require(summary.maxZLoss)
        let maxLogit = try #require(summary.maxLogit)
        #expect(maxZLoss > 0)
        #expect(maxLogit.isFinite)
        for item in summary.recentSteps {
            #expect(item.gradNorm > 0)
            #expect(item.meanUpdateToWeightRatio.isFinite)
            #expect(item.meanUpdateToWeightRatio >= 0)
            #expect(item.meanUpdateToWeightRatio < 100.0)
            #expect(item.meanStepRMS.isFinite)
            #expect(item.meanStepRMS >= 0)
            let stats = try #require(item.logitStats)
            #expect(stats.zLoss > 0)
            #expect(!item.topLayerRatios.isEmpty)
        }
        #expect(summary.recentSteps[1].meanUpdateToWeightRatio > 0)
        #expect(summary.recentSteps[1].meanStepRMS > 0)
        for layer in summary.recentSteps[1].topLayerRatios {
            #expect(layer.weightNorm > 0)
            #expect(layer.ratio.isFinite)
            #expect(layer.ratio > 0)
        }
    }

    @Test("ADPOTrainer.optimize forwards stabilityTelemetryEvery and returns populated summary")
    func adpoOptimizeForwardsStabilityTelemetry() throws {
        let model = try TinyModel.make(.qwen3Small)
        let pair = TokenizedPreferencePair(
            promptIds: [1, 2, 3],
            chosenIds: [4, 5],
            rejectedIds: [6, 7]
        )
        var options = ADPOOptimizeOptions(iterations: 2, learningRate: 1e-4, batchSize: 1)
        options.stabilityTelemetryEvery = 1
        let outcome = try ADPOTrainer.optimize(
            model: model,
            pairs: [pair],
            config: ADPOConfig(beta: 0.1),
            options: options
        )
        let summary = try #require(outcome.stabilitySummary)
        #expect(summary.sampledSteps == 2)
        #expect(summary.nonFiniteStepCount == 0)
        #expect(summary.maxGradNorm.isFinite && summary.maxGradNorm >= 0)
    }
}
