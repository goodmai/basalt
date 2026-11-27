import Foundation
import MLX
import MLXNN
import BasaltCore

package struct TrainingStabilityMonitor: Sendable, Equatable {
    package static let defaultTopK = 5

    package let sampleEvery: Int
    package let topK: Int

    package var isEnabled: Bool { sampleEvery > 0 }

    package init(sampleEvery: Int = 0, topK: Int = defaultTopK) {
        self.sampleEvery = max(0, sampleEvery)
        self.topK = max(0, topK)
    }

    package func shouldSample(step: Int) -> Bool {
        sampleEvery > 0 && step >= 0 && step % sampleEvery == 0
    }

    package func snapshotWeights(_ parameters: ModuleParameters, step: Int) -> [String: MLXArray] {
        guard shouldSample(step: step) else { return [:] }
        return Self.snapshotWeights(parameters, enabled: true)
    }

    package func recordStep(
        step: Int,
        before: [String: MLXArray],
        after: ModuleParameters,
        gradients: ModuleParameters? = nil,
        logits: MLXArray? = nil,
        into accumulator: inout TrainingStabilityAccumulator
    ) {
        guard shouldSample(step: step) else { return }
        let telemetry = Self.computeStepTelemetry(
            step: step,
            before: before,
            after: after,
            gradients: gradients,
            logits: logits,
            topK: topK
        )
        accumulator.record(telemetry)
    }

    package static func snapshotWeights(
        _ parameters: ModuleParameters,
        enabled: Bool = true
    ) -> [String: MLXArray] {
        guard enabled else { return [:] }
        let flat = parameters.flattened()
        var snapshot: [String: MLXArray] = [:]
        snapshot.reserveCapacity(flat.count)
        for (key, array) in flat where isMeasurable(array) {
            snapshot[key] = stopGradient(array)
        }
        return snapshot
    }

    package static func computeStepTelemetry(
        step: Int,
        before: [String: MLXArray],
        after: ModuleParameters,
        gradients: ModuleParameters? = nil,
        logits: MLXArray? = nil,
        topK: Int = defaultTopK
    ) -> TrainingStepTelemetry {
        let lazyLayers = buildLayerNorms(before: before, after: after)
        let lazyGradSq = buildGradNormSquared(gradients)
        let lazyLogits = logits.flatMap(lazyLogitStats)
        var pending: [MLXArray] = []
        if let lazyLayers {
            pending.append(lazyLayers.deltaSqSums)
            pending.append(lazyLayers.weightSqSums)
        }
        if let lazyGradSq {
            pending.append(lazyGradSq)
        }
        if let lazyLogits {
            pending.append(lazyLogits.zLoss)
            pending.append(lazyLogits.maxLogit)
        }
        if !pending.isEmpty { eval(pending) }
        let layerRatios = lazyLayers.map(materializeLayerRatios) ?? []
        let gradNorm = lazyGradSq.map { Double($0.item(Float.self)).squareRoot() } ?? 0.0
        let logitStats = resolveLogitStats(lazyLogits, hadInput: logits != nil)
        return TrainingStepTelemetry.from(
            step: step,
            gradNorm: gradNorm,
            layerRatios: layerRatios,
            logitStats: logitStats,
            topK: topK
        )
    }

    package static func computeLogitZLossStats(_ logits: MLXArray) -> LogitZLossStats {
        guard let lazy = lazyLogitStats(logits) else {
            return LogitZLossStats(zLoss: 0.0, maxLogit: 0.0)
        }
        eval(lazy.zLoss, lazy.maxLogit)
        return LogitZLossStats(
            zLoss: Double(lazy.zLoss.item(Float.self)),
            maxLogit: Double(lazy.maxLogit.item(Float.self))
        )
    }

    package static func lazyLogitStats(_ logits: MLXArray) -> (zLoss: MLXArray, maxLogit: MLXArray)? {
        guard isMeasurable(logits) else { return nil }
        let f32 = stopGradient(logits).asType(.float32)
        let lse = f32.ndim > 0 ? MLX.logSumExp(f32, axis: -1) : f32
        return (lse.square().mean(), f32.max())
    }

    package static func grpoClippedTokenCount(
        logPolicy: MLXArray,
        logOld: MLXArray,
        advantage: Float,
        clipEpsilon: Float
    ) -> (clipped: MLXArray, total: Int) {
        let total = logPolicy.size
        guard total > 0, logOld.size == total,
              advantage.isFinite, advantage != 0.0,
              clipEpsilon.isFinite, clipEpsilon > 0.0 else {
            return (MLXArray(Int32(0)), total)
        }
        let diff = stopGradient(logPolicy).asType(.float32) - stopGradient(logOld).asType(.float32)
        let ratio = MLX.exp(MLX.clip(diff, min: Float(-20.0), max: Float(20.0)))
        let mask = advantage > 0.0
            ? (ratio .> MLXArray(Float(1.0) + clipEpsilon))
            : (ratio .< MLXArray(Float(1.0) - clipEpsilon))
        return (mask.asType(.int32).sum(), total)
    }
}

private struct LazyLayerNorms {
    let names: [String]
    let counts: [Int]
    let deltaSqSums: MLXArray
    let weightSqSums: MLXArray
}

extension TrainingStabilityMonitor {
    private static func isMeasurable(_ array: MLXArray) -> Bool {
        array.size > 0 && array.dtype.isFloatingPoint && !array.dtype.isComplex
    }

    private static func resolveLogitStats(
        _ lazyLogits: (zLoss: MLXArray, maxLogit: MLXArray)?,
        hadInput: Bool
    ) -> LogitZLossStats? {
        if let lazyLogits {
            return LogitZLossStats(
                zLoss: Double(lazyLogits.zLoss.item(Float.self)),
                maxLogit: Double(lazyLogits.maxLogit.item(Float.self))
            )
        }
        return hadInput ? LogitZLossStats(zLoss: 0.0, maxLogit: 0.0) : nil
    }

    private static func buildLayerNorms(
        before: [String: MLXArray],
        after: ModuleParameters
    ) -> LazyLayerNorms? {
        guard !before.isEmpty else { return nil }
        let afterMap = Dictionary(after.flattened(), uniquingKeysWith: { _, latest in latest })
        var names: [String] = []
        var counts: [Int] = []
        var deltas: [MLXArray] = []
        var weights: [MLXArray] = []
        names.reserveCapacity(before.count)
        counts.reserveCapacity(before.count)
        deltas.reserveCapacity(before.count)
        weights.reserveCapacity(before.count)
        for key in before.keys.sorted() {
            guard let beforeParam = before[key],
                  let afterParam = afterMap[key],
                  isMeasurable(beforeParam),
                  isMeasurable(afterParam),
                  beforeParam.shape == afterParam.shape else {
                continue
            }
            let b32 = stopGradient(beforeParam).asType(.float32)
            let a32 = stopGradient(afterParam).asType(.float32)
            let delta = a32 - b32
            names.append(key)
            counts.append(beforeParam.size)
            deltas.append(delta.square().sum())
            weights.append(b32.square().sum())
        }
        guard !names.isEmpty else { return nil }
        return LazyLayerNorms(
            names: names,
            counts: counts,
            deltaSqSums: stacked(deltas),
            weightSqSums: stacked(weights)
        )
    }

    private static func buildGradNormSquared(_ gradients: ModuleParameters?) -> MLXArray? {
        guard let gradients else { return nil }
        let terms: [MLXArray] = gradients.flattened().compactMap { _, array in
            guard isMeasurable(array) else { return nil }
            return stopGradient(array).asType(.float32).square().sum()
        }
        guard !terms.isEmpty else { return nil }
        return stacked(terms).sum()
    }

    private static func materializeLayerRatios(_ lazy: LazyLayerNorms) -> [LayerUpdateRatio] {
        let deltaSq = lazy.deltaSqSums.asArray(Float.self)
        let weightSq = lazy.weightSqSums.asArray(Float.self)
        var ratios: [LayerUpdateRatio] = []
        ratios.reserveCapacity(lazy.names.count)
        for index in lazy.names.indices {
            let dSq = Double(deltaSq[index])
            let wSq = Double(weightSq[index])
            let updateNorm = dSq.squareRoot()
            let weightNorm = wSq.squareRoot()
            guard weightNorm > 1e-8 else { continue }
            let ratio = updateNorm / (weightNorm + TrainingStabilityMath.defaultEps)
            let stepRMS = (dSq / Double(max(1, lazy.counts[index]))).squareRoot()
            ratios.append(LayerUpdateRatio(
                layerName: lazy.names[index],
                updateNorm: updateNorm,
                weightNorm: weightNorm,
                ratio: ratio,
                stepRMS: stepRMS
            ))
        }
        return ratios
    }
}
