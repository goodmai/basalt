import BasaltCore
import Foundation
import MLX
import MLXLMCommon

public enum RecurrentStateInspector: Sendable {
    private struct PendingLayer {
        let layerIndex: Int
        let cacheKind: String
        let tensorCount: Int
        let scalarOffset: Int
    }

    public static func inspect(
        _ caches: [any KVCache],
        baselineNorms: [Int: Double] = [:],
        saturationThreshold: Double = 1e4,
        betaGuardRates: [Int: Double] = [:]
    ) -> RecurrentStateHealthSnapshot {
        var pending: [PendingLayer] = []
        var scalars: [MLXArray] = []
        pending.reserveCapacity(caches.count)

        for (index, cache) in caches.enumerated() {
            guard let cacheKind = classifyRecurrent(cache) else { continue }
            let tensors = cache.state.filter { $0.size > 0 }
            pending.append(
                PendingLayer(
                    layerIndex: index,
                    cacheKind: cacheKind,
                    tensorCount: tensors.count,
                    scalarOffset: scalars.count
                )
            )
            for tensor in tensors {
                appendTensorScalars(tensor, into: &scalars)
            }
        }

        if !scalars.isEmpty {
            eval(scalars)
        }

        let layers = pending.map { item in
            summarizeLayer(
                item,
                scalars: scalars,
                baseline: baselineNorms[item.layerIndex],
                saturationThreshold: saturationThreshold,
                betaGuardRate: betaGuardRates[item.layerIndex]
            )
        }
        return RecurrentStateHealthSnapshot(layerCount: caches.count, layers: layers)
    }

    public static func betaGuardTriggerRate(
        erase: MLXArray,
        key: MLXArray,
        decay: MLXArray
    ) -> Double {
        guard erase.size > 0, key.size > 0, decay.size > 0, erase.ndim >= 1 else { return 0.0 }
        let raw = erase.asType(.float32)
        let guarded = GatedDeltaNet2Ops.applyBetaGuard(
            erase: raw,
            key: key.asType(.float32),
            decay: decay.asType(.float32)
        )
        let eps = MLXArray(Float(1e-7), dtype: .float32)
        let triggered = (MLX.abs(guarded - raw) .> eps).any(axis: -1)
        let rate = triggered.asType(.float32).mean()
        eval(rate)
        let value = Double(rate.item(Float.self))
        return value.isFinite ? Swift.min(Swift.max(value, 0.0), 1.0) : 0.0
    }

    private static func appendTensorScalars(_ tensor: MLXArray, into scalars: inout [MLXArray]) {
        let f32 = tensor.asType(.float32)
        let finiteMask = isFinite(f32)
        let nonFinite = logicalNot(finiteMask).asType(.int32).sum()
        let zero = MLXArray(Float(0), dtype: .float32)
        let one = MLXArray(Float(1), dtype: .float32)
        let clean = MLX.where(finiteMask, f32, zero)
        let maxAbs = MLX.abs(clean).max()
        let safeScale = MLX.where(maxAbs .> zero, maxAbs, one)
        let scaled = clean / safeScale
        let frob = safeScale * (scaled * scaled).sum().sqrt()
        scalars.append(frob)
        scalars.append(maxAbs)
        scalars.append(nonFinite)
    }

    private static func summarizeLayer(
        _ pending: PendingLayer,
        scalars: [MLXArray],
        baseline: Double?,
        saturationThreshold: Double,
        betaGuardRate: Double?
    ) -> LayerStateHealth {
        var sumSq = 0.0
        var maxAbs = 0.0
        var nonFinite = 0
        for offset in 0 ..< pending.tensorCount {
            let base = pending.scalarOffset + offset * 3
            let f = Double(scalars[base].item(Float.self))
            let m = Double(scalars[base + 1].item(Float.self))
            let nf = scalars[base + 2].item(Int.self)
            if f.isFinite { sumSq += f * f }
            if m.isFinite, m > maxAbs { maxAbs = m }
            nonFinite += Swift.max(0, nf)
        }
        let frob = sumSq.squareRoot()
        let growth = baseline.flatMap { $0.isFinite && $0 > 0 ? frob / $0 : nil }
        let thresholdHit = saturationThreshold.isFinite && saturationThreshold > 0
            && (frob >= saturationThreshold || maxAbs >= saturationThreshold)
        return LayerStateHealth(
            layerIndex: pending.layerIndex,
            cacheKind: pending.cacheKind,
            stateTensorCount: pending.tensorCount,
            frobeniusNorm: frob,
            maxAbsEntry: maxAbs,
            nonFiniteCount: nonFinite
        ).withDiagnostics(
            relativeGrowthVsBaseline: growth,
            saturated: nonFinite > 0 || thresholdHit,
            betaGuardTriggerRate: betaGuardRate
        )
    }

    private static func classifyRecurrent(_ cache: any KVCache) -> String? {
        if let kind = knownRecurrentKind(cache) { return kind }
        if isStandardAttentionCache(cache) || isBasaltAttentionCache(cache) { return nil }
        return cache.state.isEmpty ? nil : String(describing: type(of: cache))
    }

    private static func knownRecurrentKind(_ cache: any KVCache) -> String? {
        if cache is MambaCache { return "mamba" }
        if cache is ConvolutionStatesCache { return "conv_states" }
        if cache is ArraysCache { return "arrays" }
        if let xlstm = cache as? XLSTMLayerCache {
            return xlstm.kind == .slstm ? "xlstm_slstm" : "xlstm_mlstm"
        }
        if cache is FastWeightLayerCache { return "fast_weight" }
        if cache is RWKV7LayerCache { return "rwkv7" }
        return nil
    }

    private static func isStandardAttentionCache(_ cache: any KVCache) -> Bool {
        cache is KVCacheSimple
            || cache is RotatingKVCache
            || cache is TurboQuantKVCache
            || cache is any QuantizedKVCacheProtocol
            || cache is any KVCacheAttentionProtocol
    }

    private static func isBasaltAttentionCache(_ cache: any KVCache) -> Bool {
        cache is BatchRotatingKVCache
            || cache is BatchKVCache
            || cache is EvictingKVCache
            || cache is Inkling2WindowKVCache
            || cache is GaugeRoPEKVCache
            || cache is DSALatentKVCache
            || cache is CacheList
    }
}
