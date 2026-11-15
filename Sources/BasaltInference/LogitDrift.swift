import Foundation
import MLX

package enum LogitDrift {
    package static func normalizedLogProbabilities(_ logits: MLXArray) -> MLXArray {
        let raw = logits.asType(.float32)
        let clean = which(
            MLX.isNaN(raw),
            MLXArray(-Float.infinity),
            which(raw .== Float.infinity, MLXArray(Float(88.0)), raw)
        )
        let lse = MLX.logSumExp(clean, axis: -1, keepDims: true)
        let vocab = max(raw.dim(-1), 1)
        let uniform = MLXArray(-Foundation.log(Float(vocab)))
        return which(MLX.isFinite(lse), clean - lse, uniform)
    }

    package static func fisherRao(logP: MLXArray, logQ: MLXArray) -> MLXArray {
        let safeLP = which(MLX.isNaN(logP), MLXArray(-Float.infinity), logP.asType(.float32))
        let safeLQ = which(MLX.isNaN(logQ), MLXArray(-Float.infinity), logQ.asType(.float32))
        let a = MLX.clip(MLX.exp(Float(0.5) * safeLP), min: Float(0), max: Float(1))
        let b = MLX.clip(MLX.exp(Float(0.5) * safeLQ), min: Float(0), max: Float(1))
        let diffNorm = MLX.sqrt(MLX.square(a - b).sum(axis: -1))
        let sumNorm = MLX.sqrt(MLX.square(a + b).sum(axis: -1))
        let raw = Float(4) * MLX.atan2(diffNorm, MLX.maximum(sumNorm, MLXArray(Float.leastNormalMagnitude)))
        let clamped = MLX.clip(raw, min: Float(0), max: Float.pi)
        return which(MLX.isFinite(clamped), clamped, MLXArray(Float(0)))
    }

    package static func fisherRaoFromLogits(_ pLogits: MLXArray, _ qLogits: MLXArray) -> MLXArray {
        fisherRao(
            logP: normalizedLogProbabilities(pLogits),
            logQ: normalizedLogProbabilities(qLogits)
        )
    }

    package static func tensors(
        logP: MLXArray,
        logQ: MLXArray
    ) -> (klDivergence: MLXArray, meanFisherRao: MLXArray, maxFisherRao: MLXArray) {
        let lp = logP.asType(.float32)
        let lq = logQ.asType(.float32)
        let p = MLX.exp(lp)
        let delta = which(p .> 0, lp - lq, Float(0))
        let kl = which(p .> 0, p * delta, Float(0)).sum(axis: -1).mean()
        let distances = fisherRao(logP: lp, logQ: lq)
        return (kl, distances.mean(), distances.max())
    }

    package static func metrics(
        logP: MLXArray,
        logQ: MLXArray
    ) -> (klDivergence: Float, meanFisherRao: Float, maxFisherRao: Float) {
        let (kl, meanFR, maxFR) = tensors(logP: logP, logQ: logQ)
        eval(kl, meanFR, maxFR)
        return (kl.item(Float.self), meanFR.item(Float.self), maxFR.item(Float.self))
    }
}
