import Foundation
import MLX
import BasaltCore

struct LeJEPALossOutput {
    let total: MLXArray
    let invariance: MLXArray
    let sigReg: MLXArray
    let effectiveLambda: Float
}

struct LeJEPALoss: Sendable {
    static let defaultReferenceSamples = 256 * 8

    var baseLambda: Float
    var referenceSamples: Int
    var sliceCount: Int
    var knotCount: Int
    var maxKnot: Float
    var viewsPerScene: Int

    init(
        baseLambda: Float = 0.02,
        referenceSamples: Int = Self.defaultReferenceSamples,
        sliceCount: Int = 1024,
        knotCount: Int = 17,
        maxKnot: Float = 3.0,
        viewsPerScene: Int = GoodViewCollator.totalViewsPerScene
    ) {
        self.baseLambda = baseLambda
        self.referenceSamples = max(1, referenceSamples)
        self.sliceCount = max(1, sliceCount)
        self.knotCount = max(2, knotCount)
        self.maxKnot = max(0.5, maxKnot)
        self.viewsPerScene = max(2, viewsPerScene)
    }

    static func effectiveLambda(
        baseLambda: Float,
        sampleCount: Int,
        referenceSamples: Int = defaultReferenceSamples
    ) -> Float {
        let denom = Float(max(1, sampleCount))
        return baseLambda * (Float(max(1, referenceSamples)) / denom)
    }

    static func unitDirections(projDim: Int, sliceCount: Int, seed: UInt64) -> MLXArray {
        let raw = MLXRandom.normal([projDim, sliceCount], dtype: .float32, key: MLXRandom.key(seed))
        let norms = MLX.maximum(MLX.sqrt(MLX.sum(raw * raw, axis: 0, keepDims: true)), MLXArray(Float(1e-9)))
        return raw / norms
    }

    static func invarianceLoss(_ projected: MLXArray, viewsPerScene: Int = GoodViewCollator.totalViewsPerScene) -> MLXArray {
        let z = projected.asType(.float32)
        let totalRows = z.dim(0)
        let dim = z.dim(1)
        let sceneCount = max(1, totalRows / viewsPerScene)
        let reshaped = z.reshaped(sceneCount, viewsPerScene, dim)
        let global0 = reshaped[0..., 0 ..< 1, 0...]
        let global1 = reshaped[0..., 1 ..< 2, 0...]
        let globalCentroid = MLXArray(Float(0.5)) * (global0 + global1)
        let diff = reshaped - globalCentroid
        return MLX.mean(MLX.sum(diff * diff, axis: -1))
    }

    func sigRegLoss(_ projected: MLXArray, directions: MLXArray? = nil, seed: UInt64 = 42) -> MLXArray {
        let z = projected.asType(.float32)
        let n = z.dim(0)
        let d = z.dim(1)
        let u = (directions ?? Self.unitDirections(projDim: d, sliceCount: sliceCount, seed: seed)).asType(.float32)
        let p = MLX.matmul(z, u)
        let step = maxKnot / Float(knotCount - 1)
        let knots = (0 ..< knotCount).map { Float($0) * step }
        let weights = (0 ..< knotCount).map { idx -> Float in
            let trap: Float = (idx == 0 || idx == knotCount - 1) ? 0.5 : 1.0
            let t = knots[idx]
            return trap * step * exp(-t * t)
        }
        let tArray = MLXArray(knots).reshaped(1, 1, knotCount)
        let wArray = MLXArray(weights).reshaped(1, knotCount)
        let targetCF = MLX.exp(MLXArray(Float(-0.5)) * tArray * tArray).squeezed(axis: 0)
        let angles = p.expandedDimensions(axes: [-1]) * tArray
        let cosMean = MLX.mean(MLX.cos(angles), axis: 0)
        let sinMean = MLX.mean(MLX.sin(angles), axis: 0)
        let realDiff = cosMean - targetCF
        let sqDist = realDiff * realDiff + sinMean * sinMean
        let integratedPerSlice = MLX.sum(sqDist * wArray, axis: -1)
        return MLXArray(Float(max(1, n))) * MLX.mean(integratedPerSlice)
    }

    func callAsFunction(_ projected: MLXArray, directions: MLXArray? = nil, seed: UInt64 = 42) -> LeJEPALossOutput {
        let inv = Self.invarianceLoss(projected, viewsPerScene: viewsPerScene)
        let sig = sigRegLoss(projected, directions: directions, seed: seed)
        let lambdaEff = Self.effectiveLambda(
            baseLambda: baseLambda, sampleCount: projected.dim(0), referenceSamples: referenceSamples)
        let total = inv + MLXArray(lambdaEff) * sig
        return LeJEPALossOutput(total: total, invariance: inv, sigReg: sig, effectiveLambda: lambdaEff)
    }
}
