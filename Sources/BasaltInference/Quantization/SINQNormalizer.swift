import BasaltCore
import Foundation
import MLX
import MLXLMCommon

package enum SINQRecipe: String, Sendable, CaseIterable, Codable {
    case rtn
    case sinq
    case asinq = "a-sinq"
}

package struct SINQConfig: Sendable, Equatable {
    package var iterations: Int
    package var damping: Float
    package var epsilon: Float
    package var minScale: Float
    package var maxScale: Float
    package var activationAlpha: Float

    package init(
        iterations: Int = 16,
        damping: Float = 0.65,
        epsilon: Float = 1e-8,
        minScale: Float = 0.125,
        maxScale: Float = 8.0,
        activationAlpha: Float = 0.5
    ) {
        self.iterations = max(1, iterations)
        self.damping = min(max(0.05, damping), 1.0)
        self.epsilon = max(1e-12, epsilon)
        self.minScale = max(1e-4, minScale)
        self.maxScale = max(self.minScale, maxScale)
        self.activationAlpha = min(max(0.0, activationAlpha), 1.0)
    }
}

package struct SINQFactors {
    package let rowScale: MLXArray
    package let colScale: MLXArray
    package let imbalanceBefore: Float
    package let imbalanceAfter: Float
}

package struct SINQSummary: Sendable, Equatable {
    package var absorbedGroups: Int = 0
    package var normalizedTensors: Int = 0
    package var routerProjectionsAdjusted: Int = 0
    package var meanImbalanceBefore: Float = 1.0
    package var meanImbalanceAfter: Float = 1.0
    package var calibrationTokens: Int = 0
    package var wallSeconds: Double = 0
}

package enum SINQNormalizer {
    package enum QuantFormat: Sendable, Equatable {
        case affine(bits: Int, groupSize: Int)
        case nf4(groupSize: Int)
        case mxfp4
    }

    package static func matrixImbalance(_ weight2D: MLXArray, epsilon: Float = 1e-8) -> Float {
        let w = weight2D.asType(.float32)
        guard w.ndim == 2, w.dim(0) > 0, w.dim(1) > 0 else { return 1.0 }
        let rowStd = sqrt(variance(w, axis: 1) + MLXArray(epsilon * epsilon)) + MLXArray(epsilon)
        let hi = rowStd.max()
        let lo = maximum(rowStd.min(), MLXArray(epsilon))
        let ratio = hi / lo
        eval(ratio)
        let val = ratio.item(Float.self)
        return val.isFinite ? val : 1.0
    }

    package static func solve(
        _ weight2D: MLXArray,
        activationStd: MLXArray? = nil,
        config: SINQConfig = SINQConfig()
    ) -> SINQFactors {
        let w = weight2D.asType(.float32)
        let rows = w.dim(0)
        let cols = w.dim(1)
        guard w.ndim == 2, rows > 0, cols > 0 else {
            return SINQFactors(
                rowScale: MLXArray.ones([max(1, rows)], dtype: .float32),
                colScale: MLXArray.ones([max(1, cols)], dtype: .float32),
                imbalanceBefore: 1.0,
                imbalanceAfter: 1.0)
        }
        let eps = MLXArray(config.epsilon)
        let epsSq = MLXArray(config.epsilon * config.epsilon)
        let actFactor = activationFactor(activationStd, cols: cols, config: config)
        let work = actFactor.map { w * $0.reshaped(1, cols) } ?? w
        let before = matrixImbalance(work, epsilon: config.epsilon)

        let logMin = MLXArray(Foundation.log(config.minScale))
        let logMax = MLXArray(Foundation.log(config.maxScale))
        let alpha = MLXArray(config.damping)
        var logS = MLXArray.zeros([rows, 1], dtype: .float32)
        var logU = MLXArray.zeros([1, cols], dtype: .float32)

        for _ in 0 ..< config.iterations {
            let scaled = work * exp(-logS - logU)
            let rStd = sqrt(variance(scaled, axis: 1, keepDims: true) + epsSq) + eps
            let cStd = sqrt(variance(scaled, axis: 0, keepDims: true) + epsSq) + eps
            let logR = log(rStd)
            let logC = log(cStd)
            logS = clip(logS + alpha * (logR - mean(logR)), min: logMin, max: logMax)
            logU = clip(logU + alpha * (logC - mean(logC)), min: logMin, max: logMax)
        }

        let sVec = exp(logS).reshaped(rows)
        let uVec = exp(logU).reshaped(cols)
        let tVec: MLXArray
        if let actFactor {
            tVec = clip(uVec / actFactor, min: MLXArray(config.minScale), max: MLXArray(config.maxScale))
        } else {
            tVec = uVec
        }
        let normalized = w / (sVec.reshaped(rows, 1) * tVec.reshaped(1, cols))
        eval(sVec, tVec, normalized)
        let after = matrixImbalance(normalized, epsilon: config.epsilon)
        return SINQFactors(rowScale: sVec, colScale: tVec, imbalanceBefore: before, imbalanceAfter: after)
    }

    private static func activationFactor(
        _ activationStd: MLXArray?, cols: Int, config: SINQConfig
    ) -> MLXArray? {
        guard let activationStd, activationStd.size == cols, config.activationAlpha > 0 else { return nil }
        let flat = abs(activationStd.asType(.float32).reshaped(cols))
        let avg = maximum(mean(flat), MLXArray(config.epsilon))
        let rel = clip(flat / avg, min: MLXArray(config.minScale), max: MLXArray(config.maxScale))
        return pow(rel, MLXArray(config.activationAlpha))
    }

    package static func quantizeNormalized(
        base: String,
        weight: MLXArray,
        rowScale: MLXArray,
        colScale: MLXArray,
        format: QuantFormat
    ) -> [String: MLXArray] {
        let origDType = weight.dtype.isFloatingPoint ? weight.dtype : .float16
        let sShape = Array(weight.shape.dropLast()) + [1]
        let tShape = Array(repeating: 1, count: weight.ndim - 1) + [weight.dim(-1)]
        let sView = rowScale.asType(.float32).reshaped(sShape)
        let tView = colScale.asType(.float32).reshaped(tShape)

        switch format {
        case .affine(let bits, let groupSize):
            let wNorm = (weight.asType(.float32) / (sView * tView)).asType(origDType)
            let (wq, scales, biases) = MLX.quantized(wNorm, groupSize: groupSize, bits: bits, mode: .affine)
            let foldedScales = (scales.asType(.float32) * sView).asType(scales.dtype)
            var out: [String: MLXArray] = [base + ".weight": wq, base + ".scales": foldedScales]
            if let biases {
                out[base + ".biases"] = (biases.asType(.float32) * sView).asType(biases.dtype)
            }
            eval(Array(out.values))
            return out

        case .nf4(let groupSize):
            let wNorm = weight.asType(.float32) / (sView * tView)
            let (codes, scales) = NF4Metal.quantized(wNorm, groupSize: groupSize)
            let foldedScales = (scales.asType(.float32) * sView).asType(.float16)
            eval(codes, foldedScales)
            return [base + ".weight": codes, base + ".scales": foldedScales]

        case .mxfp4:
            let shift = round(clip(log2(sView), min: MLXArray(Float(-4.0)), max: MLXArray(Float(4.0))))
            let dyadicS = pow(MLXArray(Float(2.0)), shift)
            let wNorm = (weight.asType(.float32) / (dyadicS * tView)).asType(origDType)
            let (wq, scales, _) = MLX.quantized(wNorm, groupSize: 32, bits: 4, mode: .mxfp4)
            let shiftedInt = clip(scales.asType(.int32) + shift.asType(.int32), min: 1, max: 254)
            let foldedScales = MLX.where(scales .> UInt8(0), shiftedInt, MLXArray(Int32(0))).asType(.uint8)
            eval(wq, foldedScales)
            return [base + ".weight": wq, base + ".scales": foldedScales]
        }
    }

    package static func scaleNormWeight(
        _ normWeight: MLXArray, colScale: MLXArray, addsUnitOffset: Bool
    ) -> MLXArray {
        let origDType = normWeight.dtype
        let w = normWeight.asType(.float32)
        let t = colScale.asType(.float32).reshaped(normWeight.shape)
        let updated = addsUnitOffset ? ((w + MLXArray(Float(1.0))) * t - MLXArray(Float(1.0))) : (w * t)
        let out = updated.asType(origDType)
        eval(out)
        return out
    }

    package static func compensateDenseConsumer(_ weight: MLXArray, colScale: MLXArray) -> MLXArray {
        let origDType = weight.dtype
        let tShape = Array(repeating: 1, count: max(0, weight.ndim - 1)) + [weight.dim(-1)]
        let compensated = (weight.asType(.float32) / colScale.asType(.float32).reshaped(tShape)).asType(origDType)
        eval(compensated)
        return compensated
    }
}
