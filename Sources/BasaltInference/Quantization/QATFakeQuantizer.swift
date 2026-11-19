import BasaltCore
import Foundation
import MLX

public enum QATFakeQuantizer {
    public static func fakeQuantize(
        _ weight: MLXArray,
        bits: Int = 4,
        groupSize: Int = 64
    ) -> MLXArray {
        let clampedBits = [2, 3, 4, 8].contains(bits) ? bits : 4
        let cols = weight.dim(-1)
        let effectiveGroup = (groupSize > 0 && cols.isMultiple(of: groupSize)) ? groupSize : cols
        let wf = weight.asType(.float32)
        let origShape = wf.shape
        let groups = cols / max(1, effectiveGroup)
        let reshaped = wf.reshaped([-1, groups, effectiveGroup])
        let minVal = reshaped.min(axis: -1, keepDims: true)
        let maxVal = reshaped.max(axis: -1, keepDims: true)
        let levels = Float((1 << clampedBits) - 1)
        let span = maximum(maxVal - minVal, MLXArray(Float(1e-7)))
        let scale = span / MLXArray(levels)
        let codes = clip(round((reshaped - minVal) / scale), min: Float(0.0), max: levels)
        let dequant = (codes * scale + minVal).reshaped(origShape)
        let ste = wf + stopGradient(dequant - wf)
        return ste.asType(weight.dtype)
    }

    public static func quantizationSQNRdB(original: MLXArray, quantized: MLXArray) -> Float {
        let orig = original.asType(.float32)
        let diff = orig - quantized.asType(.float32)
        let sig = mean(orig * orig)
        let noise = mean(diff * diff)
        eval(sig, noise)
        let s = sig.item(Float.self)
        let n = max(noise.item(Float.self), Float(1e-12))
        guard s.isFinite, s > 0 else { return 0.0 }
        return 10.0 * Foundation::log10(s / n)
    }

    public static func qatStep(
        weight: MLXArray,
        input: MLXArray,
        target: MLXArray,
        bits: Int = 4,
        groupSize: Int = 64,
        learningRate: Float = 1e-3
    ) -> (updatedWeight: MLXArray, loss: Float) {
        let xF = input.asType(.float32)
        let yF = target.asType(.float32)
        let lossAndGrad = valueAndGrad { (arrays: [MLXArray]) -> [MLXArray] in
            let wHat = fakeQuantize(arrays[0], bits: bits, groupSize: groupSize)
            let pred = matmul(xF, wHat.transposed())
            let err = pred - yF
            return [mean(err * err)]
        }
        let (losses, grads) = lossAndGrad([weight.asType(.float32)])
        let updated = (weight.asType(.float32) - MLXArray(learningRate) * grads[0]).asType(weight.dtype)
        eval(losses[0], updated)
        return (updated, losses[0].item(Float.self))
    }
}
