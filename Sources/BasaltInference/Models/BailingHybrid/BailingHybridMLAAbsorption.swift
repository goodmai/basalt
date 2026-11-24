import Foundation
import MLX
import MLXNN

package enum BailingHybridMLAAbsorption {
    package static let nvfp4ScaleDivisor: Float = 448 * 6

    package static func canAbsorb(_ kvBProj: Linear) -> Bool {
        guard kvBProj.bias == nil else { return false }
        let kind = type(of: kvBProj)
        return kind == Linear.self || kind == QuantizedLinear.self || kind == SmallMQuantizedLinear.self
    }

    package static func embedQuery(
        _ qNope: MLXArray,
        kvBProj: Linear,
        heads: Int,
        nopeDim: Int,
        vDim: Int
    ) -> MLXArray? {
        guard canAbsorb(kvBProj) else { return nil }
        let rowsPerHead = nopeDim + vDim
        if let quantized = kvBProj as? QuantizedLinear {
            let packed = quantized.weight.reshaped(1, heads, rowsPerHead, -1)[0..., 0..., ..<nopeDim, 0...]
            let scales = quantized.scales.reshaped(1, heads, rowsPerHead, -1)[0..., 0..., ..<nopeDim, 0...]
            let biases = quantized.biases?.reshaped(1, heads, rowsPerHead, -1)[0..., 0..., ..<nopeDim, 0...]
            let projected = quantizedMM(
                qNope,
                packed,
                scales: scales,
                biases: biases,
                transpose: false,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode
            )
            return applyGlobalScale(projected, globalScale: quantized.globalScale)
        }
        let weight = kvBProj.weight.reshaped(1, heads, rowsPerHead, -1)[0..., 0..., ..<nopeDim, 0...]
        return matmul(qNope, weight)
    }

    package static func unembedOutput(
        _ latent: MLXArray,
        kvBProj: Linear,
        heads: Int,
        nopeDim: Int,
        vDim: Int
    ) -> MLXArray? {
        guard canAbsorb(kvBProj) else { return nil }
        let rowsPerHead = nopeDim + vDim
        if let quantized = kvBProj as? QuantizedLinear {
            let packed = quantized.weight.reshaped(1, heads, rowsPerHead, -1)[0..., 0..., nopeDim..., 0...]
            let scales = quantized.scales.reshaped(1, heads, rowsPerHead, -1)[0..., 0..., nopeDim..., 0...]
            let biases = quantized.biases?.reshaped(1, heads, rowsPerHead, -1)[0..., 0..., nopeDim..., 0...]
            let projected = quantizedMM(
                latent,
                packed,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode
            )
            return applyGlobalScale(projected, globalScale: quantized.globalScale)
        }
        let weight = kvBProj.weight.reshaped(1, heads, rowsPerHead, -1)[0..., 0..., nopeDim..., 0...]
        return matmul(latent, weight.swappedAxes(-1, -2))
    }

    private static func applyGlobalScale(_ value: MLXArray, globalScale: MLXArray?) -> MLXArray {
        guard let globalScale else { return value }
        return (value * (globalScale / nvfp4ScaleDivisor)).asType(value.dtype)
    }
}
