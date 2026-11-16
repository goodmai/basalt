import Foundation
import MLX
import BasaltCore

package struct TeaCacheState: Sendable {
    package static let defaultCoefficients: [Float] = [
        4.98651651e+02,
        -2.83781631e+02,
        5.58554382e+01,
        -3.82021401e+00,
        2.64230861e-01,
    ]

    package let totalSteps: Int
    package let threshold: Double
    package let coefficients: [Float]
    package private(set) var stepIndex: Int = 0
    package private(set) var accumulatedDelta: Double = 0
    package private(set) var skippedSteps: Int = 0
    package private(set) var executedSteps: Int = 0
    package private(set) var skippedStepIndices: [Int] = []
    package private(set) var executedStepIndices: [Int] = []
    package private(set) var cachedResidual: MaterializedArray?
    package private(set) var previousModulatedInput: MaterializedArray?

    package init(
        totalSteps: Int,
        threshold: Double,
        coefficients: [Float] = TeaCacheState.defaultCoefficients
    ) {
        self.totalSteps = max(1, totalSteps)
        self.threshold = max(0.0, threshold)
        self.coefficients = coefficients.isEmpty ? Self.defaultCoefficients : coefficients
    }

    package var skippingEnabled: Bool {
        threshold > 0.0 && totalSteps > 4
    }

    package mutating func resolveVelocity(
        step: Int,
        modulatedInput: MLXArray,
        compute: () -> MLXArray
    ) -> MLXArray {
        stepIndex = step
        let isBoundary = step <= 0 || step >= (totalSteps - 1)
        if skippingEnabled && !isBoundary,
           let prevMod = previousModulatedInput,
           let cached = cachedResidual {
            let relL1 = Self.relativeL1Distance(current: modulatedInput, previous: prevMod)
            accumulatedDelta += Self.rescaledVelocityDelta(relL1: relL1, coefficients: coefficients)
            if accumulatedDelta < threshold {
                skippedSteps += 1
                skippedStepIndices.append(step)
                return cached
            }
        }
        accumulatedDelta = 0.0
        let fresh = compute()
        previousModulatedInput = modulatedInput.materialized()
        cachedResidual = fresh.materialized()
        executedSteps += 1
        executedStepIndices.append(step)
        return fresh
    }

    package static func relativeL1Distance(current: MLXArray, previous: MLXArray) -> Float {
        let cur = current.asType(.float32)
        let prev = previous.asType(.float32)
        let diffMean = mean(abs(cur - prev))
        let baseMean = maximum(mean(abs(prev)), MLXArray(Float(1e-6)))
        let ratio = diffMean / baseMean
        eval(ratio)
        let value = ratio.item(Float.self)
        return value.isFinite ? max(0.0, value) : 1.0
    }

    package static func rescaledVelocityDelta(relL1: Float, coefficients: [Float] = defaultCoefficients) -> Double {
        guard relL1.isFinite, relL1 > 0 else { return 0.0 }
        let clamped = min(relL1, 0.25)
        var poly: Float = 0.0
        for coeff in coefficients {
            poly = poly * clamped + coeff
        }
        let positivePoly = max(Float(0.01), poly)
        let ramp = min(Float(1.0), relL1 / 0.12)
        let overflow = max(Float(0.0), relL1 - 0.25)
        return Double(positivePoly * ramp + overflow)
    }
}

package enum TinyAutoencoderPreview {
    package static func shouldEmitPreview(step: Int, totalSteps: Int, previewEverySteps: Int) -> Bool {
        guard previewEverySteps > 0, totalSteps > 1 else { return false }
        let completedStep = step + 1
        guard completedStep < totalSteps else { return false }
        return completedStep.isMultiple(of: previewEverySteps)
    }

    package static func makePartialFrame(
        state: MLXArray,
        step: Int,
        request: ImageGenerationRequest,
        isLatent: Bool,
        includeAlpha: Bool
    ) throws(BasaltError) -> ImagePartialFrame {
        let outChannels = includeAlpha && request.outputFormat != .jpeg ? 4 : 3
        let projected: MLXArray
        if isLatent {
            projected = projectLatentTAEF(
                state,
                targetHeight: request.height,
                targetWidth: request.width,
                outChannels: outChannels
            )
        } else {
            projected = projectPixelState(
                state,
                targetHeight: request.height,
                targetWidth: request.width,
                outChannels: outChannels
            )
        }
        let encoded = try ImagePNGCodec.encode(
            projected,
            format: request.outputFormat,
            forceOpaque: outChannels == 3
        )
        return ImagePartialFrame(
            step: step + 1,
            totalSteps: request.steps,
            imageData: encoded.data,
            format: encoded.format,
            dimensions: ImageDimensions(width: request.width, height: request.height)
        )
    }

    private static func projectLatentTAEF(
        _ latent: MLXArray,
        targetHeight: Int,
        targetWidth: Int,
        outChannels: Int
    ) -> MLXArray {
        let nhwc = latent.ndim == 3 ? latent[.newAxis] : latent
        let inChannels = nhwc.dim(3)
        let basis = taefProjectionBasis(inChannels: inChannels, outChannels: outChannels)
        let weight = MLXArray(basis, [inChannels, outChannels])
        let lowRes = sigmoid(matmul(nhwc.asType(.float32), weight))
        return upsampleBilinearGrid(lowRes[0], height: targetHeight, width: targetWidth)
    }

    private static func projectPixelState(
        _ pixels: MLXArray,
        targetHeight: Int,
        targetWidth: Int,
        outChannels: Int
    ) -> MLXArray {
        let hwc = (pixels.ndim == 4 ? pixels[0] : pixels).asType(.float32)
        let avail = hwc.dim(2)
        let sliced: MLXArray
        if avail >= outChannels {
            sliced = sigmoid(hwc[0..., 0..., ..<outChannels])
        } else {
            let alphaPad = MLXArray.ones([hwc.dim(0), hwc.dim(1), outChannels - avail], dtype: .float32)
            sliced = concatenated([sigmoid(hwc), alphaPad], axis: 2)
        }
        if sliced.dim(0) == targetHeight && sliced.dim(1) == targetWidth {
            return sliced
        }
        return upsampleBilinearGrid(sliced, height: targetHeight, width: targetWidth)
    }

    private static func taefProjectionBasis(inChannels: Int, outChannels: Int) -> [Float] {
        var weights = [Float](repeating: 0, count: inChannels * outChannels)
        let scale = 1.0 / Float(max(1, inChannels)).squareRoot()
        for cIn in 0 ..< inChannels {
            for cOut in 0 ..< outChannels {
                let angle = Double((cIn + 1) * (cOut + 2)) * 0.67
                weights[cIn * outChannels + cOut] = Float(cos(angle)) * scale
            }
        }
        return weights
    }

    package static func upsampleBilinearGrid(_ hwc: MLXArray, height: Int, width: Int) -> MLXArray {
        let inH = hwc.dim(0)
        let inW = hwc.dim(1)
        if inH == height && inW == width { return hwc }
        let rowIdx = clip(
            floor((MLXArray(0 ..< height).asType(.float32) + 0.5) * Float(inH) / Float(height)).asType(.int32),
            min: 0,
            max: max(0, inH - 1)
        )
        let colIdx = clip(
            floor((MLXArray(0 ..< width).asType(.float32) + 0.5) * Float(inW) / Float(width)).asType(.int32),
            min: 0,
            max: max(0, inW - 1)
        )
        return hwc.take(rowIdx, axis: 0).take(colIdx, axis: 1)
    }
}
