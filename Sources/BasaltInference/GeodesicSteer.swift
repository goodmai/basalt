import Foundation
import MLX
import BasaltCore

package struct ContextGateTensor {
    package let unitVector: MLXArray
    package let threshold: Float
    package let sharpness: Float

    package init(_ gate: ContextSteerGate, fallbackDirection: MLXArray) {
        let raw = gate.contextVector.map { MLXArray($0).asType(.float32) } ?? fallbackDirection.asType(.float32)
        let norm = MLX.sqrt(MLX.square(raw).sum())
        self.unitVector = which(norm .>= Float(1e-12), raw / MLX.maximum(norm, MLXArray(Float(1e-12))), MLXArray.zeros(like: raw))
        self.threshold = gate.threshold
        self.sharpness = gate.sharpness
    }

    package func scale(_ baseScale: Float, hidden: MLXArray) -> MLXArray {
        let h = hidden.asType(.float32)
        let eps = MLXArray(Float(1e-12))
        let hNorm = MLX.sqrt(MLX.square(h).sum(axis: -1, keepDims: true))
        let vNormSq = MLX.square(unitVector).sum()
        let hHat = h / MLX.maximum(hNorm, eps)
        let cosine = (hHat * unitVector).sum(axis: -1, keepDims: true)
        let weight = MLX.sigmoid(sharpness * (cosine - threshold))
        return which((hNorm .>= Float(1e-12)) .&& (vNormSq .>= Float(1e-12)), baseScale * weight, MLXArray(Float(0)))
    }
}

package struct SteerStep {
    package let direction: MLXArray
    package let scale: Float
    package let geometry: SteerGeometry
    package let contextGate: ContextGateTensor?

    package init(direction: MLXArray, scale: Float, geometry: SteerGeometry, contextGate: ContextSteerGate? = nil) {
        let f32 = direction.asType(.float32)
        self.direction = f32
        self.scale = scale
        self.geometry = geometry
        self.contextGate = contextGate.map { ContextGateTensor($0, fallbackDirection: f32) }
    }
}

package enum GeodesicSteer {
    package static func apply(hidden: MLXArray, step: SteerStep) -> MLXArray {
        guard step.scale != 0 else { return hidden }
        guard let gate = step.contextGate else {
            return apply(hidden: hidden, direction: step.direction, scale: step.scale, geometry: step.geometry)
        }
        let alpha = gate.scale(step.scale, hidden: hidden)
        switch step.geometry {
        case .additive:
            return hidden + (alpha * step.direction).asType(hidden.dtype)
        case .geodesic:
            return applyGeodesicAngles(hidden: hidden, direction: step.direction, angles: alpha)
        }
    }

    package static func apply(
        hidden: MLXArray,
        direction: MLXArray,
        scale: Float,
        geometry: SteerGeometry
    ) -> MLXArray {
        guard scale != 0 else { return hidden }
        switch geometry {
        case .additive:
            let delta = scale * direction.asType(.float32)
            return hidden + delta.asType(hidden.dtype)
        case .geodesic:
            return applyGeodesic(hidden: hidden, direction: direction, angle: scale)
        }
    }

    private static func applyGeodesic(hidden: MLXArray, direction: MLXArray, angle: Float) -> MLXArray {
        guard let frame = tangentFrame(hidden: hidden, direction: direction) else { return hidden }
        let rotated = Foundation.cos(angle) * frame.hHat + Foundation.sin(angle) * frame.unitPerp
        return finishGeodesic(rotated: rotated, frame: frame, dtype: hidden.dtype)
    }

    private static func applyGeodesicAngles(hidden: MLXArray, direction: MLXArray, angles: MLXArray) -> MLXArray {
        guard let frame = tangentFrame(hidden: hidden, direction: direction) else { return hidden }
        let rotated = MLX.cos(angles) * frame.hHat + MLX.sin(angles) * frame.unitPerp
        return finishGeodesic(rotated: rotated, frame: frame, dtype: hidden.dtype)
    }

    private struct TangentFrame {
        let h: MLXArray
        let hNorm: MLXArray
        let uNorm: MLXArray
        let hHat: MLXArray
        let unitPerp: MLXArray
    }

    private static func tangentFrame(
        hidden: MLXArray,
        direction: MLXArray
    ) -> TangentFrame? {
        let dim = hidden.dim(-1)
        guard dim >= 2 else { return nil }
        let h = hidden.asType(.float32)
        let u = direction.asType(.float32)
        let eps = MLXArray(Float(1e-12))
        let hNorm = MLX.sqrt(MLX.square(h).sum(axis: -1, keepDims: true))
        let uNorm = MLX.sqrt(MLX.square(u).sum(axis: -1, keepDims: true))
        let hHat = h / MLX.maximum(hNorm, eps)
        let uHat = u / MLX.maximum(uNorm, eps)
        let rawPerp = uHat - (hHat * uHat).sum(axis: -1, keepDims: true) * hHat
        let perpNorm = MLX.sqrt(MLX.square(rawPerp).sum(axis: -1, keepDims: true))
        let unitPerp = which(
            perpNorm .>= Float(1e-6),
            rawPerp / MLX.maximum(perpNorm, eps),
            orthogonalFallback(hHat: hHat, dim: dim)
        )
        return TangentFrame(h: h, hNorm: hNorm, uNorm: uNorm, hHat: hHat, unitPerp: unitPerp)
    }

    private static func finishGeodesic(
        rotated: MLXArray,
        frame: TangentFrame,
        dtype: DType
    ) -> MLXArray {
        let eps = MLXArray(Float(1e-12))
        let rotatedNorm = MLX.sqrt(MLX.square(rotated).sum(axis: -1, keepDims: true))
        let exactUnit = rotated / MLX.maximum(rotatedNorm, eps)
        let steered = which((frame.hNorm .>= Float(1e-12)) .&& (frame.uNorm .>= Float(1e-12)), frame.hNorm * exactUnit, frame.h)
        return steered.asType(dtype)
    }

    private static func orthogonalFallback(hHat: MLXArray, dim: Int) -> MLXArray {
        let eps = MLXArray(Float(1e-12))
        let e0 = MLXArray((0 ..< dim).map { $0 == 0 ? Float(1) : Float(0) })
        let e1 = MLXArray((0 ..< dim).map { $0 == 1 ? Float(1) : Float(0) })
        let h0 = (hHat * e0).sum(axis: -1, keepDims: true)
        let h1 = (hHat * e1).sum(axis: -1, keepDims: true)
        let basis = which(MLX.abs(h0) .<= MLX.abs(h1), e0, e1)
        let raw = basis - (basis * hHat).sum(axis: -1, keepDims: true) * hHat
        let norm = MLX.sqrt(MLX.square(raw).sum(axis: -1, keepDims: true))
        return raw / MLX.maximum(norm, eps)
    }
}
