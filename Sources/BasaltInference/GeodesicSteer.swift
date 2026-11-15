import Foundation
import MLX
import BasaltCore

package struct SteerStep {
    package let direction: MLXArray
    package let scale: Float
    package let geometry: SteerGeometry

    package init(direction: MLXArray, scale: Float, geometry: SteerGeometry) {
        self.direction = direction.asType(.float32)
        self.scale = scale
        self.geometry = geometry
    }
}

package enum GeodesicSteer {
    package static func apply(hidden: MLXArray, step: SteerStep) -> MLXArray {
        apply(hidden: hidden, direction: step.direction, scale: step.scale, geometry: step.geometry)
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
        let dim = hidden.dim(-1)
        guard dim >= 2 else { return hidden }
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
        let rotated = Foundation.cos(angle) * hHat + Foundation.sin(angle) * unitPerp
        let rotatedNorm = MLX.sqrt(MLX.square(rotated).sum(axis: -1, keepDims: true))
        let exactUnit = rotated / MLX.maximum(rotatedNorm, eps)
        let steered = which((hNorm .>= Float(1e-12)) .&& (uNorm .>= Float(1e-12)), hNorm * exactUnit, h)
        return steered.asType(hidden.dtype)
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
