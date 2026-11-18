import Foundation

public struct CognitiveNullspace: Sendable, Codable, Equatable {
    public let dimension: Int
    public let basis: [[Float]]
    public let singularValues: [Float]

    public var rank: Int { basis.count }

    public init(dimension: Int, basis: [[Float]], singularValues: [Float] = []) throws(BasaltError) {
        guard dimension > 0 else {
            throw .invalidRequestStructure(details: "cognitive nullspace dimension must be positive")
        }
        guard basis.allSatisfy({ $0.count == dimension && $0.allSatisfy(\.isFinite) }) else {
            throw .invalidRequestStructure(details: "every cognitive basis vector must have \(dimension) finite values")
        }
        let orthonormal = Self.orthonormalizeColumns(basis.map { $0.map(Double.init) }, dimension: dimension, maxRank: basis.count, tolerance: 1e-6)
        self.dimension = dimension
        self.basis = orthonormal.columns.map { $0.map(Float.init) }
        self.singularValues = singularValues.isEmpty ? orthonormal.norms.map(Float.init) : Array(singularValues.prefix(orthonormal.columns.count))
    }

    public static func from(
        anchors: [[Float]],
        maxRank: Int? = nil,
        tolerance: Float = 1e-5
    ) throws(BasaltError) -> CognitiveNullspace {
        guard let first = anchors.first, !first.isEmpty else {
            throw .invalidRequestStructure(details: "cognitive nullspace extraction needs at least one non-empty anchor vector")
        }
        let dimension = first.count
        guard anchors.allSatisfy({ $0.count == dimension && $0.allSatisfy(\.isFinite) }) else {
            throw .invalidRequestStructure(details: "all cognitive anchor vectors must share the same finite dimension \(dimension)")
        }
        if let maxRank, maxRank <= 0 {
            throw .invalidRequestStructure(details: "nullspaceMaxRank must be positive")
        }
        let cap = min(maxRank ?? dimension, dimension)
        let extracted = pivotedOrthonormalBasis(anchors.map { $0.map(Double.init) }, dimension: dimension, maxRank: cap, tolerance: Double(max(tolerance, 1e-9)))
        return try CognitiveNullspace(
            dimension: dimension,
            basis: extracted.columns.map { $0.map(Float.init) },
            singularValues: extracted.norms.map(Float.init)
        )
    }

    public func project(_ vector: [Float]) throws(BasaltError) -> [Float] {
        guard vector.count == dimension, vector.allSatisfy(\.isFinite) else {
            throw .invalidRequestStructure(details: "vector to project must have \(dimension) finite values")
        }
        var work = vector.map(Double.init)
        for _ in 0 ..< 2 {
            for col in basis {
                var dot = 0.0
                for i in 0 ..< dimension { dot += Double(col[i]) * work[i] }
                for i in 0 ..< dimension { work[i] -= dot * Double(col[i]) }
            }
        }
        return work.map(Float.init)
    }

    public func maxProjectionMagnitude(_ vector: [Float]) -> Float {
        guard vector.count == dimension, !basis.isEmpty else { return 0 }
        var peak = 0.0
        for col in basis {
            var dot = 0.0
            for i in 0 ..< dimension { dot += Double(col[i]) * Double(vector[i]) }
            peak = max(peak, abs(dot))
        }
        return Float(peak)
    }

    public func l2ProjectionMagnitude(_ vector: [Float]) -> Float {
        guard vector.count == dimension, !basis.isEmpty else { return 0 }
        var sumSq = 0.0
        for col in basis {
            var dot = 0.0
            for i in 0 ..< dimension { dot += Double(col[i]) * Double(vector[i]) }
            sumSq += dot * dot
        }
        return Float(sumSq.squareRoot())
    }

    public func orthogonalityDefect() -> Float {
        guard !basis.isEmpty else { return 0 }
        var sumSq = 0.0
        for i in basis.indices {
            for j in basis.indices {
                var dot = 0.0
                for k in 0 ..< dimension { dot += Double(basis[i][k]) * Double(basis[j][k]) }
                let diff = dot - (i == j ? 1.0 : 0.0)
                sumSq += diff * diff
            }
        }
        return Float(sumSq.squareRoot())
    }
}

extension CognitiveNullspace {
    static func orthonormalizeColumns(
        _ candidates: [[Double]],
        dimension: Int,
        maxRank: Int,
        tolerance: Double
    ) -> (columns: [[Double]], norms: [Double]) {
        var accepted: [[Double]] = []
        var norms: [Double] = []
        var peak = 0.0
        for candidate in candidates {
            guard accepted.count < maxRank else { break }
            var residual = candidate
            for _ in 0 ..< 2 {
                for u in accepted {
                    let proj = dot(u, residual)
                    subtractScaled(u, by: proj, from: &residual)
                }
            }
            let norm = dot(residual, residual).squareRoot()
            peak = max(peak, norm)
            guard norm > max(tolerance * max(peak, 1.0), 1e-12) else { continue }
            let scale = 1.0 / norm
            accepted.append(residual.map { $0 * scale })
            norms.append(norm)
        }
        return (accepted, norms)
    }

    static func pivotedOrthonormalBasis(
        _ rows: [[Double]],
        dimension: Int,
        maxRank: Int,
        tolerance: Double
    ) -> (columns: [[Double]], norms: [Double]) {
        var residuals = rows
        var used = Array(repeating: false, count: rows.count)
        var accepted: [[Double]] = []
        var norms: [Double] = []
        var peak = 0.0
        while accepted.count < maxRank {
            var bestIndex = -1
            var bestNormSq = 0.0
            for i in residuals.indices where !used[i] {
                let normSq = dot(residuals[i], residuals[i])
                if normSq > bestNormSq {
                    bestNormSq = normSq
                    bestIndex = i
                }
            }
            guard bestIndex >= 0 else { break }
            used[bestIndex] = true
            var v = residuals[bestIndex]
            for u in accepted { subtractScaled(u, by: dot(u, v), from: &v) }
            let norm = dot(v, v).squareRoot()
            if accepted.isEmpty { peak = norm }
            guard peak > 1e-12, norm > tolerance * peak else { break }
            let u = v.map { $0 / norm }
            accepted.append(u)
            norms.append(norm)
            for i in residuals.indices where !used[i] {
                subtractScaled(u, by: dot(u, residuals[i]), from: &residuals[i])
            }
        }
        return (accepted, norms)
    }

    private static func dot(_ a: [Double], _ b: [Double]) -> Double {
        var sum = 0.0
        for i in a.indices { sum += a[i] * b[i] }
        return sum
    }

    private static func subtractScaled(_ u: [Double], by scale: Double, from target: inout [Double]) {
        guard scale != 0 else { return }
        for i in target.indices { target[i] -= scale * u[i] }
    }
}
