import Foundation

public struct DiLoCoOuterConfig: Sendable, Codable, Equatable {
    public var innerSteps: Int
    public var outerLearningRate: Float
    public var nesterovMomentum: Float

    public init(
        innerSteps: Int = 32,
        outerLearningRate: Float = 0.7,
        nesterovMomentum: Float = 0.9
    ) {
        self.innerSteps = max(1, min(512, innerSteps))
        self.outerLearningRate = outerLearningRate
        self.nesterovMomentum = min(0.999, max(0.0, nesterovMomentum))
    }
}

public struct DiLoCoOuterOptimizer: Sendable {
    public let config: DiLoCoOuterConfig
    public private(set) var outerStepCount: Int = 0
    private var velocity: [String: [Float]] = [:]

    public init(config: DiLoCoOuterConfig = DiLoCoOuterConfig()) {
        self.config = config
    }

    public mutating func step(
        globalParameters: [String: [Float]],
        workerPseudoGradients: [(tokenCount: Int, delta: [String: [Float]])]
    ) -> [String: [Float]] {
        let totalTokens = max(1, workerPseudoGradients.reduce(0) { $0 + max(0, $1.tokenCount) })
        outerStepCount += 1
        let mu = config.nesterovMomentum
        let lr = config.outerLearningRate
        var updatedGlobal: [String: [Float]] = [:]
        for (key, theta) in globalParameters {
            let count = theta.count
            var avgDelta = [Float](repeating: 0, count: count)
            for worker in workerPseudoGradients where worker.tokenCount > 0 {
                guard let d = worker.delta[key], d.count == count else { continue }
                let w = Float(worker.tokenCount) / Float(totalTokens)
                for i in 0 ..< count {
                    avgDelta[i] += w * d[i]
                }
            }
            var v = velocity[key] ?? [Float](repeating: 0, count: count)
            if v.count != count { v = [Float](repeating: 0, count: count) }
            var nextTheta = [Float](repeating: 0, count: count)
            for i in 0 ..< count {
                let nextV = mu * v[i] + avgDelta[i]
                v[i] = nextV
                let nesterovGrad = mu * nextV + avgDelta[i]
                nextTheta[i] = theta[i] - lr * nesterovGrad
            }
            velocity[key] = v
            updatedGlobal[key] = nextTheta
        }
        return updatedGlobal
    }
}

public struct QJLSketchPacket: Sendable, Codable, Equatable {
    public let dimension: Int
    public let sketchRows: Int
    public let l2Norm: Float
    public let seed: UInt64
    public let packedSigns: [UInt64]

    public var wireBytes: Int {
        16 + packedSigns.count * 8
    }
}

public enum QJLSketchCompressor {
    public static func encode(
        _ vector: [Float],
        sketchRows: Int? = nil,
        seed: UInt64
    ) -> QJLSketchPacket {
        let d = vector.count
        let m = max(8, sketchRows ?? d)
        var normSq: Double = 0
        for v in vector { normSq += Double(v) * Double(v) }
        let norm = Float(normSq.squareRoot())
        guard d > 0, norm > 1e-12 else {
            return QJLSketchPacket(dimension: d, sketchRows: m, l2Norm: 0, seed: seed, packedSigns: [])
        }
        let words = (m + 63) / 64
        var signs = [UInt64](repeating: 0, count: words)
        for row in 0 ..< m {
            var dot: Float = 0
            for col in 0 ..< d {
                dot += gaussianEntry(row: row, col: col, seed: seed) * vector[col]
            }
            if dot >= 0 {
                signs[row >> 6] |= (UInt64(1) << UInt64(row & 63))
            }
        }
        return QJLSketchPacket(dimension: d, sketchRows: m, l2Norm: norm, seed: seed, packedSigns: signs)
    }

    public static func decode(_ packet: QJLSketchPacket) -> [Float] {
        let d = packet.dimension
        let m = packet.sketchRows
        guard d > 0, m > 0, packet.l2Norm > 0 else {
            return [Float](repeating: 0, count: max(0, d))
        }
        let scale = Float((Double.pi / 2.0).squareRoot()) * packet.l2Norm / Float(m)
        var out = [Float](repeating: 0, count: d)
        for row in 0 ..< m {
            let word = row >> 6
            let bit = word < packet.packedSigns.count
                ? ((packet.packedSigns[word] >> UInt64(row & 63)) & 1)
                : 0
            let sgn: Float = bit == 1 ? 1.0 : -1.0
            let factor = scale * sgn
            for col in 0 ..< d {
                out[col] += factor * gaussianEntry(row: row, col: col, seed: packet.seed)
            }
        }
        return out
    }

    private static func gaussianEntry(row: Int, col: Int, seed: UInt64) -> Float {
        let idx = UInt64(row) &* 1_000_003 &+ UInt64(col)
        let u1 = max(1e-7, uniform01(seed ^ (idx &* 0x9E37_79B9_7F4A_7C15)))
        let u2 = uniform01((seed &+ 0xD1B5_4A32_D192_ED03) ^ (idx &* 0xBF58_476D_1CE4_E5B9))
        let radius = (-2.0 * log(u1)).squareRoot()
        return Float(radius * cos(2.0 * .pi * u2))
    }

    private static func uniform01(_ x: UInt64) -> Double {
        var z = x
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        return Double(z >> 11) / Double(UInt64(1) << 53)
    }
}

public struct PowerSGDLowRankPacket: Sendable, Codable, Equatable {
    public let rows: Int
    public let cols: Int
    public let rank: Int
    public let leftP: [Float]
    public let rightQ: [Float]

    public var wireFloats: Int {
        leftP.count + rightQ.count
    }
}

public enum PowerSGDCompressor {
    public static func compress(
        matrix: [Float],
        rows: Int,
        cols: Int,
        rank: Int,
        warmQ: inout [Float]
    ) -> PowerSGDLowRankPacket {
        let r = max(1, min(rank, min(rows, cols)))
        if warmQ.count != cols * r {
            warmQ = initialOrthonormalQ(cols: cols, rank: r)
        }
        var pMat = multiplyMat(a: matrix, aRows: rows, aCols: cols, b: warmQ, bCols: r)
        orthonormalizeColumns(&pMat, rows: rows, cols: r)
        let qMat = multiplyTransposeA(a: matrix, aRows: rows, aCols: cols, b: pMat, bCols: r)
        warmQ = qMat
        orthonormalizeColumns(&warmQ, rows: cols, cols: r)
        return PowerSGDLowRankPacket(rows: rows, cols: cols, rank: r, leftP: pMat, rightQ: qMat)
    }

    public static func decompress(_ packet: PowerSGDLowRankPacket) -> [Float] {
        var out = [Float](repeating: 0, count: packet.rows * packet.cols)
        let r = packet.rank
        for i in 0 ..< packet.rows {
            for j in 0 ..< packet.cols {
                var sum: Float = 0
                for k in 0 ..< r {
                    sum += packet.leftP[i * r + k] * packet.rightQ[j * r + k]
                }
                out[i * packet.cols + j] = sum
            }
        }
        return out
    }

    private static func initialOrthonormalQ(cols: Int, rank: Int) -> [Float] {
        var q = [Float](repeating: 0, count: cols * rank)
        for i in 0 ..< cols {
            for k in 0 ..< rank {
                q[i * rank + k] = Float(sin(Double((i + 1) * (k + 2)) * 0.37))
            }
        }
        orthonormalizeColumns(&q, rows: cols, cols: rank)
        return q
    }

    private static func multiplyMat(a: [Float], aRows: Int, aCols: Int, b: [Float], bCols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: aRows * bCols)
        for i in 0 ..< aRows {
            for k in 0 ..< bCols {
                var s: Float = 0
                for j in 0 ..< aCols { s += a[i * aCols + j] * b[j * bCols + k] }
                out[i * bCols + k] = s
            }
        }
        return out
    }

    private static func multiplyTransposeA(a: [Float], aRows: Int, aCols: Int, b: [Float], bCols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: aCols * bCols)
        for j in 0 ..< aCols {
            for k in 0 ..< bCols {
                var s: Float = 0
                for i in 0 ..< aRows { s += a[i * aCols + j] * b[i * bCols + k] }
                out[j * bCols + k] = s
            }
        }
        return out
    }

    private static func orthonormalizeColumns(_ mat: inout [Float], rows: Int, cols: Int) {
        for k in 0 ..< cols {
            for prev in 0 ..< k {
                var dot: Double = 0
                for i in 0 ..< rows { dot += Double(mat[i * cols + prev]) * Double(mat[i * cols + k]) }
                let proj = Float(dot)
                for i in 0 ..< rows { mat[i * cols + k] -= proj * mat[i * cols + prev] }
            }
            var normSq: Double = 0
            for i in 0 ..< rows {
                let v = Double(mat[i * cols + k])
                normSq += v * v
            }
            let inv = normSq > 1e-16 ? Float(1.0 / normSq.squareRoot()) : 0
            if inv > 0 {
                for i in 0 ..< rows { mat[i * cols + k] *= inv }
            } else {
                for i in 0 ..< rows { mat[i * cols + k] = 0 }
            }
        }
    }
}

public struct EF21ResidualState: Sendable {
    public private(set) var estimate: [Float]
    public private(set) var warmQ: [Float] = []

    public init(dimension: Int) {
        self.estimate = [Float](repeating: 0, count: max(0, dimension))
    }

    public mutating func stepPowerSGD(
        target: [Float],
        rows: Int,
        cols: Int,
        rank: Int
    ) -> (estimate: [Float], residualNorm: Float, packet: PowerSGDLowRankPacket) {
        if estimate.count != target.count {
            estimate = [Float](repeating: 0, count: target.count)
        }
        let diff = zip(target, estimate).map { $0 - $1 }
        let pkt = PowerSGDCompressor.compress(matrix: diff, rows: rows, cols: cols, rank: rank, warmQ: &warmQ)
        let compressedDiff = PowerSGDCompressor.decompress(pkt)
        var resSq: Double = 0
        for i in target.indices {
            estimate[i] += compressedDiff[i]
            let err = Double(target[i] - estimate[i])
            resSq += err * err
        }
        return (estimate, Float(resSq.squareRoot()), pkt)
    }
}
