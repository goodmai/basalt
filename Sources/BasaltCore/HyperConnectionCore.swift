import Foundation

public enum HyperConnectionVariant: String, Codable, Sendable, CaseIterable {
    case hc
    case mhc
    case ihc
}

public enum HyperConnectionGateMode: String, Codable, Sendable, CaseIterable {
    case sigmoid
    case monographSoftmaxSoftplus = "monograph_softmax_softplus"
}

public enum HyperConnectionCollapseMode: String, Codable, Sendable, CaseIterable {
    case mean
    case sum
}

public struct HyperConnectionSpec: Codable, Sendable, Equatable {
    public var variant: HyperConnectionVariant
    public var expansionRate: Int
    public var sinkhornIterations: Int
    public var sinkhornEps: Float
    public var gateMode: HyperConnectionGateMode
    public var collapseMode: HyperConnectionCollapseMode
    public var dynamic: Bool
    public var alphaInit: Float

    enum CodingKeys: String, CodingKey {
        case variant = "hc_variant"
        case expansionRate = "hc_expansion_rate"
        case sinkhornIterations = "hc_sinkhorn_iterations"
        case sinkhornEps = "hc_sinkhorn_eps"
        case gateMode = "hc_gate_mode"
        case collapseMode = "hc_collapse_mode"
        case dynamic = "hc_dynamic"
        case alphaInit = "hc_alpha_init"
    }

    public init(
        variant: HyperConnectionVariant = .mhc,
        expansionRate: Int = 4,
        sinkhornIterations: Int = 20,
        sinkhornEps: Float = 1e-6,
        gateMode: HyperConnectionGateMode = .sigmoid,
        collapseMode: HyperConnectionCollapseMode = .mean
    ) {
        self.variant = variant
        self.expansionRate = expansionRate
        self.sinkhornIterations = sinkhornIterations
        self.sinkhornEps = sinkhornEps
        self.gateMode = gateMode
        self.collapseMode = collapseMode
        self.dynamic = true
        self.alphaInit = 0.01
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        variant = try c.decodeIfPresent(HyperConnectionVariant.self, forKey: .variant) ?? .mhc
        expansionRate = try c.decodeIfPresent(Int.self, forKey: .expansionRate) ?? 4
        sinkhornIterations = try c.decodeIfPresent(Int.self, forKey: .sinkhornIterations) ?? 20
        sinkhornEps = try c.decodeIfPresent(Float.self, forKey: .sinkhornEps) ?? 1e-6
        gateMode = try c.decodeIfPresent(HyperConnectionGateMode.self, forKey: .gateMode) ?? .sigmoid
        collapseMode = try c.decodeIfPresent(HyperConnectionCollapseMode.self, forKey: .collapseMode) ?? .mean
        dynamic = try c.decodeIfPresent(Bool.self, forKey: .dynamic) ?? true
        alphaInit = try c.decodeIfPresent(Float.self, forKey: .alphaInit) ?? 0.01
    }

    public func with(dynamic: Bool? = nil, alphaInit: Float? = nil) -> HyperConnectionSpec {
        var copy = self
        if let dynamic { copy.dynamic = dynamic }
        if let alphaInit { copy.alphaInit = alphaInit }
        return copy
    }

    public func validate() throws(BasaltError) {
        guard expansionRate >= 1 else {
            throw .modelConfigurationInvalid(
                path: "hyper_connection",
                reason: "hc_expansion_rate must be >= 1, got \(expansionRate)"
            )
        }
        guard sinkhornIterations >= 1 else {
            throw .modelConfigurationInvalid(
                path: "hyper_connection",
                reason: "hc_sinkhorn_iterations must be >= 1, got \(sinkhornIterations)"
            )
        }
        guard sinkhornEps.isFinite && sinkhornEps > 0 else {
            throw .modelConfigurationInvalid(
                path: "hyper_connection",
                reason: "hc_sinkhorn_eps must be finite and > 0, got \(sinkhornEps)"
            )
        }
        guard alphaInit.isFinite && alphaInit >= 0 else {
            throw .modelConfigurationInvalid(
                path: "hyper_connection",
                reason: "hc_alpha_init must be finite and >= 0, got \(alphaInit)"
            )
        }
    }
}

public enum BirkhoffMath: Sendable {
    public static func identity(n: Int) -> [Float] {
        guard n > 0 else { return [] }
        var result = [Float](repeating: 0, count: n * n)
        for index in 0 ..< n {
            result[index * n + index] = 1
        }
        return result
    }

    public static func sinkhornCPU(
        logits: [Float],
        n: Int,
        iterations: Int = 20,
        eps: Float = 1e-6
    ) -> [Float] {
        guard n > 0, logits.count == n * n, iterations >= 1, eps.isFinite, eps > 0 else {
            return []
        }
        var p = [Float](repeating: 0, count: n * n)
        for row in 0 ..< n {
            let base = row * n
            var rowMax = -Float.infinity
            for col in 0 ..< n {
                let value = logits[base + col]
                if value > rowMax { rowMax = value }
            }
            for col in 0 ..< n {
                p[base + col] = Float(Foundation.exp(Double(logits[base + col] - rowMax)))
            }
        }
        for _ in 0 ..< iterations {
            normalizeRows(&p, n: n, eps: eps)
            normalizeCols(&p, n: n, eps: eps)
        }
        return p
    }

    public static func amaxGain(matrix: [Float], n: Int) -> Float {
        guard n > 0, matrix.count == n * n else { return 0 }
        var maxRow: Float = 0
        for row in 0 ..< n {
            let base = row * n
            var sum: Float = 0
            for col in 0 ..< n {
                sum += abs(matrix[base + col])
            }
            if sum > maxRow { maxRow = sum }
        }
        var maxCol: Float = 0
        for col in 0 ..< n {
            var sum: Float = 0
            for row in 0 ..< n {
                sum += abs(matrix[row * n + col])
            }
            if sum > maxCol { maxCol = sum }
        }
        return max(maxRow, maxCol)
    }

    public static func compose(matrices: [[Float]], n: Int) -> [Float] {
        guard n > 0 else { return [] }
        var acc = identity(n: n)
        for matrix in matrices where matrix.count == n * n {
            acc = multiply(matrix, acc, n: n)
        }
        return acc
    }

    public static func multiply(_ lhs: [Float], _ rhs: [Float], n: Int) -> [Float] {
        guard n > 0, lhs.count == n * n, rhs.count == n * n else { return [] }
        var out = [Float](repeating: 0, count: n * n)
        for row in 0 ..< n {
            let rowBase = row * n
            for col in 0 ..< n {
                var acc: Float = 0
                for k in 0 ..< n {
                    acc += lhs[rowBase + k] * rhs[k * n + col]
                }
                out[rowBase + col] = acc
            }
        }
        return out
    }

    public static func rowColDeviations(
        matrix: [Float],
        n: Int
    ) -> (minEntry: Float, maxRowError: Float, maxColError: Float) {
        guard n > 0, matrix.count == n * n else { return (0, 0, 0) }
        var minEntry = Float.infinity
        var maxRowError: Float = 0
        for row in 0 ..< n {
            let base = row * n
            var sum: Float = 0
            for col in 0 ..< n {
                let value = matrix[base + col]
                if value < minEntry { minEntry = value }
                sum += value
            }
            maxRowError = max(maxRowError, abs(sum - 1))
        }
        var maxColError: Float = 0
        for col in 0 ..< n {
            var sum: Float = 0
            for row in 0 ..< n {
                sum += matrix[row * n + col]
            }
            maxColError = max(maxColError, abs(sum - 1))
        }
        return (minEntry, maxRowError, maxColError)
    }

    private static func normalizeRows(_ p: inout [Float], n: Int, eps: Float) {
        for row in 0 ..< n {
            let base = row * n
            var sum: Float = 0
            for col in 0 ..< n {
                sum += p[base + col]
            }
            let inv = 1 / (sum + eps)
            for col in 0 ..< n {
                p[base + col] *= inv
            }
        }
    }

    private static func normalizeCols(_ p: inout [Float], n: Int, eps: Float) {
        for col in 0 ..< n {
            var sum: Float = 0
            for row in 0 ..< n {
                sum += p[row * n + col]
            }
            let inv = 1 / (sum + eps)
            for row in 0 ..< n {
                p[row * n + col] *= inv
            }
        }
    }
}
