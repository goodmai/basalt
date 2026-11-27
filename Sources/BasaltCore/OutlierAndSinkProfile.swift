import Foundation

public enum QuantizabilityRiskTier: String, Sendable, Codable, Equatable, CaseIterable {
    case safe
    case watch
    case sensitive
}

public struct OutlierProfileRequest: Sendable, Codable, Equatable {
    public static let promptCount = 1...32
    public static let sinkTokenRange = 1...64

    public var prompts: [String]
    public var sinkTokenCount: Int
    public var deadNeuronThreshold: Double
    public var layers: [Int]?

    public init(
        prompts: [String],
        sinkTokenCount: Int = 1,
        deadNeuronThreshold: Double = 1e-3,
        layers: [Int]? = nil
    ) {
        self.prompts = prompts
        self.sinkTokenCount = sinkTokenCount
        self.deadNeuronThreshold = deadNeuronThreshold
        self.layers = layers
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.prompts = try container.decode([String].self, forKey: .prompts)
        self.sinkTokenCount = try container.decodeIfPresent(Int.self, forKey: .sinkTokenCount) ?? 1
        self.deadNeuronThreshold = try container.decodeIfPresent(Double.self, forKey: .deadNeuronThreshold) ?? 1e-3
        self.layers = try container.decodeIfPresent([Int].self, forKey: .layers)
    }

    public func validate(layerCount: Int? = nil) throws(BasaltError) {
        guard Self.promptCount.contains(prompts.count) else {
            throw .invalidRequestStructure(details: "outlier_profile: prompts must contain \(Self.promptCount) items")
        }
        guard prompts.allSatisfy({ !$0.isEmpty }) else {
            throw .invalidRequestStructure(details: "outlier_profile: prompts must be non-empty strings")
        }
        guard Self.sinkTokenRange.contains(sinkTokenCount) else {
            throw .invalidRequestStructure(details: "outlier_profile: sinkTokenCount must lie in \(Self.sinkTokenRange)")
        }
        guard deadNeuronThreshold.isFinite, deadNeuronThreshold >= 0 else {
            throw .invalidRequestStructure(details: "outlier_profile: deadNeuronThreshold must be finite and non-negative")
        }
        if let layers {
            guard !layers.isEmpty, layers.allSatisfy({ $0 >= 0 }) else {
                throw .invalidRequestStructure(details: "outlier_profile: layers must be a non-empty list of non-negative indices")
            }
            if let layerCount, layers.contains(where: { $0 >= layerCount }) {
                throw .invalidRequestStructure(details: "outlier_profile: layers must lie in 0..<\(layerCount)")
            }
        }
    }

    enum CodingKeys: String, CodingKey {
        case prompts, sinkTokenCount, deadNeuronThreshold, layers
    }
}

public struct HeadSinkMetrics: Sendable, Codable, Equatable {
    public var headIndex: Int
    public var meanSinkMass: Double
    public var maxSinkMass: Double

    public init(headIndex: Int, meanSinkMass: Double, maxSinkMass: Double) {
        self.headIndex = headIndex
        self.meanSinkMass = meanSinkMass
        self.maxSinkMass = maxSinkMass
    }
}

public struct LayerOutlierMetrics: Sendable, Codable, Equatable {
    public var layerIndex: Int
    public var excessKurtosis: Double
    public var crestFactor: Double
    public var maxAbsActivation: Double
    public var rmsActivation: Double
    public var deadNeuronFraction: Double?
    public var meanAttentionSinkMass: Double?
    public var heads: [HeadSinkMetrics]
    public var riskTier: QuantizabilityRiskTier

    public init(
        layerIndex: Int,
        excessKurtosis: Double,
        crestFactor: Double,
        maxAbsActivation: Double,
        rmsActivation: Double,
        deadNeuronFraction: Double? = nil
    ) {
        self.layerIndex = layerIndex
        self.excessKurtosis = excessKurtosis
        self.crestFactor = crestFactor
        self.maxAbsActivation = maxAbsActivation
        self.rmsActivation = rmsActivation
        self.deadNeuronFraction = deadNeuronFraction
        self.meanAttentionSinkMass = nil
        self.heads = []
        self.riskTier = OutlierAndSinkMath.classifyQuantizabilityRisk(
            excessKurtosis: excessKurtosis,
            crestFactor: crestFactor
        )
    }

    public func withHeads(_ heads: [HeadSinkMetrics]) -> LayerOutlierMetrics {
        var copy = self
        copy.heads = heads
        if heads.isEmpty {
            copy.meanAttentionSinkMass = nil
        } else {
            copy.meanAttentionSinkMass = heads.reduce(0.0) { $0 + $1.meanSinkMass } / Double(heads.count)
        }
        return copy
    }
}

public struct OutlierProfileReport: Sendable, Codable, Equatable {
    public var promptCount: Int
    public var totalTokens: Int
    public var layerCount: Int
    public var maxKurtosisLayer: Int?
    public var maxCrestFactorLayer: Int?
    public var sensitiveLayers: [Int]
    public var layers: [LayerOutlierMetrics]

    public init(
        promptCount: Int,
        totalTokens: Int,
        layers: [LayerOutlierMetrics],
        layerCount: Int? = nil
    ) {
        let sorted = layers.sorted { $0.layerIndex < $1.layerIndex }
        self.promptCount = promptCount
        self.totalTokens = totalTokens
        self.layers = sorted
        self.layerCount = layerCount ?? sorted.count
        self.maxKurtosisLayer = Self.peakLayer(in: sorted, by: \.excessKurtosis)
        self.maxCrestFactorLayer = Self.peakLayer(in: sorted, by: \.crestFactor)
        self.sensitiveLayers = sorted.filter { $0.riskTier == .sensitive }.map(\.layerIndex)
    }

    private static func peakLayer(
        in layers: [LayerOutlierMetrics],
        by keyPath: KeyPath<LayerOutlierMetrics, Double>
    ) -> Int? {
        var bestIndex: Int?
        var bestValue = -Double.infinity
        for layer in layers {
            let value = layer[keyPath: keyPath]
            if value.isFinite, bestIndex == nil || value > bestValue {
                bestIndex = layer.layerIndex
                bestValue = value
            }
        }
        return bestIndex
    }
}

public struct OutlierMomentAccumulator: Sendable, Equatable {
    public var count: Int
    public var mean: Double
    public var m2: Double
    public var m3: Double
    public var m4: Double
    public var maxAbs: Double

    public init(
        count: Int = 0,
        mean: Double = 0,
        m2: Double = 0,
        m3: Double = 0,
        m4: Double = 0,
        maxAbs: Double = 0
    ) {
        self.count = max(0, count)
        self.mean = mean
        self.m2 = max(0, m2)
        self.m3 = m3
        self.m4 = max(0, m4)
        self.maxAbs = max(0, maxAbs)
    }

    public static func from(values: [Double]) -> OutlierMomentAccumulator {
        guard !values.isEmpty else { return OutlierMomentAccumulator() }
        let n = Double(values.count)
        let mean = values.reduce(0.0, +) / n
        var m2 = 0.0
        var m3 = 0.0
        var m4 = 0.0
        var maxAbs = 0.0
        for x in values {
            let d = x - mean
            let d2 = d * d
            m2 += d2
            m3 += d2 * d
            m4 += d2 * d2
            let ax = abs(x)
            if ax > maxAbs { maxAbs = ax }
        }
        return OutlierMomentAccumulator(
            count: values.count,
            mean: mean,
            m2: m2,
            m3: m3,
            m4: m4,
            maxAbs: maxAbs
        )
    }

    public mutating func merge(_ other: OutlierMomentAccumulator) {
        guard other.count > 0 else { return }
        guard count > 0 else {
            self = other
            return
        }
        let na = Double(count)
        let nb = Double(other.count)
        let n = na + nb
        let delta = other.mean - mean
        let delta2 = delta * delta
        let delta3 = delta2 * delta
        let delta4 = delta2 * delta2
        let newM2 = m2 + other.m2 + delta2 * (na * nb / n)
        let newM3 = m3 + other.m3
            + delta3 * (na * nb * (na - nb) / (n * n))
            + 3.0 * delta * (na * other.m2 - nb * m2) / n
        let newM4 = m4 + other.m4
            + delta4 * (na * nb * (na * na - na * nb + nb * nb) / (n * n * n))
            + 6.0 * delta2 * (na * na * other.m2 + nb * nb * m2) / (n * n)
            + 4.0 * delta * (na * other.m3 - nb * m3) / n
        mean += delta * (nb / n)
        m2 = max(0, newM2)
        m3 = newM3
        m4 = max(0, newM4)
        maxAbs = max(maxAbs, other.maxAbs)
        count += other.count
    }

    public var excessKurtosis: Double {
        guard count > 0 else { return 0 }
        let n = Double(count)
        let variance = m2 / n
        let sigma = variance.squareRoot()
        guard sigma.isFinite, sigma > 1e-12 else { return 0 }
        let value = (m4 / n) / (variance * variance) - 3.0
        return value.isFinite ? value : 0
    }

    public var rms: Double {
        guard count > 0 else { return 0 }
        let meanSq = m2 / Double(count) + mean * mean
        let value = max(0, meanSq).squareRoot()
        return value.isFinite ? value : 0
    }

    public var crestFactor: Double {
        let rootMeanSq = rms
        guard rootMeanSq > 1e-12 else { return 0 }
        let ratio = maxAbs / rootMeanSq
        return ratio.isFinite ? ratio : 0
    }
}

public enum OutlierAndSinkMath: Sendable {
    public static func excessKurtosis(_ values: [Double]) -> Double {
        OutlierMomentAccumulator.from(values: values).excessKurtosis
    }

    public static func crestFactor(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        var maxAbs = 0.0
        var sumSq = 0.0
        for x in values {
            let ax = abs(x)
            if ax > maxAbs { maxAbs = ax }
            sumSq += x * x
        }
        let rms = (sumSq / Double(values.count)).squareRoot()
        guard rms.isFinite, rms > 1e-12 else { return 0 }
        let ratio = maxAbs / rms
        return ratio.isFinite ? ratio : 0
    }

    public static func deadNeuronFraction(_ values: [Double], threshold: Double = 1e-3) -> Double {
        guard !values.isEmpty else { return 0 }
        let dead = values.reduce(0) { $0 + (abs($1) < threshold ? 1 : 0) }
        return Double(dead) / Double(values.count)
    }

    public static func attentionSinkMass(
        weights: [[Double]],
        sinkCount: Int = 1
    ) -> (mean: Double, max: Double) {
        let kCount = max(1, sinkCount)
        let start = weights.count > kCount ? kCount : 0
        var total = 0.0
        var peak = 0.0
        var validRows = 0
        for query in start ..< weights.count {
            let row = weights[query]
            guard !row.isEmpty else { continue }
            let limit = min(kCount, row.count)
            var mass = 0.0
            for key in 0 ..< limit {
                mass += row[key]
            }
            let clamped = mass.isFinite ? min(1.0, max(0.0, mass)) : 0.0
            total += clamped
            if clamped > peak { peak = clamped }
            validRows += 1
        }
        guard validRows > 0 else { return (0.0, 0.0) }
        return (total / Double(validRows), peak)
    }

    public static func classifyQuantizabilityRisk(
        excessKurtosis: Double,
        crestFactor: Double
    ) -> QuantizabilityRiskTier {
        if !excessKurtosis.isFinite || !crestFactor.isFinite {
            return .sensitive
        }
        if excessKurtosis >= 50.0 || crestFactor >= 15.0 {
            return .sensitive
        }
        if excessKurtosis >= 10.0 || crestFactor >= 8.0 {
            return .watch
        }
        return .safe
    }
}
