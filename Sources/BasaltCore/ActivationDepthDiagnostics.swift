import Foundation

public struct DepthDiagnosticsRequest: Sendable, Codable, Equatable {
    public static let sequenceLengthRange = 1...2048
    public static let rankSequenceCap = 512

    public var prompt: String
    public var includeRank: Bool = true
    public var softcapThresholdFraction: Double = 0.95
    public var maxSequenceLength: Int = 512
    public var layers: [Int]? = nil

    public init(
        prompt: String,
        includeRank: Bool = true,
        softcapThresholdFraction: Double = 0.95,
        maxSequenceLength: Int = 512,
        layers: [Int]? = nil
    ) {
        self.prompt = prompt
        self.includeRank = includeRank
        self.softcapThresholdFraction = softcapThresholdFraction
        self.maxSequenceLength = maxSequenceLength
        self.layers = layers
    }

    public func validate(layerCount: Int? = nil) throws(BasaltError) {
        guard !prompt.isEmpty else {
            throw .invalidRequestStructure(details: "depth_diagnostics: prompt must not be empty")
        }
        guard Self.sequenceLengthRange.contains(maxSequenceLength) else {
            throw .invalidRequestStructure(
                details: "depth_diagnostics: maxSequenceLength must be in \(Self.sequenceLengthRange), got \(maxSequenceLength)"
            )
        }
        guard softcapThresholdFraction.isFinite, softcapThresholdFraction > 0, softcapThresholdFraction <= 1.0 else {
            throw .invalidRequestStructure(
                details: "depth_diagnostics: softcapThresholdFraction must be in (0, 1], got \(softcapThresholdFraction)"
            )
        }
        if let layers {
            try Self.validateLayers(layers, layerCount: layerCount)
        }
    }

    private static func validateLayers(_ layers: [Int], layerCount: Int?) throws(BasaltError) {
        guard !layers.isEmpty else {
            throw .invalidRequestStructure(details: "depth_diagnostics: layers must list at least one layer index")
        }
        if let bad = layers.first(where: { $0 < 0 }) {
            throw .invalidRequestStructure(details: "depth_diagnostics: layer index \(bad) must not be negative")
        }
        if let layerCount, let outOfBounds = layers.first(where: { $0 >= layerCount }) {
            throw .invalidRequestStructure(
                details: "depth_diagnostics: layer \(outOfBounds) is outside the model's \(layerCount) layers"
            )
        }
    }
}

public struct LayerDepthMetrics: Sendable, Codable, Equatable {
    public var layerIndex: Int
    public var residualNormIn: Double
    public var residualNormOut: Double
    public var residualNormGrowth: Double
    public var attentionContributionNorm: Double?
    public var mlpContributionNorm: Double?
    public var attnToMLPRatio: Double?
    public var preSoftmaxSMax: [Double]
    public var maxPreSoftmaxLogit: Double
    public var softcapValue: Double?
    public var softcapSaturationRate: Double?
    public var effectiveRank: Double?

    public init(
        layerIndex: Int,
        residualNormIn: Double,
        residualNormOut: Double,
        preSoftmaxSMax: [Double] = [],
        effectiveRank: Double? = nil
    ) {
        self.layerIndex = layerIndex
        self.residualNormIn = residualNormIn
        self.residualNormOut = residualNormOut
        self.residualNormGrowth = DepthDiagnosticsMath.residualNormGrowth(normIn: residualNormIn, normOut: residualNormOut)
        self.preSoftmaxSMax = preSoftmaxSMax
        self.maxPreSoftmaxLogit = preSoftmaxSMax.filter(\.isFinite).max() ?? 0.0
        self.effectiveRank = effectiveRank
    }

    public mutating func setSublayerNorms(attention: Double?, mlp: Double?) {
        attentionContributionNorm = attention
        mlpContributionNorm = mlp
        if let attention, let mlp {
            attnToMLPRatio = DepthDiagnosticsMath.attnToMLPRatio(attnNorm: attention, mlpNorm: mlp)
        } else {
            attnToMLPRatio = nil
        }
    }

    public mutating func setSoftcap(value: Double?, saturationRate: Double?) {
        softcapValue = value
        softcapSaturationRate = saturationRate
    }
}

public struct DepthDiagnosticsReport: Sendable, Codable, Equatable {
    public var promptTokenCount: Int
    public var layerCount: Int
    public var maxSMaxOverall: Double
    public var maxSMaxLayer: Int
    public var meanResidualGrowth: Double
    public var minEffectiveRank: Double?
    public var layers: [LayerDepthMetrics]

    public init(promptTokenCount: Int, layerCount: Int, layers: [LayerDepthMetrics]) {
        let summary = Self.summarize(layers)
        self.promptTokenCount = promptTokenCount
        self.layerCount = layerCount
        self.maxSMaxOverall = summary.maxSMaxOverall
        self.maxSMaxLayer = summary.maxSMaxLayer
        self.meanResidualGrowth = summary.meanResidualGrowth
        self.minEffectiveRank = summary.minEffectiveRank
        self.layers = layers
    }

    private struct Summary {
        let maxSMaxOverall: Double
        let maxSMaxLayer: Int
        let meanResidualGrowth: Double
        let minEffectiveRank: Double?
    }

    private static func summarize(_ layers: [LayerDepthMetrics]) -> Summary {
        var bestSMax = -Double.infinity
        var bestLayer = layers.first?.layerIndex ?? 0
        var growthSum = 0.0
        var growthCount = 0
        var minRank: Double?
        for layer in layers {
            if layer.maxPreSoftmaxLogit.isFinite, layer.maxPreSoftmaxLogit > bestSMax {
                bestSMax = layer.maxPreSoftmaxLogit
                bestLayer = layer.layerIndex
            }
            if layer.residualNormGrowth.isFinite {
                growthSum += layer.residualNormGrowth
                growthCount += 1
            }
            if let rank = layer.effectiveRank, rank.isFinite {
                minRank = minRank.map { min($0, rank) } ?? rank
            }
        }
        return Summary(
            maxSMaxOverall: bestSMax.isFinite ? bestSMax : 0.0,
            maxSMaxLayer: bestLayer,
            meanResidualGrowth: growthCount > 0 ? growthSum / Double(growthCount) : 0.0,
            minEffectiveRank: minRank
        )
    }
}

public enum DepthDiagnosticsMath: Sendable {
    public static func effectiveRank(singularValues: [Double]) -> Double {
        let positive = singularValues.filter { $0.isFinite && $0 > 0 }
        guard let peak = positive.max(), peak > 0 else { return 0.0 }
        let total = positive.reduce(0.0) { $0 + $1 / peak }
        guard total > 0, total.isFinite else { return 0.0 }
        var entropy = 0.0
        for value in positive {
            let p = (value / peak) / total
            if p > 0 { entropy -= p * Foundation.log(p) }
        }
        return Foundation.exp(max(0.0, entropy))
    }

    public static func residualNormGrowth(normIn: Double, normOut: Double, eps: Double = 1e-12) -> Double {
        guard normIn.isFinite, normOut.isFinite else { return 0.0 }
        let safeEps = (eps.isFinite && eps > 0) ? eps : 1e-12
        return max(0.0, normOut) / max(max(0.0, normIn), safeEps)
    }

    public static func attnToMLPRatio(attnNorm: Double, mlpNorm: Double, eps: Double = 1e-12) -> Double {
        guard attnNorm.isFinite, mlpNorm.isFinite else { return 0.0 }
        let safeEps = (eps.isFinite && eps > 0) ? eps : 1e-12
        return max(0.0, attnNorm) / max(max(0.0, mlpNorm), safeEps)
    }

    public static func softcapSaturationFraction(
        logits: [Double],
        softcap: Double,
        thresholdFraction: Double = 0.95
    ) -> Double {
        guard softcap.isFinite, softcap > 0, thresholdFraction.isFinite else { return 0.0 }
        var finiteCount = 0
        var saturatedCount = 0
        for logit in logits where logit.isFinite {
            finiteCount += 1
            if abs(Foundation.tanh(logit / softcap)) >= thresholdFraction {
                saturatedCount += 1
            }
        }
        guard finiteCount > 0 else { return 0.0 }
        return Double(saturatedCount) / Double(finiteCount)
    }
}
