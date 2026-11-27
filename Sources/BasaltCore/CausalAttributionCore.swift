import Foundation

public struct CausalSweepOptions: Sendable, Codable, Equatable {
    public var layers: [Int]?
    public var includePatching: Bool

    public init(layers: [Int]? = nil, includePatching: Bool = true) {
        self.layers = layers
        self.includePatching = includePatching
    }
}

public struct CausalSweepRequest: Sendable, Codable, Equatable {
    public var cleanPrompt: String
    public var corruptedPrompt: String
    public var targetToken: String?
    public var foilToken: String?
    public var targetTokenID: Int?
    public var foilTokenID: Int?
    public var layers: [Int]?
    public var includePatching: Bool = true

    public init(
        cleanPrompt: String,
        corruptedPrompt: String,
        targetToken: String? = nil,
        foilToken: String? = nil,
        layers: [Int]? = nil,
        includePatching: Bool = true
    ) {
        self.cleanPrompt = cleanPrompt
        self.corruptedPrompt = corruptedPrompt
        self.targetToken = targetToken
        self.foilToken = foilToken
        self.targetTokenID = nil
        self.foilTokenID = nil
        self.layers = layers
        self.includePatching = includePatching
    }

    public init(
        cleanPrompt: String,
        corruptedPrompt: String,
        targetTokenID: Int,
        foilTokenID: Int,
        layers: [Int]? = nil,
        includePatching: Bool = true
    ) {
        self.cleanPrompt = cleanPrompt
        self.corruptedPrompt = corruptedPrompt
        self.targetToken = nil
        self.foilToken = nil
        self.targetTokenID = targetTokenID
        self.foilTokenID = foilTokenID
        self.layers = layers
        self.includePatching = includePatching
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        cleanPrompt = try container.decode(String.self, forKey: .cleanPrompt)
        corruptedPrompt = try container.decode(String.self, forKey: .corruptedPrompt)
        targetToken = try container.decodeIfPresent(String.self, forKey: .targetToken)
        foilToken = try container.decodeIfPresent(String.self, forKey: .foilToken)
        targetTokenID = try container.decodeIfPresent(Int.self, forKey: .targetTokenID)
        foilTokenID = try container.decodeIfPresent(Int.self, forKey: .foilTokenID)
        layers = try container.decodeIfPresent([Int].self, forKey: .layers)
        includePatching = try container.decodeIfPresent(Bool.self, forKey: .includePatching) ?? true
    }

    enum CodingKeys: String, CodingKey {
        case cleanPrompt, corruptedPrompt, targetToken, foilToken, targetTokenID, foilTokenID, layers, includePatching
    }

    public func validate(layers totalLayers: Int? = nil) throws(BasaltError) {
        guard !cleanPrompt.isEmpty, !corruptedPrompt.isEmpty else {
            throw .invalidRequestStructure(details: "causal_sweep requires non-empty cleanPrompt and corruptedPrompt")
        }
        try Self.validateEndpoint(id: targetTokenID, text: targetToken, label: "target")
        try Self.validateEndpoint(id: foilTokenID, text: foilToken, label: "foil")
        if let targetID = targetTokenID, let foilID = foilTokenID, targetID == foilID {
            throw .invalidRequestStructure(details: "causal_sweep: targetTokenID and foilTokenID must differ")
        }
        try Self.validateLayers(layers, totalLayers: totalLayers)
    }

    private static func validateEndpoint(id: Int?, text: String?, label: String) throws(BasaltError) {
        if id != nil, let text, !text.isEmpty {
            throw .invalidRequestStructure(details: "causal_sweep: specify \(label)Token or \(label)TokenID, not both")
        }
        if let id {
            guard id >= 0 else {
                throw .invalidRequestStructure(details: "causal_sweep: \(label)TokenID must be non-negative, got \(id)")
            }
            return
        }
        guard let text, !text.isEmpty else {
            throw .invalidRequestStructure(details: "causal_sweep requires \(label)Token or \(label)TokenID")
        }
    }

    private static func validateLayers(_ selected: [Int]?, totalLayers: Int?) throws(BasaltError) {
        guard let selected else { return }
        guard !selected.isEmpty else {
            throw .invalidRequestStructure(details: "causal_sweep: layers must be non-empty when specified")
        }
        if let negative = selected.first(where: { $0 < 0 }) {
            throw .invalidRequestStructure(details: "causal_sweep: layer index \(negative) must be non-negative")
        }
        if let totalLayers, let outOfBounds = selected.first(where: { $0 >= totalLayers }) {
            throw .invalidRequestStructure(details: "causal_sweep: layer \(outOfBounds) is outside 0..<\(totalLayers)")
        }
    }
}

public struct LayerCausalScore: Sendable, Codable, Equatable {
    public var layerIndex: Int
    public var directLogitAttribution: Double
    public var patchedLogitDiff: Double?
    public var causalRecovery: Double?

    public init(
        layerIndex: Int,
        directLogitAttribution: Double,
        patchedLogitDiff: Double? = nil,
        causalRecovery: Double? = nil
    ) {
        self.layerIndex = layerIndex
        self.directLogitAttribution = directLogitAttribution
        self.patchedLogitDiff = patchedLogitDiff
        self.causalRecovery = causalRecovery
    }
}

public struct CausalSweepReport: Sendable, Codable, Equatable {
    public var targetTokenID: Int
    public var foilTokenID: Int
    public var cleanLogitDiff: Double
    public var corruptedLogitDiff: Double
    public var embeddingDLA: Double
    public var totalDLASum: Double
    public var conservationError: Double
    public var topDLALayer: Int?
    public var topCausalLayer: Int?
    public var layers: [LayerCausalScore]

    public init(
        targetTokenID: Int,
        foilTokenID: Int,
        cleanLogitDiff: Double,
        corruptedLogitDiff: Double,
        embeddingDLA: Double,
        layers: [LayerCausalScore]
    ) {
        let sum = embeddingDLA + layers.reduce(0.0) { $0 + $1.directLogitAttribution }
        self.targetTokenID = targetTokenID
        self.foilTokenID = foilTokenID
        self.cleanLogitDiff = cleanLogitDiff
        self.corruptedLogitDiff = corruptedLogitDiff
        self.embeddingDLA = embeddingDLA
        self.totalDLASum = sum
        self.conservationError = abs(sum - cleanLogitDiff)
        self.topDLALayer = CausalAttributionMath.topDLALayer(in: layers)
        self.topCausalLayer = CausalAttributionMath.topCausalLayer(in: layers)
        self.layers = layers
    }

    public func withTotalDLASum(_ totalDLASum: Double) -> CausalSweepReport {
        var copy = self
        copy.totalDLASum = totalDLASum
        copy.conservationError = abs(totalDLASum - cleanLogitDiff)
        return copy
    }
}

public enum CausalAttributionMath: Sendable {
    public static func dot(_ lhs: Span<Double>, _ rhs: Span<Double>) -> Double {
        let count = min(lhs.count, rhs.count)
        var sum = 0.0
        for index in 0 ..< count {
            sum += lhs[index] * rhs[index]
        }
        return sum
    }

    public static func unembedDirection(targetRow: [Double], foilRow: [Double]) -> [Double] {
        let count = min(targetRow.count, foilRow.count)
        var out: [Double] = []
        out.reserveCapacity(count)
        for index in 0 ..< count {
            out.append(targetRow[index] - foilRow[index])
        }
        return out
    }

    public static func directLogitAttributions(
        embeddingState: [Double],
        layerDeltas: [[Double]],
        direction: [Double],
        biasDiff: Double = 0
    ) -> (embeddingDLA: Double, layerDLAs: [Double], totalSum: Double) {
        let embeddingDLA = dot(embeddingState.span, direction.span)
        var layerDLAs: [Double] = []
        layerDLAs.reserveCapacity(layerDeltas.count)
        var total = embeddingDLA + biasDiff
        for delta in layerDeltas {
            let value = dot(delta.span, direction.span)
            layerDLAs.append(value)
            total += value
        }
        return (embeddingDLA, layerDLAs, total)
    }

    public static func causalRecovery(
        cleanDiff: Double,
        corruptedDiff: Double,
        patchedDiff: Double,
        eps: Double = 1e-8
    ) -> Double {
        let denominator = cleanDiff - corruptedDiff
        guard denominator.isFinite, abs(denominator) >= eps else { return 0 }
        let numerator = patchedDiff - corruptedDiff
        guard numerator.isFinite else { return 0 }
        return numerator / denominator
    }

    public static func topDLALayer(in layers: [LayerCausalScore]) -> Int? {
        var best: (index: Int, magnitude: Double)?
        for score in layers where score.directLogitAttribution.isFinite {
            let magnitude = abs(score.directLogitAttribution)
            guard let current = best else {
                best = (score.layerIndex, magnitude)
                continue
            }
            if magnitude > current.magnitude || (magnitude == current.magnitude && score.layerIndex < current.index) {
                best = (score.layerIndex, magnitude)
            }
        }
        return best?.index
    }

    public static func topCausalLayer(in layers: [LayerCausalScore]) -> Int? {
        var best: (index: Int, recovery: Double)?
        for score in layers {
            guard let recovery = score.causalRecovery, recovery.isFinite else { continue }
            guard let current = best else {
                best = (score.layerIndex, recovery)
                continue
            }
            if recovery > current.recovery || (recovery == current.recovery && score.layerIndex < current.index) {
                best = (score.layerIndex, recovery)
            }
        }
        return best?.index
    }
}
