import Foundation

public struct BradleyTerryRouter: Sendable, Codable {
    public let strongEmbedding: [Double]
    public let weakEmbedding: [Double]
    public let weights: [Double]
    public let projection: [[Double]]

    public init(strongEmbedding: [Double], weakEmbedding: [Double], weights: [Double], projection: [[Double]]) throws {
        let d = weights.count
        guard d > 0, strongEmbedding.count == d, weakEmbedding.count == d, projection.count == d,
              let featureCount = projection.first?.count, featureCount > 0,
              projection.allSatisfy({ $0.count == featureCount })
        else {
            throw BradleyTerryError.shapeMismatch
        }
        self.strongEmbedding = strongEmbedding
        self.weakEmbedding = weakEmbedding
        self.weights = weights
        self.projection = projection
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            strongEmbedding: try container.decode([Double].self, forKey: .strongEmbedding),
            weakEmbedding: try container.decode([Double].self, forKey: .weakEmbedding),
            weights: try container.decode([Double].self, forKey: .weights),
            projection: try container.decode([[Double]].self, forKey: .projection)
        )
    }

    public static func load(from url: URL) throws -> BradleyTerryRouter {
        try JSONDecoder().decode(BradleyTerryRouter.self, from: Data(contentsOf: url))
    }

    public var featureCount: Int { projection[0].count }

    public func strongWinProbability(for prompt: String) -> Double {
        let features = Self.hashedFeatures(prompt, count: featureCount)
        var logit = 0.0
        for i in weights.indices {
            let projected = zip(projection[i], features).reduce(0.0) { $0 + $1.0 * $1.1 }
            logit += weights[i] * (strongEmbedding[i] - weakEmbedding[i]) * projected
        }
        return 1 / (1 + exp(-logit))
    }

    public static func hashedFeatures(_ text: String, count: Int) -> [Double] {
        var features = [Double](repeating: 0, count: count)
        for word in text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }) {
            features[Int(fnv1a(word) % UInt64(count))] += 1
        }
        let norm = features.reduce(0.0) { $0 + $1 * $1 }.squareRoot()
        return norm > 0 ? features.map { $0 / norm } : features
    }

    public static func calibrateThreshold(scores: [Double], strongFraction: Double) -> Double {
        guard !scores.isEmpty else { return 0.5 }
        let sorted = scores.sorted(by: >)
        let strongCount = Int((Double(sorted.count) * min(max(strongFraction, 0), 1)).rounded())
        if strongCount == 0 { return 1 }
        if strongCount >= sorted.count { return min(sorted[sorted.count - 1], 1) - .ulpOfOne }
        return (sorted[strongCount - 1] + sorted[strongCount]) / 2
    }

    private static func fnv1a(_ word: Substring) -> UInt64 {
        word.utf8.reduce(UInt64(0xcbf29ce484222325)) { ($0 ^ UInt64($1)) &* 0x100000001b3 }
    }
}

public enum BradleyTerryError: Error, Equatable {
    case shapeMismatch
}

public struct StickyRoutingPolicy: Sendable {
    public enum Tier: Sendable, Equatable {
        case strong
        case weak
    }

    public let strongModel: String
    public let weakModel: String
    public let threshold: Double
    public let cacheStickinessPenalty: Double
    public let switchCostTokenScale: Double

    public init(
        strongModel: String, weakModel: String, threshold: Double = 0.5,
        cacheStickinessPenalty: Double = 0.1, switchCostTokenScale: Double = 4_096
    ) {
        self.strongModel = strongModel
        self.weakModel = weakModel
        self.threshold = threshold
        self.cacheStickinessPenalty = cacheStickinessPenalty
        self.switchCostTokenScale = switchCostTokenScale
    }

    public func switchCost(warmPrefixTokens: Int) -> Double {
        let tokens = Double(max(0, warmPrefixTokens))
        return tokens / (tokens + switchCostTokenScale)
    }

    public func tier(strongWinProbability p: Double, previous: Tier?, warmPrefixTokens: Int) -> Tier {
        let margin = p - threshold
        guard let previous else { return margin > 0 ? .strong : .weak }
        let penalty = cacheStickinessPenalty * switchCost(warmPrefixTokens: warmPrefixTokens)
        switch previous {
        case .weak: return margin > penalty ? .strong : .weak
        case .strong: return margin < -penalty ? .weak : .strong
        }
    }

    public func model(for tier: Tier) -> String {
        tier == .strong ? strongModel : weakModel
    }
}
