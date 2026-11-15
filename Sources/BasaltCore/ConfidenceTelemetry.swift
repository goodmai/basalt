import Foundation

public struct ConfidenceTelemetry: Codable, Sendable, Equatable {
    public static let bins = 10
    public static let entropyStep = 0.5
    public var responses = 0
    public var minProbability = [Int](repeating: 0, count: ConfidenceTelemetry.bins)
    public var meanProbability = [Int](repeating: 0, count: ConfidenceTelemetry.bins)
    public var meanEntropy = [Int](repeating: 0, count: ConfidenceTelemetry.bins)
    public var minMargin = [Int](repeating: 0, count: ConfidenceTelemetry.bins)
    public var belnap = BelnapDistribution()

    public init() {}

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        responses = try container.decodeIfPresent(Int.self, forKey: .responses) ?? 0
        minProbability = try container.decodeIfPresent([Int].self, forKey: .minProbability) ?? [Int](repeating: 0, count: Self.bins)
        meanProbability = try container.decodeIfPresent([Int].self, forKey: .meanProbability) ?? [Int](repeating: 0, count: Self.bins)
        meanEntropy = try container.decodeIfPresent([Int].self, forKey: .meanEntropy) ?? [Int](repeating: 0, count: Self.bins)
        minMargin = try container.decodeIfPresent([Int].self, forKey: .minMargin) ?? [Int](repeating: 0, count: Self.bins)
        belnap = try container.decodeIfPresent(BelnapDistribution.self, forKey: .belnap) ?? BelnapDistribution()
    }

    public mutating func record(_ confidence: ResponseConfidence, thresholds: EpistemicThresholds = .default) {
        responses += 1
        Self.add(confidence.min, to: &minProbability, step: 1.0 / Double(Self.bins))
        Self.add(confidence.mean, to: &meanProbability, step: 1.0 / Double(Self.bins))
        if let entropy = confidence.meanEntropy { Self.add(entropy, to: &meanEntropy, step: Self.entropyStep) }
        if let margin = confidence.minMargin { Self.add(margin, to: &minMargin, step: 1.0 / Double(Self.bins)) }
        belnap.record(confidence.belnapState(thresholds: thresholds))
    }

    static func add(_ value: Double, to bins: inout [Int], step: Double) {
        guard value.isFinite else { return }
        bins[min(bins.count - 1, max(0, Int(value / step)))] += 1
    }

    enum CodingKeys: String, CodingKey {
        case responses, belnap
        case minProbability = "min_probability", meanProbability = "mean_probability", meanEntropy = "mean_entropy", minMargin = "min_margin"
    }
}
