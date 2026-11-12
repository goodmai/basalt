import Foundation

public struct ConfidenceTelemetry: Codable, Sendable, Equatable {
    public static let bins = 10
    public static let entropyStep = 0.5
    public var responses = 0
    public var minProbability = [Int](repeating: 0, count: ConfidenceTelemetry.bins)
    public var meanProbability = [Int](repeating: 0, count: ConfidenceTelemetry.bins)
    public var meanEntropy = [Int](repeating: 0, count: ConfidenceTelemetry.bins)
    public var minMargin = [Int](repeating: 0, count: ConfidenceTelemetry.bins)

    public init() {}

    public mutating func record(_ confidence: ResponseConfidence) {
        responses += 1
        Self.add(confidence.min, to: &minProbability, step: 1.0 / Double(Self.bins))
        Self.add(confidence.mean, to: &meanProbability, step: 1.0 / Double(Self.bins))
        if let entropy = confidence.meanEntropy { Self.add(entropy, to: &meanEntropy, step: Self.entropyStep) }
        if let margin = confidence.minMargin { Self.add(margin, to: &minMargin, step: 1.0 / Double(Self.bins)) }
    }

    static func add(_ value: Double, to bins: inout [Int], step: Double) {
        guard value.isFinite else { return }
        bins[min(bins.count - 1, max(0, Int(value / step)))] += 1
    }

    enum CodingKeys: String, CodingKey {
        case responses
        case minProbability = "min_probability", meanProbability = "mean_probability", meanEntropy = "mean_entropy", minMargin = "min_margin"
    }
}
