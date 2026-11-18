import Synchronization

public struct ResponseConfidence: Codable, Sendable, Equatable {
    public let min: Double
    public let mean: Double
    public var meanEntropy: Double?
    public var minMargin: Double?
    public var highConfidenceDraftRejections: Int
    public var totalHighConfidenceDrafts: Int

    public init(
        min: Double,
        mean: Double,
        highConfidenceDraftRejections: Int = 0,
        totalHighConfidenceDrafts: Int = 0
    ) {
        self.min = min
        self.mean = mean
        self.highConfidenceDraftRejections = max(0, highConfidenceDraftRejections)
        self.totalHighConfidenceDrafts = max(0, totalHighConfidenceDrafts)
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        min = try container.decode(Double.self, forKey: .min)
        mean = try container.decode(Double.self, forKey: .mean)
        meanEntropy = try container.decodeIfPresent(Double.self, forKey: .meanEntropy)
        minMargin = try container.decodeIfPresent(Double.self, forKey: .minMargin)
        highConfidenceDraftRejections = max(0, try container.decodeIfPresent(Int.self, forKey: .highConfidenceDraftRejections) ?? 0)
        totalHighConfidenceDrafts = max(0, try container.decodeIfPresent(Int.self, forKey: .totalHighConfidenceDrafts) ?? 0)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(min, forKey: .min)
        try container.encode(mean, forKey: .mean)
        try container.encodeIfPresent(meanEntropy, forKey: .meanEntropy)
        try container.encodeIfPresent(minMargin, forKey: .minMargin)
        if highConfidenceDraftRejections > 0 || totalHighConfidenceDrafts > 0 {
            try container.encode(highConfidenceDraftRejections, forKey: .highConfidenceDraftRejections)
            try container.encode(totalHighConfidenceDrafts, forKey: .totalHighConfidenceDrafts)
        }
    }

    enum CodingKeys: String, CodingKey {
        case min, mean
        case meanEntropy = "mean_entropy", minMargin = "min_margin"
        case highConfidenceDraftRejections = "high_confidence_draft_rejections"
        case totalHighConfidenceDrafts = "total_high_confidence_drafts"
    }

    public var draftRefutationRate: Double {
        totalHighConfidenceDrafts > 0
            ? EvidencePair.clamp(Double(highConfidenceDraftRejections) / Double(totalHighConfidenceDrafts))
            : 0.0
    }

    public func evidencePair(entropyCeiling: Double = 4.0) -> EvidencePair {
        let probSupport = EvidencePair.clamp(mean)
        let ceiling = entropyCeiling.isFinite && entropyCeiling > 0 ? entropyCeiling : 4.0
        let support: Double
        if let entropy = meanEntropy, entropy.isFinite {
            let entropySupport = EvidencePair.clamp(1.0 - Swift.max(0.0, entropy) / ceiling)
            support = 0.5 * (probSupport + entropySupport)
        } else {
            support = probSupport
        }
        let refutation = totalHighConfidenceDrafts > 0
            ? draftRefutationRate
            : Swift.max(0.0, probSupport - EvidencePair.clamp(min))
        return EvidencePair(support: support, refutation: refutation)
    }

    public func belnapState(thresholds: EpistemicThresholds = .default) -> BelnapState {
        BelnapState.project(
            evidencePair(entropyCeiling: thresholds.entropyCeiling),
            thresholds: thresholds
        )
    }

    public static func summarize(_ values: [Double]) -> ResponseConfidence? {
        let valid = values.filter { $0.isFinite && (0.0...1.0).contains($0) }
        guard let minimum = valid.min() else { return nil }
        let sum = valid.reduce(0.0, +)
        return ResponseConfidence(min: minimum, mean: sum / Double(valid.count))
    }

    public func merged(with other: ResponseConfidence, weightSelf: Int, weightOther: Int) -> ResponseConfidence {
        let w1 = Double(max(1, weightSelf))
        let w2 = Double(max(1, weightOther))
        var merged = ResponseConfidence(
            min: Swift.min(min, other.min),
            mean: (mean * w1 + other.mean * w2) / (w1 + w2),
            highConfidenceDraftRejections: highConfidenceDraftRejections + other.highConfidenceDraftRejections,
            totalHighConfidenceDrafts: totalHighConfidenceDrafts + other.totalHighConfidenceDrafts
        )
        if let a = meanEntropy, let b = other.meanEntropy { merged.meanEntropy = (a * w1 + b * w2) / (w1 + w2) } else { merged.meanEntropy = meanEntropy ?? other.meanEntropy }
        merged.minMargin = [minMargin, other.minMargin].compactMap { $0 }.min()
        return merged
    }
}

public final class ConfidenceProbe: Sendable {
    private let recorded = Mutex<[Double]>([])
    private let shape = Mutex<(entropy: [Double], margin: [Double])>(([], []))
    private let drafts = Mutex<(rejections: Int, total: Int)>((0, 0))

    public init() {}

    public var values: [Double] { recorded.withLock { $0 } }

    public var draftCounts: (rejections: Int, total: Int) { drafts.withLock { $0 } }

    public var summary: ResponseConfidence? {
        guard var summary = ResponseConfidence.summarize(recorded.withLock { $0 }) else { return nil }
        let (entropy, margin) = shape.withLock { $0 }
        let entropies = entropy.filter(\.isFinite)
        summary.meanEntropy = entropies.isEmpty ? nil : entropies.reduce(0, +) / Double(entropies.count)
        summary.minMargin = margin.filter(\.isFinite).min()
        let counts = drafts.withLock { $0 }
        summary.highConfidenceDraftRejections = counts.rejections
        summary.totalHighConfidenceDrafts = counts.total
        return summary
    }

    public func recordHighConfidenceDrafts(rejected: Int, total: Int) {
        guard total > 0 else { return }
        let validTotal = max(0, total)
        let validRejected = Swift.min(validTotal, max(0, rejected))
        drafts.withLock {
            $0.rejections += validRejected
            $0.total += validTotal
        }
    }

    public func record(_ maxProbability: Double, entropy: Double, margin: Double) {
        record(maxProbability)
        shape.withLock { $0.entropy.append(entropy); $0.margin.append(margin) }
    }

    public func record(_ maxProbability: Double) {
        guard maxProbability.isFinite, (0.0...1.0).contains(maxProbability) else { return }
        recorded.withLock { $0.append(maxProbability) }
    }
}

public final class LogitProbe: Sendable {
    public struct Step: Sendable, Equatable {
        public var token: Int
        public var ids: [Int]
        public var logprobs: [Float]

        public init(token: Int, ids: [Int], logprobs: [Float]) {
            self.token = token
            self.ids = ids
            self.logprobs = logprobs
        }
    }

    public let topK: Int
    private let recorded = Mutex<[Step]>([])

    public init(topK: Int) {
        self.topK = max(0, topK)
    }

    public var steps: [Step] { recorded.withLock { $0 } }

    public func record(ids: [Int], logprobs: [Float]) {
        recorded.withLock { $0.append(Step(token: -1, ids: ids, logprobs: logprobs)) }
    }

    public func sampled(_ token: Int) {
        recorded.withLock { steps in
            guard let last = steps.indices.last else { return }
            steps[last].token = token
        }
    }
}
