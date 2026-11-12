import Synchronization

public struct ResponseConfidence: Codable, Sendable, Equatable {
    public let min: Double
    public let mean: Double
    public var meanEntropy: Double?
    public var minMargin: Double?

    public init(min: Double, mean: Double) {
        self.min = min
        self.mean = mean
    }

    enum CodingKeys: String, CodingKey {
        case min, mean
        case meanEntropy = "mean_entropy", minMargin = "min_margin"
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
        var merged = ResponseConfidence(min: Swift.min(min, other.min), mean: (mean * w1 + other.mean * w2) / (w1 + w2))
        if let a = meanEntropy, let b = other.meanEntropy { merged.meanEntropy = (a * w1 + b * w2) / (w1 + w2) } else { merged.meanEntropy = meanEntropy ?? other.meanEntropy }
        merged.minMargin = [minMargin, other.minMargin].compactMap { $0 }.min()
        return merged
    }
}

public final class ConfidenceProbe: Sendable {
    private let recorded = Mutex<[Double]>([])
    private let shape = Mutex<(entropy: [Double], margin: [Double])>(([], []))

    public init() {}

    public var values: [Double] { recorded.withLock { $0 } }

    public var summary: ResponseConfidence? {
        guard var summary = ResponseConfidence.summarize(recorded.withLock { $0 }) else { return nil }
        let (entropy, margin) = shape.withLock { $0 }
        let entropies = entropy.filter(\.isFinite)
        summary.meanEntropy = entropies.isEmpty ? nil : entropies.reduce(0, +) / Double(entropies.count)
        summary.minMargin = margin.filter(\.isFinite).min()
        return summary
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
        self.topK = topK
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
