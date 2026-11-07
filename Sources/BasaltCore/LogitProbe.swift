import Synchronization

public struct ResponseConfidence: Codable, Sendable, Equatable {
    public let min: Double
    public let mean: Double

    public init(min: Double, mean: Double) {
        self.min = min
        self.mean = mean
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
        return ResponseConfidence(
            min: Swift.min(min, other.min),
            mean: (mean * w1 + other.mean * w2) / (w1 + w2)
        )
    }
}

public final class ConfidenceProbe: Sendable {
    private let recorded = Mutex<[Double]>([])

    public init() {}

    public var values: [Double] { recorded.withLock { $0 } }

    public var summary: ResponseConfidence? {
        ResponseConfidence.summarize(recorded.withLock { $0 })
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
