import Synchronization

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
