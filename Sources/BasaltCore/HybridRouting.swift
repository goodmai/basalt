import Foundation

public enum CloudFallbackReason: String, Sendable, Equatable {
    case memory
    case contextDegradation
    case capability
}

public struct CloudAssessment: Sendable, Equatable {
    public let capabilityGap: Double
    public let estimatedCost: Double
    public let remainingBudget: Double?
    public let latencySeconds: Double

    public init(capabilityGap: Double, estimatedCost: Double, remainingBudget: Double?, latencySeconds: Double) {
        self.capabilityGap = capabilityGap
        self.estimatedCost = estimatedCost
        self.remainingBudget = remainingBudget
        self.latencySeconds = latencySeconds
    }
}

public protocol CloudFallback: Sendable {
    func assess(_ request: GenerationRequest) async -> CloudAssessment?
    func generate(_ request: GenerationRequest) async throws -> GenerationResponse
    func generateStream(_ request: GenerationRequest) async throws -> AsyncThrowingStream<StreamChunk, Error>
}

public struct HybridCandidate: Sendable, Equatable {
    public let capability: Double
    public let degradation: Double
    public let latencySeconds: Double
    public let cost: Double

    public init(capability: Double, degradation: Double, latencySeconds: Double, cost: Double) {
        self.capability = capability
        self.degradation = degradation
        self.latencySeconds = latencySeconds
        self.cost = cost
    }
}

public struct HybridArbiter: Sendable, Equatable {
    public let capabilityWeight: Double
    public let latencyWeight: Double
    public let costWeight: Double
    public let maxLatencySeconds: Double
    public let localContextLimit: Int?

    public init(
        capabilityWeight: Double = 1, latencyWeight: Double = 0, costWeight: Double = 1,
        maxLatencySeconds: Double = 300, localContextLimit: Int? = nil
    ) {
        self.capabilityWeight = capabilityWeight
        self.latencyWeight = latencyWeight
        self.costWeight = costWeight
        self.maxLatencySeconds = maxLatencySeconds
        self.localContextLimit = localContextLimit
    }

    public func utility(_ candidate: HybridCandidate, remainingBudget: Double?) -> Double {
        let costShare = remainingBudget.map { $0 > 0 ? candidate.cost / $0 : .infinity } ?? candidate.cost
        return capabilityWeight * candidate.capability * (1 - candidate.degradation)
            - latencyWeight * candidate.latencySeconds / maxLatencySeconds
            - costWeight * costShare
    }

    public func localDegradation(contextTokens: Int) -> Double {
        guard let localContextLimit else { return 0 }
        return contextTokens > localContextLimit ? 1 : 0
    }

    public func cloudReason(
        local: HybridCandidate, localFits: Bool, cloud: HybridCandidate, remainingBudget: Double?
    ) -> CloudFallbackReason? {
        if let remainingBudget, cloud.cost > remainingBudget { return nil }
        if !localFits { return .memory }
        guard utility(cloud, remainingBudget: remainingBudget) > utility(local, remainingBudget: remainingBudget) else {
            return nil
        }
        return local.degradation > 0 ? .contextDegradation : .capability
    }
}
