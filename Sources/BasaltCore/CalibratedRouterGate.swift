import Foundation

public enum RouterDomainClass: String, Sendable, Codable, CaseIterable, Equatable, Hashable {
    case general
    case coding
    case legal
    case medical
    case finance
    case security

    public var isRegulatedByDefault: Bool {
        switch self {
        case .legal, .medical, .finance, .security: true
        case .general, .coding: false
        }
    }
}

public enum RouterExecutionTier: String, Sendable, Codable, Equatable, CaseIterable {
    case fastLocal = "fast_local"
    case reasoning
    case symbolicVerified = "symbolic_verified"
}

public struct CalibratedRouterConfig: Sendable, Codable, Equatable {
    public var temperature: Double
    public var fastConfidenceThreshold: Double
    public var regulatedConfidenceThreshold: Double
    public var maxFastTTFTSeconds: Double
    public var regulatedDomains: Set<RouterDomainClass>

    public init(
        temperature: Double = 1.0,
        fastConfidenceThreshold: Double = 0.85,
        regulatedConfidenceThreshold: Double = 0.92,
        maxFastTTFTSeconds: Double = 0.8,
        regulatedDomains: Set<RouterDomainClass> = Set(RouterDomainClass.allCases.filter(\.isRegulatedByDefault))
    ) {
        self.temperature = temperature
        self.fastConfidenceThreshold = fastConfidenceThreshold
        self.regulatedConfidenceThreshold = regulatedConfidenceThreshold
        self.maxFastTTFTSeconds = maxFastTTFTSeconds
        self.regulatedDomains = regulatedDomains
    }

    public func validate() throws(BasaltError) {
        guard temperature.isFinite, SystemOneCalibrationCore.temperatureBounds.contains(temperature) else {
            throw .invalidRequestStructure(details: "calibrated router: temperature must lie in \(SystemOneCalibrationCore.temperatureBounds)")
        }
        guard fastConfidenceThreshold.isFinite, (0.0 ... 1.0).contains(fastConfidenceThreshold) else {
            throw .invalidRequestStructure(details: "calibrated router: fastConfidenceThreshold must lie in 0...1")
        }
        guard regulatedConfidenceThreshold.isFinite, (0.0 ... 1.0).contains(regulatedConfidenceThreshold) else {
            throw .invalidRequestStructure(details: "calibrated router: regulatedConfidenceThreshold must lie in 0...1")
        }
        guard maxFastTTFTSeconds.isFinite, maxFastTTFTSeconds > 0 else {
            throw .invalidRequestStructure(details: "calibrated router: maxFastTTFTSeconds must be positive")
        }
    }

    enum CodingKeys: String, CodingKey {
        case temperature
        case fastConfidenceThreshold = "fast_confidence_threshold"
        case regulatedConfidenceThreshold = "regulated_confidence_threshold"
        case maxFastTTFTSeconds = "max_fast_ttft_seconds"
        case regulatedDomains = "regulated_domains"
    }
}

public struct CalibratedRouterDecision: Sendable, Codable, Equatable {
    public var domain: RouterDomainClass
    public var calibratedConfidence: Double
    public var calibratedProbabilities: [Double]
    public var tier: RouterExecutionTier
    public var escalatedForRegulation: Bool
    public var temperature: Double

    public init(
        domain: RouterDomainClass,
        calibratedConfidence: Double,
        calibratedProbabilities: [Double],
        tier: RouterExecutionTier,
        escalatedForRegulation: Bool,
        temperature: Double
    ) {
        self.domain = domain
        self.calibratedConfidence = calibratedConfidence
        self.calibratedProbabilities = calibratedProbabilities
        self.tier = tier
        self.escalatedForRegulation = escalatedForRegulation
        self.temperature = temperature
    }

    enum CodingKeys: String, CodingKey {
        case domain, tier, temperature
        case calibratedConfidence = "calibrated_confidence"
        case calibratedProbabilities = "calibrated_probabilities"
        case escalatedForRegulation = "escalated_for_regulation"
    }
}

public struct CalibratedRouterGate: Sendable, Equatable {
    public var config: CalibratedRouterConfig

    public init(config: CalibratedRouterConfig = CalibratedRouterConfig()) {
        self.config = config
    }

    public static func calibrate(
        samples: [CalibrationSample],
        holdoutFraction: Double = 0.2,
        baseConfig: CalibratedRouterConfig = CalibratedRouterConfig()
    ) -> (gate: CalibratedRouterGate, fit: CalibrationFit) {
        let fit = SystemOneCalibrationCore.fit(samples, holdoutFraction: holdoutFraction)
        var updated = baseConfig
        updated.temperature = fit.readout.temperature
        return (CalibratedRouterGate(config: updated), fit)
    }

    public func route(
        rawDomainLogits: [Double],
        domains: [RouterDomainClass] = RouterDomainClass.allCases,
        estimatedFastTTFTSeconds: Double? = nil
    ) throws(BasaltError) -> CalibratedRouterDecision {
        try config.validate()
        guard !rawDomainLogits.isEmpty, rawDomainLogits.count == domains.count else {
            throw .invalidRequestStructure(details: "calibrated router: logits count must match domains count")
        }
        guard rawDomainLogits.allSatisfy(\.isFinite) else {
            throw .invalidRequestStructure(details: "calibrated router: logits must be finite")
        }
        let readout = SystemOneReadout(temperature: config.temperature)
        let probs = SystemOneCalibrationCore.probabilities(kind: .choice, rawScores: rawDomainLogits, readout: readout)
        var bestIndex = 0
        var bestProb = probs[0]
        for idx in 1 ..< probs.count where probs[idx] > bestProb {
            bestProb = probs[idx]
            bestIndex = idx
        }
        let chosen = domains[bestIndex]
        let isRegulated = config.regulatedDomains.contains(chosen)
        let fastWithinBudget = estimatedFastTTFTSeconds.map { $0 <= config.maxFastTTFTSeconds } ?? true
        let tier: RouterExecutionTier
        let escalatedForReg: Bool
        if isRegulated {
            if bestProb >= config.regulatedConfidenceThreshold {
                tier = .reasoning
                escalatedForReg = true
            } else {
                tier = .symbolicVerified
                escalatedForReg = true
            }
        } else if bestProb >= config.fastConfidenceThreshold && fastWithinBudget {
            tier = .fastLocal
            escalatedForReg = false
        } else {
            tier = .reasoning
            escalatedForReg = false
        }
        return CalibratedRouterDecision(
            domain: chosen,
            calibratedConfidence: bestProb,
            calibratedProbabilities: probs,
            tier: tier,
            escalatedForRegulation: escalatedForReg,
            temperature: config.temperature
        )
    }

    public func hybridCandidate(
        for decision: CalibratedRouterDecision,
        localLatencySeconds: Double,
        cost: Double = 0
    ) -> HybridCandidate {
        let capability: Double = switch decision.tier {
        case .fastLocal: min(1.0, max(0.0, decision.calibratedConfidence))
        case .reasoning: min(1.0, max(0.5, decision.calibratedConfidence))
        case .symbolicVerified: decision.escalatedForRegulation ? 0.5 : decision.calibratedConfidence
        }
        let degradation = decision.tier == .fastLocal && localLatencySeconds > config.maxFastTTFTSeconds ? 0.25 : 0.0
        return HybridCandidate(
            capability: capability,
            degradation: degradation,
            latencySeconds: max(0.0, localLatencySeconds),
            cost: max(0.0, cost)
        )
    }
}
