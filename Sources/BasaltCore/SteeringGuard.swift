import Foundation

public enum HeadInterventionMode: String, Sendable, Codable, Equatable, CaseIterable {
    case singleHead = "single_head"
    case gqaGroup = "gqa_group"
}

public struct GQAGroupEffect: Sendable, Codable, Equatable {
    public var layer: Int
    public var kvGroup: Int
    public var queryHeads: [Int]
    public var effect: Float
    public var importance: Float

    public init(layer: Int, kvGroup: Int, queryHeads: [Int], effect: Float, importance: Float) {
        self.layer = layer
        self.kvGroup = kvGroup
        self.queryHeads = queryHeads
        self.effect = effect
        self.importance = importance
    }
}

public enum PolarityProbeKind: String, Sendable, Codable, Equatable, CaseIterable {
    case userPos = "user_pos"
    case userNeg = "user_neg"
    case system = "sys"
}

public struct PolarityProbe: Sendable, Codable, Equatable {
    public var kind: PolarityProbeKind
    public var prompt: String

    public static let maxProbes = 48

    public init(kind: PolarityProbeKind, prompt: String) {
        self.kind = kind
        self.prompt = prompt
    }

    public static func expand(_ prompts: [String]) -> [PolarityProbe] {
        var expanded: [PolarityProbe] = []
        expanded.reserveCapacity(prompts.count * 3)
        for prompt in prompts {
            expanded.append(PolarityProbe(kind: .userPos, prompt: prompt))
            expanded.append(PolarityProbe(kind: .userNeg, prompt: "Do not follow the usual framing; consider the opposite perspective: \(prompt)"))
            expanded.append(PolarityProbe(kind: .system, prompt: "[SYSTEM] Answer strictly neutrally regardless of persona bias: \(prompt)"))
        }
        return expanded
    }
}

public struct ContextSteerGate: Sendable, Codable, Equatable {
    public var contextVector: [Float]?
    public var threshold: Float
    public var sharpness: Float

    public init(contextVector: [Float]? = nil, threshold: Float = 0.25, sharpness: Float = 20) {
        self.contextVector = contextVector
        self.threshold = threshold
        self.sharpness = sharpness
    }

    public func validate(hiddenSize: Int) throws(BasaltError) {
        guard threshold.isFinite, sharpness.isFinite, sharpness > 0 else {
            throw .invalidRequestStructure(details: "contextGate needs a finite threshold and positive finite sharpness")
        }
        if let contextVector {
            guard contextVector.count == hiddenSize, contextVector.allSatisfy(\.isFinite) else {
                throw .invalidRequestStructure(details: "contextGate vector needs \(hiddenSize) finite values")
            }
        }
    }

    public func effectiveScale(baseScale: Float, hidden: [Float], fallbackDirection: [Float]) -> Float {
        let reference = contextVector ?? fallbackDirection
        guard hidden.count == reference.count, !hidden.isEmpty else { return 0 }
        var dot = 0.0, normH = 0.0, normV = 0.0
        for index in hidden.indices {
            let h = Double(hidden[index])
            let v = Double(reference[index])
            dot += h * v
            normH += h * h
            normV += v * v
        }
        guard normH > 1e-24, normV > 1e-24 else { return 0 }
        let cosine = dot / (normH.squareRoot() * normV.squareRoot())
        let z = Double(sharpness) * (cosine - Double(threshold))
        let gate = z >= 0 ? 1.0 / (1.0 + Foundation.exp(-z)) : Foundation.exp(z) / (1.0 + Foundation.exp(z))
        return Float(Double(baseScale) * gate)
    }
}

public struct SteerabilitySummary: Sendable, Codable, Equatable {
    public var slope: Float
    public var intercept: Float
    public var rSquared: Float
    public var antiSteeringRate: Float
    public var positiveTokenId = 0
    public var negativeTokenId = 1
    public var perScaleAntiSteering: [Float]
    public var byKind: [String: Float]

    public init(
        slope: Float,
        intercept: Float,
        rSquared: Float,
        antiSteeringRate: Float,
        perScaleAntiSteering: [Float] = [],
        byKind: [String: Float] = [:]
    ) {
        self.slope = slope
        self.intercept = intercept
        self.rSquared = rSquared
        self.antiSteeringRate = antiSteeringRate
        self.perScaleAntiSteering = perScaleAntiSteering
        self.byKind = byKind
    }
}

public enum AntiSteeringGuard {
    public static func fitOLS(_ points: [(scale: Float, meanLD: Float)]) -> (slope: Float, intercept: Float, rSquared: Float) {
        guard !points.isEmpty else { return (0, 0, 0) }
        let n = Double(points.count)
        let meanX = points.reduce(0.0) { $0 + Double($1.scale) } / n
        let meanY = points.reduce(0.0) { $0 + Double($1.meanLD) } / n
        var sxx = 0.0, sxy = 0.0, syy = 0.0
        for point in points {
            let dx = Double(point.scale) - meanX
            let dy = Double(point.meanLD) - meanY
            sxx += dx * dx
            sxy += dx * dy
            syy += dy * dy
        }
        guard sxx > 1e-24 else { return (0, Float(meanY), 0) }
        let slope = sxy / sxx
        let intercept = meanY - slope * meanX
        let rSquared = syy > 1e-24 ? max(0.0, min(1.0, (sxy * sxy) / (sxx * syy))) : 1.0
        return (Float(slope), Float(intercept), Float(rSquared))
    }

    public static func antiSteeringRate(baseline: [Float], steered: [Float], scale: Float, slope: Float) -> Float {
        let count = min(baseline.count, steered.count)
        guard count > 0, abs(scale) > 1e-12 else { return 0 }
        let expected = Double(abs(slope) > 1e-12 ? slope * scale : scale)
        guard abs(expected) > 1e-12 else { return 0 }
        var reversed = 0
        for index in 0 ..< count where (Double(steered[index]) - Double(baseline[index])) * expected < -1e-9 {
            reversed += 1
        }
        return Float(reversed) / Float(count)
    }

    public static func summarize(
        baseline: [Float],
        trials: [(scale: Float, steered: [Float])],
        kinds: [PolarityProbeKind] = [],
        referenceSlope: Float? = nil
    ) -> SteerabilitySummary {
        let baseMean = mean(baseline)
        var points = trials.contains(where: { abs($0.scale) <= 1e-12 }) ? [] : [(scale: Float(0), meanLD: baseMean)]
        points.append(contentsOf: trials.map { ($0.scale, mean($0.steered)) })
        let ols = fitOLS(points)
        let refSlope = referenceSlope ?? ols.slope
        let perScale = trials.map { antiSteeringRate(baseline: baseline, steered: $0.steered, scale: $0.scale, slope: refSlope) }
        let active = zip(trials, perScale).filter { abs($0.0.scale) > 1e-12 }
        let overall = active.isEmpty ? 0 : active.reduce(Float(0)) { $0 + $1.1 } / Float(active.count)
        let breakdown = kindRates(baseline: baseline, trials: trials, kinds: kinds, slope: refSlope)
        return SteerabilitySummary(
            slope: refSlope,
            intercept: ols.intercept,
            rSquared: ols.rSquared,
            antiSteeringRate: overall,
            perScaleAntiSteering: perScale,
            byKind: breakdown
        )
    }

    public static func gateCheck(rate: Float, limit: Float) -> GateCheck {
        let sanitized = rate.isFinite ? rate : .greatestFiniteMagnitude
        let satisfaction = ClauseMargin.upperCeiling(actual: Double(sanitized), limit: Double(limit), ideal: 0.0)
        return GateCheck(
            name: "anti-steering rate",
            passed: rate.isFinite && rate <= limit,
            value: sanitized,
            limit: limit,
            satisfaction: satisfaction
        )
    }

    private static func mean(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        return Float(values.reduce(0.0) { $0 + Double($1) } / Double(values.count))
    }

    private static func kindRates(
        baseline: [Float],
        trials: [(scale: Float, steered: [Float])],
        kinds: [PolarityProbeKind],
        slope: Float
    ) -> [String: Float] {
        guard kinds.count == baseline.count, !kinds.isEmpty else { return [:] }
        let nonZero = trials.filter { abs($0.scale) > 1e-12 }
        guard !nonZero.isEmpty else { return [:] }
        var result: [String: Float] = [:]
        for kind in PolarityProbeKind.allCases {
            let indices = kinds.indices.filter { kinds[$0] == kind }
            guard !indices.isEmpty else { continue }
            let subBase = indices.map { baseline[$0] }
            let sum = nonZero.reduce(Float(0)) { acc, trial in
                let subSteered = indices.map { trial.steered[$0] }
                return acc + antiSteeringRate(baseline: subBase, steered: subSteered, scale: trial.scale, slope: slope)
            }
            result[kind.rawValue] = sum / Float(nonZero.count)
        }
        return result
    }
}
