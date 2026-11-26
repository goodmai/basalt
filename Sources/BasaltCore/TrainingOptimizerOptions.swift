import Foundation

public struct TrainingOptimizerOptions: Sendable, Equatable, Codable {
    public static let defaultWeightDecay: Float = 0.0
    public static let defaultQKClipTau: Float = 100.0
    public static let defaultQKClipAlpha: Float = 0.5
    public static let defaultSOAPPrecondFreq: Int = 10
    public static let defaultSOAPMaxPrecondDim: Int = 1024
    public static let defaultSOAPShampooBeta: Float = 0.95
    public static let defaultSOAPBeta1: Float = 0.95
    public static let defaultSOAPBeta2: Float = 0.95
    public static let defaultSOAPEps: Float = 1e-8

    public static let precondFreqRange = 1 ... 10_000
    public static let maxPrecondDimRange = 1 ... 4_096

    public var kind: TrainingOptimizerKind
    public var weightDecay: Float
    public var qkClipTau: Float
    public var qkClipAlpha: Float
    public var soapPrecondFreq: Int
    public var soapMaxPrecondDim: Int
    public var soapShampooBeta: Float
    public var soapBeta1: Float
    public var soapBeta2: Float
    public var soapEps: Float

    public init(
        kind: TrainingOptimizerKind = .adamw,
        weightDecay: Float = defaultWeightDecay
    ) {
        self.kind = kind
        self.weightDecay = weightDecay
        self.qkClipTau = Self.defaultQKClipTau
        self.qkClipAlpha = Self.defaultQKClipAlpha
        self.soapPrecondFreq = Self.defaultSOAPPrecondFreq
        self.soapMaxPrecondDim = Self.defaultSOAPMaxPrecondDim
        self.soapShampooBeta = Self.defaultSOAPShampooBeta
        self.soapBeta1 = Self.defaultSOAPBeta1
        self.soapBeta2 = Self.defaultSOAPBeta2
        self.soapEps = Self.defaultSOAPEps
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.kind = try c.decodeIfPresent(TrainingOptimizerKind.self, forKey: .kind) ?? .adamw
        self.weightDecay = try c.decodeIfPresent(Float.self, forKey: .weightDecay) ?? Self.defaultWeightDecay
        self.qkClipTau = try c.decodeIfPresent(Float.self, forKey: .qkClipTau) ?? Self.defaultQKClipTau
        self.qkClipAlpha = try c.decodeIfPresent(Float.self, forKey: .qkClipAlpha) ?? Self.defaultQKClipAlpha
        self.soapPrecondFreq = try c.decodeIfPresent(Int.self, forKey: .soapPrecondFreq) ?? Self.defaultSOAPPrecondFreq
        self.soapMaxPrecondDim = try c.decodeIfPresent(Int.self, forKey: .soapMaxPrecondDim) ?? Self.defaultSOAPMaxPrecondDim
        self.soapShampooBeta = try c.decodeIfPresent(Float.self, forKey: .soapShampooBeta) ?? Self.defaultSOAPShampooBeta
        self.soapBeta1 = try c.decodeIfPresent(Float.self, forKey: .soapBeta1) ?? Self.defaultSOAPBeta1
        self.soapBeta2 = try c.decodeIfPresent(Float.self, forKey: .soapBeta2) ?? Self.defaultSOAPBeta2
        self.soapEps = try c.decodeIfPresent(Float.self, forKey: .soapEps) ?? Self.defaultSOAPEps
    }

    public func validate(prefix: String = "optimizer") throws(BasaltError) {
        try requireNonNegative(weightDecay, name: "weightDecay", prefix: prefix)
        try requirePositive(qkClipTau, name: "qkClipTau", prefix: prefix)
        try requireUnitClosed(qkClipAlpha, name: "qkClipAlpha", prefix: prefix)
        try requireInt(soapPrecondFreq, in: Self.precondFreqRange, name: "soapPrecondFreq", prefix: prefix)
        try requireInt(soapMaxPrecondDim, in: Self.maxPrecondDimRange, name: "soapMaxPrecondDim", prefix: prefix)
        try requireDecay(soapShampooBeta, name: "soapShampooBeta", prefix: prefix)
        try requireDecay(soapBeta1, name: "soapBeta1", prefix: prefix)
        try requireDecay(soapBeta2, name: "soapBeta2", prefix: prefix)
        try requirePositive(soapEps, name: "soapEps", prefix: prefix)
    }

    public func validateAdapterCompatible(prefix: String = "optimizer") throws(BasaltError) {
        try validate(prefix: prefix)
        guard kind != .muonclip else {
            throw .invalidRequestStructure(
                details: "\(prefix): muonclip requires full-parameter training with trainable Q/K projections (or trainable QK-norms); adapter-based trainers freeze base weights — use muon instead"
            )
        }
    }

    private func requireNonNegative(_ value: Float, name: String, prefix: String) throws(BasaltError) {
        guard value.isFinite, value >= 0 else {
            throw .invalidRequestStructure(details: "\(prefix): \(name) must be finite and non-negative")
        }
    }

    private func requirePositive(_ value: Float, name: String, prefix: String) throws(BasaltError) {
        guard value.isFinite, value > 0 else {
            throw .invalidRequestStructure(details: "\(prefix): \(name) must be finite and positive")
        }
    }

    private func requireUnitClosed(_ value: Float, name: String, prefix: String) throws(BasaltError) {
        guard value.isFinite, value >= 0, value <= 1 else {
            throw .invalidRequestStructure(details: "\(prefix): \(name) must lie in [0, 1]")
        }
    }

    private func requireDecay(_ value: Float, name: String, prefix: String) throws(BasaltError) {
        guard value.isFinite, value >= 0, value < 1 else {
            throw .invalidRequestStructure(details: "\(prefix): \(name) must lie in [0, 1)")
        }
    }

    private func requireInt(_ value: Int, in range: ClosedRange<Int>, name: String, prefix: String) throws(BasaltError) {
        guard range.contains(value) else {
            throw .invalidRequestStructure(details: "\(prefix): \(name) must be in \(range)")
        }
    }
}

public enum MatrixOptimizerMath {
    public static func matchedRMSScale(rows: Int, cols: Int) -> Float {
        guard rows > 0, cols > 0 else { return 0 }
        return 0.2 * Float(max(rows, cols)).squareRoot()
    }

    public static func gqaSafeQKClipScales(
        headMaxLogits: [Float],
        tau: Float,
        alpha: Float = TrainingOptimizerOptions.defaultQKClipAlpha,
        kvHeads: Int
    ) -> (qScales: [Float], kScales: [Float]) {
        let qHeads = headMaxLogits.count
        let onesQ = Array(repeating: Float(1.0), count: qHeads)
        let onesK = Array(repeating: Float(1.0), count: max(0, kvHeads))
        guard qHeads > 0, kvHeads > 0, qHeads % kvHeads == 0, tau.isFinite, tau > 0 else {
            return (onesQ, onesK)
        }
        let clampedAlpha = Double(min(max(alpha.isFinite ? alpha : 0.5, 0.0), 1.0))
        let groupSize = qHeads / kvHeads
        let gammas: [Double] = headMaxLogits.map { logit in
            guard logit.isFinite, logit > tau else { return 1.0 }
            return min(1.0, Double(tau) / Double(logit))
        }
        var qScales = onesQ
        var kScales = onesK
        for kvIndex in 0 ..< kvHeads {
            let start = kvIndex * groupSize
            let end = start + groupSize
            var sharedGamma = 0.0
            for head in start ..< end {
                sharedGamma = max(sharedGamma, gammas[head])
            }
            let sigmaK = Foundation.pow(sharedGamma, 1.0 - clampedAlpha)
            let safeK = sigmaK > 0 ? sigmaK : 1.0
            kScales[kvIndex] = Float(min(1.0, safeK))
            for head in start ..< end {
                qScales[head] = Float(min(1.0, gammas[head] / safeK))
            }
        }
        return (qScales, kScales)
    }

    public static func soapAxisPlan(
        rows: Int,
        cols: Int,
        maxPrecondDim: Int
    ) -> (useLeft: Bool, useRight: Bool) {
        guard rows > 0, cols > 0, maxPrecondDim > 1 else {
            return (false, false)
        }
        return (useLeft: rows > 1 && rows <= maxPrecondDim, useRight: cols > 1 && cols <= maxPrecondDim)
    }

    public static func isSOAPEligible(shape: [Int], maxPrecondDim: Int) -> Bool {
        guard shape.count == 2 else { return false }
        let plan = soapAxisPlan(rows: shape[0], cols: shape[1], maxPrecondDim: maxPrecondDim)
        return plan.useLeft || plan.useRight
    }
}
