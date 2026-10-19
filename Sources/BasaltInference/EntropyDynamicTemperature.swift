import MLX
import MLXLMCommon

public struct EntropyDynamicTemperature: LogitProcessor {
    public static let defaultN: Float = 0.8
    public static let defaultTheta: Float = 1.5
    public static let defaultGammaRep: Float = 0.5
    public static let temperatureRange: ClosedRange<Float> = 0.05...2.0

    static let probeTokens = 16
    static let repeatThreshold = 4
    static let minTokens = 64
    static let retainTokens = 512
    static let readbackLag = 2

    let baseTemperature: Float
    let n: Float
    let theta: Float
    let gammaRep: Float
    let topP: Float?
    let minP: Float?

    private(set) var history: [Int] = []
    private var awaitingReadback: [MLXArray] = []
    private(set) var repetition: Float = 0

    public init(
        baseTemperature: Float,
        n: Float = defaultN, theta: Float = defaultTheta, gammaRep: Float = defaultGammaRep,
        topP: Float? = nil, minP: Float? = nil
    ) {
        self.baseTemperature = baseTemperature
        self.topP = topP.flatMap { $0 > 0 && $0 < 1 ? $0 : nil }
        self.minP = minP.flatMap { $0 > 0 ? $0 : nil }
        self.n = n
        self.theta = theta
        self.gammaRep = gammaRep
    }

    static func entropy(_ logits: MLXArray) -> MLXArray {
        let z = logits.asType(.float32)
        let logp = z - logSumExp(z, axis: -1, keepDims: true)
        let p = exp(logp)
        let h = -sum(MLX.where(p .> 0, p * logp, MLXArray(Float(0))), axis: -1, keepDims: true)
        return maximum(h, MLXArray(Float(0)))
    }

    func temperature(entropy: MLXArray, repetition: Float) -> MLXArray {
        let t = baseTemperature * pow(MLXArray(n), theta / entropy) + gammaRep * repetition
        return clip(t, min: min(Self.temperatureRange.lowerBound, baseTemperature), max: Self.temperatureRange.upperBound)
    }

    public mutating func prompt(_ prompt: MLXArray) {
    }

    public func process(logits: MLXArray) -> MLXArray {
        let t = temperature(entropy: Self.entropy(logits), repetition: repetition)
        return filteredByTopPMinP(logits) * (baseTemperature / t)
    }

    func filteredByTopPMinP(_ logits: MLXArray) -> MLXArray {
        guard topP != nil || minP != nil else { return logits }
        let z = logits.asType(.float32)
        let logp = z - logSumExp(z, axis: -1, keepDims: true)
        let negInf = MLXArray(-Float.infinity)
        var keep = logp .> negInf
        if let topP {
            let order = argSort(logp, axis: -1)
            let cumulative = cumsum(exp(takeAlong(logp, order, axis: -1)), axis: -1)
            let keepSorted = cumulative .> (1 - topP)
            keep = keep .&& putAlong(keep, order, values: keepSorted, axis: -1)
        }
        if let minP {
            keep = keep .&& (logp .>= logp.max(axis: -1, keepDims: true) + log(MLXArray(minP)))
        }
        return MLX.where(keep, z, negInf)
    }

    public mutating func didSample(token: MLXArray) {
        awaitingReadback.append(token)
        while awaitingReadback.count > Self.readbackLag {
            history.append(awaitingReadback.removeFirst().item(Int.self))
        }
        if history.count > Self.retainTokens {
            history.removeFirst(history.count - Self.retainTokens)
        }
        repetition = history.count < Self.minTokens ? 0 : Float(
            RepetitionGuard.score(tokens: history, probeSize: Self.probeTokens, threshold: Self.repeatThreshold))
    }
}
