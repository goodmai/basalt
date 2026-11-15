import Foundation

public enum AdapterMethod: Sendable, Codable, Equatable {
    case lora(rank: Int, scale: Float = 20.0)
    case phm(hypercomplexDim: Int, factorRank: Int = 1, sharedBasis: Bool = true, scale: Float = 20.0)

    public static let allowedRanks = 1...64
    public static let allowedHypercomplexDims = 1...64

    public var scale: Float {
        switch self {
        case .lora(_, let scale), .phm(_, _, _, let scale):
            scale
        }
    }

    public func parameterCount(inputDim: Int, outputDim: Int, layerCount: Int = 1) -> Int {
        guard layerCount > 0, inputDim > 0, outputDim > 0 else { return 0 }
        switch self {
        case .lora(let rank, _):
            guard rank > 0 else { return 0 }
            return layerCount * rank * (inputDim + outputDim)
        case .phm(let hypercomplexDim, let factorRank, let sharedBasis, _):
            guard hypercomplexDim > 0, factorRank > 0 else { return 0 }
            let basisCube = hypercomplexDim * hypercomplexDim * hypercomplexDim
            let basisParams = sharedBasis ? basisCube : layerCount * basisCube
            return basisParams + layerCount * factorRank * (inputDim + outputDim)
        }
    }

    public func parameterCount(projections: [(inputDim: Int, outputDim: Int)]) -> Int {
        let valid = projections.filter { $0.inputDim > 0 && $0.outputDim > 0 }
        guard !valid.isEmpty else { return 0 }
        switch self {
        case .lora(let rank, _):
            guard rank > 0 else { return 0 }
            return valid.reduce(0) { $0 + rank * ($1.inputDim + $1.outputDim) }
        case .phm(let hypercomplexDim, let factorRank, let sharedBasis, _):
            guard hypercomplexDim > 0, factorRank > 0 else { return 0 }
            let basisCube = hypercomplexDim * hypercomplexDim * hypercomplexDim
            let basisParams = sharedBasis ? basisCube : valid.count * basisCube
            return basisParams + valid.reduce(0) { $0 + factorRank * ($1.inputDim + $1.outputDim) }
        }
    }

    public func matchingPHM(
        hypercomplexDim: Int,
        sharedBasis: Bool = true,
        inputDim: Int,
        outputDim: Int,
        layerCount: Int = 1
    ) throws(BasaltError) -> AdapterMethod {
        try validate(prefix: "adapter")
        guard layerCount > 0 else {
            throw .invalidRequestStructure(details: "adapter: layerCount (\(layerCount)) must be positive")
        }
        let budget = parameterCount(inputDim: inputDim, outputDim: outputDim, layerCount: layerCount)
        let basisCube = hypercomplexDim * hypercomplexDim * hypercomplexDim
        let basisCost = sharedBasis ? basisCube : layerCount * basisCube
        let perRankCost = max(1, layerCount * (inputDim + outputDim))
        let remaining = max(0, budget - basisCost)
        let matchedRank = min(Self.allowedRanks.upperBound, max(1, (remaining + perRankCost / 2) / perRankCost))
        let candidate = AdapterMethod.phm(
            hypercomplexDim: hypercomplexDim,
            factorRank: matchedRank,
            sharedBasis: sharedBasis,
            scale: scale
        )
        try candidate.validate(inputDim: inputDim, outputDim: outputDim, prefix: "adapter")
        return candidate
    }

    public func validate(prefix: String = "adapter") throws(BasaltError) {
        switch self {
        case .lora(let rank, let scale):
            guard Self.allowedRanks.contains(rank) else {
                throw .invalidRequestStructure(details: "\(prefix): rank must be in \(Self.allowedRanks)")
            }
            guard scale.isFinite else {
                throw .invalidRequestStructure(details: "\(prefix): scale must be finite")
            }
        case .phm(let hypercomplexDim, let factorRank, _, let scale):
            guard Self.allowedHypercomplexDims.contains(hypercomplexDim) else {
                throw .invalidRequestStructure(details: "\(prefix): hypercomplexDim must be in \(Self.allowedHypercomplexDims)")
            }
            guard Self.allowedRanks.contains(factorRank) else {
                throw .invalidRequestStructure(details: "\(prefix): factorRank must be in \(Self.allowedRanks)")
            }
            guard scale.isFinite else {
                throw .invalidRequestStructure(details: "\(prefix): scale must be finite")
            }
        }
    }

    public func validate(inputDim: Int, outputDim: Int, prefix: String = "adapter") throws(BasaltError) {
        try validate(prefix: prefix)
        guard inputDim > 0, outputDim > 0 else {
            throw .invalidRequestStructure(details: "\(prefix): inputDim (\(inputDim)) and outputDim (\(outputDim)) must be positive")
        }
        if case .phm(let hypercomplexDim, _, _, _) = self {
            guard inputDim % hypercomplexDim == 0, outputDim % hypercomplexDim == 0 else {
                throw .invalidRequestStructure(
                    details: "\(prefix): inputDim (\(inputDim)) and outputDim (\(outputDim)) must be divisible by hypercomplexDim (\(hypercomplexDim))"
                )
            }
        }
    }
}

public struct AdapterSpec: Sendable, Codable, Equatable {
    public var method: AdapterMethod
    public var layers: [Int]?
    public var targetSuffixes: [String]?

    public init(method: AdapterMethod, layers: [Int]? = nil, targetSuffixes: [String]? = nil) {
        self.method = method
        self.layers = layers
        self.targetSuffixes = targetSuffixes
    }

    public func validate(prefix: String = "adapter") throws(BasaltError) {
        try method.validate(prefix: prefix)
        if let layers, layers.isEmpty || layers.contains(where: { $0 < 0 }) {
            throw .invalidRequestStructure(details: "\(prefix): layers must be a non-empty list of non-negative layer indices")
        }
        if let targetSuffixes, targetSuffixes.isEmpty || targetSuffixes.contains(where: \.isEmpty) {
            throw .invalidRequestStructure(details: "\(prefix): targetSuffixes must be non-empty strings")
        }
    }
}
