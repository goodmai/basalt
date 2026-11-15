import Foundation

public enum OrthogonalParameterization: String, Sendable, Codable, CaseIterable, Equatable {
    case matrixExp
    case cayley
}

public enum AdapterFamily: Sendable, Equatable {
    case lora(scale: Float = 20.0)
    case phm(hypercomplexDim: Int, sharedBasis: Bool = true, scale: Float = 20.0)
    case oft(parameterization: OrthogonalParameterization = .cayley, blockShared: Bool = false)
}

public enum AdapterMethod: Sendable, Codable, Equatable {
    case lora(rank: Int, scale: Float = 20.0)
    case phm(hypercomplexDim: Int, factorRank: Int = 1, sharedBasis: Bool = true, scale: Float = 20.0)
    case oft(blockSize: Int, parameterization: OrthogonalParameterization = .cayley, blockShared: Bool = false)

    public static let allowedRanks = 1...64
    public static let allowedHypercomplexDims = 1...64
    public static let allowedBlockSizes = 2...256

    public var scale: Float {
        switch self {
        case .lora(_, let scale), .phm(_, _, _, let scale):
            scale
        case .oft:
            1.0
        }
    }

    public static func oftLayerParameterCount(inputDim: Int, blockSize: Int, blockShared: Bool) -> Int {
        guard inputDim > 0, blockSize >= 2, inputDim.isMultiple(of: blockSize) else { return 0 }
        let freePerBlock = blockSize * (blockSize - 1) / 2
        let blockCount = blockShared ? 1 : inputDim / blockSize
        return blockCount * freePerBlock
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
        case .oft(let blockSize, _, let blockShared):
            return layerCount * Self.oftLayerParameterCount(
                inputDim: inputDim,
                blockSize: blockSize,
                blockShared: blockShared
            )
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
        case .oft(let blockSize, _, let blockShared):
            return valid.reduce(0) {
                $0 + Self.oftLayerParameterCount(
                    inputDim: $1.inputDim,
                    blockSize: blockSize,
                    blockShared: blockShared
                )
            }
        }
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
        case .oft(let blockSize, _, _):
            guard Self.allowedBlockSizes.contains(blockSize) else {
                throw .invalidRequestStructure(details: "\(prefix): blockSize must be in \(Self.allowedBlockSizes)")
            }
        }
    }

    public func validate(inputDim: Int, outputDim: Int, prefix: String = "adapter") throws(BasaltError) {
        try validate(prefix: prefix)
        guard inputDim > 0, outputDim > 0 else {
            throw .invalidRequestStructure(details: "\(prefix): inputDim (\(inputDim)) and outputDim (\(outputDim)) must be positive")
        }
        switch self {
        case .lora:
            return
        case .phm(let hypercomplexDim, _, _, _):
            guard inputDim % hypercomplexDim == 0, outputDim % hypercomplexDim == 0 else {
                throw .invalidRequestStructure(
                    details: "\(prefix): inputDim (\(inputDim)) and outputDim (\(outputDim)) must be divisible by hypercomplexDim (\(hypercomplexDim))"
                )
            }
        case .oft(let blockSize, _, _):
            guard inputDim.isMultiple(of: blockSize) else {
                throw .invalidRequestStructure(
                    details: "\(prefix): inputDim (\(inputDim)) must be divisible by blockSize (\(blockSize))"
                )
            }
        }
    }
}

extension AdapterMethod {
    public func matchingPHM(
        hypercomplexDim: Int,
        sharedBasis: Bool = true,
        inputDim: Int,
        outputDim: Int,
        layerCount: Int = 1
    ) throws(BasaltError) -> AdapterMethod {
        guard layerCount > 0 else {
            throw .invalidRequestStructure(details: "adapter: layerCount (\(layerCount)) must be positive")
        }
        let projections = Array(repeating: (inputDim: inputDim, outputDim: outputDim), count: layerCount)
        return try matchingPHM(hypercomplexDim: hypercomplexDim, sharedBasis: sharedBasis, scale: scale, projections: projections)
    }

    public func matchingPHM(
        hypercomplexDim: Int,
        sharedBasis: Bool = true,
        scale targetScale: Float? = nil,
        projections: [(inputDim: Int, outputDim: Int)]
    ) throws(BasaltError) -> AdapterMethod {
        try validate(prefix: "adapter")
        let valid = try Self.validatedProjections(projections)
        let budget = parameterCount(projections: valid)
        let basisCube = hypercomplexDim * hypercomplexDim * hypercomplexDim
        let basisCost = sharedBasis ? basisCube : valid.count * basisCube
        let perRankCost = max(1, valid.reduce(0) { $0 + $1.inputDim + $1.outputDim })
        let remaining = max(0, budget - basisCost)
        let matchedRank = min(Self.allowedRanks.upperBound, max(1, (remaining + perRankCost / 2) / perRankCost))
        let candidate = AdapterMethod.phm(
            hypercomplexDim: hypercomplexDim,
            factorRank: matchedRank,
            sharedBasis: sharedBasis,
            scale: targetScale ?? scale
        )
        for proj in valid {
            try candidate.validate(inputDim: proj.inputDim, outputDim: proj.outputDim, prefix: "adapter")
        }
        return candidate
    }

    public func matchingOFT(
        parameterization: OrthogonalParameterization = .cayley,
        blockShared: Bool = false,
        projections: [(inputDim: Int, outputDim: Int)]
    ) throws(BasaltError) -> AdapterMethod {
        try validate(prefix: "adapter")
        let valid = try Self.validatedProjections(projections)
        let budget = parameterCount(projections: valid)
        let candidates = Self.allowedBlockSizes.filter { b in
            valid.allSatisfy { $0.inputDim.isMultiple(of: b) }
        }
        guard let bestBlock = candidates.min(by: { lhs, rhs in
            let lhsGap = abs(AdapterMethod.oft(blockSize: lhs, parameterization: parameterization, blockShared: blockShared).parameterCount(projections: valid) - budget)
            let rhsGap = abs(AdapterMethod.oft(blockSize: rhs, parameterization: parameterization, blockShared: blockShared).parameterCount(projections: valid) - budget)
            return lhsGap == rhsGap ? lhs < rhs : lhsGap < rhsGap
        }) else {
            throw .invalidRequestStructure(details: "adapter: no OFT blockSize in \(Self.allowedBlockSizes) divides all target input dimensions")
        }
        let candidate = AdapterMethod.oft(blockSize: bestBlock, parameterization: parameterization, blockShared: blockShared)
        for proj in valid {
            try candidate.validate(inputDim: proj.inputDim, outputDim: proj.outputDim, prefix: "adapter")
        }
        return candidate
    }

    public func matchingOFT(
        parameterization: OrthogonalParameterization = .cayley,
        blockShared: Bool = false,
        inputDim: Int,
        outputDim: Int,
        layerCount: Int = 1
    ) throws(BasaltError) -> AdapterMethod {
        guard layerCount > 0 else {
            throw .invalidRequestStructure(details: "adapter: layerCount (\(layerCount)) must be positive")
        }
        let projections = Array(repeating: (inputDim: inputDim, outputDim: outputDim), count: layerCount)
        return try matchingOFT(parameterization: parameterization, blockShared: blockShared, projections: projections)
    }

    public func matchingLoRA(
        scale targetScale: Float? = nil,
        projections: [(inputDim: Int, outputDim: Int)]
    ) throws(BasaltError) -> AdapterMethod {
        try validate(prefix: "adapter")
        let valid = try Self.validatedProjections(projections)
        let budget = parameterCount(projections: valid)
        let perRankCost = max(1, valid.reduce(0) { $0 + $1.inputDim + $1.outputDim })
        let matchedRank = min(Self.allowedRanks.upperBound, max(1, (budget + perRankCost / 2) / perRankCost))
        let candidate = AdapterMethod.lora(rank: matchedRank, scale: targetScale ?? scale)
        try candidate.validate(prefix: "adapter")
        return candidate
    }

    public func matchingBudget(
        to family: AdapterFamily,
        projections: [(inputDim: Int, outputDim: Int)]
    ) throws(BasaltError) -> AdapterMethod {
        switch family {
        case .lora(let targetScale):
            return try matchingLoRA(scale: targetScale, projections: projections)
        case .phm(let hypercomplexDim, let sharedBasis, let targetScale):
            return try matchingPHM(hypercomplexDim: hypercomplexDim, sharedBasis: sharedBasis, scale: targetScale, projections: projections)
        case .oft(let parameterization, let blockShared):
            return try matchingOFT(parameterization: parameterization, blockShared: blockShared, projections: projections)
        }
    }

    public func matchingBudget(
        to family: AdapterFamily,
        inputDim: Int,
        outputDim: Int,
        layerCount: Int = 1
    ) throws(BasaltError) -> AdapterMethod {
        guard layerCount > 0 else {
            throw .invalidRequestStructure(details: "adapter: layerCount (\(layerCount)) must be positive")
        }
        let projections = Array(repeating: (inputDim: inputDim, outputDim: outputDim), count: layerCount)
        return try matchingBudget(to: family, projections: projections)
    }

    private static func validatedProjections(
        _ projections: [(inputDim: Int, outputDim: Int)]
    ) throws(BasaltError) -> [(inputDim: Int, outputDim: Int)] {
        guard !projections.isEmpty, projections.allSatisfy({ $0.inputDim > 0 && $0.outputDim > 0 }) else {
            throw .invalidRequestStructure(details: "adapter: projections must be non-empty with positive dimensions")
        }
        return projections
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

    public func matchingBudget(
        to family: AdapterFamily,
        projections: [(inputDim: Int, outputDim: Int)]
    ) throws(BasaltError) -> AdapterSpec {
        try validate(prefix: "adapter")
        let matched = try method.matchingBudget(to: family, projections: projections)
        return AdapterSpec(method: matched, layers: layers, targetSuffixes: targetSuffixes)
    }

    public func matchingBudget(
        to family: AdapterFamily,
        inputDim: Int,
        outputDim: Int,
        layerCount: Int = 1
    ) throws(BasaltError) -> AdapterSpec {
        try validate(prefix: "adapter")
        let matched = try method.matchingBudget(to: family, inputDim: inputDim, outputDim: outputDim, layerCount: layerCount)
        return AdapterSpec(method: matched, layers: layers, targetSuffixes: targetSuffixes)
    }
}
