import Foundation

public enum QuantizedLRPolicy {
    public static let defaultQuantizedDivisor: Float = 40.0

    public static func effectiveLearningRate(
        baseLR: Float,
        weightBits: Int?,
        overrideDivisor: Float? = nil
    ) throws(BasaltError) -> Float {
        guard baseLR.isFinite, baseLR > 0 else {
            throw .invalidRequestStructure(details: "quantized LR policy: baseLR must be finite and positive")
        }
        if let overrideDivisor {
            guard overrideDivisor.isFinite, overrideDivisor > 0 else {
                throw .invalidRequestStructure(details: "quantized LR policy: overrideDivisor must be finite and positive")
            }
        }
        guard let weightBits, weightBits <= 8 else {
            return baseLR
        }
        guard weightBits >= 2 else {
            throw .invalidRequestStructure(details: "quantized LR policy: weightBits must be at least 2")
        }
        return baseLR / (overrideDivisor ?? defaultQuantizedDivisor)
    }

    public static func recommendedBatchSize(parameterMillions: Double, quantized: Bool) -> Int {
        guard parameterMillions.isFinite, parameterMillions > 0 else { return 1 }
        if !quantized {
            return parameterMillions <= 100 ? 8 : (parameterMillions <= 300 ? 4 : 2)
        }
        if parameterMillions <= 100 { return 8 }
        if parameterMillions <= 300 { return 4 }
        if parameterMillions <= 900 { return 2 }
        return 1
    }
}

public struct AdaLoRAMatrixBudget: Sendable, Codable, Equatable {
    public var modulePath: String
    public var currentRank: Int
    public var minRank: Int
    public var maxRank: Int
    public var singularImportances: [Float]

    public init(
        modulePath: String,
        currentRank: Int,
        minRank: Int = 1,
        maxRank: Int = 64,
        singularImportances: [Float]
    ) {
        self.modulePath = modulePath
        self.currentRank = currentRank
        self.minRank = minRank
        self.maxRank = maxRank
        self.singularImportances = singularImportances
    }

    enum CodingKeys: String, CodingKey {
        case modulePath = "module_path"
        case currentRank = "current_rank"
        case minRank = "min_rank"
        case maxRank = "max_rank"
        case singularImportances = "singular_importances"
    }
}

public struct AdaLoRARankScheduleConfig: Sendable, Codable, Equatable {
    public var initRank: Int
    public var targetAverageRank: Int
    public var warmupSteps: Int
    public var pruneInterval: Int
    public var importanceBeta: Float

    public init(
        initRank: Int = 16,
        targetAverageRank: Int = 8,
        warmupSteps: Int = 10,
        pruneInterval: Int = 10,
        importanceBeta: Float = 0.85
    ) {
        self.initRank = initRank
        self.targetAverageRank = targetAverageRank
        self.warmupSteps = warmupSteps
        self.pruneInterval = pruneInterval
        self.importanceBeta = importanceBeta
    }

    public func validate() throws(BasaltError) {
        guard AdapterMethod.allowedRanks.contains(initRank) else {
            throw .invalidRequestStructure(details: "AdaLoRA: initRank must be in \(AdapterMethod.allowedRanks)")
        }
        guard (1 ... initRank).contains(targetAverageRank) else {
            throw .invalidRequestStructure(details: "AdaLoRA: targetAverageRank must lie in 1...\(initRank)")
        }
        guard warmupSteps >= 0, pruneInterval >= 1 else {
            throw .invalidRequestStructure(details: "AdaLoRA: warmupSteps must be >= 0 and pruneInterval >= 1")
        }
        guard importanceBeta.isFinite, (0.0 ..< 1.0).contains(importanceBeta) else {
            throw .invalidRequestStructure(details: "AdaLoRA: importanceBeta must lie in [0, 1)")
        }
    }
}

public enum AdaLoRARankScheduler {
    public static func updateImportance(
        previous: [Float],
        gradientSensitivity: [Float],
        beta: Float = 0.85
    ) -> [Float] {
        guard !gradientSensitivity.isEmpty else { return previous }
        let clampedBeta = min(0.999, max(0.0, beta.isFinite ? beta : 0.85))
        return gradientSensitivity.enumerated().map { idx, sens in
            let clean = abs(sens.isFinite ? sens : 0.0)
            let prev = idx < previous.count ? max(0.0, previous[idx]) : clean
            return clampedBeta * prev + (1.0 - clampedBeta) * clean
        }
    }

    public static func pruneToBudget(
        matrices: [AdaLoRAMatrixBudget],
        totalRankBudget: Int
    ) throws(BasaltError) -> [AdaLoRAMatrixBudget] {
        guard !matrices.isEmpty else { return [] }
        let minSum = matrices.reduce(0) { $0 + max(1, $1.minRank) }
        guard totalRankBudget >= minSum else {
            throw .invalidRequestStructure(details: "AdaLoRA: totalRankBudget (\(totalRankBudget)) < sum of minRanks (\(minSum))")
        }
        var allocated = matrices.map { max(1, min($0.minRank, $0.singularImportances.count)) }
        var remaining = max(0, totalRankBudget - allocated.reduce(0, +))
        var candidates: [(matrixIdx: Int, score: Float)] = []
        for (mIdx, matrix) in matrices.enumerated() {
            let sorted = matrix.singularImportances.map { max(0.0, $0.isFinite ? $0 : 0.0) }.sorted(by: >)
            let upper = min(matrix.maxRank, sorted.count)
            if allocated[mIdx] < upper {
                for rankIdx in allocated[mIdx] ..< upper {
                    candidates.append((mIdx, sorted[rankIdx]))
                }
            }
        }
        candidates.sort { $0.score > $1.score }
        for cand in candidates {
            guard remaining > 0 else { break }
            allocated[cand.matrixIdx] += 1
            remaining -= 1
        }
        return matrices.enumerated().map { idx, matrix in
            var next = matrix
            next.currentRank = allocated[idx]
            return next
        }
    }
}

public struct TwoStageSFTDPOPolicy: Sendable, Codable, Equatable {
    public var baseLearningRate: Float
    public var dpoLearningRateRatio: Float
    public var dpoBeta: Float
    public var freezeBaseWeights: Bool
    public var detachReferenceLogits: Bool
    public var dpoPairsPath: String?

    public init(
        baseLearningRate: Float = 1e-4,
        dpoLearningRateRatio: Float = 0.2,
        dpoBeta: Float = 0.1,
        freezeBaseWeights: Bool = true,
        detachReferenceLogits: Bool = true,
        dpoPairsPath: String? = nil
    ) {
        self.baseLearningRate = baseLearningRate
        self.dpoLearningRateRatio = dpoLearningRateRatio
        self.dpoBeta = dpoBeta
        self.freezeBaseWeights = freezeBaseWeights
        self.detachReferenceLogits = detachReferenceLogits
        self.dpoPairsPath = dpoPairsPath
    }

    public func resolvedStages(weightBits: Int?) throws(BasaltError) -> (sftLR: Float, dpoLR: Float, beta: Float) {
        guard dpoLearningRateRatio.isFinite, dpoLearningRateRatio > 0, dpoLearningRateRatio <= 1.0 else {
            throw .invalidRequestStructure(details: "SFT->DPO: dpoLearningRateRatio must lie in (0, 1]")
        }
        guard dpoBeta.isFinite, dpoBeta > 0 else {
            throw .invalidRequestStructure(details: "SFT->DPO: dpoBeta must be positive")
        }
        let sftLR = try QuantizedLRPolicy.effectiveLearningRate(baseLR: baseLearningRate, weightBits: weightBits)
        return (sftLR, sftLR * dpoLearningRateRatio, dpoBeta)
    }
}
