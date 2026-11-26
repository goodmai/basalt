import Foundation

public enum BlockDiffusionConfigError: Error, Equatable, Sendable {
    case invalidBlockSize(Int)
    case invalidMaxStepsPerBlock(Int)
    case invalidConfidenceThreshold(Float)
    case invalidFactorParallelUnmasking(Float)
    case invalidRemaskThreshold(Float)
    case invalidTop2MarginThreshold(Float)
}

public struct BlockDiffusionConfig: Equatable, Sendable {
    public var blockSize: Int
    public var confidenceThreshold: Float
    public var maxStepsPerBlock: Int
    public var maskTokenId: Int
    public var remaskThreshold: Float?
    public var dualCacheEnabled: Bool
    public var factorParallelUnmasking: Float?
    public var top2MarginThreshold: Float?
    public var bidirectionalPrompt = false
    public var prefixCacheOnly = false
    public var reuseSuffixCache = false
    public var maxSuffixLength: Int? = nil

    public init(
        blockSize: Int = 32,
        confidenceThreshold: Float = 0.9,
        maxStepsPerBlock: Int? = nil,
        maskTokenId: Int = 126_336,
        remaskThreshold: Float? = nil,
        dualCacheEnabled: Bool = true
    ) {
        let clampedBlock = max(1, blockSize)
        self.blockSize = clampedBlock
        self.confidenceThreshold = confidenceThreshold
        self.maxStepsPerBlock = max(1, maxStepsPerBlock ?? clampedBlock)
        self.maskTokenId = maskTokenId
        self.remaskThreshold = remaskThreshold
        self.dualCacheEnabled = dualCacheEnabled
        self.factorParallelUnmasking = nil
        self.top2MarginThreshold = nil
    }

    public func withFactorParallelUnmasking(_ factor: Float?) -> Self {
        var copy = self
        copy.factorParallelUnmasking = factor
        return copy
    }

    public func withTop2MarginThreshold(_ margin: Float?) -> Self {
        var copy = self
        copy.top2MarginThreshold = margin
        return copy
    }

    public static func live(environment: [String: String] = ProcessInfo.processInfo.environment) -> Self {
        var config = Self(blockSize: 8, confidenceThreshold: 0.9, maxStepsPerBlock: 8)
        config.reuseSuffixCache = true
        config.maxSuffixLength = 128
        config.factorParallelUnmasking = 1.25
        if let raw = environment["BASALT_DLLM_BLOCK_SIZE"], let size = Int(raw), size >= 1 {
            config.blockSize = size
            config.maxStepsPerBlock = size
        }
        if let raw = environment["BASALT_DLLM_THRESHOLD"], let tau = Float(raw), tau.isFinite, (0.0 ... 1.0).contains(tau) {
            config.confidenceThreshold = tau
            if environment["BASALT_DLLM_FACTOR"] == nil {
                config.factorParallelUnmasking = nil
            }
        }
        if let raw = environment["BASALT_DLLM_FACTOR"] {
            if let factor = Float(raw), factor.isFinite, factor > 0 {
                config.factorParallelUnmasking = factor
            } else if raw == "0" || raw == "off" {
                config.factorParallelUnmasking = nil
            }
        }
        if environment["BASALT_DLLM_PREFIX_ONLY"] == "1" {
            config.prefixCacheOnly = true
        }
        if let raw = environment["BASALT_DLLM_REUSE_SUFFIX"] {
            config.reuseSuffixCache = raw != "0"
        }
        if let raw = environment["BASALT_DLLM_MAX_SUFFIX"], let cap = Int(raw), cap >= 0 {
            config.maxSuffixLength = cap
        }
        return config
    }

    public func validated() throws(BlockDiffusionConfigError) -> Self {
        guard blockSize >= 1 else { throw .invalidBlockSize(blockSize) }
        guard maxStepsPerBlock >= 1 else { throw .invalidMaxStepsPerBlock(maxStepsPerBlock) }
        guard confidenceThreshold.isFinite, (0.0 ... 1.0).contains(confidenceThreshold) else {
            throw .invalidConfidenceThreshold(confidenceThreshold)
        }
        if let factor = factorParallelUnmasking, !factor.isFinite || factor <= 0 {
            throw .invalidFactorParallelUnmasking(factor)
        }
        if let remask = remaskThreshold, !remask.isFinite || remask < 0 || remask >= 1.0 {
            throw .invalidRemaskThreshold(remask)
        }
        if let margin = top2MarginThreshold, !margin.isFinite || margin < 0 || margin > 1.0 {
            throw .invalidTop2MarginThreshold(margin)
        }
        return self
    }
}

public enum FastDLLMUnmasker: Sendable {
    private struct RankedPosition {
        let index: Int
        let confidence: Float
    }

    public static func selectPositions(
        confidences: [Float],
        maskedMask: [Bool],
        config: BlockDiffusionConfig,
        top2Margins: [Float]? = nil
    ) -> [Int] {
        let ranked = rankedMaskedPositions(confidences: confidences, maskedMask: maskedMask)
        guard !ranked.isEmpty else { return [] }

        if let factor = config.factorParallelUnmasking {
            return selectByFactor(ranked: ranked, factor: factor)
        }
        if config.confidenceThreshold >= 1.0, config.top2MarginThreshold == nil {
            return [ranked[0].index]
        }
        return selectByThreshold(ranked: ranked, config: config, top2Margins: top2Margins)
    }

    public static func selectRemaskPositions(
        confidences: [Float],
        maskedMask: [Bool],
        config: BlockDiffusionConfig,
        pinnedMask: [Bool]? = nil,
        maxRemaskCount: Int? = nil
    ) -> [Int] {
        guard let remaskThreshold = config.remaskThreshold, remaskThreshold > 0 else { return [] }
        if let limit = maxRemaskCount, limit <= 0 { return [] }

        let count = min(confidences.count, maskedMask.count)
        var candidates: [RankedPosition] = []
        candidates.reserveCapacity(count)

        for index in 0 ..< count where !maskedMask[index] {
            if let pinnedMask, index < pinnedMask.count, pinnedMask[index] { continue }
            let score = sanitizeConfidence(confidences[index])
            if score < remaskThreshold {
                candidates.append(RankedPosition(index: index, confidence: score))
            }
        }
        guard !candidates.isEmpty else { return [] }

        candidates.sort { lhs, rhs in
            lhs.confidence != rhs.confidence ? lhs.confidence < rhs.confidence : lhs.index < rhs.index
        }
        let selected = maxRemaskCount.map { Array(candidates.prefix($0)) } ?? candidates
        return selected.map(\.index).sorted()
    }

    private static func rankedMaskedPositions(confidences: [Float], maskedMask: [Bool]) -> [RankedPosition] {
        let count = min(confidences.count, maskedMask.count)
        var candidates: [RankedPosition] = []
        candidates.reserveCapacity(count)
        for index in 0 ..< count where maskedMask[index] {
            candidates.append(RankedPosition(index: index, confidence: sanitizeConfidence(confidences[index])))
        }
        candidates.sort { lhs, rhs in
            lhs.confidence != rhs.confidence ? lhs.confidence > rhs.confidence : lhs.index < rhs.index
        }
        return candidates
    }

    private static func selectByFactor(ranked: [RankedPosition], factor: Float) -> [Int] {
        var selected: [Int] = []
        selected.reserveCapacity(ranked.count)
        for (rank, entry) in ranked.enumerated() {
            let clamped = max(Float(0), entry.confidence)
            if Float(rank + 1) * (1.0 - clamped) < factor {
                selected.append(entry.index)
            } else {
                break
            }
        }
        if selected.isEmpty {
            selected.append(ranked[0].index)
        }
        return selected.sorted()
    }

    private static func selectByThreshold(
        ranked: [RankedPosition],
        config: BlockDiffusionConfig,
        top2Margins: [Float]?
    ) -> [Int] {
        var selected: [Int] = []
        selected.reserveCapacity(ranked.count)
        for entry in ranked {
            let meetsConfidence = entry.confidence >= config.confidenceThreshold
            let meetsMargin = meetsMarginThreshold(index: entry.index, threshold: config.top2MarginThreshold, margins: top2Margins)
            if meetsConfidence || meetsMargin {
                selected.append(entry.index)
            }
        }
        if selected.isEmpty {
            selected.append(ranked[0].index)
        }
        return selected.sorted()
    }

    private static func meetsMarginThreshold(index: Int, threshold: Float?, margins: [Float]?) -> Bool {
        guard let threshold, let margins, index < margins.count else { return false }
        let value = margins[index]
        return value.isFinite && value >= threshold
    }

    private static func sanitizeConfidence(_ value: Float) -> Float {
        guard value.isFinite else { return -.infinity }
        return min(max(value, 0), 1)
    }
}
