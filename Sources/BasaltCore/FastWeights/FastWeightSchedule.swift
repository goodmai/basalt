import Foundation

public struct FastWeightChunkSlice: Sendable, Equatable {
    public let range: Range<Int>
    public let stepStart: Int
    public let stepEnd: Int
    public let completesChunk: Bool
    public let chunkSize: Int

    public init(range: Range<Int>, stepStart: Int, stepEnd: Int, completesChunk: Bool, chunkSize: Int) {
        self.range = range
        self.stepStart = stepStart
        self.stepEnd = stepEnd
        self.completesChunk = completesChunk
        self.chunkSize = chunkSize
    }

    public var count: Int { range.count }
}

public struct FastWeightSchedule: Sendable, Equatable {
    public let chunkSize: Int
    public private(set) var offset: Int
    public private(set) var stepInChunk: Int

    public init(chunkSize: Int, offset: Int = 0, stepInChunk: Int? = nil) {
        let clampedChunk = max(1, chunkSize)
        let clampedOffset = max(0, offset)
        self.chunkSize = clampedChunk
        self.offset = clampedOffset
        if let explicit = stepInChunk {
            self.stepInChunk = ((explicit % clampedChunk) + clampedChunk) % clampedChunk
        } else {
            self.stepInChunk = clampedOffset % clampedChunk
        }
    }

    public mutating func planSlices(for tokenCount: Int) -> [FastWeightChunkSlice] {
        guard tokenCount > 0 else { return [] }
        var result: [FastWeightChunkSlice] = []
        var cursor = 0
        while cursor < tokenCount {
            let room = chunkSize - stepInChunk
            let take = min(room, tokenCount - cursor)
            let nextStep = stepInChunk + take
            let completes = nextStep == chunkSize
            result.append(FastWeightChunkSlice(
                range: cursor ..< (cursor + take),
                stepStart: stepInChunk,
                stepEnd: nextStep,
                completesChunk: completes,
                chunkSize: chunkSize
            ))
            cursor += take
            offset += take
            stepInChunk = completes ? 0 : nextStep
        }
        return result
    }

    public static func slices(
        chunkSize: Int,
        stepInChunk: Int,
        tokenCount: Int
    ) -> (slices: [FastWeightChunkSlice], nextStepInChunk: Int) {
        var schedule = FastWeightSchedule(chunkSize: chunkSize, offset: 0, stepInChunk: stepInChunk)
        let planned = schedule.planSlices(for: tokenCount)
        return (planned, schedule.stepInChunk)
    }
}

public enum FastWeightFootprint: Sendable {
    public static let tttModelTypes: Set<String> = ["ttt"]
    public static let titansModelTypes: Set<String> = ["titans", "titans_mac", "titans_mag", "titans_mal"]
    public static let lactModelTypes: Set<String> = ["lact_swiglu", "lact"]

    public static func isFastWeightModelType(_ modelType: String?) -> Bool {
        guard let normalized = modelType?.lowercased() else { return false }
        return tttModelTypes.contains(normalized)
            || titansModelTypes.contains(normalized)
            || lactModelTypes.contains(normalized)
    }

    public static func tttLayerStateBytes(
        config: TTTConfiguration,
        stateElementBytes: Int = 4,
        activationBytes: Int = 2
    ) -> Int64 {
        let h = Int64(config.numAttentionHeads)
        let d = Int64(config.headDim)
        let weightElements: Int64 = switch config.variant {
        case .linear:
            h * (d * d + d)
        case .mlp:
            h * (2 * d * d * Int64(config.mlpExpansion) + d * Int64(config.mlpExpansion) + d)
        }
        let convHistory = Int64(max(0, config.convKernel - 1))
        let convBytes = convHistory * 2 * Int64(config.hiddenSize) * Int64(activationBytes)
        return 2 * weightElements * Int64(stateElementBytes) + convBytes
    }

    public static func tttStateBytes(
        config: TTTConfiguration,
        stateElementBytes: Int = 4,
        activationBytes: Int = 2
    ) -> Int64 {
        Int64(config.numHiddenLayers) * tttLayerStateBytes(
            config: config,
            stateElementBytes: stateElementBytes,
            activationBytes: activationBytes
        )
    }

    public static func titansLayerStateBytes(
        config: TitansConfiguration,
        stateElementBytes: Int = 4,
        activationBytes: Int = 2
    ) -> Int64 {
        let h = Int64(config.numAttentionHeads)
        let d = Int64(config.headDim)
        let m = Int64(config.memoryHiddenDim)
        let weightElements: Int64 = config.memoryDepth == 1
            ? h * (d * d + d)
            : h * (2 * d * m + m + d)
        let convHistory = Int64(max(0, config.convKernel - 1))
        let convBytes = convHistory * 3 * Int64(config.hiddenSize) * Int64(activationBytes)
        return 3 * weightElements * Int64(stateElementBytes) + convBytes
    }

    public static func titansStateBytes(
        config: TitansConfiguration,
        stateElementBytes: Int = 4,
        activationBytes: Int = 2
    ) -> Int64 {
        Int64(config.numHiddenLayers) * titansLayerStateBytes(
            config: config,
            stateElementBytes: stateElementBytes,
            activationBytes: activationBytes
        )
    }

    public static func lactLayerStateBytes(
        config: LaCTConfiguration,
        stateElementBytes: Int = 4,
        activationBytes: Int = 2
    ) -> Int64 {
        let h = Int64(config.numFastWeightHeads)
        let d = Int64(config.fastWeightHeadDim)
        let m = Int64(config.fastWeightHiddenDim)
        let weightElements: Int64 = switch config.memoryMode {
        case .linear: h * d * d
        case .swiglu: h * 3 * m * d
        }
        let copies: Int64 = config.updateRule == .sgd ? 2 : 3
        let convHistory = Int64(max(0, config.convKernel - 1))
        let convBytes = convHistory * 3 * h * d * Int64(activationBytes)
        return copies * weightElements * Int64(stateElementBytes) + convBytes
    }

    public static func lactStateBytes(
        config: LaCTConfiguration,
        stateElementBytes: Int = 4,
        activationBytes: Int = 2
    ) -> Int64 {
        Int64(config.numHiddenLayers) * lactLayerStateBytes(
            config: config,
            stateElementBytes: stateElementBytes,
            activationBytes: activationBytes
        )
    }

    public static func configFixedStateBytes(_ c: [String: Any]) -> Int64 {
        guard let modelType = (c["model_type"] as? String)?.lowercased(),
              isFastWeightModelType(modelType),
              let data = try? JSONSerialization.data(withJSONObject: c) else { return 0 }
        let decoder = JSONDecoder()
        if tttModelTypes.contains(modelType), let cfg = try? decoder.decode(TTTConfiguration.self, from: data) {
            return tttStateBytes(config: cfg)
        }
        if titansModelTypes.contains(modelType), let cfg = try? decoder.decode(TitansConfiguration.self, from: data) {
            return titansStateBytes(config: cfg)
        }
        if lactModelTypes.contains(modelType), let cfg = try? decoder.decode(LaCTConfiguration.self, from: data) {
            return lactStateBytes(config: cfg)
        }
        return 0
    }
}
