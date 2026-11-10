import BasaltCore

public struct EngineConfiguration: Sendable {
    public var reasoningEffort: String? = nil
    public var gpuCacheLimit: Int? = 512 << 20
    public var promptCacheBytes: Int? = nil
    public var kvPrecision: KVPrecision? = nil
    public var quantizedKVStart = KVCacheGeometry.defaultQuantizedKVStart
    public var kvZones: KVZones? = nil
    public var kvHeadSplit: Double? = nil
    public var questTopK: Int? = nil
    public var repetitionPenalty: Float? = nil
    public var topK: Int? = nil
    public var minP: Float? = nil
    public var seed: UInt64? = nil
    public var cacheBlendRatio: Float? = nil
    public var speculativeLookup = false
    public var speculativeRecycling = false
    public var mtp = false
    public var mtpBlockSize = 4
    public var mtpDraftHead = false
    public var experimentalSamplers = false
    public var grammarJumpForward = false
    public var hybridAnchors = 8
    public var batchSlots = 1
    public var wiresWorkingSet = false
    public var expertStreaming: ExpertStreaming? = nil
    public var probesModel = false
    public var recordConfidence = false
    public var visionTowerBundle: String? = nil
}

extension EngineConfiguration {
    public static let standard = EngineConfiguration()

    var normalized: EngineConfiguration {
        var configuration = self
        configuration.batchSlots = max(1, batchSlots)
        configuration.mtpBlockSize = max(2, mtpBlockSize)
        return configuration
    }
}
