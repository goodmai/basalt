import Foundation

public enum KVCacheTier: String, Sendable, Codable, CaseIterable, Equatable, Hashable {
    case l0GPU = "l0_gpu"
    case l1HostRAM = "l1_host_ram"
    case l2SharedNVMe = "l2_shared_nvme"
}

public enum KVMemoryTopology: String, Sendable, Codable, Equatable {
    case unifiedMemory = "unified_memory"
    case discreteVRAM = "discrete_vram"
}

public struct TieredKVConfig: Sendable, Codable, Equatable {
    public var topology: KVMemoryTopology
    public var chunkTokens: Int
    public var l0CapacityBytes: Int64
    public var l1CapacityBytes: Int64
    public var l2CapacityBytes: Int64
    public var unifiedMemoryBudgetBytes: Int64?
    public var prefetchLookaheadLayers: Int

    public init(
        topology: KVMemoryTopology = .unifiedMemory,
        chunkTokens: Int = 256,
        l0CapacityBytes: Int64 = 512 * 1_048_576,
        l1CapacityBytes: Int64 = 4_096 * 1_048_576,
        l2CapacityBytes: Int64 = 32_768 * 1_048_576,
        prefetchLookaheadLayers: Int = 1
    ) {
        self.topology = topology
        self.chunkTokens = chunkTokens
        self.l0CapacityBytes = l0CapacityBytes
        self.l1CapacityBytes = l1CapacityBytes
        self.l2CapacityBytes = l2CapacityBytes
        self.unifiedMemoryBudgetBytes = nil
        self.prefetchLookaheadLayers = prefetchLookaheadLayers
    }

    public var effectiveUnifiedBudgetBytes: Int64 {
        let sum = l0CapacityBytes + l1CapacityBytes
        guard let cap = unifiedMemoryBudgetBytes else { return sum }
        return min(cap, sum)
    }

    public func validate() throws(BasaltError) {
        guard chunkTokens >= 1 else {
            throw .invalidRequestStructure(details: "tiered KV: chunkTokens must be >= 1")
        }
        guard l0CapacityBytes > 0, l1CapacityBytes >= 0, l2CapacityBytes >= 0 else {
            throw .invalidRequestStructure(details: "tiered KV: capacities must be non-negative and L0 > 0")
        }
        if let cap = unifiedMemoryBudgetBytes, cap < l0CapacityBytes {
            throw .invalidRequestStructure(details: "tiered KV: unifiedMemoryBudgetBytes must be >= l0CapacityBytes")
        }
        guard prefetchLookaheadLayers >= 0 else {
            throw .invalidRequestStructure(details: "tiered KV: prefetchLookaheadLayers must be >= 0")
        }
    }

    public func capacity(for tier: KVCacheTier) -> Int64 {
        switch tier {
        case .l0GPU: l0CapacityBytes
        case .l1HostRAM: l1CapacityBytes
        case .l2SharedNVMe: l2CapacityBytes
        }
    }
}

public struct TieredKVBlock: Sendable, Codable, Equatable {
    public var prefixHash: String
    public var tokenCount: Int
    public var byteSize: Int64
    public var layerCount: Int
    public var tier: KVCacheTier
    public var lastAccessTick: UInt64

    public init(
        prefixHash: String,
        tokenCount: Int,
        byteSize: Int64,
        layerCount: Int,
        tier: KVCacheTier,
        lastAccessTick: UInt64
    ) {
        self.prefixHash = prefixHash
        self.tokenCount = tokenCount
        self.byteSize = byteSize
        self.layerCount = layerCount
        self.tier = tier
        self.lastAccessTick = lastAccessTick
    }
}

public struct TieredKVMetrics: Sendable, Codable, Equatable {
    public var l0Hits: Int
    public var l1Hits: Int
    public var l2Hits: Int
    public var misses: Int
    public var demotions: Int
    public var evictions: Int
    public var l0Bytes: Int64
    public var l1Bytes: Int64
    public var l2Bytes: Int64
    public var zeroCopyPromotions: Int

    public var hitRate: Double {
        let hits = l0Hits + l1Hits + l2Hits
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0.0
    }

    public var unifiedRAMBytes: Int64 {
        l0Bytes + l1Bytes
    }

    public init(
        l0Hits: Int = 0,
        l1Hits: Int = 0,
        l2Hits: Int = 0,
        misses: Int = 0,
        demotions: Int = 0,
        evictions: Int = 0
    ) {
        self.l0Hits = l0Hits
        self.l1Hits = l1Hits
        self.l2Hits = l2Hits
        self.misses = misses
        self.demotions = demotions
        self.evictions = evictions
        self.l0Bytes = 0
        self.l1Bytes = 0
        self.l2Bytes = 0
        self.zeroCopyPromotions = 0
    }
}

public struct LayerPrefetchStep: Sendable, Codable, Equatable {
    public var computeLayer: Int
    public var prefetchLayer: Int
    public var sourceTier: KVCacheTier
    public var bytesPerLayer: Int64
    public var zeroCopyRebind: Bool

    public init(
        computeLayer: Int,
        prefetchLayer: Int,
        sourceTier: KVCacheTier,
        bytesPerLayer: Int64,
        zeroCopyRebind: Bool = false
    ) {
        self.computeLayer = computeLayer
        self.prefetchLayer = prefetchLayer
        self.sourceTier = sourceTier
        self.bytesPerLayer = bytesPerLayer
        self.zeroCopyRebind = zeroCopyRebind
    }
}

public struct TieredKVLookupResult: Sendable, Codable, Equatable {
    public var matchedTokens: Int
    public var deepestTierUsed: KVCacheTier?
    public var matchedBlocks: [TieredKVBlock]
    public var prefetchSchedule: [LayerPrefetchStep]

    public init(
        matchedTokens: Int,
        deepestTierUsed: KVCacheTier?,
        matchedBlocks: [TieredKVBlock],
        prefetchSchedule: [LayerPrefetchStep]
    ) {
        self.matchedTokens = matchedTokens
        self.deepestTierUsed = deepestTierUsed
        self.matchedBlocks = matchedBlocks
        self.prefetchSchedule = prefetchSchedule
    }
}

public struct TieredKVHierarchyCore: Sendable, Equatable {
    public var config: TieredKVConfig
    private var blocks: [String: TieredKVBlock]
    private var tick: UInt64
    private var metricsState: TieredKVMetrics

    public init(config: TieredKVConfig = .init()) {
        self.config = config
        self.blocks = [:]
        self.tick = 0
        self.metricsState = TieredKVMetrics()
    }

    public var metrics: TieredKVMetrics {
        var out = metricsState
        out.l0Bytes = usedBytes(in: .l0GPU)
        out.l1Bytes = usedBytes(in: .l1HostRAM)
        out.l2Bytes = usedBytes(in: .l2SharedNVMe)
        return out
    }

    public func aggregateChunks(tokens: [Int]) -> [[Int]] {
        guard !tokens.isEmpty, config.chunkTokens > 0 else { return [] }
        var chunks: [[Int]] = []
        var idx = 0
        while idx < tokens.count {
            let end = min(tokens.count, idx + config.chunkTokens)
            chunks.append(Array(tokens[idx ..< end]))
            idx = end
        }
        return chunks
    }

    public mutating func storeChunk(
        prefixHash: String,
        tokenCount: Int,
        byteSize: Int64,
        layerCount: Int
    ) throws(BasaltError) {
        try config.validate()
        guard byteSize > 0, tokenCount > 0, layerCount >= 1 else {
            throw .invalidRequestStructure(details: "tiered KV: chunk byteSize, tokenCount, and layerCount must be positive")
        }
        tick += 1
        blocks[prefixHash] = TieredKVBlock(
            prefixHash: prefixHash,
            tokenCount: tokenCount,
            byteSize: byteSize,
            layerCount: layerCount,
            tier: .l0GPU,
            lastAccessTick: tick
        )
        rebalanceOffloading()
    }

    public mutating func lookupLongestPrefix(chunkHashes: [String]) -> TieredKVLookupResult {
        var matched: [TieredKVBlock] = []
        var deepest: KVCacheTier?
        var prefetch: [LayerPrefetchStep] = []
        for hash in chunkHashes {
            guard var block = blocks[hash] else {
                metricsState.misses += 1
                break
            }
            tick += 1
            let sourceTier = block.tier
            recordHit(for: sourceTier)
            deepest = deeperTier(deepest, sourceTier)
            if sourceTier != .l0GPU {
                let isZeroCopy = config.topology == .unifiedMemory && sourceTier == .l1HostRAM
                if isZeroCopy {
                    metricsState.zeroCopyPromotions += 1
                }
                prefetch.append(contentsOf: buildPrefetchSchedule(for: block, sourceTier: sourceTier, zeroCopy: isZeroCopy))
                block.tier = .l0GPU
            }
            block.lastAccessTick = tick
            blocks[hash] = block
            matched.append(block)
        }
        rebalanceOffloading()
        return TieredKVLookupResult(
            matchedTokens: matched.reduce(0) { $0 + $1.tokenCount },
            deepestTierUsed: deepest,
            matchedBlocks: matched,
            prefetchSchedule: prefetch
        )
    }

    public func block(for prefixHash: String) -> TieredKVBlock? {
        blocks[prefixHash]
    }

    private mutating func recordHit(for tier: KVCacheTier) {
        switch tier {
        case .l0GPU: metricsState.l0Hits += 1
        case .l1HostRAM: metricsState.l1Hits += 1
        case .l2SharedNVMe: metricsState.l2Hits += 1
        }
    }

    private func deeperTier(_ current: KVCacheTier?, _ incoming: KVCacheTier) -> KVCacheTier {
        let rank: (KVCacheTier) -> Int = { tier in
            switch tier {
            case .l0GPU: 0
            case .l1HostRAM: 1
            case .l2SharedNVMe: 2
            }
        }
        guard let current else { return incoming }
        return rank(incoming) > rank(current) ? incoming : current
    }

    private func buildPrefetchSchedule(
        for block: TieredKVBlock,
        sourceTier: KVCacheTier,
        zeroCopy: Bool
    ) -> [LayerPrefetchStep] {
        guard config.prefetchLookaheadLayers > 0, block.layerCount > 1 else { return [] }
        let perLayer = zeroCopy ? 0 : max(1, block.byteSize / Int64(block.layerCount))
        let stride = config.prefetchLookaheadLayers
        var steps: [LayerPrefetchStep] = []
        for layer in 0 ..< max(0, block.layerCount - stride) {
            steps.append(LayerPrefetchStep(
                computeLayer: layer,
                prefetchLayer: layer + stride,
                sourceTier: sourceTier,
                bytesPerLayer: perLayer,
                zeroCopyRebind: zeroCopy
            ))
        }
        return steps
    }

    private mutating func rebalanceOffloading() {
        demoteOverflow(from: .l0GPU, to: .l1HostRAM)
        if config.topology == .unifiedMemory {
            enforceUnifiedMemoryBudget()
        }
        demoteOverflow(from: .l1HostRAM, to: .l2SharedNVMe)
        demoteOverflow(from: .l2SharedNVMe, to: nil)
    }

    private mutating func enforceUnifiedMemoryBudget() {
        let limit = config.effectiveUnifiedBudgetBytes
        while usedBytes(in: .l0GPU) + usedBytes(in: .l1HostRAM) > limit {
            guard let victimKey = oldestKey(in: .l1HostRAM) ?? oldestKey(in: .l0GPU) else { break }
            if config.l2CapacityBytes > 0 {
                blocks[victimKey]?.tier = .l2SharedNVMe
                metricsState.demotions += 1
            } else {
                blocks.removeValue(forKey: victimKey)
                metricsState.evictions += 1
            }
        }
    }

    private mutating func demoteOverflow(from tier: KVCacheTier, to nextTier: KVCacheTier?) {
        let limit = config.capacity(for: tier)
        while usedBytes(in: tier) > limit {
            guard let victimKey = oldestKey(in: tier) else { break }
            if let nextTier, config.capacity(for: nextTier) > 0 {
                blocks[victimKey]?.tier = nextTier
                metricsState.demotions += 1
            } else {
                blocks.removeValue(forKey: victimKey)
                metricsState.evictions += 1
            }
        }
    }

    private func usedBytes(in tier: KVCacheTier) -> Int64 {
        blocks.values.reduce(0) { $0 + ($1.tier == tier ? $1.byteSize : 0) }
    }

    private func oldestKey(in tier: KVCacheTier) -> String? {
        blocks.values
            .filter { $0.tier == tier }
            .min { $0.lastAccessTick < $1.lastAccessTick }?
            .prefixHash
    }
}
