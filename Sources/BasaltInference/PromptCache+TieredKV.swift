import BasaltCore
import Foundation

struct TieredPromptCacheTracker: Sendable, Equatable {
    private(set) var core: TieredKVHierarchyCore

    init(ramBudgetBytes: Int64 = 512 * 1_048_576, diskBudgetBytes: Int64 = 32_768 * 1_048_576) {
        self.core = TieredKVHierarchyCore(config: Self.makeConfig(ramBudgetBytes: ramBudgetBytes, diskBudgetBytes: diskBudgetBytes))
    }

    var metrics: TieredKVMetrics { core.metrics }

    mutating func reconfigure(ramBudgetBytes: Int64, diskBudgetBytes: Int64 = 32_768 * 1_048_576) {
        core.config = Self.makeConfig(ramBudgetBytes: ramBudgetBytes, diskBudgetBytes: diskBudgetBytes)
    }

    mutating func reset(ramBudgetBytes: Int64, diskBudgetBytes: Int64 = 32_768 * 1_048_576) {
        core = TieredKVHierarchyCore(config: Self.makeConfig(ramBudgetBytes: ramBudgetBytes, diskBudgetBytes: diskBudgetBytes))
    }

    mutating func recordStore(namespace: String, tokens: [Int], bytes: Int64, layerCount: Int) {
        guard bytes > 0, !tokens.isEmpty else { return }
        let chunks = core.aggregateChunks(tokens: tokens)
        guard !chunks.isEmpty else { return }
        let layers = max(1, layerCount)
        let totalTokens = max(1, tokens.count)
        let hashes = Self.prefixHashes(namespace: namespace, chunks: chunks)
        for (chunk, hash) in zip(chunks, hashes) {
            let chunkBytes = max(1, (bytes * Int64(chunk.count)) / Int64(totalTokens))
            try? core.storeChunk(prefixHash: hash, tokenCount: chunk.count, byteSize: chunkBytes, layerCount: layers)
        }
    }

    @discardableResult
    mutating func recordLookup(namespace: String, tokens: [Int]) -> TieredKVLookupResult {
        let chunks = core.aggregateChunks(tokens: tokens)
        let hashes = Self.prefixHashes(namespace: namespace, chunks: chunks)
        return core.lookupLongestPrefix(chunkHashes: hashes)
    }

    static func makeConfig(ramBudgetBytes: Int64, diskBudgetBytes: Int64) -> TieredKVConfig {
        let ram = max(Int64(1_048_576), ramBudgetBytes)
        let l0 = max(Int64(524_288), ram / 4)
        let l1 = max(Int64(0), ram - l0)
        var config = TieredKVConfig(
            topology: .unifiedMemory,
            chunkTokens: 256,
            l0CapacityBytes: l0,
            l1CapacityBytes: l1,
            l2CapacityBytes: max(Int64(0), diskBudgetBytes),
            prefetchLookaheadLayers: 1)
        config.unifiedMemoryBudgetBytes = ram
        return config
    }

    static func prefixHashes(namespace: String, chunks: [[Int]]) -> [String] {
        var state: UInt64 = 14_695_981_039_346_656_037
        for byte in namespace.utf8 {
            state ^= UInt64(byte)
            state &*= 1_099_511_628_211
        }
        var hashes: [String] = []
        hashes.reserveCapacity(chunks.count)
        for chunk in chunks {
            for token in chunk {
                state ^= UInt64(bitPattern: Int64(token))
                state &*= 1_099_511_628_211
            }
            hashes.append(String(state, radix: 16))
        }
        return hashes
    }
}
