import Foundation
import Testing
@testable import BasaltCore

@Suite("Tiered L0/L1/L2 KV Cache (LMCache 2026) and Adaptive Heterogeneous Planner (INRIA PLAFRIM COMPAS 2026) (#620)")
struct TieredKVAndHeterogeneousPlannerTests {
    @Test("TieredKVHierarchyCore chunks tokens, performs 3-pointer L0->L1->L2 offloading, longest-prefix lookup, layer prefetch, and OpenMetrics export")
    func tieredKVLifecycleAndPrefetch() throws {
        try TieredKVConfig().validate()
        #expect(throws: BasaltError.self) { try TieredKVConfig(chunkTokens: 0).validate() }
        #expect(throws: BasaltError.self) { try TieredKVConfig(l0CapacityBytes: 0).validate() }
        #expect(throws: BasaltError.self) { try TieredKVConfig(prefetchLookaheadLayers: -1).validate() }

        var kv = TieredKVHierarchyCore(config: TieredKVConfig(
            chunkTokens: 4,
            l0CapacityBytes: 100,
            l1CapacityBytes: 100,
            l2CapacityBytes: 100,
            prefetchLookaheadLayers: 1
        ))
        #expect(kv.metrics.hitRate == 0.0)
        #expect(kv.aggregateChunks(tokens: [1, 2, 3, 4, 5, 6]).count == 2)
        #expect(kv.aggregateChunks(tokens: []).isEmpty)
        #expect(throws: BasaltError.self) {
            try kv.storeChunk(prefixHash: "bad", tokenCount: 0, byteSize: 50, layerCount: 4)
        }

        try kv.storeChunk(prefixHash: "c1", tokenCount: 4, byteSize: 80, layerCount: 4)
        try kv.storeChunk(prefixHash: "c2", tokenCount: 4, byteSize: 80, layerCount: 4)
        try kv.storeChunk(prefixHash: "c3", tokenCount: 4, byteSize: 80, layerCount: 4)
        #expect(kv.block(for: "c3")?.tier == .l0GPU)
        #expect(kv.block(for: "c2")?.tier == .l1HostRAM)
        #expect(kv.block(for: "c1")?.tier == .l2SharedNVMe)
        #expect(kv.metrics.demotions == 3 && kv.metrics.evictions == 0)

        let lookup = kv.lookupLongestPrefix(chunkHashes: ["c3", "c2", "c1", "missing"])
        #expect(lookup.matchedTokens == 12)
        #expect(lookup.deepestTierUsed == .l2SharedNVMe)
        #expect(lookup.prefetchSchedule.count == 6)
        #expect(lookup.prefetchSchedule[0].computeLayer == 0 && lookup.prefetchSchedule[0].prefetchLayer == 1)
        #expect(lookup.prefetchSchedule[0].zeroCopyRebind && lookup.prefetchSchedule[0].bytesPerLayer == 0)
        #expect(!lookup.prefetchSchedule[3].zeroCopyRebind && lookup.prefetchSchedule[3].bytesPerLayer > 0)
        #expect(kv.metrics.zeroCopyPromotions == 1)
        #expect(kv.metrics.l0Hits == 1 && kv.metrics.l1Hits == 1 && kv.metrics.l2Hits == 1 && kv.metrics.misses == 1)
        #expect(abs(kv.metrics.hitRate - 0.75) < 1e-6)

        var umaCapConfig = TieredKVConfig(
            topology: .unifiedMemory,
            chunkTokens: 4,
            l0CapacityBytes: 100,
            l1CapacityBytes: 100,
            l2CapacityBytes: 200
        )
        umaCapConfig.unifiedMemoryBudgetBytes = 120
        var umaKV = TieredKVHierarchyCore(config: umaCapConfig)
        try umaKV.storeChunk(prefixHash: "u1", tokenCount: 4, byteSize: 80, layerCount: 4)
        try umaKV.storeChunk(prefixHash: "u2", tokenCount: 4, byteSize: 80, layerCount: 4)
        #expect(umaKV.metrics.unifiedRAMBytes <= 120)
        #expect(umaKV.block(for: "u1")?.tier == .l2SharedNVMe)

        var badUma = TieredKVConfig(topology: .unifiedMemory, l0CapacityBytes: 100, l1CapacityBytes: 100)
        badUma.unifiedMemoryBudgetBytes = 50
        #expect(throws: BasaltError.self) { try badUma.validate() }

        var discreteKV = TieredKVHierarchyCore(config: TieredKVConfig(
            topology: .discreteVRAM,
            chunkTokens: 4,
            l0CapacityBytes: 100,
            l1CapacityBytes: 100,
            l2CapacityBytes: 100
        ))
        try discreteKV.storeChunk(prefixHash: "d1", tokenCount: 4, byteSize: 80, layerCount: 4)
        try discreteKV.storeChunk(prefixHash: "d2", tokenCount: 4, byteSize: 80, layerCount: 4)
        let discreteLookup = discreteKV.lookupLongestPrefix(chunkHashes: ["d1"])
        #expect(discreteLookup.prefetchSchedule.first?.zeroCopyRebind == false)
        #expect((discreteLookup.prefetchSchedule.first?.bytesPerLayer ?? 0) > 0)

        try kv.storeChunk(prefixHash: "c4", tokenCount: 4, byteSize: 80, layerCount: 4)
        #expect(kv.metrics.evictions >= 1)

        var snapshot = ServerTelemetrySnapshot()
        snapshot.tieredKV = kv.metrics
        let exported = TelemetryExport.openMetrics(snapshot)
        #expect(exported.contains("basalt_tiered_kv_hits_total{tier=\"l0_gpu\"} 1"))
        #expect(exported.contains("basalt_tiered_kv_hits_total{tier=\"l2_shared_nvme\"} 1"))
        #expect(exported.contains("basalt_tiered_kv_hit_ratio"))
    }

    @Test("AdaptiveHeterogeneousPlanner balances runtime vs energy across heterogeneous nodes and respects memory capacity")
    func adaptiveHeterogeneousSplitting() throws {
        #expect(throws: BasaltError.self) {
            _ = try AdaptiveHeterogeneousPlanner.plan(
                workload: SplittableWorkload(workloadId: "w", totalWorkUnits: 0),
                nodes: [HeterogeneousNodeProfile(nodeId: "n1", effectiveTFLOPS: 20, powerWatts: 40, availableMemoryBytes: 1_000_000)]
            )
        }
        #expect(throws: BasaltError.self) {
            _ = try AdaptiveHeterogeneousPlanner.plan(
                workload: SplittableWorkload(workloadId: "w", totalWorkUnits: 10, memoryBytesPerUnit: -1),
                nodes: [HeterogeneousNodeProfile(nodeId: "n1", effectiveTFLOPS: 20, powerWatts: 40, availableMemoryBytes: 1_000_000)]
            )
        }
        #expect(throws: BasaltError.self) {
            _ = try AdaptiveHeterogeneousPlanner.plan(
                workload: SplittableWorkload(workloadId: "w", totalWorkUnits: 10, energyWeight: 1.5),
                nodes: [HeterogeneousNodeProfile(nodeId: "n1", effectiveTFLOPS: 20, powerWatts: 40, availableMemoryBytes: 1_000_000)]
            )
        }
        #expect(throws: BasaltError.self) {
            _ = try AdaptiveHeterogeneousPlanner.plan(
                workload: SplittableWorkload(workloadId: "w", totalWorkUnits: 10),
                nodes: []
            )
        }

        let peer = PeerCapability(peerId: "mac-ultra", chipName: "M3 Ultra", unifiedMemoryBytes: 64_000_000_000, measuredTFLOPS: 60, thermalFactor: 1.0, linkGbps: 80)
        let fastNode = HeterogeneousNodeProfile(capability: peer, powerWatts: 140)
        let frugalNode = HeterogeneousNodeProfile(nodeId: "mac-mini", effectiveTFLOPS: 20, powerWatts: 22, availableMemoryBytes: 16_000_000_000, linkGbps: 40)

        let latencyPlan = try AdaptiveHeterogeneousPlanner.plan(
            workload: SplittableWorkload(workloadId: "prefill-fast", totalWorkUnits: 100, minChunkUnits: 5, memoryBytesPerUnit: 1_000_000, energyWeight: 0.0),
            nodes: [fastNode, frugalNode]
        )
        #expect(latencyPlan.unassignedUnits == 0)
        let fastUnitsLatency = latencyPlan.assignments.first { $0.nodeId == "mac-ultra" }?.assignedUnits ?? 0
        let frugalUnitsLatency = latencyPlan.assignments.first { $0.nodeId == "mac-mini" }?.assignedUnits ?? 0
        #expect(fastUnitsLatency > frugalUnitsLatency)

        let ecoPlan = try AdaptiveHeterogeneousPlanner.plan(
            workload: SplittableWorkload(workloadId: "prefill-eco", totalWorkUnits: 100, minChunkUnits: 5, memoryBytesPerUnit: 1_000_000, energyWeight: 0.95),
            nodes: [fastNode, frugalNode]
        )
        #expect(ecoPlan.unassignedUnits == 0)
        #expect(ecoPlan.totalEnergyJoules < latencyPlan.totalEnergyJoules)

        let tightMem = try AdaptiveHeterogeneousPlanner.plan(
            workload: SplittableWorkload(workloadId: "tight", totalWorkUnits: 50, minChunkUnits: 5, memoryBytesPerUnit: 1_000_000_000, energyWeight: 0.2),
            nodes: [HeterogeneousNodeProfile(nodeId: "small", effectiveTFLOPS: 10, powerWatts: 15, availableMemoryBytes: 10_000_000_000)]
        )
        #expect(tightMem.assignments.first?.assignedUnits == 10 && tightMem.unassignedUnits == 40)
    }
}
