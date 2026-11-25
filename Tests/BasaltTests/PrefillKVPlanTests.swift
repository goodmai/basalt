import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Prefill KV preallocation and synchronized donation eliminate 2x global KV transients (#769)", .serialized)
struct PrefillKVPlanTests {
    @Test("Short prefills below the threshold or models with no KVCacheSimple layers leave the plan inactive")
    func inactiveCases() {
        let short = PrefillKVPlan.make(promptTokens: PrefillKVPlan.thresholdTokens - 1, simpleLayerCount: 8, maxTokens: 32)
        #expect(!short.isActive)
        #expect(short.initialSteps.isEmpty)
        #expect(short.decodeStep == PrefillKVPlan.defaultStep)

        let warmed = PrefillKVPlan.make(
            promptTokens: 2_000, reusedTokens: 1_500, simpleLayerCount: 8, maxTokens: 32)
        #expect(!warmed.isActive)

        let noSimple = PrefillKVPlan.make(promptTokens: 32_768, simpleLayerCount: 0, maxTokens: 32)
        #expect(!noSimple.isActive)
    }

    @Test("Long cold and warm prefills stagger KVCacheSimple capacities and restore a disjoint decode stride")
    func staggeredPlanMath() {
        let cold = PrefillKVPlan.make(promptTokens: 127_078, simpleLayerCount: 8, maxTokens: 32)
        #expect(cold.isActive)
        #expect(cold.decodeStep == 8 * 256)
        #expect(cold.initialSteps == (0 ..< 8).map { 127_078 + 32 + $0 * 256 })

        let unbounded = PrefillKVPlan.make(promptTokens: 32_768, simpleLayerCount: 5, maxTokens: nil)
        #expect(unbounded.decodeStep == 5 * 256)
        #expect(unbounded.initialSteps == (0 ..< 5).map { 32_768 + 256 + $0 * 256 })

        let hugeMax = PrefillKVPlan.make(promptTokens: 16_384, reusedTokens: 4_096, simpleLayerCount: 16, maxTokens: 8_192)
        #expect(hugeMax.decodeStep == 8 * 256)
        #expect(hugeMax.initialSteps.count == 16)
        #expect(hugeMax.initialSteps[0] == 12_288 + PrefillKVPlan.maxDecodeReserve)
        #expect(hugeMax.initialSteps[8] == hugeMax.initialSteps[0])

        let quantized = PrefillKVPlan.make(
            promptTokens: 32_768, simpleLayerCount: 8, maxTokens: 512, quantizesAfterPrefill: true)
        #expect(quantized.initialSteps[0] == 32_768)
    }

    @Test("Applying and restoring a plan updates only KVCacheSimple layers and preserves rotating caches")
    func applyAndRestoreOnlyTouchesSimpleLayers() {
        let rotating = RotatingKVCache(maxSize: 1024, keep: 0)
        let simple0 = KVCacheSimple()
        let simple1 = KVCacheSimple()
        let layers: [any KVCache] = [rotating, simple0, simple1]
        let plan = PrefillKVPlan.make(promptTokens: 4_096, simpleLayerCount: 2, maxTokens: 64)

        plan.apply(to: layers)
        #expect(simple0.step == 4_096 + 64)
        #expect(simple1.step == 4_096 + 64 + 256)

        plan.restore(layers)
        #expect(simple0.step == 512)
        #expect(simple1.step == 512)
    }

    @Test("Staggered capacities keep decode reallocations disjoint across layers over multiple growth cycles")
    func decodeGrowthNeverCollidesAcrossGroups() {
        let layerCount = 5
        let plan = PrefillKVPlan.make(promptTokens: 1_024, simpleLayerCount: layerCount, maxTokens: 4)
        let layers = (0 ..< layerCount).map { _ in KVCacheSimple() }
        plan.apply(to: layers)
        let prefillK = MLXArray.zeros([1, 1, 1_024, 16], dtype: .bfloat16)
        for layer in layers {
            _ = layer.update(keys: prefillK, values: prefillK)
        }
        eval(layers.flatMap { $0.innerState() })
        plan.restore(layers)

        var capacities = layers.map { $0.innerState()[0].dim(2) }
        #expect(capacities == plan.initialSteps)

        let tokenK = MLXArray.zeros([1, 1, 1, 16], dtype: .bfloat16)
        var totalGrowths = 0
        for _ in 0 ..< (2 * plan.decodeStep + 16) {
            var grewThisStep = 0
            for (index, layer) in layers.enumerated() {
                _ = layer.update(keys: tokenK, values: tokenK)
                let newCap = layer.innerState()[0].dim(2)
                if newCap != capacities[index] {
                    grewThisStep += 1
                    capacities[index] = newCap
                }
            }
            #expect(grewThisStep <= 1)
            totalGrowths += grewThisStep
            eval(layers.flatMap { $0.innerState() })
        }
        #expect(totalGrowths >= layerCount * 2)
    }

    @Test("Preallocated KVCacheSimple holds one constant buffer capacity across all prefill chunks and matches stock output")
    func preallocatedPrefillAvoidsReallocationAcrossChunks() {
        let layerCount = 8
        let totalTokens = 1_536
        let chunkSize = 64
        let headDim = 64
        let q = MLXArray.ones([1, 2, chunkSize, headDim], dtype: .bfloat16) * Float(0.01)
        let kChunk = MLXArray.ones([1, 1, chunkSize, headDim], dtype: .bfloat16) * Float(0.01)
        let vChunk = MLXArray.ones([1, 1, chunkSize, headDim], dtype: .bfloat16) * Float(0.01)
        eval(q, kChunk, vChunk)

        func run(usePlan: Bool) -> (reallocations: Int, checksum: Float) {
            let layers = (0 ..< layerCount).map { _ in KVCacheSimple() }
            let plan = PrefillKVPlan.make(promptTokens: totalTokens, simpleLayerCount: layerCount, maxTokens: 32)
            if usePlan { plan.apply(to: layers) }
            var capacities = Array(repeating: 0, count: layerCount)
            var reallocations = 0
            var lastOut = q
            for _ in stride(from: 0, to: totalTokens, by: chunkSize) {
                for (index, layer) in layers.enumerated() {
                    let (k, v) = layer.update(keys: kChunk, values: vChunk)
                    let cap = layer.innerState()[0].dim(2)
                    if capacities[index] != 0, cap != capacities[index] {
                        reallocations += 1
                    }
                    capacities[index] = cap
                    lastOut = MLXFast.scaledDotProductAttention(
                        queries: q, keys: k, values: v, scale: 1.0 / Float(headDim).squareRoot(), mask: .causal)
                }
                eval([lastOut] + layers.flatMap { $0.innerState() })
            }
            if usePlan { plan.restore(layers) }
            return (reallocations, lastOut.asType(.float32).sum().item(Float.self))
        }

        let stock = run(usePlan: false)
        let planned = run(usePlan: true)
        #expect(stock.reallocations == layerCount * (totalTokens / 256 - 1))
        #expect(planned.reallocations == 0)
        #expect(abs(stock.checksum - planned.checksum) < 1e-2)
    }

    @Test("PrefillKVPlan matches stock TokenIterator tokens bit-for-bit on TinyModel",
          arguments: [TinyModel.Family.gemma4KEqV, TinyModel.Family.qwen35])
    func tinyModelExactParity(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let promptLen = 1_100
        let tokens = MLXArray((0 ..< promptLen).map { Int32(($0 * 17 + 3) % 256) })
        let input = LMInput(tokens: tokens)
        var params = GenerateParameters(maxTokens: 8, temperature: 0)
        params.prefill.stepSize = 128

        let stockIter = try TokenIterator(input: input, model: model, parameters: params)
        let stockTokens = Array(stockIter)

        let cache = try model.newCache(parameters: params)
        let plan = PrefillKVPlan.make(
            promptTokens: promptLen,
            simpleLayerCount: PrefillKVPlan.simpleLayers(in: cache).count,
            maxTokens: params.maxTokens)
        #expect(plan.isActive)
        plan.apply(to: cache)
        var synced = params
        synced.prefill = plan.synchronizing(params.prefill)
        let plannedIter: TokenIterator
        do {
            defer { plan.restore(cache) }
            plannedIter = try TokenIterator(input: input, model: model, cache: cache, parameters: synced)
        }
        for layer in PrefillKVPlan.simpleLayers(in: cache) {
            #expect(layer.step == plan.decodeStep)
        }
        let plannedTokens = Array(plannedIter)
        #expect(plannedTokens == stockTokens)
    }
}

extension PrefillKVPlanTests {
    @Test("Draining PromptCacheTree.store releases source KVCacheSimple buffers and preserves exact leased contents")
    func drainingStoreReleasesSourceAndPreservesLeasedKV() {
        let count = 32
        let layers = (0 ..< 4).map { layerIdx -> KVCacheSimple in
            let cache = KVCacheSimple()
            let k = MLXArray((0 ..< (count * 16)).map { Float($0 + layerIdx * 1000) }).reshaped([1, 1, count, 16]).asType(.bfloat16)
            let v = (k + Float(0.5)).asType(.bfloat16)
            _ = cache.update(keys: k, values: v)
            eval(cache.innerState())
            return cache
        }
        let expected = layers.map { $0.state.map { $0.asType(.float32).asArray(Float.self) } }
        let promptCache = PromptCache(layers)
        let tokens = Array(100 ..< (100 + count))
        var tree = PromptCacheTree(byteBudget: ByteCount(1 << 24))
        tree.store("shared", tokens: tokens, cache: promptCache, drainSource: true)

        for layer in layers {
            #expect(layer.offset == 0)
            #expect(layer.innerState().reduce(0) { $0 + $1.nbytes } == 0)
        }
        #expect(tree.storedPrefix(of: tokens, in: "shared") == count)
        guard let lease = tree.lease("shared", for: tokens + [999]) else {
            Issue.record("expected lease after draining store")
            return
        }
        #expect(lease.reusedTokens == count)
        for (index, layer) in lease.cache.layers.enumerated() {
            let actual = layer.state.map { $0.asType(.float32).asArray(Float.self) }
            #expect(actual == expected[index])
        }
        #expect(tree.checkInvariants(leases: 1).isEmpty)
        _ = consume lease
    }

    @Test("Oversized hybrid snapshots are refused before slicing and do not evict smaller cached entries")
    func oversizedHybridSnapshotSkipsStoreAndKeepsExistingEntries() throws {
        let model = try TinyModel.make(.gemma4KEqV)
        let smallTokens = Array(1 ..< 17)
        let largeTokens = Array(100 ..< 228)

        func snapshotted(_ tokens: [Int]) throws -> PromptCache {
            let cache = PromptCache(try model.newCache(parameters: nil))
            _ = model(MLXArray(tokens.map(Int32.init))[.newAxis], cache: cache.layers)
            eval(cache.layers.flatMap { $0.innerState() })
            return cache.snapshot(at: tokens.count, state: nil)
        }

        let smallSnap = try snapshotted(smallTokens)
        let largeSnap = try snapshotted(largeTokens)
        var probe = PromptCacheTree(byteBudget: ByteCount(1 << 24))
        probe.store("shared", tokens: smallTokens, cache: smallSnap)
        let exactSmallBytes = probe.byteCount
        #expect(exactSmallBytes > 0)

        var tree = PromptCacheTree(byteBudget: exactSmallBytes + ByteCount(64))
        tree.store("shared", tokens: smallTokens, cache: smallSnap)
        #expect(tree.storedPrefix(of: smallTokens, in: "shared") == smallTokens.count)

        tree.store("shared", tokens: largeTokens, cache: largeSnap, drainSource: true)
        #expect(tree.storedPrefix(of: largeTokens, in: "shared") == 0)
        #expect(tree.storedPrefix(of: smallTokens, in: "shared") == smallTokens.count)
        #expect(tree.checkInvariants(leases: 0).isEmpty)
    }
}
