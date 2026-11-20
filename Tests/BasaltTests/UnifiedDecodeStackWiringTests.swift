import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("Unified live decode stack wiring (#634)")
struct UnifiedDecodeStackWiringTests {
    @Test("DraftBackoff wires DynamicTreeBudget for thermal state, low power, batch size, and acceptance EMA")
    func dynamicTreeBudgetBackoffWiring() {
        var backoff = DraftBackoff()
        backoff.record(acceptedDrafts: 12, drafted: 12)
        #expect(backoff.effectiveLimit(thermalLevel: .nominal, isLowPowerModeEnabled: false, activeBatchSize: 1) == backoff.draftLimit)
        let serious = backoff.effectiveLimit(thermalLevel: .serious, isLowPowerModeEnabled: false, activeBatchSize: 1)
        let critical = backoff.effectiveLimit(thermalLevel: .critical, isLowPowerModeEnabled: false, activeBatchSize: 1)
        let batched = backoff.effectiveLimit(thermalLevel: .nominal, isLowPowerModeEnabled: false, activeBatchSize: 4)
        #expect(serious < backoff.draftLimit)
        #expect(critical <= serious)
        #expect(batched < backoff.draftLimit)

        for _ in 0 ..< 8 {
            backoff.record(acceptedDrafts: 1, drafted: 10)
        }
        #expect(backoff.acceptanceEMA.value < 0.35)
        #expect(backoff.effectiveLimit(thermalLevel: .nominal) <= 6)
    }

    @Test("SmallMQuantizedLinear.install enables small-M Metal verify path on 4-bit affine MACaQQuantizedLinear")
    func smallMMACaQParity() {
        let w = MLXRandom.normal([64, 128]).asType(.bfloat16)
        let preScale = MLXArray.ones([128], dtype: .bfloat16)
        let rotated = MACaQMath.rotateBlocks(w.asType(.float32), blockSize: 64)
        let (qw, sc, bi) = quantized(rotated.asType(.bfloat16), groupSize: 64, bits: 4, mode: .affine)
        let spec = MACaQLayerSpec(bits: 4, groupSize: 64, mode: .affine, hadamardBlock: 64)
        let layer = MACaQQuantizedLinear(
            packed: MACaQPackedWeights(weight: qw, scales: sc, biases: bi),
            bias: nil, preScale: preScale, spec: spec)

        let x = MLXRandom.normal([1, 4, 128]).asType(.bfloat16)
        let baseline = layer(x).asType(.float32)
        let installed = SmallMQuantizedLinear.install(in: layer)
        #expect(installed == 1)
        #expect(layer.useSmallMKernel)
        let smallM = layer(x).asType(.float32)
        let maxDiff = MLX.abs(baseline - smallM).max().item(Float.self)
        #expect(maxDiff < 0.05)
    }

    @Test("PromptCacheTree wires TieredKVHierarchyCore and exports L0/L1/L2 telemetry")
    func tieredKVPromptCacheWiring() throws {
        var tree = PromptCacheTree(byteBudget: ByteCount(16 * 1_048_576))
        let k = MLXArray.zeros([1, 2, 300, 32], dtype: .float16)
        let v = MLXArray.zeros([1, 2, 300, 32], dtype: .float16)
        let layer = KVCacheSimple()
        _ = layer.update(keys: k, values: v)
        let cache = PromptCache([layer])
        let tokens = Array(1 ... 300)
        tree.store("ns", tokens: tokens, cache: cache)
        let leased = tree.lease("ns", for: tokens + [301]) != nil
        #expect(leased)
        let taken = tree.takeTelemetryDelta()
        let delta = try #require(taken)
        let tiered = try #require(delta.tiered)
        #expect(tiered.l0Hits + tiered.l1Hits >= 1)
        #expect(tiered.unifiedRAMBytes > 0)
    }

    @Test("FreshCache and GenerationPlan route SageMX, MoBA, and KVarN backends cleanly")
    func freshCacheAndPlanRouting() {
        let sagePlan = GenerationPlan.make(
            GenerationRequest(prompt: "hi"),
            GenerationPlan.Settings(attentionBackend: .sagemx, capabilities: .unrestricted),
            GenerationPlan.Facts())
        #expect(sagePlan.attentionBackend == .sagemx)

        let mobaFresh = FreshCache.plan(options: LaunchOptions(params: GenerateParameters(), attentionBackend: .moba, mobaTopK: 6))
        #expect(mobaFresh == .sageMX(.moba(topBlocks: 6)))

        let kvarnFresh = FreshCache.plan(options: LaunchOptions(params: GenerateParameters(), kvPrecision: .kvarnQJL))
        #expect(kvarnFresh == .kvarn(useQJL: true))
    }

    @Test("SageMXKVCache (sagemx & moba) and KVarNKVCache decode through KVCacheAttentionProtocol on TinyModel")
    func liveAttentionAndKVarNBackendsOnTinyModel() throws {
        let model = try TinyModel.make(.qwen3HeadDim32)
        #expect(try SageMXKVCache.modelAttendsThroughCache(model))
        #expect(try KVarNKVCache.modelAttendsThroughCache(model, useQJL: true))

        let prompt = Array(repeating: [3, 5, 7, 11, 13, 17, 19, 23], count: 16).flatMap { $0 }
        let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
        let params = GenerateParameters(maxTokens: 6, temperature: 0)

        let baseline = Array(try TokenIterator(input: input, model: model, parameters: params))
        let sageCache = SageMXKVCache.replacingFullAttention(try model.newCache(parameters: nil), mode: .sagemx)
        let sageTokens = Array(try TokenIterator(input: input, model: model, cache: sageCache, parameters: params))
        #expect(sageTokens.count == baseline.count)
        #expect(sageCache.compactMap { $0 as? SageMXKVCache }.allSatisfy { $0.kernelSteps > 0 })

        let kvarnCache = KVarNKVCache.replacingFullAttention(try model.newCache(parameters: nil), useQJL: true)
        let kvarnTokens = Array(try TokenIterator(input: input, model: model, cache: kvarnCache, parameters: params))
        #expect(kvarnTokens.count == baseline.count)
        #expect(kvarnCache.compactMap { $0 as? KVarNKVCache }.allSatisfy { $0.compressedSteps > 0 })

        let longPrompt = Array(repeating: [3, 5, 7, 11, 13, 17, 19, 23], count: 40).flatMap { $0 }
        let longInput = LMInput(tokens: MLXArray(longPrompt.map(Int32.init)))
        let mobaCache = SageMXKVCache.replacingFullAttention(try model.newCache(parameters: nil), mode: .moba(topBlocks: 2))
        let mobaTokens = Array(try TokenIterator(input: longInput, model: model, cache: mobaCache, parameters: params))
        #expect(mobaTokens.count == 6)
        #expect(mobaCache.compactMap { $0 as? SageMXKVCache }.allSatisfy { $0.kernelSteps > 0 })
    }
}
