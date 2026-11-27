import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

private final class StubRecurrentCache: KVCache {
    var offset = 0
    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }
    var state: [MLXArray]
    var metaState: [String] = []

    init(state: [MLXArray]) {
        self.state = state
    }

    func innerState() -> [MLXArray] { state }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        StubRecurrentCache(state: state)
    }
}

@Suite("Hybrid recurrent state health and <think> reasoning telemetry (#860)", .serialized)
struct HybridAndReasoningTelemetryTests {
    struct LiveFamilySpec: Sendable, CustomTestStringConvertible {
        let family: TinyModel.Family
        let expectedLayers: Int
        let expectedRecurrent: Int
        let expectedKind: String

        var testDescription: String { family.rawValue }
    }

    static let liveFamilies: [LiveFamilySpec] = [
        LiveFamilySpec(family: .gdn2, expectedLayers: 4, expectedRecurrent: 3, expectedKind: "arrays"),
        LiveFamilySpec(family: .qwen35, expectedLayers: 4, expectedRecurrent: 2, expectedKind: "mamba"),
        LiveFamilySpec(family: .xlstm, expectedLayers: 3, expectedRecurrent: 3, expectedKind: "xlstm_mlstm"),
        LiveFamilySpec(family: .mamba3Siso, expectedLayers: 3, expectedRecurrent: 3, expectedKind: "arrays"),
        LiveFamilySpec(family: .tttLinear, expectedLayers: 2, expectedRecurrent: 2, expectedKind: "fast_weight"),
    ]

    @Test("nGramRepetitionRate returns exact repetition fractions on unique, period-4 loop, constant, and short sequences (#228)")
    func nGramRepetitionRateExactMath() {
        let unique = [1, 2, 3, 4, 5, 6, 7, 8]
        #expect(ReasoningTelemetryMath.nGramRepetitionRate(tokens: unique, n: 4) == 0.0)

        let period4 = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        let period4Rate = ReasoningTelemetryMath.nGramRepetitionRate(tokens: period4, n: 4)
        #expect(Swift.abs(period4Rate - 0.75) < 1e-12)

        let constant = [7, 7, 7, 7, 7, 7]
        #expect(ReasoningTelemetryMath.nGramRepetitionRate(tokens: constant, n: 4) == 1.0)

        #expect(ReasoningTelemetryMath.nGramRepetitionRate(tokens: [], n: 4) == 0.0)
        #expect(ReasoningTelemetryMath.nGramRepetitionRate(tokens: [1, 2], n: 4) == 0.0)
        #expect(ReasoningTelemetryMath.nGramRepetitionRate(tokens: [1, 2, 3, 4], n: 4) == 0.0)
        #expect(ReasoningTelemetryMath.nGramRepetitionRate(tokens: period4, n: 0) == 0.0)

        let bigramLoop = [1, 2, 1, 2, 1]
        let bigramRate = ReasoningTelemetryMath.nGramRepetitionRate(tokens: bigramLoop, n: 2)
        #expect(Swift.abs(bigramRate - (2.0 / 3.0)) < 1e-12)
    }

    @Test("splitThinkingAndAnswer handles closed, unclosed budget-truncated, untagged, prompt-opened, and multi-block traces")
    func splitThinkingAndAnswerCases() {
        let closed = ReasoningTelemetryMath.splitThinkingAndAnswer(text: "<think>step 1\nstep 2</think>Final 42")
        #expect(closed.thinking == "step 1\nstep 2" && closed.answer == "Final 42" && !closed.unclosed)

        let unclosed = ReasoningTelemetryMath.splitThinkingAndAnswer(text: "<think>still checking edge cases")
        #expect(unclosed.thinking == "still checking edge cases" && unclosed.answer.isEmpty && unclosed.unclosed)

        let plain = ReasoningTelemetryMath.splitThinkingAndAnswer(text: "Direct answer without tags.")
        #expect(plain.thinking.isEmpty && plain.answer == "Direct answer without tags." && !plain.unclosed)

        let promptOpened = ReasoningTelemetryMath.splitThinkingAndAnswer(text: "implicit thought</think>42")
        #expect(promptOpened.thinking == "implicit thought" && promptOpened.answer == "42" && !promptOpened.unclosed)

        let multi = ReasoningTelemetryMath.splitThinkingAndAnswer(text: "<think>a</think>x<think>b</think>y")
        #expect(multi.thinking == "ab" && multi.answer == "xy" && !multi.unclosed)

        let emptyTag = ReasoningTelemetryMath.splitThinkingAndAnswer(text: "<think>x</think>", openTag: "", closeTag: "</think>")
        #expect(emptyTag.thinking.isEmpty && emptyTag.answer == "<think>x</think>" && !emptyTag.unclosed)
    }

    @Test("ThinkingTelemetryReducer computes ratios, loop rate, budget truncation rate, entropy drop, window eviction, and reset")
    func thinkingTelemetryReducerAndEntropyDrop() throws {
        let drop = try #require(ReasoningTelemetryMath.postThinkEntropyDrop(thinkingEntropy: 2.2, answerEntropy: 0.7))
        #expect(Swift.abs(drop - 1.5) < 1e-12)
        #expect(ReasoningTelemetryMath.postThinkEntropyDrop(thinkingEntropy: nil, answerEntropy: 0.7) == nil)
        #expect(ReasoningTelemetryMath.postThinkEntropyDrop(thinkingEntropy: .nan, answerEntropy: 0.5) == nil)
        #expect(ReasoningTelemetryMath.postThinkEntropyDrop(thinkingEntropy: 1.0, answerEntropy: .infinity) == nil)

        var reducer = ThinkingTelemetryReducer(windowCapacity: 2)
        #expect(reducer.isEmpty && reducer.snapshot() == ReasoningTelemetrySnapshot())

        let turn1 = ReasoningTurnRecord(
            thinkingTokens: 100, answerTokens: 50, hitTokenBudgetBeforeClose: false,
            thinkingTokenIDs: [1, 2, 3, 4, 5, 6, 7, 8], meanThinkingEntropy: 2.0, meanAnswerEntropy: 0.5
        )
        let turn2 = ReasoningTurnRecord(
            thinkingTokens: 200, answerTokens: 50, hitTokenBudgetBeforeClose: true,
            thinkingTokenIDs: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            meanThinkingEntropy: 1.8, meanAnswerEntropy: 0.8
        )
        reducer.record(turn1)
        reducer.record(turn2)

        let snap = reducer.snapshot()
        #expect(snap.sampleCount == 2 && snap.totalThinkingTokens == 300 && snap.totalAnswerTokens == 100)
        #expect(Swift.abs(snap.thinkingToAnswerRatio - 3.0) < 1e-12)
        #expect(Swift.abs(snap.meanNGramLoopRate - 0.375) < 1e-12)
        #expect(Swift.abs(snap.budgetTruncationRate - 0.5) < 1e-12)
        let meanDrop = try #require(snap.meanPostThinkEntropyDrop)
        #expect(Swift.abs(meanDrop - 1.25) < 1e-12)

        let turn3 = ReasoningTurnRecord(
            thinkingTokens: 40, answerTokens: 0, hitTokenBudgetBeforeClose: true,
            thinkingTokenIDs: [9, 9, 9, 9, 9], meanThinkingEntropy: nil, meanAnswerEntropy: nil
        )
        reducer.record(turn3)
        let evicted = reducer.snapshot()
        #expect(evicted.sampleCount == 2 && evicted.totalThinkingTokens == 240 && evicted.totalAnswerTokens == 50)
        #expect(Swift.abs(evicted.budgetTruncationRate - 1.0) < 1e-12)
        #expect(Swift.abs(evicted.meanNGramLoopRate - 0.875) < 1e-12)
        let remainingDrop = try #require(evicted.meanPostThinkEntropyDrop)
        #expect(Swift.abs(remainingDrop - 1.0) < 1e-12)

        reducer.reset()
        #expect(reducer.isEmpty && reducer.snapshot() == ReasoningTelemetrySnapshot())
    }

    @Test("OpenMetrics rendering and Codable round-trip preserve recurrent state health and reasoning telemetry")
    func openMetricsAndCodableRoundTrip() throws {
        let layer0 = LayerStateHealth(
            layerIndex: 0, cacheKind: "arrays", stateTensorCount: 4,
            frobeniusNorm: 12.5, maxAbsEntry: 1.75, nonFiniteCount: 0
        ).withDiagnostics(relativeGrowthVsBaseline: 1.25, saturated: false, betaGuardTriggerRate: 0.25)
        let layer1 = LayerStateHealth(
            layerIndex: 1, cacheKind: "mamba", stateTensorCount: 2,
            frobeniusNorm: 12000.0, maxAbsEntry: 500.0, nonFiniteCount: 2
        ).withDiagnostics(relativeGrowthVsBaseline: 4.0, saturated: true, betaGuardTriggerRate: 0.75)
        let health = RecurrentStateHealthSnapshot(layerCount: 4, layers: [layer0, layer1])
        #expect(health.recurrentLayerCount == 2 && health.totalNonFiniteCount == 2 && health.saturatedLayerCount == 1)
        #expect(health.meanBetaGuardRate == 0.5 && health.maxFrobeniusNorm == 12000.0 && health.maxAbsEntry == 500.0)

        let encodedHealth = try JSONEncoder().encode(health)
        let decodedHealth = try JSONDecoder().decode(RecurrentStateHealthSnapshot.self, from: encodedHealth)
        #expect(decodedHealth == health)

        let healthText = TelemetryExport.renderRecurrentStateHealth(health)
        #expect(healthText.contains("# TYPE basalt_recurrent_state_frobenius_norm gauge"))
        #expect(healthText.contains("basalt_recurrent_state_frobenius_norm{layer=\"0\",kind=\"arrays\"} 12.5"))
        #expect(healthText.contains("basalt_recurrent_state_max_abs{layer=\"1\",kind=\"mamba\"} 500"))
        #expect(healthText.contains("basalt_recurrent_state_non_finite_total 2"))
        #expect(healthText.contains("basalt_recurrent_state_saturated_layers 1"))
        #expect(healthText.contains("basalt_recurrent_state_beta_guard_rate{layer=\"0\",kind=\"arrays\"} 0.25"))

        let reasoning = ReasoningTelemetrySnapshot(
            sampleCount: 4, totalThinkingTokens: 600, totalAnswerTokens: 200,
            meanNGramLoopRate: 0.125, budgetTruncationRate: 0.25, meanPostThinkEntropyDrop: 0.85
        )
        let encodedReasoning = try JSONEncoder().encode(reasoning)
        let decodedReasoning = try JSONDecoder().decode(ReasoningTelemetrySnapshot.self, from: encodedReasoning)
        #expect(decodedReasoning == reasoning)

        let reasoningText = TelemetryExport.renderReasoningTelemetry(reasoning)
        #expect(reasoningText.contains("basalt_reasoning_thinking_to_answer_ratio 3"))
        #expect(reasoningText.contains("basalt_reasoning_ngram_loop_rate 0.125"))
        #expect(reasoningText.contains("basalt_reasoning_budget_truncation_rate 0.25"))
        #expect(reasoningText.contains("basalt_reasoning_post_think_entropy_drop 0.85"))

        let noDrop = ReasoningTelemetrySnapshot(sampleCount: 1, totalThinkingTokens: 10, totalAnswerTokens: 10)
        #expect(!TelemetryExport.renderReasoningTelemetry(noDrop).contains("basalt_reasoning_post_think_entropy_drop"))
    }
}

extension HybridAndReasoningTelemetryTests {
    @Test("RecurrentStateInspector measures live TinyModel recurrent/hybrid caches with CPU parity and zero state mutation",
          arguments: liveFamilies)
    func liveTinyModelInspectionParityAndImmutability(spec: LiveFamilySpec) throws {
        let model = try TinyModel.make(spec.family)
        let cache = try model.newCache(parameters: nil)

        let initial = RecurrentStateInspector.inspect(cache)
        #expect(initial.layerCount == spec.expectedLayers && initial.recurrentLayerCount == spec.expectedRecurrent)
        #expect(initial.maxFrobeniusNorm == 0 && initial.totalNonFiniteCount == 0 && initial.saturatedLayerCount == 0)

        let step1 = model(MLXArray([Int32(3), 7, 11, 19])[.newAxis], cache: cache)
        eval(step1)
        let step2 = model(MLXArray([Int32(23)])[.newAxis], cache: cache)
        eval(step2)

        let beforeBytes = Self.snapshotBytes(cache)
        let cpuStats = Self.cpuReferenceStats(cache)
        let health = RecurrentStateInspector.inspect(cache)
        let afterBytes = Self.snapshotBytes(cache)

        #expect(beforeBytes == afterBytes)
        #expect(health.layerCount == spec.expectedLayers && health.recurrentLayerCount == spec.expectedRecurrent)
        #expect(health.totalNonFiniteCount == 0 && health.saturatedLayerCount == 0)
        #expect(health.maxFrobeniusNorm.isFinite && health.maxFrobeniusNorm > 0)

        for layer in health.layers {
            #expect(layer.cacheKind == spec.expectedKind && layer.stateTensorCount > 0)
            #expect(layer.frobeniusNorm.isFinite && layer.frobeniusNorm > 0)
            #expect(layer.maxAbsEntry.isFinite && layer.maxAbsEntry > 0)
            #expect(layer.nonFiniteCount == 0 && !layer.saturated)
            let expected = try #require(cpuStats[layer.layerIndex])
            #expect(Swift.abs(layer.frobeniusNorm - expected.frob) <= Swift.max(1e-3, expected.frob * 1e-4))
            #expect(Swift.abs(layer.maxAbsEntry - expected.maxAbs) <= Swift.max(1e-4, expected.maxAbs * 1e-4))
        }
    }

    @Test("RecurrentStateInspector tracks baseline relative growth, saturation threshold, non-finite entries, and custom KVCache")
    func growthSaturationNonFiniteAndCustomCache() throws {
        let model = try TinyModel.make(.gdn2)
        let cache = try model.newCache(parameters: nil)
        let out1 = model(MLXArray([Int32(5), 9, 13, 17])[.newAxis], cache: cache)
        eval(out1)
        let baseHealth = RecurrentStateInspector.inspect(cache)
        let baselines = Dictionary(uniqueKeysWithValues: baseHealth.layers.map { ($0.layerIndex, $0.frobeniusNorm) })

        let out2 = model(MLXArray([Int32(21), 25, 29, 33])[.newAxis], cache: cache)
        eval(out2)
        let grown = RecurrentStateInspector.inspect(cache, baselineNorms: baselines)
        for layer in grown.layers {
            let base = try #require(baselines[layer.layerIndex])
            let ratio = try #require(layer.relativeGrowthVsBaseline)
            #expect(Swift.abs(ratio - (layer.frobeniusNorm / base)) < 1e-9)
        }
        let saturated = RecurrentStateInspector.inspect(cache, saturationThreshold: 1e-6)
        #expect(saturated.saturatedLayerCount == 3 && saturated.layers.allSatisfy(\.saturated))

        let taintedTensor = MLXArray([Float(3.0), Float(4.0), Float.nan, Float.infinity, -Float.infinity])
        let arrays = ArraysCache(size: 1)
        arrays[0] = taintedTensor
        let stub = StubRecurrentCache(state: [taintedTensor])
        let attention = KVCacheSimple()
        let mixed = RecurrentStateInspector.inspect([arrays, attention, stub], betaGuardRates: [0: 1.0, 2: 0.5])
        #expect(mixed.layerCount == 3 && mixed.recurrentLayerCount == 2)
        #expect(mixed.totalNonFiniteCount == 6 && mixed.saturatedLayerCount == 2)
        #expect(mixed.meanBetaGuardRate == 0.75)
        for layer in mixed.layers {
            #expect(layer.nonFiniteCount == 3 && layer.saturated)
            #expect(Swift.abs(layer.frobeniusNorm - 5.0) < 1e-4)
            #expect(Swift.abs(layer.maxAbsEntry - 4.0) < 1e-4)
        }
    }

    @Test("betaGuardTriggerRate matches GatedDeltaNet2Ops.applyBetaGuard on expansive, contractive, and mixed heads")
    func betaGuardTriggerRateMatchesGDN2() {
        let dk = 32
        let invSqrt = Float(1.0 / Double(dk).squareRoot())
        let keyRow = [Float](repeating: invSqrt, count: dk)
        let expansiveRow = (0 ..< dk).map { $0 < dk / 2 ? Float(2.0) : Float(0.0) }
        let weakDecayRow = [Float](repeating: 0.95, count: dk)
        let strongDecayRow = [Float](repeating: 0.5, count: dk)

        let key1 = MLXArray(keyRow, [1, 1, 1, dk])
        let erase1 = MLXArray(expansiveRow, [1, 1, 1, dk])
        let rateTriggered = RecurrentStateInspector.betaGuardTriggerRate(
            erase: erase1, key: key1, decay: MLXArray(weakDecayRow, [1, 1, 1, dk])
        )
        let rateUntouched = RecurrentStateInspector.betaGuardTriggerRate(
            erase: erase1, key: key1, decay: MLXArray(strongDecayRow, [1, 1, 1, dk])
        )
        #expect(Swift.abs(rateTriggered - 1.0) < 1e-6)
        #expect(Swift.abs(rateUntouched) < 1e-6)

        let mixedKey = MLXArray(keyRow + keyRow, [1, 1, 2, dk])
        let mixedErase = MLXArray(expansiveRow + expansiveRow, [1, 1, 2, dk])
        let mixedDecay = MLXArray(weakDecayRow + strongDecayRow, [1, 1, 2, dk])
        let rateMixed = RecurrentStateInspector.betaGuardTriggerRate(erase: mixedErase, key: mixedKey, decay: mixedDecay)
        #expect(Swift.abs(rateMixed - 0.5) < 1e-6)
    }

    @Test("MLXInferenceEngine wired with TelemetryTracker populates state_health on TinyModel.gdn2 generation")
    func gdn2EngineGenerationPopulatesStateHealth() async throws {
        let tracker = TelemetryTracker()
        let engine = MLXInferenceEngine(.standard, telemetry: tracker)
        await engine.attach(try TinyModel.container(.gdn2))
        let response = try await engine.generate(request: GenerationRequest(prompt: "hello world", maxTokens: 3, temperature: 0))
        #expect(response.completionTokens > 0)
        let snapshot = await tracker.snapshot()
        let health = try #require(snapshot.stateHealth)
        #expect(!health.layers.isEmpty)
        #expect(health.layers.allSatisfy { $0.frobeniusNorm.isFinite && $0.frobeniusNorm >= 0 })
        #expect(health.layers.allSatisfy { $0.nonFiniteCount == 0 })
        let metrics = TelemetryExport.renderRecurrentStateHealth(health)
        #expect(metrics.contains("basalt_recurrent_state_frobenius_norm"))
    }

    private static func snapshotBytes(_ caches: [any KVCache]) -> [[Data]] {
        caches.map { cache in
            cache.state.map { $0.asData(access: .copy).data }
        }
    }

    private static func cpuReferenceStats(_ caches: [any KVCache]) -> [Int: (frob: Double, maxAbs: Double)] {
        var result: [Int: (frob: Double, maxAbs: Double)] = [:]
        for (index, cache) in caches.enumerated() {
            var sumSq = 0.0
            var maxAbs = 0.0
            for tensor in cache.state where tensor.size > 0 {
                let floats = tensor.asType(.float32).asArray(Float.self)
                for value in floats where value.isFinite {
                    let d = Double(value)
                    sumSq += d * d
                    maxAbs = Swift.max(maxAbs, Swift.abs(d))
                }
            }
            result[index] = (sumSq.squareRoot(), maxAbs)
        }
        return result
    }
}
