import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
@testable import BasaltCore
@testable import BasaltInference

private enum ADPOTestFixtures {
    struct IntentionalAbort: Error {}

    static func quantizedTinyModel() throws -> any LanguageModel {
        let model = try TinyModel.make(.qwen3Small)
        quantize(model: model, groupSize: 32, bits: 4, mode: .affine) { _, module in
            module is Linear
        }
        try model.prepare()
        eval(model)
        return model
    }

    static func samplePairs() -> [TokenizedPreferencePair] {
        [
            TokenizedPreferencePair(
                promptIds: [10, 14, 18, 255],
                chosenIds: [21, 22, 23, 24, 25, 26],
                rejectedIds: [110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160]
            ),
            TokenizedPreferencePair(
                promptIds: [31, 35, 39, 43, 255],
                chosenIds: [50, 52, 54, 56, 58, 60, 62, 64, 66],
                rejectedIds: [180, 185, 190, 195, 200]
            ),
        ]
    }

    static func parameterBits(_ model: Module) -> [String: [UInt32]] {
        let flat = model.parameters().flattened()
        eval(flat.map(\.1))
        var out: [String: [UInt32]] = [:]
        for (key, array) in flat {
            if array.dtype == .uint32 {
                out[key] = array.asArray(UInt32.self)
            } else {
                out[key] = array.asType(.float32).asArray(Float.self).map(\.bitPattern)
            }
        }
        return out
    }

    static func randomizeTrainable(_ model: Module, seed: UInt64) {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            let updates: [(String, MLXArray)] = model.trainableParameters().flattened().map { key, array in
                (key, MLXRandom.normal(array.shape, dtype: array.dtype) * MLXArray(Float(0.25), dtype: array.dtype))
            }
            model.update(parameters: .unflattened(updates))
            eval(model.trainableParameters().flattened().map(\.1))
            model.invalidateCompiledTraces()
        }
    }
}

@Suite("ADPO / cADPO segment objective equivalence and zero-copy reference pass (#571)", .serialized)
struct ADPOObjectiveAndReferenceTests {
    @Test("ADPO with segments = 1 matches standard DPO loss and parameter gradients within 1e-6")
    func adpoOneSegmentMatchesStandardDPO() throws {
        let cases: [([Float], [Float], [Float], [Float], Float)] = [
            ([-1.2, -0.8, -1.5, -0.4, -0.9], [-2.1, -1.7, -0.6, -2.4, -1.9, -1.1, -0.9, -2.0], [-1.4, -1.0, -1.6, -0.5, -1.1], [-1.8, -1.5, -0.7, -2.1, -1.7, -1.0, -1.1, -1.8], 0.1),
            ([-0.3, -0.5, -0.2, -0.7, -0.4, -0.6, -0.9], [-1.9, -2.3, -1.4], [-0.8, -0.9, -0.6, -1.0, -0.7, -0.8, -1.1], [-1.2, -1.5, -1.1], 0.5),
            ([-2.5], [-0.4, -0.6, -0.8, -1.0], [-1.5], [-0.9, -0.7, -0.9, -1.2], 1.0),
        ]
        for (cw, cl, rw, rl, beta) in cases {
            let dpo = ADPOTrainer.standardDPOLoss(
                chosenLogProbs: MLXArray(cw),
                rejectedLogProbs: MLXArray(cl),
                refChosenLogProbs: MLXArray(rw),
                refRejectedLogProbs: MLXArray(rl),
                beta: beta
            )
            let adpo = ADPOTrainer.segmentLoss(
                chosenLogProbs: MLXArray(cw),
                rejectedLogProbs: MLXArray(cl),
                refChosenLogProbs: MLXArray(rw),
                refRejectedLogProbs: MLXArray(rl),
                config: ADPOConfig(objective: .adpo, segments: 1, beta: beta)
            ).loss
            let dpoMode = ADPOTrainer.segmentLoss(
                chosenLogProbs: MLXArray(cw),
                rejectedLogProbs: MLXArray(cl),
                refChosenLogProbs: MLXArray(rw),
                refRejectedLogProbs: MLXArray(rl),
                config: ADPOConfig(objective: .dpo, segments: 1, beta: beta)
            ).loss
            eval(dpo, adpo, dpoMode)
            #expect(Swift.abs(adpo.item(Float.self) - dpo.item(Float.self)) < 1e-6)
            #expect(Swift.abs(dpoMode.item(Float.self) - dpo.item(Float.self)) < 1e-6)
        }
        try assertGradientEquivalenceOnTinyModel()
    }

    private func assertGradientEquivalenceOnTinyModel() throws {
        let model = try TinyModel.make(.qwen3Small)
        _ = try MLXInferenceEngine.attachConfiguredAdapter(to: model, layers: [0, 1], method: .lora(rank: 4, scale: 8.0))
        defer { _ = PHMAdapterApplicator.revert(in: model) }
        let pair = ADPOTestFixtures.samplePairs()[0]
        let ref = ADPOTrainer.referenceLogProbs(for: pair, in: model)
        ADPOTestFixtures.randomizeTrainable(model, seed: 571)
        let adpoCfg = ADPOConfig(objective: .adpo, segments: 1, beta: 0.2)
        let adpoGradFn = valueAndGrad(model: model as Module) { _, _ in
            let cw = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.chosenIds)
            let cl = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.rejectedIds)
            return [ADPOTrainer.segmentLoss(chosenLogProbs: cw, rejectedLogProbs: cl, refChosenLogProbs: MLXArray(ref.chosen), refRejectedLogProbs: MLXArray(ref.rejected), config: adpoCfg).loss]
        }
        let dpoGradFn = valueAndGrad(model: model as Module) { _, _ in
            let cw = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.chosenIds)
            let cl = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.rejectedIds)
            return [ADPOTrainer.standardDPOLoss(chosenLogProbs: cw, rejectedLogProbs: cl, refChosenLogProbs: MLXArray(ref.chosen), refRejectedLogProbs: MLXArray(ref.rejected), beta: 0.2)]
        }
        let (adpoVal, adpoGrads) = adpoGradFn(model as Module, [])
        let (dpoVal, dpoGrads) = dpoGradFn(model as Module, [])
        let flatA = Dictionary(uniqueKeysWithValues: adpoGrads.flattened())
        let flatD = Dictionary(uniqueKeysWithValues: dpoGrads.flattened())
        eval(adpoVal + dpoVal + Array(flatA.values) + Array(flatD.values))
        #expect(Swift.abs(adpoVal[0].item(Float.self) - dpoVal[0].item(Float.self)) < 1e-6)
        #expect(!flatA.isEmpty)
        for (key, gradA) in flatA {
            let gradD = try #require(flatD[key])
            let diff = MLX.abs(gradA - gradD).max().item(Float.self)
            #expect(diff < 1e-6, "\(key) diff \(diff)")
        }
    }

    @Test("Padding-free segment alignment handles unequal lengths and clamps m cleanly", arguments: [1, 4, 8, 128])
    func paddingFreeSegmentAlignment(requestedSegments: Int) throws {
        for (cwLen, clLen) in [(6, 17), (23, 9), (1, 14)] {
            let (chosen, rejected) = ADPOSegments.aligned(chosenLength: cwLen, rejectedLength: clLen, segments: requestedSegments)
            let expectedCount = min(requestedSegments, min(cwLen, clLen))
            #expect(chosen.count == expectedCount && rejected.count == expectedCount)
            #expect(chosen.first?.lowerBound == 0 && chosen.last?.upperBound == cwLen)
            #expect(rejected.first?.lowerBound == 0 && rejected.last?.upperBound == clLen)
            #expect(chosen.allSatisfy { !$0.isEmpty } && rejected.allSatisfy { !$0.isEmpty })
            for idx in 0 ..< (expectedCount - 1) {
                #expect(chosen[idx].upperBound == chosen[idx + 1].lowerBound)
                #expect(rejected[idx].upperBound == rejected[idx + 1].lowerBound)
            }
        }
        let model = try TinyModel.make(.qwen3Small)
        let prompt = [10, 12, 255]
        let shortChosen = [21, 22, 23, 24, 25, 26]
        let longRejected = Array(100 ..< 115)
        let cwA = ADPOTrainer.tokenLogProbs(model: model, promptIds: prompt, responseIds: shortChosen)
        let clLong = ADPOTrainer.tokenLogProbs(model: model, promptIds: prompt, responseIds: longRejected)
        eval(cwA, clLong)
        let clamp128 = ADPOTrainer.segmentLoss(
            chosenLogProbs: cwA,
            rejectedLogProbs: clLong,
            refChosenLogProbs: cwA - MLXArray(Float(0.2)),
            refRejectedLogProbs: clLong + MLXArray(Float(0.8)),
            config: ADPOConfig(objective: .adpo, segments: 128, beta: 0.1)
        )
        let clamp6 = ADPOTrainer.segmentLoss(
            chosenLogProbs: cwA,
            rejectedLogProbs: clLong,
            refChosenLogProbs: cwA - MLXArray(Float(0.2)),
            refRejectedLogProbs: clLong + MLXArray(Float(0.8)),
            config: ADPOConfig(objective: .adpo, segments: 6, beta: 0.1)
        )
        let cadpo = ADPOTrainer.segmentLoss(
            chosenLogProbs: cwA,
            rejectedLogProbs: clLong,
            refChosenLogProbs: cwA - MLXArray(Float(0.2)),
            refRejectedLogProbs: clLong + MLXArray(Float(0.8)),
            config: ADPOConfig(objective: .cadpo, segments: 6, beta: 0.1, conservatism: 0.7)
        )
        eval(clamp128.loss, clamp6.loss, clamp6.margin, cadpo.margin)
        #expect(Swift.abs(clamp128.loss.item(Float.self) - clamp6.loss.item(Float.self)) < 1e-6)
        #expect(cadpo.margin.item(Float.self) < clamp6.margin.item(Float.self))
    }

    @Test("Zero-copy reference pass on 4-bit QLoRA shares quantized weights and restores parameters bit-identically")
    func zeroCopyReferencePassRestoresBitIdentically() throws {
        let model = try ADPOTestFixtures.quantizedTinyModel()
        let pair = ADPOTestFixtures.samplePairs()[0]
        let baseWeightIDs = model.namedModules().compactMap { path, mod -> (String, ObjectIdentifier)? in
            (mod as? QuantizedLinear).map { (path, ObjectIdentifier($0.weight)) }
        }
        #expect(!baseWeightIDs.isEmpty)
        let pristineW = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.chosenIds)
        let pristineL = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.rejectedIds)
        eval(pristineW, pristineL)
        let pristineChosen = pristineW.asArray(Float.self)
        let pristineRejected = pristineL.asArray(Float.self)

        _ = try MLXInferenceEngine.attachConfiguredAdapter(to: model, layers: [0, 1], method: .lora(rank: 4, scale: 10.0))
        defer { _ = PHMAdapterApplicator.revert(in: model) }
        ADPOTestFixtures.randomizeTrainable(model, seed: 99)
        let adaptedBefore = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.chosenIds)
        eval(adaptedBefore)
        let adaptedVals = adaptedBefore.asArray(Float.self)
        let maxShift = zip(adaptedVals, pristineChosen).map { Swift.abs($0 - $1) }.max() ?? 0
        #expect(maxShift > 1e-2)

        let bitsBefore = ADPOTestFixtures.parameterBits(model)
        let ref: ADPOReferenceLogProbs = ADPOTrainer.withReferencePolicy(in: model) {
            let currentIDs = model.namedModules().compactMap { path, mod -> (String, ObjectIdentifier)? in
                (mod as? QuantizedLinear).map { (path, ObjectIdentifier($0.weight)) }
            }
            #expect(Dictionary(uniqueKeysWithValues: currentIDs) == Dictionary(uniqueKeysWithValues: baseWeightIDs))
            let allDisabled = model.namedModules().compactMap { $0.1 as? LoRALayer }.allSatisfy { !$0.loraEnabled }
            #expect(allDisabled)
            let zeroMax = model.trainableParameters().flattened().map { MLX.abs($0.1).max().item(Float.self) }.max() ?? 1
            #expect(zeroMax == 0)
            let cw = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.chosenIds)
            let cl = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.rejectedIds)
            eval(cw, cl)
            return ADPOReferenceLogProbs(chosen: cw.asArray(Float.self), rejected: cl.asArray(Float.self))
        }
        for (lhs, rhs) in zip(ref.chosen, pristineChosen) { #expect(Swift.abs(lhs - rhs) < 1e-6) }
        for (lhs, rhs) in zip(ref.rejected, pristineRejected) { #expect(Swift.abs(lhs - rhs) < 1e-6) }
        #expect(throws: ADPOTestFixtures.IntentionalAbort.self) {
            try ADPOTrainer.withReferencePolicy(in: model) { throw ADPOTestFixtures.IntentionalAbort() }
        }
        #expect(ADPOTestFixtures.parameterBits(model) == bitsBefore)
        let adaptedAfter = ADPOTrainer.tokenLogProbs(model: model, promptIds: pair.promptIds, responseIds: pair.chosenIds)
        eval(adaptedAfter)
        #expect(adaptedAfter.asArray(Float.self).map(\.bitPattern) == adaptedVals.map(\.bitPattern))
    }
}

@Suite("ADPO / cADPO optimization, EMA QLoRA shadow weights and Sharpening-Tax guard (#571)", .serialized)
struct ADPOTrainingEMAAndSharpeningTests {
    @Test("cADPO on 4-bit QLoRA TinyModel monotonically increases preference margin, tracks EMA shadow weights and reports Sharpening-Tax metrics")
    func cadpoMonotoneMarginEMAAndDiagnostics() throws {
        let model = try ADPOTestFixtures.quantizedTinyModel()
        _ = try withRandomState(MLXRandom.RandomState(seed: 571)) {
            try MLXInferenceEngine.attachConfiguredAdapter(to: model, layers: [0, 1], method: .lora(rank: 4, scale: 10.0))
        }
        defer { _ = PHMAdapterApplicator.revert(in: model) }

        var config = ADPOConfig(objective: .cadpo, segments: 8, beta: 0.1, conservatism: 0.85, emaDecay: 0.85)
        config.diversitySamples = 4
        let options = ADPOOptimizeOptions(iterations: 8, learningRate: 5e-4, batchSize: 4, emaDecay: 0.85)
        let outcome = try ADPOTrainer.optimize(
            model: model,
            pairs: ADPOTestFixtures.samplePairs(),
            config: config,
            options: options
        )
        #expect(outcome.losses.count == 8 && outcome.margins.count == 8)
        for step in 0 ..< (outcome.margins.count - 1) {
            #expect(outcome.margins[step + 1] > outcome.margins[step], "\(outcome.margins)")
        }
        #expect(outcome.marginAfter > outcome.marginBefore)
        let firstLoss = try #require(outcome.losses.first)
        let lastLoss = try #require(outcome.losses.last)
        #expect(lastLoss < firstLoss)
        #expect(outcome.emaApplied && outcome.emaUpdates == 8)
        #expect(!outcome.sharpeningGuardTriggered && outcome.earlyStoppedStep == nil)
        #expect(outcome.entropyBefore > 0 && outcome.entropyAfter > 0)
        let expectedTax = 1.0 - (outcome.entropyAfter / outcome.entropyBefore)
        #expect(Swift.abs(outcome.sharpeningTax - expectedTax) < 1e-6)
        #expect((0.0 ... 1.0).contains(outcome.passAtKDiversity))
    }

    @Test("EMAWeightShadow tracks exact exponential moving average and restores live weights bit-identically")
    func emaShadowTrackingAndSwap() throws {
        let model = try ADPOTestFixtures.quantizedTinyModel()
        _ = try MLXInferenceEngine.attachConfiguredAdapter(to: model, layers: [0], method: .lora(rank: 2, scale: 10.0))
        defer { _ = PHMAdapterApplicator.revert(in: model) }
        let targetKey = try #require(model.trainableParameters().flattened().first?.0)
        var ema = EMAWeightShadow(decay: 0.9, model: model)
        var manual = try #require(ema.values(for: targetKey))
        for step in 1 ... 3 {
            ADPOTestFixtures.randomizeTrainable(model, seed: UInt64(100 + step))
            let live = try #require(model.trainableParameters().flattened().first { $0.0 == targetKey }?.1)
            eval(live)
            let liveVals = live.asType(.float32).asArray(Float.self)
            manual = zip(manual, liveVals).map { 0.9 * $0 + 0.1 * $1 }
            ema.update(from: model)
            let tracked = try #require(ema.values(for: targetKey))
            for (lhs, rhs) in zip(tracked, manual) {
                #expect(Swift.abs(lhs - rhs) < 1e-6)
            }
        }
        let liveBits = ADPOTestFixtures.parameterBits(model)
        let shadowMatch: [Float] = try ema.withShadowWeights(in: model) {
            let swapped = try #require(model.trainableParameters().flattened().first { $0.0 == targetKey }?.1)
            eval(swapped)
            return swapped.asType(.float32).asArray(Float.self)
        }
        for (lhs, rhs) in zip(shadowMatch, manual) { #expect(Swift.abs(lhs - rhs) < 1e-5) }
        #expect(ADPOTestFixtures.parameterBits(model) == liveBits)
    }

    @Test("Sharpening-Tax guard stops training early and rolls back collapsing updates")
    func sharpeningTaxGuardEarlyStop() throws {
        let model = try TinyModel.make(.qwen3Small)
        _ = try withRandomState(MLXRandom.RandomState(seed: 571)) {
            try MLXInferenceEngine.attachConfiguredAdapter(to: model, layers: [0, 1], method: .lora(rank: 4, scale: 10.0))
        }
        defer { _ = PHMAdapterApplicator.revert(in: model) }
        let pairs = ADPOTestFixtures.samplePairs()
        let initialEntropy = ADPOTrainer.policyEntropy(model: model, pairs: pairs)
        var strictConfig = ADPOConfig(objective: .adpo, segments: 4, beta: 0.5, emaDecay: nil)
        strictConfig.minEntropy = initialEntropy + 1.0
        strictConfig.diversitySamples = 2
        let initialBits = ADPOTestFixtures.parameterBits(model)
        let outcome = try ADPOTrainer.optimize(
            model: model,
            pairs: pairs,
            config: strictConfig,
            options: ADPOOptimizeOptions(iterations: 10, learningRate: 1e-3, batchSize: 2)
        )
        #expect(outcome.sharpeningGuardTriggered)
        #expect(outcome.earlyStoppedStep == 1)
        #expect(outcome.losses.count == 1)
        #expect(ADPOTestFixtures.parameterBits(model) == initialBits)
        #expect(ADPOMetrics.passAtKDiversity([[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]) == 0)
        #expect(ADPOMetrics.passAtKDiversity([[[1, 2], [3, 4], [5, 6]]]) == 1)
    }

    @Test("YAML config round-trips ADPO settings and engine.train runs end-to-end ADPO self-generation alignment")
    func yamlConfigAndEngineTrainIntegration() async throws {
        let parsed = try ADPOConfig.parseYAML("train: { method: adpo, segments: 128, ema: 0.995 }")
        #expect(parsed.objective == .adpo && parsed.segments == 128 && parsed.emaDecay == 0.995)
        let roundTrip = try ADPOConfig.parseYAML(parsed.toYAMLInline())
        #expect(roundTrip == parsed)
        let cadpoYAML = try ADPOConfig.parseYAML("train:\n  method: cadpo\n  segments: 64\n  conservatism: 0.75\n  ema: none")
        #expect(cadpoYAML.objective == .cadpo && cadpoYAML.segments == 64 && cadpoYAML.conservatism == 0.75 && cadpoYAML.emaDecay == nil)
        for bad in ["train: { method: dpo, segments: 8 }", "train: { method: adpo, ema: 1.5 }", "train: { unknown: 1 }"] {
            #expect(throws: BasaltError.self) { try ADPOConfig.parseYAML(bad) }
        }

        let directory = FileManager.default.temporaryDirectory.appending(path: "adpo-train-\(UUID().uuidString)/adapter")
        defer { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }
        let engine = try await TinyModel.engine(.qwen3Small)
        var gate = GateRequest(heldOut: ["What is 3 + 4?"], maxKL: 50.0, maxPerplexityRise: 50.0)
        gate.target = []
        var request = TrainRequest(tasks: .arithmetic(count: 5, seed: 11), gate: gate)
        request.layers = [0, 1]
        request.rank = 4
        request.iterations = 6
        request.learningRate = 5e-4
        request.output = directory.path
        var adpo = ADPOConfig(objective: .adpo, segments: 16, beta: 0.1, emaDecay: 0.9)
        adpo.diversitySamples = 3
        request.adpo = adpo

        let report = try await engine.train(request)
        #expect(report.verified > 0 && !report.losses.isEmpty)
        let marginBefore = try #require(report.preferenceMarginBefore)
        let marginAfter = try #require(report.preferenceMarginAfter)
        #expect(marginAfter > marginBefore)
        #expect(report.entropyBefore != nil && report.entropyAfter != nil && report.sharpeningTax != nil)
        #expect(report.passAtKDiversity != nil && report.emaApplied == true)
        #expect(FileManager.default.fileExists(atPath: directory.appending(component: "adapters.safetensors").path))
    }
}
