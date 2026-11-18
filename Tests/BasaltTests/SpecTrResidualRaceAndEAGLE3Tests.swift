import Foundation
import MLX
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Lossless OT & Residual Race speculative verification, EAGLE-3 fusion & DiffuCD (#553)")
struct SpecTrResidualRaceAndEAGLE3Tests {
    @Test("Fused Metal speculative_verify_ot_residual_race matches CPU reference bit-for-bit at T=0 and T>0 (#228)")
    func gpuAndCPUReferenceExactParity() {
        MLXRandom.seed(5530)
        let (steps, cands, vocab) = (5, 3, 96)
        let tLogits = MLXRandom.normal([steps + 1, vocab], dtype: .float32)
        let dLogits = MLXRandom.normal([steps, cands, vocab], dtype: .float32)

        var dTokHost = [Int32](repeating: 0, count: steps * cands)
        for s in 0 ..< steps {
            let win = tLogits[s].argMax().item(Int32.self)
            for m in 0 ..< cands {
                dTokHost[s * cands + m] = (s < 3 && m == cands - 1) ? win : Int32((s * 7 + m * 11) % vocab)
            }
        }
        let dTokens = MLXArray(dTokHost).reshaped(steps, cands)

        let tHost: [[Float]] = (0 ... steps).map { tLogits[$0].asArray(Float.self) }
        let dHost: [[[Float]]] = (0 ..< steps).map { s in
            (0 ..< cands).map { m in dLogits[s, m].asArray(Float.self) }
        }
        let tokHost: [[Int32]] = (0 ..< steps).map { s in
            (0 ..< cands).map { m in dTokHost[s * cands + m] }
        }

        for temp: Float in [0.0, 0.7, 1.0] {
            for mode in [SpeculativeVerifyMode.residualRace, .specTrOT] {
                for seed: UInt32 in [42, 1337, 99991] {
                    let gpu = SpecTrAndResidualRaceMetal.verifyGPU(
                        targetLogits: tLogits, draftLogits: dLogits, draftTokens: dTokens,
                        temperature: temp, mode: mode, seed: seed)
                    let cpu = SpecTrAndResidualRaceMetal.verifyCPU(
                        targetLogits: tHost, draftLogits: dHost, draftTokens: tokHost,
                        temperature: temp, mode: mode, seed: seed)
                    #expect(gpu == cpu)
                }
            }
        }
    }

    @Test("Residual Race and SpecTr k-draft OT preserve exact target distribution (TV < 0.012 over 30,000 samples) and k=3 beats k=1 acceptance")
    func exactTargetMeasurePreservationAndAcceptanceGain() {
        let targetLogits: [Float] = [1.4, 0.7, -0.2, 0.3, -1.1, 0.9, -0.5, 0.1]
        let draftLogits: [Float] = [0.6, 1.2, 0.4, -0.3, -0.8, 0.5, 0.2, -0.4]
        let temp: Float = 0.85
        let targetProbs = SpecTrAndResidualRaceMetal.softmaxHost(targetLogits, temperature: temp)
        let trials = 30_000

        var countsRace = [Int](repeating: 0, count: targetProbs.count)
        var countsOT3 = [Int](repeating: 0, count: targetProbs.count)
        var acc1 = 0
        var acc3 = 0

        for i in 0 ..< trials {
            let baseSeed = UInt32(100_000 + i * 17)
            let d1 = SpecTrAndResidualRaceMetal.draftWithRaceKey(
                logits: draftLogits, temperature: temp, seed: baseSeed, step: 0)
            let outRace = SpecTrAndResidualRaceMetal.verifyCPU(
                targetLogits: [targetLogits, targetLogits],
                draftLogits: [[draftLogits]],
                draftTokens: [[d1]],
                temperature: temp,
                mode: .residualRace,
                seed: baseSeed ^ 0xABCD_1234
            )
            countsRace[Int(outRace.emittedTokens[0])] += 1
            acc1 += outRace.acceptedCount

            let c0 = SpecTrAndResidualRaceMetal.draftWithRaceKey(logits: draftLogits, temperature: temp, seed: baseSeed, step: 1)
            let c1 = SpecTrAndResidualRaceMetal.draftWithRaceKey(logits: draftLogits, temperature: temp, seed: baseSeed, step: 2)
            let c2 = SpecTrAndResidualRaceMetal.draftWithRaceKey(logits: draftLogits, temperature: temp, seed: baseSeed, step: 3)
            let outOT = SpecTrAndResidualRaceMetal.verifyCPU(
                targetLogits: [targetLogits, targetLogits],
                draftLogits: [[draftLogits, draftLogits, draftLogits]],
                draftTokens: [[c0, c1, c2]],
                temperature: temp,
                mode: .specTrOT,
                seed: baseSeed ^ 0x55AA_77CC
            )
            countsOT3[Int(outOT.emittedTokens[0])] += 1
            acc3 += outOT.acceptedCount
        }

        var tvRace: Float = 0
        var tvOT3: Float = 0
        for v in 0 ..< targetProbs.count {
            tvRace += 0.5 * abs(Float(countsRace[v]) / Float(trials) - targetProbs[v])
            tvOT3 += 0.5 * abs(Float(countsOT3[v]) / Float(trials) - targetProbs[v])
        }
        #expect(tvRace < 0.012)
        #expect(tvOT3 < 0.012)
        #expect(acc3 > acc1)
    }

    @Test("EAGLE-3 3-layer fusion head quantizes cleanly and DiffuCD vanishes below minTimestep")
    func eagle3HeadAndDiffuCDSchedule() {
        MLXRandom.seed(25030)
        let head = EAGLE3FusionHead(hiddenSize: 64, vocabSize: 128, tapConfig: .standard(numLayers: 28))
        quantize(model: head, groupSize: 64, bits: 4)
        let low = MLXRandom.normal([1, 4, 64], dtype: .float32)
        let mid = MLXRandom.normal([1, 4, 64], dtype: .float32)
        let high = MLXRandom.normal([1, 4, 64], dtype: .float32)
        let (topIdx, logits) = head.topKCandidates(low: low, mid: mid, high: high, k: 3)
        eval(topIdx, logits)
        #expect(topIdx.shape == [1, 4, 3])
        #expect(logits.shape == [1, 4, 128])
        #expect(EAGLE3FusionHead.trimRollbackCount(draftedSteps: 6, acceptedSteps: 2) == 4)

        let sched = DiffuCDScheduler(alphaMax: 0.5, gamma: 2.0, minTimestep: 0.2)
        #expect(sched.scale(at: 0.1) == 0)
        #expect(abs(sched.scale(at: 1.0) - 0.5) < 1e-6)
        let strong = MLXArray([Float(2.0), Float(4.0)])
        let weak = MLXArray([Float(1.0), Float(2.0)])
        let guided = sched.extrapolate(strongPrediction: strong, weakPrediction: weak, normalizedTimestep: 1.0)
        eval(guided)
        #expect(abs(guided - MLXArray([Float(2.5), Float(5.0)])).max().item(Float.self) < 1e-5)
    }
}
