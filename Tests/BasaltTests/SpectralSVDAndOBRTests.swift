import Foundation
import MLX
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Spectral compression & Hessian restoration: SVD-LLM V2, SAES-SVD, HiSVD, OBR, MoE-SVD & MAEP (#552)")
struct SpectralSVDAndOBRTests {
    @Test("SVD-LLM V2 whitened truncation satisfies exact Eckart-Young-Mirsky output error identity ||X W^T - X W_r^T||_F^2 == sum_{i>r} sigma_i^2")
    func whitenedSVDEckartYoungIdentity() {
        let rng = MLXRandom.RandomState(seed: 5520)
        let (m, n, k, rank) = (64, 24, 32, 12)
        let scales = MLXArray((0 ..< k).map { Float(exp(Double($0) * 0.12)) }).reshaped(1, k)
        let x = MLXRandom.normal([m, k], dtype: .float32, key: rng) * scales
        let w = MLXRandom.normal([n, k], dtype: .float32, key: rng)
        let h = matmul(x.T, x)

        let res = InputWhitenedSVD.factorize(weight: w, hessian: h, rank: rank, epsilonFloor: 1e-12)
        let wHat = res.reconstructedWeight()
        let outDiff = matmul(x, w.T) - matmul(x, wHat.T)
        let frobSq = (outDiff * outDiff).sum().item(Float.self)
        let relDiff = abs(frobSq - res.tailEnergy) / max(res.tailEnergy, 1e-6)
        #expect(relDiff < 1e-3)

        let layer = WhitenedLowRankLinear(result: res)
        let layerOut = layer(x)
        eval(layerOut)
        let layerErr = abs(layerOut - matmul(x, wHat.T)).max().item(Float.self)
        #expect(layerErr < 1e-4)
    }

    @Test("SAES-SVD drift compensation reduces cross-layer output error vs uncompensated SVD and HiSVD respects budget")
    func saesDriftCompensationAndHiSVDBudget() {
        let rng = MLXRandom.RandomState(seed: 5521)
        let (m, n, k, rank) = (96, 20, 24, 14)
        let xClean = MLXRandom.normal([m, k], dtype: .float32, key: rng)
        let drift = matmul(xClean, MLXRandom.normal([k, k], dtype: .float32, key: rng) * Float(0.18))
        let xDrifted = xClean + drift
        let w = MLXRandom.normal([n, k], dtype: .float32, key: rng)
        let yClean = matmul(xClean, w.T)

        let naive = InputWhitenedSVD.factorize(weight: w, hessian: matmul(xClean.T, xClean), rank: rank)
        let saes = InputWhitenedSVD.factorizeWithDriftCompensation(
            weight: w, cleanInput: xClean, driftedInput: xDrifted, rank: rank)

        let errNaive = ((yClean - matmul(xDrifted, naive.reconstructedWeight().T)).square()).sum().item(Float.self)
        let errSAES = ((yClean - matmul(xDrifted, saes.reconstructedWeight().T)).square()).sum().item(Float.self)
        #expect(errSAES < errNaive)

        let summaries = [
            LayerSpectrumSummary(layerId: "sharp", outputs: 64, inputs: 64, singularValues: (0 ..< 32).map { $0 < 4 ? 10.0 : 0.1 }),
            LayerSpectrumSummary(layerId: "flat", outputs: 64, inputs: 64, singularValues: [Float](repeating: 3.0, count: 32)),
        ]
        let alloc = InputWhitenedSVD.allocateGlobalRanksHiSVD(layers: summaries, targetParamRatio: 0.45, minRank: 2)
        let sharpRank = alloc["sharp"] ?? 0
        let flatRank = alloc["flat"] ?? 0
        #expect(flatRank > sharpRank)
        let usedParams = (sharpRank + flatRank) * (64 + 64)
        #expect(usedParams <= Int(Float(2 * 64 * 64) * 0.45) + 1)
    }

    @Test("OBR closed-form Hessian restoration strictly improves both pruned MSE and post-MACaQ quantization MSE")
    func obrImprovesPrunedAndMACaQQuantizedError() throws {
        let rng = MLXRandom.RandomState(seed: 25091)
        let (m, n, k) = (128, 16, 64)
        let latent = MLXRandom.normal([m, 16], dtype: .float32, key: rng)
        let mixer = MLXRandom.normal([16, k], dtype: .float32, key: rng)
        let x = matmul(latent, mixer) + MLXRandom.normal([m, k], dtype: .float32, key: rng) * Float(0.15)
        let w = MLXRandom.normal([n, k], dtype: .float32, key: rng)
        let h = matmul(x.T, x)

        let prunedCols = Array(48 ..< 64)
        let obr = OptimalBrainRestoration.restoreColumns(weight: w, hessian: h, prunedIndices: prunedCols)
        #expect(obr.restoredLoss < obr.uncompensatedLoss * 0.5)

        var uncompMask = [Float](repeating: 1.0, count: k)
        for c in prunedCols { uncompMask[c] = 0.0 }
        let wUncomp = w * MLXArray(uncompMask).reshaped(1, k)

        func quantizeMACaQMatrix(_ mat: MLXArray) -> MLXArray {
            let pre = MACaQMath.balancePreScale(weight: mat, meanSquareActivation: diagonal(h), alpha: 0.75)
            let frame = MACaQMath.transformFrame(weight: mat, hessian: h, preScale: pre, blockSize: 32, ridge: 1.0)
            let (wq, ws, wb) = MLX.quantized(frame.weight.asType(.float16), groupSize: 32, bits: 4, mode: .affine)
            let pSpec = PVCodeGrid.Spec(groupSize: 32, bits: 4, mode: .affine)
            var grid = MACaQMath.makeGridState(packedWeight: wq, scales: ws, biases: wb, spec: pSpec).state
            _ = MACaQMath.flipRefine(state: &grid, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: 32)
            let packedW = PVCodeGrid.packCodes(grid.codes.asType(.uint8), bits: 4)
            let spec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: .affine, hadamardBlock: 32)
            let layer = MACaQQuantizedLinear(
                packed: MACaQPackedWeights(weight: packedW, scales: ws, biases: wb),
                bias: nil, preScale: pre.asType(.float16), spec: spec)
            return layer.unrotatedDenseWeight().asType(.float32)
        }

        let qUncomp = quantizeMACaQMatrix(wUncomp)
        let qOBR = quantizeMACaQMatrix(obr.restoredWeight)
        let yRef = matmul(x, w.T)
        let mseUncomp = ((yRef - matmul(x, qUncomp.T)).square()).mean().item(Float.self)
        let mseOBR = ((yRef - matmul(x, qOBR.T)).square()).mean().item(Float.self)
        #expect(mseOBR < mseUncomp)
    }

    @Test("SharedBasisSwitchLinear is exact at full rank and MAEP protects vision-critical experts under text-heavy calibration")
    func sharedBasisMoEAndMAEPScoring() {
        let rng = MLXRandom.RandomState(seed: 26110)
        let (numExperts, n, k) = (4, 12, 16)
        let x = MLXRandom.normal([20, k], dtype: .float32, key: rng)
        let h = matmul(x.T, x)
        let experts = (0 ..< numExperts).map { _ in MLXRandom.normal([n, k], dtype: .float32, key: rng) }

        let fullMoE = SharedBasisSwitchLinear.factorizeExperts(expertWeights: experts, hessian: h, rank: k, epsilonFloor: 1e-10)
        let idx = MLXArray([Int32]((0 ..< 40).map { Int32($0 % numExperts) })).reshaped(20, 2)
        let rw = MLXArray([Float](repeating: 0.5, count: 40)).reshaped(20, 2)
        let got = fullMoE(x, routerIndices: idx, routerWeights: rw)
        eval(got)

        var maxErr: Float = 0
        let idxHost = idx.asArray(Int32.self)
        for t in 0 ..< 20 {
            let e0 = Int(idxHost[t * 2])
            let e1 = Int(idxHost[t * 2 + 1])
            let ref = 0.5 * matmul(x[t ..< (t + 1)], experts[e0].T) + 0.5 * matmul(x[t ..< (t + 1)], experts[e1].T)
            eval(ref)
            maxErr = max(maxErr, abs(ref.squeezed(axis: 0) - got[t]).max().item(Float.self))
        }
        #expect(maxErr < 1e-3)

        var samples: [MAEPExpertScorer.ModalitySample] = []
        for _ in 0 ..< 20 {
            samples.append(.init(isVision: false, topExperts: [0, 1], topWeights: [0.6, 0.4], expertOutputs: [[1.0, 0.0], [0.2, 0.2]]))
        }
        samples.append(.init(isVision: true, attentionScore: 0.2, topExperts: [2, 3], topWeights: [0.85, 0.15], expertOutputs: [[4.0, -3.0], [0.1, 0.1]]))
        let scores = MAEPExpertScorer.scoreExperts(samples: samples, numExperts: 4, visionBalance: 0.5)
        #expect(scores[2] > scores[1])
        #expect(scores[2] > scores[3])
    }
}
