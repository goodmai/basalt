import BasaltCore
import Foundation
import MLX
import MLXNN

package struct OBRRestorationResult {
    package var restoredWeight: MLXArray
    package var uncompensatedLoss: Float
    package var restoredLoss: Float
}

package enum OptimalBrainRestoration {
    package static func restoreColumns(
        weight: MLXArray,
        hessian: MLXArray,
        prunedIndices: [Int],
        replacementColumns: MLXArray? = nil,
        damping: Float = 1e-4
    ) -> OBRRestorationResult {
        let wf = weight.asType(.float32)
        let hf = 0.5 * (hessian.asType(.float32) + hessian.asType(.float32).T)
        let n = wf.dim(0)
        let k = wf.dim(1)
        let prunedSet = Set(prunedIndices.filter { $0 >= 0 && $0 < k })
        guard !prunedSet.isEmpty, prunedSet.count < k else {
            return OBRRestorationResult(restoredWeight: wf, uncompensatedLoss: 0, restoredLoss: 0)
        }
        let eList = prunedSet.sorted()
        let rList = (0 ..< k).filter { !prunedSet.contains($0) }
        let eArr = MLXArray(eList.map(Int32.init))
        let rArr = MLXArray(rList.map(Int32.init))

        let wE = wf.take(eArr, axis: 1)
        let wHatE = replacementColumns?.asType(.float32) ?? MLXArray.zeros([n, eList.count], dtype: .float32)
        let errE = wHatE - wE

        let hRR = hf.take(rArr, axis: 0).take(rArr, axis: 1)
        let hRE = hf.take(rArr, axis: 0).take(eArr, axis: 1)
        let traceScale = max((hf.diagonal().sum().item(Float.self)) / Float(max(1, k)), 1e-8)
        let reg = hRR + MLXArray.eye(rList.count, dtype: .float32) * (damping * traceScale)
        let deltaWRT = -MLXLinalg.solve(reg, matmul(hRE, errE.T), stream: .cpu)
        eval(deltaWRT)
        let deltaWR = deltaWRT.T

        var uncompCols: [MLXArray] = Array(repeating: MLXArray(0), count: k)
        var restCols: [MLXArray] = Array(repeating: MLXArray(0), count: k)
        for (pos, colIdx) in eList.enumerated() {
            let c = wHatE[0..., pos ..< (pos + 1)]
            uncompCols[colIdx] = c
            restCols[colIdx] = c
        }
        let wR = wf.take(rArr, axis: 1)
        let wRUpdated = wR + deltaWR
        for (pos, colIdx) in rList.enumerated() {
            uncompCols[colIdx] = wR[0..., pos ..< (pos + 1)]
            restCols[colIdx] = wRUpdated[0..., pos ..< (pos + 1)]
        }
        let wUncomp = concatenated(uncompCols, axis: 1)
        let wRestored = concatenated(restCols, axis: 1)
        let diffUncomp = wUncomp - wf
        let diffRest = wRestored - wf
        let lossUncomp = (matmul(diffUncomp, hf) * diffUncomp).sum().item(Float.self)
        let lossRest = (matmul(diffRest, hf) * diffRest).sum().item(Float.self)
        return OBRRestorationResult(
            restoredWeight: wRestored,
            uncompensatedLoss: lossUncomp,
            restoredLoss: lossRest
        )
    }
}

package final class SharedBasisSwitchLinear: Module {
    @ModuleInfo(key: "shared_basis") package var sharedBasis: Linear
    @ParameterInfo(key: "expert_cores") package var expertCores: MLXArray
    package let numExperts: Int
    package let rank: Int

    package init(sharedBasisWeight: MLXArray, expertCores: MLXArray) {
        self.numExperts = expertCores.dim(0)
        self.rank = expertCores.dim(2)
        self._sharedBasis.wrappedValue = Linear(weight: sharedBasisWeight, bias: nil)
        self._expertCores.wrappedValue = expertCores
        super.init()
    }

    package static func factorizeExperts(
        expertWeights: [MLXArray],
        hessian: MLXArray,
        rank: Int,
        epsilonFloor: Float = 1e-6
    ) -> SharedBasisSwitchLinear {
        precondition(!expertWeights.isEmpty)
        let k = expertWeights[0].dim(1)
        let r = min(max(1, rank), k)
        let (s, sInv) = InputWhitenedSVD.computeWhiteningPair(hessian: hessian, epsilonFloor: epsilonFloor)
        let stacked = concatenated(expertWeights.map { $0.asType(.float32) }, axis: 0)
        let (_, _, vt) = MLXLinalg.svd(matmul(stacked, s), stream: .cpu)
        eval(vt)
        let vtR = vt[0 ..< r, 0...]
        let vR = vtR.T
        let sharedWeight = matmul(vtR, sInv)
        let cores = stacked.reshaped(expertWeights.count, expertWeights[0].dim(0), k)
        let projectedCores = cores.SqueezeMapped { wE in matmul(matmul(wE, s), vR) }
        eval(sharedWeight, projectedCores)
        return SharedBasisSwitchLinear(sharedBasisWeight: sharedWeight, expertCores: projectedCores)
    }

    package func callAsFunction(_ x: MLXArray, routerIndices: MLXArray, routerWeights: MLXArray) -> MLXArray {
        let origType = x.dtype
        let targetDType = (sharedBasis as? QuantizedLinear)?.scales.dtype ?? sharedBasis.weight.dtype
        let safeDType = targetDType.isFloatingPoint ? targetDType : origType
        let hShared = sharedBasis(x.asType(safeDType))
        let tokens = hShared.size / rank
        let topK = routerIndices.dim(-1)
        let flatH = hShared.reshaped(tokens, 1, 1, rank)
        let gatheredCores = expertCores.take(routerIndices.reshaped(tokens * topK), axis: 0)
            .reshaped(tokens, topK, expertCores.dim(1), rank)
        let routedOut = matmul(flatH, gatheredCores.swappedAxes(-1, -2)).squeezed(axis: 2)
        let weighted = (routedOut * routerWeights.reshaped(tokens, topK, 1)).sum(axis: 1)
        let outShape = Array(x.shape.dropLast()) + [expertCores.dim(1)]
        return weighted.reshaped(outShape).asType(origType)
    }
}

private extension MLXArray {
    func SqueezeMapped(_ transform: (MLXArray) -> MLXArray) -> MLXArray {
        var slices: [MLXArray] = []
        for idx in 0 ..< dim(0) {
            slices.append(transform(self[idx]))
        }
        return stacked(slices, axis: 0)
    }
}

package enum MAEPExpertScorer {
    package struct ModalitySample: Sendable {
        package var isVision: Bool
        package var attentionScore: Float
        package var topExperts: [Int]
        package var topWeights: [Float]
        package var expertOutputs: [[Float]]

        package init(
            isVision: Bool,
            attentionScore: Float = 1.0,
            topExperts: [Int],
            topWeights: [Float],
            expertOutputs: [[Float]]
        ) {
            self.isVision = isVision
            self.attentionScore = attentionScore
            self.topExperts = topExperts
            self.topWeights = topWeights
            self.expertOutputs = expertOutputs
        }
    }

    package static func scoreExperts(
        samples: [ModalitySample],
        numExperts: Int,
        visionBalance: Float = 0.5
    ) -> [Float] {
        var textScore = [Float](repeating: 0, count: numExperts)
        var visionScore = [Float](repeating: 0, count: numExperts)
        var textCount: Float = 0
        var visionCount: Float = 0

        for sample in samples where sample.topExperts.count >= 2 {
            let k = sample.topExperts.count
            let d = sample.expertOutputs.first?.count ?? 0
            guard d > 0 else { continue }
            let sumW = max(sample.topWeights.reduce(0, +), 1e-8)
            var fullOut = [Float](repeating: 0, count: d)
            for j in 0 ..< k {
                let w = sample.topWeights[j]
                for c in 0 ..< d { fullOut[c] += w * sample.expertOutputs[j][c] }
            }
            let sampleScale: Float = sample.isVision ? (1.0 / max(sample.attentionScore, 0.1)) : 1.0
            if sample.isVision { visionCount += sampleScale } else { textCount += 1.0 }

            for maskPos in 0 ..< k {
                let remSum = sumW - sample.topWeights[maskPos]
                let renorm: Float = remSum > 1e-8 ? (sumW / remSum) : 0
                var pertSq: Float = 0
                for c in 0 ..< d {
                    var maskedVal: Float = 0
                    for j in 0 ..< k where j != maskPos {
                        maskedVal += (sample.topWeights[j] * renorm) * sample.expertOutputs[j][c]
                    }
                    let diff = fullOut[c] - maskedVal
                    pertSq += diff * diff
                }
                let eId = sample.topExperts[maskPos]
                if eId >= 0 && eId < numExperts {
                    if sample.isVision {
                        visionScore[eId] += pertSq * sampleScale
                    } else {
                        textScore[eId] += pertSq
                    }
                }
            }
        }
        let tNorm = max(textScore.reduce(0, +) / max(textCount, 1.0), 1e-12)
        let vNorm = max(visionScore.reduce(0, +) / max(visionCount, 1.0), 1e-12)
        let beta = min(max(visionBalance, 0), 1)
        return (0 ..< numExperts).map { e in
            let tPart = (textScore[e] / max(textCount, 1.0)) / tNorm
            let vPart = (visionScore[e] / max(visionCount, 1.0)) / vNorm
            return (1.0 - beta) * tPart + beta * vPart
        }
    }
}
