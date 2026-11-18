import BasaltCore
import Foundation
import MLX
import MLXNN

package struct WhitenedSVDResult {
    package var factorA: MLXArray
    package var factorB: MLXArray
    package var singularValues: MLXArray
    package var tailEnergy: Float
    package var rank: Int

    package func reconstructedWeight() -> MLXArray {
        matmul(factorA, factorB)
    }
}

package struct LayerSpectrumSummary: Sendable {
    package var layerId: String
    package var outputs: Int
    package var inputs: Int
    package var singularValues: [Float]

    package init(layerId: String, outputs: Int, inputs: Int, singularValues: [Float]) {
        self.layerId = layerId
        self.outputs = outputs
        self.inputs = inputs
        self.singularValues = singularValues
    }
}

package enum InputWhitenedSVD {
    package static func computeWhiteningPair(
        hessian: MLXArray,
        epsilonFloor: Float = 1e-6
    ) -> (s: MLXArray, sInv: MLXArray) {
        let sym = 0.5 * (hessian.asType(.float32) + hessian.asType(.float32).T)
        let (eigVals, eigVecs) = MLXLinalg.eigh(sym, stream: .cpu)
        eval(eigVals, eigVecs)
        let valsHost = eigVals.asArray(Float.self)
        let maxVal = max(valsHost.max() ?? 1.0, 1e-12)
        let floorVal = max(epsilonFloor * maxVal, 1e-18)
        let sqrtVals = MLXArray(valsHost.map { max($0, floorVal).squareRoot() })
        let invSqrtVals = MLXArray(valsHost.map { 1.0 / max($0, floorVal).squareRoot() })
        let s = eigVecs * sqrtVals.reshaped(1, -1)
        let sInv = invSqrtVals.reshaped(-1, 1) * eigVecs.T
        eval(s, sInv)
        return (s, sInv)
    }

    package static func factorize(
        weight: MLXArray,
        hessian: MLXArray,
        rank: Int,
        epsilonFloor: Float = 1e-6
    ) -> WhitenedSVDResult {
        let wf = weight.asType(.float32)
        let maxRank = min(wf.dim(0), wf.dim(1))
        let r = min(max(1, rank), maxRank)
        let (s, sInv) = computeWhiteningPair(hessian: hessian, epsilonFloor: epsilonFloor)
        let ws = matmul(wf, s)
        let (u, sigma, vt) = MLXLinalg.svd(ws, stream: .cpu)
        eval(u, sigma, vt)
        let sigHost = sigma.asArray(Float.self)
        var tail: Float = 0
        if r < sigHost.count {
            for idx in r ..< sigHost.count {
                tail += sigHost[idx] * sigHost[idx]
            }
        }
        let sqrtSigR = MLXArray(sigHost.prefix(r).map { max($0, 0).squareRoot() })
        let uR = u[0..., 0 ..< r]
        let vtR = vt[0 ..< r, 0...]
        let factorA = uR * sqrtSigR.reshaped(1, r)
        let factorB = matmul(sqrtSigR.reshaped(r, 1) * vtR, sInv)
        eval(factorA, factorB)
        return WhitenedSVDResult(
            factorA: factorA,
            factorB: factorB,
            singularValues: sigma,
            tailEnergy: tail,
            rank: r
        )
    }

    package static func factorizeWithDriftCompensation(
        weight: MLXArray,
        cleanInput: MLXArray,
        driftedInput: MLXArray,
        rank: Int,
        ridge: Float = 1e-5,
        epsilonFloor: Float = 1e-6
    ) -> WhitenedSVDResult {
        let wf = weight.asType(.float32)
        let xClean = cleanInput.asType(.float32)
        let xDrift = driftedInput.asType(.float32)
        let k = xDrift.dim(-1)
        let hDrift = matmul(xDrift.T, xDrift)
        let reg = hDrift + MLXArray.eye(k, dtype: .float32) * ridge
        let cross = matmul(xDrift.T, matmul(xClean, wf.T))
        let targetWT = MLXLinalg.solve(reg, cross, stream: .cpu)
        eval(targetWT)
        return factorize(
            weight: targetWT.T,
            hessian: hDrift,
            rank: rank,
            epsilonFloor: epsilonFloor
        )
    }

    package static func allocateGlobalRanksHiSVD(
        layers: [LayerSpectrumSummary],
        targetParamRatio: Float,
        minRank: Int = 1
    ) -> [String: Int] {
        let clampedRatio = min(max(targetParamRatio, 0.05), 1.0)
        let totalDense = layers.reduce(0) { $0 + $1.outputs * $1.inputs }
        let paramBudget = Int(Float(totalDense) * clampedRatio)
        var ranks: [String: Int] = [:]
        var usedParams = 0
        struct Candidate {
            var layerIdx: Int
            var normEnergy: Float
            var cost: Int
        }
        var candidates: [Candidate] = []
        for (idx, layer) in layers.enumerated() {
            let maxUseful = min(min(layer.outputs, layer.inputs), (layer.outputs * layer.inputs) / max(1, layer.outputs + layer.inputs))
            let baseR = min(max(1, minRank), max(1, maxUseful))
            ranks[layer.layerId] = baseR
            let costPerRank = layer.outputs + layer.inputs
            usedParams += baseR * costPerRank
            let energies = layer.singularValues.map { $0 * $0 }
            let sumE = max(energies.reduce(0, +), 1e-12)
            if baseR < maxUseful && baseR < energies.count {
                for rIdx in baseR ..< min(maxUseful, energies.count) {
                    candidates.append(Candidate(
                        layerIdx: idx,
                        normEnergy: (energies[rIdx] / sumE) / Float(max(1, costPerRank)),
                        cost: costPerRank
                    ))
                }
            }
        }
        candidates.sort { $0.normEnergy > $1.normEnergy }
        for cand in candidates {
            if usedParams + cand.cost > paramBudget { continue }
            let id = layers[cand.layerIdx].layerId
            ranks[id, default: 1] += 1
            usedParams += cand.cost
        }
        return ranks
    }
}

package final class WhitenedLowRankLinear: Module, UnaryLayer {
    @ModuleInfo(key: "down_proj") package var downProj: Linear
    @ModuleInfo(key: "up_proj") package var upProj: Linear
    package let rank: Int

    package init(result: WhitenedSVDResult, bias: MLXArray? = nil, dtype: DType = .float32) {
        self.rank = result.rank
        self._downProj.wrappedValue = Linear(weight: result.factorB.asType(dtype), bias: nil)
        self._upProj.wrappedValue = Linear(weight: result.factorA.asType(dtype), bias: bias?.asType(dtype))
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        upProj(downProj(x))
    }
}
