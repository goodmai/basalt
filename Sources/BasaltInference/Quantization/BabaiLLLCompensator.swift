import BasaltCore
import Foundation
import MLX

package struct LLLReductionResult: Sendable {
    package var reducedBasis: [Float]
    package var unimodularTransform: [Int32]
    package var dimension: Int
    package var defectBefore: Float
    package var defectAfter: Float
}

package struct TurboBoAResult {
    package var packed: MACaQPackedWeights
    package var channelOrder: MLXArray?
    package var lossRTN: Float
    package var lossBoA: Float
}

package enum BabaiLLLCompensator {
    package static func lllReduce(basis: [Float], dim n: Int, delta: Float = 0.75, maxPasses: Int = 64) -> LLLReductionResult {
        precondition(basis.count == n * n && n >= 1)
        var b = basis
        var u = [Int32](repeating: 0, count: n * n)
        for i in 0 ..< n { u[i * n + i] = 1 }
        let defBefore = orthogonalityDefect(b, dim: n)
        var (gs, mu, bNormSq) = gramSchmidt(b, dim: n)
        var k = 1
        var passes = 0
        while k < n && passes < maxPasses * n {
            passes += 1
            for j in stride(from: k - 1, through: 0, by: -1) {
                let q = Int32(rintf(mu[k * n + j]))
                if q != 0 {
                    for r in 0 ..< n {
                        b[r * n + k] -= Float(q) * b[r * n + j]
                        u[r * n + k] -= q * u[r * n + j]
                    }
                    (gs, mu, bNormSq) = gramSchmidt(b, dim: n)
                }
            }
            let lhs = bNormSq[k]
            let rhs = (delta - mu[k * n + (k - 1)] * mu[k * n + (k - 1)]) * bNormSq[k - 1]
            if lhs >= rhs - 1e-7 {
                k += 1
            } else {
                for r in 0 ..< n {
                    b.swapAt(r * n + k, r * n + (k - 1))
                    u.swapAt(r * n + k, r * n + (k - 1))
                }
                (gs, mu, bNormSq) = gramSchmidt(b, dim: n)
                k = max(k - 1, 1)
            }
        }
        _ = gs
        let defAfter = orthogonalityDefect(b, dim: n)
        return LLLReductionResult(
            reducedBasis: b, unimodularTransform: u, dimension: n,
            defectBefore: defBefore, defectAfter: defAfter)
    }

    private static func gramSchmidt(_ b: [Float], dim n: Int) -> (gs: [Float], mu: [Float], normSq: [Float]) {
        var gs = b
        var mu = [Float](repeating: 0, count: n * n)
        var normSq = [Float](repeating: 0, count: n)
        for j in 0 ..< n {
            for i in 0 ..< j {
                var dot: Float = 0
                for r in 0 ..< n { dot += b[r * n + j] * gs[r * n + i] }
                let m = normSq[i] > 1e-12 ? (dot / normSq[i]) : 0
                mu[j * n + i] = m
                for r in 0 ..< n { gs[r * n + j] -= m * gs[r * n + i] }
            }
            var nsq: Float = 0
            for r in 0 ..< n { nsq += gs[r * n + j] * gs[r * n + j] }
            normSq[j] = max(nsq, 1e-12)
        }
        return (gs, mu, normSq)
    }

    private static func orthogonalityDefect(_ b: [Float], dim n: Int) -> Float {
        let (_, _, gsNormSq) = gramSchmidt(b, dim: n)
        var logProdCol: Float = 0
        var logDetSq: Float = 0
        for j in 0 ..< n {
            var colSq: Float = 0
            for r in 0 ..< n { colSq += b[r * n + j] * b[r * n + j] }
            logProdCol += 0.5 * logf(max(colSq, 1e-20))
            logDetSq += 0.5 * logf(max(gsNormSq[j], 1e-20))
        }
        return expf(min(logProdCol - logDetSq, 80.0))
    }
}

extension BabaiLLLCompensator {
    package struct QuantizeSpec: Sendable {
        package var bits: Int
        package var groupSize: Int
        package var mode: QuantizationMode
        package var numSteps: Int
        package var reorderChannels: Bool
        package var ridge: Float

        package init(
            bits: Int = 4,
            groupSize: Int = 32,
            mode: QuantizationMode = .affine,
            numSteps: Int = 8,
            reorderChannels: Bool = false,
            ridge: Float = 0.02
        ) {
            self.bits = bits
            self.groupSize = groupSize
            self.mode = mode
            self.numSteps = max(1, numSteps)
            self.reorderChannels = reorderChannels
            self.ridge = ridge
        }
    }

    package static func turboBoAQuantize(
        weight: MLXArray,
        hessian: MLXArray,
        spec: QuantizeSpec = .init()
    ) -> TurboBoAResult {
        let wfFull = weight.asType(.float32)
        let k = wfFull.dim(1)
        var hf = Float(0.5) * (hessian.asType(.float32) + hessian.asType(.float32).T)
        let damp = spec.ridge * maximum(mean(diagonal(hf)), Float(1e-8))
        hf = hf + eye(k, dtype: .float32) * damp

        let order: MLXArray? = spec.reorderChannels ? argSort(-diagonal(hf), axis: 0).asType(.int32) : nil
        let wfOrig: MLXArray = if let order { wfFull.take(order, axis: 1) } else { wfFull }
        let hPerm: MLXArray = if let order { hf.take(order, axis: 0).take(order, axis: 1) } else { hf }

        let (rtnWQ, rtnS, rtnB) = quantizeBlock(wfOrig, hessDiag: diagonal(hPerm), spec: spec)
        let rtnDeq = MLX.dequantized(rtnWQ, scales: rtnS, biases: rtnB, groupSize: spec.groupSize, bits: spec.bits, mode: spec.mode).asType(.float32)
        let lossRTN = MACaQMath.hessianLoss(values: rtnDeq, target: wfOrig, hessian: hPerm)

        var wWork = wfOrig
        let groupsTotal = k / spec.groupSize
        let steps = min(spec.numSteps, groupsTotal)
        let groupsPerStep = max(1, (groupsTotal + steps - 1) / steps)
        var wqChunks: [MLXArray] = []
        var sChunks: [MLXArray] = []
        var bChunks: [MLXArray] = []
        let hDiag = diagonal(hPerm)

        var startGroup = 0
        while startGroup < groupsTotal {
            let endGroup = min(groupsTotal, startGroup + groupsPerStep)
            let c0 = startGroup * spec.groupSize
            let c1 = endGroup * spec.groupSize
            let slice = wWork[0..., c0 ..< c1]
            let (wq, ws, wb) = quantizeBlock(slice, hessDiag: hDiag[c0 ..< c1], spec: spec)
            wqChunks.append(wq)
            sChunks.append(ws)
            if let wb { bChunks.append(wb) }
            if c1 < k {
                let deqSlice = MLX.dequantized(wq, scales: ws, biases: wb, groupSize: spec.groupSize, bits: spec.bits, mode: spec.mode).asType(.float32)
                let err = slice - deqSlice
                let hRR = hPerm[c1 ..< k, c1 ..< k]
                let hRE = hPerm[c1 ..< k, c0 ..< c1]
                let propT = MLXLinalg.solve(hRR, matmul(hRE, err.T), stream: .cpu)
                eval(propT)
                wWork = concatenated([wWork[0..., 0 ..< c1], wWork[0..., c1 ..< k] + propT.T], axis: 1)
                eval(wWork)
            }
            startGroup = endGroup
        }

        let finalWQ = concatenated(wqChunks, axis: 1)
        let finalS = concatenated(sChunks, axis: 1)
        let finalB: MLXArray? = bChunks.isEmpty ? nil : concatenated(bChunks, axis: 1)
        let boaDeq = MLX.dequantized(finalWQ, scales: finalS, biases: finalB, groupSize: spec.groupSize, bits: spec.bits, mode: spec.mode).asType(.float32)
        let lossBoA = MACaQMath.hessianLoss(values: boaDeq, target: wfOrig, hessian: hPerm)
        if lossBoA <= lossRTN {
            return TurboBoAResult(
                packed: MACaQPackedWeights(weight: finalWQ, scales: finalS, biases: finalB),
                channelOrder: order, lossRTN: lossRTN, lossBoA: lossBoA)
        }
        return TurboBoAResult(
            packed: MACaQPackedWeights(weight: rtnWQ, scales: rtnS, biases: rtnB),
            channelOrder: order, lossRTN: lossRTN, lossBoA: lossRTN)
    }

    private static func quantizeBlock(
        _ slice: MLXArray,
        hessDiag: MLXArray,
        spec: QuantizeSpec
    ) -> (MLXArray, MLXArray, MLXArray?) {
        if spec.mode == .mxfp4 {
            let screened = MACaQMath.screenMXFP4Exponents(weight: slice, hessianDiag: hessDiag)
            let packed = MACaQMath.packMXFP4(sidx: screened.sidx, e8m0: screened.e8m0)
            return (packed.weight, packed.scales, nil)
        }
        return MLX.quantized(slice.asType(.float16), groupSize: spec.groupSize, bits: spec.bits, mode: spec.mode)
    }
}
