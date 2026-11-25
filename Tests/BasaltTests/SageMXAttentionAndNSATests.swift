import Foundation
import MLX
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Fused Metal quantized & sparse attention: MXAttention (UOS + PNQ), SageAttention2++/3, and NSA/MoBA (#551)")
struct SageMXAttentionAndNSATests {
    @Test("MXAttention PNQ preserves exact partition-of-unity on constant V and matches pure CPU float32 reference (#228)")
    func mxAttentionPNQAndCPUReferenceParity() {
        withRandomState(MLXRandom.RandomState(seed: 5510)) { mxAttentionPNQAndCPUReferenceParityBody() }
    }

    private func mxAttentionPNQAndCPUReferenceParityBody() {
        let (heads, lQ, lK, d) = (2, 12, 24, 64)
        let q = MLXRandom.normal([heads, lQ, d], dtype: .float32) + MLXRandom.normal([heads, 1, d], dtype: .float32) * Float(3.0)
        let k = MLXRandom.normal([heads, lK, d], dtype: .float32) + MLXRandom.normal([heads, 1, d], dtype: .float32) * Float(4.0)
        let vConst = MLXArray.ones([heads, lK, d], dtype: .float32) * Float(2.5)
        let scale = 1.0 / Float(d).squareRoot()

        let constOut = SageMXAttentionMetal.attention(
            q: q, k: k, v: vConst,
            options: .init(scale: scale, causal: false, useUOS: true, usePNQ: true, meanCenter: true))
        eval(constOut)
        #expect(abs(constOut - Float(2.5)).max().item(Float.self) < 1e-4)

        let vRand = MLXRandom.normal([heads, lK, d], dtype: .float32)
        for causal in [false, true] {
            for dtype in [DType.float32, .float16, .bfloat16] {
                let opts = SageMXAttentionMetal.Options(
                    scale: scale, causal: causal, useUOS: true, usePNQ: true, meanCenter: true)
                let qCast = q.asType(dtype)
                let kCast = k.asType(dtype)
                let vCast = vRand.asType(dtype)
                let gpu = SageMXAttentionMetal.attention(q: qCast, k: kCast, v: vCast, options: opts).asType(.float32)
                eval(gpu)
                let cpu = SageMXAttentionMetal.cpuAttention(
                    q: qCast.asType(.float32).asArray(Float.self),
                    k: kCast.asType(.float32).asArray(Float.self),
                    v: vCast.asType(.float32).asArray(Float.self),
                    shape: .init(heads: heads, lQ: lQ, lK: lK, d: d),
                    options: opts)
                let tol: Float = dtype == .bfloat16 ? 3e-2 : (dtype == .float16 ? 8e-3 : 5e-4)
                let diff = zip(gpu.asArray(Float.self), cpu).map { abs($0 - $1) }.max() ?? 0
                #expect(diff <= tol)
            }
        }
    }

    @Test("SageAttention token-mean centering is algebraically exact under softmax")
    func sageMeanCenteringExactIdentity() {
        let rng = MLXRandom.RandomState(seed: 24102)
        let (heads, lQ, lK, d) = (2, 8, 16, 64)
        let q = MLXRandom.normal([heads, lQ, d], dtype: .float32, key: rng) + Float(5.0)
        let k = MLXRandom.normal([heads, lK, d], dtype: .float32, key: rng) - Float(3.0)
        let scale = 1.0 / Float(d).squareRoot()

        let refProbs = softmax(matmul(q, k.swappedAxes(-1, -2)) * scale, axis: -1)
        let qMean = mean(q, axis: 1, keepDims: true)
        let kMean = mean(k, axis: 1, keepDims: true)
        let qC = q - qMean
        let kC = k - kMean
        let deltaS = matmul(qMean, kC.swappedAxes(-1, -2))
        let centeredProbs = softmax((matmul(qC, kC.swappedAxes(-1, -2)) + deltaS) * scale, axis: -1)
        eval(refProbs, centeredProbs)
        #expect(abs(refProbs - centeredProbs).max().item(Float.self) < 1e-5)
    }

    @Test("MoBA / DeepSeek NSA fused Metal block-sparse attention matches gathered MLX reference (#228)")
    func mobaAndNSABlockSparseParity() {
        let rng = MLXRandom.RandomState(seed: 2502)
        let (heads, lQ, lK, d) = (2, 6, 64, 64)
        let blockSize = 16
        let topBlocks = 2
        let scale = 1.0 / Float(d).squareRoot()
        let q = MLXRandom.normal([heads, lQ, d], dtype: .float32, key: rng)
        let k = MLXRandom.normal([heads, lK, d], dtype: .float32, key: rng)
        let v = MLXRandom.normal([heads, lK, d], dtype: .float32, key: rng)

        let sel = BlockSparseRoutingMetal.selectMoBABlocks(q: q, k: k, blockSize: blockSize, topBlocks: topBlocks)
        eval(sel)
        let selHost = sel.asArray(Int32.self)

        for dtype in [DType.float32, .float16, .bfloat16] {
            let gpu = BlockSparseRoutingMetal.sparseSelectedAttention(
                q: q.asType(dtype), k: k.asType(dtype), v: v.asType(dtype),
                selectedBlocks: sel, blockSize: blockSize, scale: scale
            ).asType(.float32)
            eval(gpu)

            let qF = q.asType(dtype).asType(.float32)
            let kF = k.asType(dtype).asType(.float32)
            let vF = v.asType(dtype).asType(.float32)
            var maxDiff: Float = 0
            for h in 0 ..< heads {
                for qi in 0 ..< lQ {
                    var tokIndices: [Int32] = []
                    for sb in 0 ..< topBlocks {
                        let bId = Int(selHost[(h * lQ + qi) * topBlocks + sb])
                        for off in 0 ..< blockSize { tokIndices.append(Int32(bId * blockSize + off)) }
                    }
                    let idxArr = MLXArray(tokIndices)
                    let kSub = kF[h].take(idxArr, axis: 0)
                    let vSub = vF[h].take(idxArr, axis: 0)
                    let qVec = qF[h, qi].reshaped(1, d)
                    let p = softmax(matmul(qVec, kSub.T) * scale, axis: -1)
                    let refRow = matmul(p, vSub).squeezed(axis: 0)
                    let gotRow = gpu[h, qi]
                    eval(refRow, gotRow)
                    maxDiff = max(maxDiff, abs(refRow - gotRow).max().item(Float.self))
                }
            }
            let tol: Float = dtype == .bfloat16 ? 2.5e-2 : (dtype == .float16 ? 6e-3 : 5e-4)
            #expect(maxDiff <= tol)
        }

        let nsaOut = BlockSparseRoutingMetal.nsaThreeBranchAttention(
            q: q, k: k, v: v, config: .init(blockSize: 16, topBlocks: 2, slidingWindow: 16, scale: scale))
        eval(nsaOut)
        #expect(nsaOut.shape == [heads, lQ, d])
        let finite = nsaOut.asArray(Float.self).allSatisfy { $0.isFinite }
        #expect(finite)
    }
}
