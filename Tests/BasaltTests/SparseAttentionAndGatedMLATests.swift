import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

@Suite("R4: NSA serve wiring, DSA Lightning Indexer Metal kernels, and Gated MLA (#845)", .serialized)
struct SparseAttentionAndGatedMLATests {
    @Test("DSA Lightning Indexer Metal kernel matches pure-MLX and CPU reference with exact top-k indices across dtypes (#228)")
    func dsaIndexerScoreAndTopKParity() {
        let (batch, lQ, lK, hI, dI, topK) = (2, 6, 18, 4, 24, 5)
        var rng = SplitMix64(seed: 84501)
        let qHost = (0 ..< batch * lQ * hI * dI).map { _ in (Float(rng.next() & 0xFFFF) / 32768.0 - 1.0) * 0.45 }
        let kHost = (0 ..< batch * lK * dI).map { _ in (Float(rng.next() & 0xFFFF) / 32768.0 - 1.0) * 0.45 }
        let wHost = (0 ..< batch * lQ * hI).map { _ in (Float(rng.next() & 0xFFFF) / 65536.0 + 0.15) * 0.5 }
        let qF32 = MLXArray(qHost, [batch, lQ, hI, dI])
        let kF32 = MLXArray(kHost, [batch, lK, dI])
        let wF32 = MLXArray(wHost, [batch, lQ, hI])

        for causal in [false, true] {
            let qOffset = causal ? (lK - lQ) : 0
            for dtype in [DType.float32, .float16, .bfloat16] {
                let qCast = qF32.asType(dtype)
                let kCast = kF32.asType(dtype)
                let metalScores = DSALightningIndexerMetal.score(
                    q: qCast, k: kCast, weights: wF32,
                    options: .init(causal: causal, qOffset: qOffset, useMetal: true))
                let mlxScores = DSALightningIndexerMetal.score(
                    q: qCast, k: kCast, weights: wF32,
                    options: .init(causal: causal, qOffset: qOffset, useMetal: false))
                let metalIdx = DSALightningIndexerMetal.selectTopK(
                    scores: metalScores, topK: topK, causal: causal, qOffset: qOffset)
                let mlxIdx = DSALightningIndexerMetal.selectTopK(
                    scores: mlxScores, topK: topK, causal: causal, qOffset: qOffset)
                eval(metalScores, mlxScores, metalIdx, mlxIdx)

                let cpuScores = SparseAttentionCPUOracle.dsaIndexerScore(
                    q: qCast.asType(.float32).asArray(Float.self),
                    k: kCast.asType(.float32).asArray(Float.self),
                    w: wHost,
                    shape: .init(batch: batch, lQ: lQ, lK: lK, hI: hI, dI: dI),
                    causal: causal, qOffset: qOffset)
                let cpuIdx = SparseAttentionCPUOracle.selectTopK(
                    scores: cpuScores,
                    spec: .init(batch: batch, lQ: lQ, lK: lK, topK: topK, causal: causal, qOffset: qOffset))
                let tol: Float = dtype == .bfloat16 ? 2e-2 : (dtype == .float16 ? 3e-3 : 1e-4)
                let metalErr = SparseAttentionCPUOracle.maxFiniteDiff(metalScores.asArray(Float.self), cpuScores)
                let mlxErr = SparseAttentionCPUOracle.maxFiniteDiff(mlxScores.asArray(Float.self), cpuScores)
                #expect(metalErr <= tol && mlxErr <= tol)
                if dtype == .float32 {
                    #expect(metalIdx.asArray(Int32.self) == cpuIdx)
                    #expect(mlxIdx.asArray(Int32.self) == cpuIdx)
                }
            }
        }
    }

    @Test("DSA Lightning Indexer breaks exact ties deterministically with smaller index winning across Metal, pure-MLX, and CPU (#228)")
    func dsaIndexerDeterministicTieBreak() {
        let (batch, lQ, lK, hI, dI) = (1, 3, 10, 2, 8)
        let qArr = MLXArray.ones([batch, lQ, hI, dI], dtype: .float32)
        let wArr = MLXArray.ones([batch, lQ, hI], dtype: .float32)
        var kVals = [Float](repeating: -1.0, count: lK * dI)
        for s in [2, 5, 8] {
            for d in 0 ..< dI { kVals[s * dI + d] = 1.0 }
        }
        for s in [1, 4, 7] {
            for d in 0 ..< dI { kVals[s * dI + d] = 2.0 }
        }
        let kArr = MLXArray(kVals, [batch, lK, dI])

        for useMetal in [true, false] {
            let scores = DSALightningIndexerMetal.score(
                q: qArr, k: kArr, weights: wArr, options: .init(causal: false, useMetal: useMetal))
            let picked = DSALightningIndexerMetal.selectTopK(scores: scores, topK: 8, causal: false)
            eval(scores, picked)
            let row0 = Array(picked.asArray(Int32.self).prefix(8))
            #expect(row0 == [1, 4, 7, 2, 5, 8, 0, 3])
        }

        let causalScores = DSALightningIndexerMetal.score(
            q: qArr, k: kArr, weights: wArr, options: .init(causal: true, qOffset: 0, useMetal: true))
        let causalPicked = DSALightningIndexerMetal.selectTopK(scores: causalScores, topK: 4, causal: true, qOffset: 0)
        eval(causalPicked)
        let flat = causalPicked.asArray(Int32.self)
        #expect(Array(flat[0 ..< 4]) == [0, -1, -1, -1])
        #expect(Array(flat[4 ..< 8]) == [1, 0, -1, -1])
        #expect(Array(flat[8 ..< 12]) == [1, 2, 0, -1])
    }

    @Test("DSA sparse latent attention Metal kernel matches pure-MLX and CPU oracle across RoPE, NoPE, value head gate, and dtypes (#228)")
    func dsaSparseLatentAttentionParity() {
        let (batch, heads, lQ, lK, topK) = (2, 3, 4, 14, 5)
        let idxData: [Int32] = (0 ..< batch * lQ * topK).map { i in
            let slot = i % topK
            let row = (i / topK) % lQ
            return slot == topK - 1 && row == 0 ? -1 : Int32((row * 3 + slot * 2) % lK)
        }
        let topIdx = MLXArray(idxData, [batch, lQ, topK])
        for (dc, dr, hasVGate) in [(24, 16, false), (32, 0, false), (64, 16, true), (32, 0, true)] {
            let qAbs = MLXArray(SparseAttentionCPUOracle.pattern(batch * heads * lQ * dc, mul: 17, mod: 31, scale: 0.09), [batch, heads, lQ, dc])
            let cKV = MLXArray(SparseAttentionCPUOracle.pattern(batch * lK * dc, mul: 13, mod: 29, scale: 0.09), [batch, 1, lK, dc])
            let qPe: MLXArray? = dr > 0 ? MLXArray(SparseAttentionCPUOracle.pattern(batch * heads * lQ * dr, mul: 19, mod: 37, scale: 0.08), [batch, heads, lQ, dr]) : nil
            let kPe: MLXArray? = dr > 0 ? MLXArray(SparseAttentionCPUOracle.pattern(batch * lK * dr, mul: 23, mod: 41, scale: 0.08), [batch, 1, lK, dr]) : nil
            let vGate: MLXArray? = hasVGate ? sigmoid(MLXArray(SparseAttentionCPUOracle.pattern(batch * heads * lK, mul: 7, mod: 19, scale: 0.3), [batch, heads, lK, 1])) : nil
            let scale = 1.0 / Float(dc + dr).squareRoot()

            for dtype in [DType.float32, .float16, .bfloat16] {
                let inputs = DSALightningIndexerMetal.SparseLatentInputs(
                    qAbsorbed: qAbs.asType(dtype), qPe: qPe?.asType(dtype),
                    cKV: cKV.asType(dtype), kPe: kPe?.asType(dtype),
                    topIndices: topIdx, valueHeadGate: vGate)
                let metalOut = DSALightningIndexerMetal.sparseLatentAttention(
                    inputs, options: .init(scale: scale, causal: true, qOffset: lK - lQ, useMetal: true))
                let mlxOut = DSALightningIndexerMetal.sparseLatentAttention(
                    inputs, options: .init(scale: scale, causal: true, qOffset: lK - lQ, useMetal: false))
                eval(metalOut, mlxOut)

                let host = SparseAttentionCPUOracle.SparseLatentHost(
                    qAbs: inputs.qAbsorbed.asType(.float32).asArray(Float.self),
                    qPe: inputs.qPe?.asType(.float32).asArray(Float.self),
                    cKV: inputs.cKV.asType(.float32).asArray(Float.self),
                    kPe: inputs.kPe?.asType(.float32).asArray(Float.self),
                    vGate: vGate?.asArray(Float.self),
                    topIndices: idxData)
                let spec = SparseAttentionCPUOracle.SparseLatentSpec(
                    batch: batch, heads: heads, lQ: lQ, lK: lK, dc: dc, dr: dr,
                    topK: topK, scale: scale, causal: true, qOffset: lK - lQ)
                let cpuOut = SparseAttentionCPUOracle.dsaSparseLatentAttention(host: host, spec: spec)
                let tol: Float = dtype == .bfloat16 ? 2.5e-2 : (dtype == .float16 ? 5e-3 : 1e-4)
                let metalErr = SparseAttentionCPUOracle.maxFiniteDiff(metalOut.asType(.float32).asArray(Float.self), cpuOut)
                let mlxErr = SparseAttentionCPUOracle.maxFiniteDiff(mlxOut.asType(.float32).asArray(Float.self), cpuOut)
                #expect(metalErr <= tol && mlxErr <= tol)
            }
        }
    }

    @Test("DSA sparse MLA equals dense MLA within 1e-4 FP32 and matches greedy argmax tokens exactly when k >= L_K (#228)")
    func dsaDenseEquivalenceWhenTopKExceedsLength() {
        withRandomState(MLXRandom.RandomState(seed: 8451)) {
            var denseCfg = GatedMLAConfiguration(
                hiddenSize: 64, numHeads: 4, kvLoraRank: 32, qkNopeHeadDim: 16, qkRopeHeadDim: 16, vHeadDim: 16)
            denseCfg.qLoraRank = 24
            denseCfg.outputGate = .headWise
            denseCfg.useOutputNorm = true

            var sparseCfg = denseCfg
            sparseCfg.indexer = DSALightningIndexer.Config(indexHeads: 4, indexHeadDim: 16, topK: 24, ropeHeadDim: 8)

            let denseAttn = GatedMLAAttention(denseCfg)
            let sparseAttn = GatedMLAAttention(sparseCfg)
            sparseAttn.update(parameters: denseAttn.parameters())

            let hidden = MLXArray(SparseAttentionCPUOracle.pattern(1 * 16 * 64, mul: 19, mod: 37, scale: 0.08), [1, 16, 64])
            let denseOut = denseAttn(hidden, mask: .causal, cache: nil)
            let sparseOut = sparseAttn(hidden, mask: .causal, cache: nil)
            let lmHead = Linear(64, 64, bias: false)
            let denseTokens = argMax(lmHead(denseOut), axis: -1).asType(.int32)
            let sparseTokens = argMax(lmHead(sparseOut), axis: -1).asType(.int32)
            eval(denseOut, sparseOut, denseTokens, sparseTokens)

            let maxDiff = abs(denseOut - sparseOut).max().item(Float.self)
            #expect(maxDiff < 1e-4)
            #expect(denseTokens.asArray(Int32.self) == sparseTokens.asArray(Int32.self))
        }
    }
}

extension SparseAttentionAndGatedMLATests {
    @Test("NSA GQA selection indices, causal prefill with partial tail block, and dynamic [B, H, L_Q, 3] gate tensor match CPU oracle (#228)")
    func nsaGQASelectionCausalityAndDynamicGatesParity() {
        let (batch, qHeads, kvHeads, lQ, lK, d) = (1, 8, 2, 10, 53, 32)
        let (blockSize, topBlocks, window) = (16, 2, 12)
        let scale = 1.0 / Float(d).squareRoot()
        let q = MLXArray(SparseAttentionCPUOracle.pattern(batch * qHeads * lQ * d, mul: 17, mod: 31, scale: 0.1), [batch, qHeads, lQ, d])
        let k = MLXArray(SparseAttentionCPUOracle.pattern(batch * kvHeads * lK * d, mul: 13, mod: 29, scale: 0.1), [batch, kvHeads, lK, d])
        let v = MLXArray(SparseAttentionCPUOracle.pattern(batch * kvHeads * lK * d, mul: 11, mod: 23, scale: 0.1), [batch, kvHeads, lK, d])

        let kComp = BlockSparseRoutingMetal.blockCentroids(k.reshaped(kvHeads, lK, d), blockSize: blockSize)
        let qGrouped = q.reshaped(kvHeads, qHeads / kvHeads, lQ, d)
        let compWeights = softmax(matmul(qGrouped, kComp.expandedDimensions(axis: 1).swappedAxes(-1, -2)) * scale, axis: -1)
        let gpuBlocks = BlockSparseRoutingMetal.selectNSABlocks(
            compWeights: compWeights, blockSize: blockSize, topBlocks: topBlocks, lK: lK, causal: false, qOffset: 0)
        eval(compWeights, gpuBlocks)
        let cpuBlocks = SparseAttentionCPUOracle.selectNSABlocks(
            compWeights: compWeights.asArray(Float.self),
            kvHeads: kvHeads, gqaGroup: qHeads / kvHeads, lQ: lQ, numBlocks: 4, topBlocks: topBlocks)
        #expect(gpuBlocks.asArray(Int32.self) == cpuBlocks)

        let cfg = BlockSparseRoutingMetal.Config(
            blockSize: blockSize, topBlocks: topBlocks, slidingWindow: window, scale: scale, causal: true)
        let staticOut = BlockSparseRoutingMetal.nsaThreeBranchAttention(q: q, k: k, v: v, config: cfg, qOffset: 0)
        let staticGates = broadcast(MLXArray([Float(0.2), 0.5, 0.3]).reshaped(1, 1, 1, 3), to: [batch, qHeads, lQ, 3])
        let dynamicSame = BlockSparseRoutingMetal.nsaThreeBranchAttention(
            q: q, k: k, v: v, config: cfg, gateTensor: staticGates, qOffset: 0)
        let varyingGates = softmax(
            MLXArray(SparseAttentionCPUOracle.pattern(batch * qHeads * lQ * 3, mul: 7, mod: 19, scale: 0.25), [batch, qHeads, lQ, 3]),
            axis: -1)
        let dynamicVarying = BlockSparseRoutingMetal.nsaThreeBranchAttention(
            q: q, k: k, v: v, config: cfg, gateTensor: varyingGates, qOffset: 0)
        let compCfg = BlockSparseRoutingMetal.Config(
            blockSize: blockSize, topBlocks: topBlocks, slidingWindow: window, scale: scale, gates: [1, 0, 0], causal: true)
        let selCfg = BlockSparseRoutingMetal.Config(
            blockSize: blockSize, topBlocks: topBlocks, slidingWindow: window, scale: scale, gates: [0, 1, 0], causal: true)
        let winCfg = BlockSparseRoutingMetal.Config(
            blockSize: blockSize, topBlocks: topBlocks, slidingWindow: window, scale: scale, gates: [0, 0, 1], causal: true)
        let compOnly = BlockSparseRoutingMetal.nsaThreeBranchAttention(q: q, k: k, v: v, config: compCfg, qOffset: 0)
        let selOnly = BlockSparseRoutingMetal.nsaThreeBranchAttention(q: q, k: k, v: v, config: selCfg, qOffset: 0)
        let winOnly = BlockSparseRoutingMetal.nsaThreeBranchAttention(q: q, k: k, v: v, config: winCfg, qOffset: 0)
        let g0 = varyingGates[0..., 0..., 0..., 0 ..< 1]
        let g1 = varyingGates[0..., 0..., 0..., 1 ..< 2]
        let g2 = varyingGates[0..., 0..., 0..., 2 ..< 3]
        let hasCompMask = (MLXArray(Int32(0) ..< Int32(lQ)) .>= Int32(blockSize - 1)).reshaped(1, 1, lQ, 1)
        let effG0 = MLX.where(hasCompMask, g0, MLXArray(Float(0)))
        let normScale = (g0 + g1 + g2) / (effG0 + g1 + g2)
        let manualDyn = (compOnly * effG0 + selOnly * g1 + winOnly * g2) * normScale
        eval(staticOut, dynamicSame, dynamicVarying, manualDyn)
        #expect(abs(staticOut - dynamicSame).max().item(Float.self) < 1e-5)
        #expect(abs(dynamicVarying - manualDyn).max().item(Float.self) < 1e-5)

        let kFuturePerturbed = concatenated([k[0..., 0..., 0 ..< 10, 0...], k[0..., 0..., 10 ..< lK, 0...] + Float(50.0)], axis: 2)
        let vFuturePerturbed = concatenated([v[0..., 0..., 0 ..< 10, 0...], v[0..., 0..., 10 ..< lK, 0...] - Float(50.0)], axis: 2)
        let perturbedOut = BlockSparseRoutingMetal.nsaThreeBranchAttention(
            q: q, k: kFuturePerturbed, v: vFuturePerturbed, config: cfg, qOffset: 0)
        eval(perturbedOut)
        #expect(abs(staticOut - perturbedOut).max().item(Float.self) < 1e-6)
    }

    @Test("NSA matches stock dense SDPA when gates = [0, 1, 0] and topBlocks >= ceil(L_K / blockSize) for causal and non-causal GQA with partial tail block (#228)")
    func nsaDenseEquivalenceWithPartialTailBlock() {
        withRandomState(MLXRandom.RandomState(seed: 8452)) {
            let (batch, qHeads, kvHeads, lQ, lK, d) = (1, 8, 2, 12, 53, 32)
            let blockSize = 16
            let topBlocks = (lK + blockSize - 1) / blockSize
            let scale = 1.0 / Float(d).squareRoot()
            let q = MLXArray(SparseAttentionCPUOracle.pattern(batch * qHeads * lQ * d, mul: 17, mod: 31, scale: 0.12), [batch, qHeads, lQ, d])
            let k = MLXArray(SparseAttentionCPUOracle.pattern(batch * kvHeads * lK * d, mul: 13, mod: 29, scale: 0.12), [batch, kvHeads, lK, d])
            let v = MLXArray(SparseAttentionCPUOracle.pattern(batch * kvHeads * lK * d, mul: 11, mod: 23, scale: 0.12), [batch, kvHeads, lK, d])
            let proj = Linear(d, 32, bias: false)

            for causal in [false, true] {
                let cfg = BlockSparseRoutingMetal.Config(
                    blockSize: blockSize, topBlocks: topBlocks, slidingWindow: 16,
                    scale: scale, gates: [0.0, 1.0, 0.0], causal: causal)
                let nsaOut = BlockSparseRoutingMetal.nsaThreeBranchAttention(q: q, k: k, v: v, config: cfg)
                let refMask: MLXFast.ScaledDotProductAttentionMaskMode = causal ? .causal : .none
                let denseOut = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: refMask)
                let nsaIds = argMax(proj(nsaOut), axis: -1).asType(.int32)
                let denseIds = argMax(proj(denseOut), axis: -1).asType(.int32)
                eval(nsaOut, denseOut, nsaIds, denseIds)

                let diff = abs(nsaOut - denseOut).max().item(Float.self)
                #expect(diff < 1e-4)
                #expect(nsaIds.asArray(Int32.self) == denseIds.asArray(Int32.self))
            }
        }
    }

    @Test("Live SageMXKVCache(.nsa) decodes on TinyModel with TokenIterator parity and preserves centroid invariants across copy and trim (#228)")
    func liveSageMXNSACacheDecodeTrimAndCopy() throws {
        let model = try TinyModel.make(.qwen3HeadDim32)
        let prompt = Array(repeating: [3, 5, 7, 11, 13, 17, 19, 23], count: 34).flatMap { $0 }
        let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
        let params = GenerateParameters(maxTokens: 4, temperature: 0)
        let caches = SageMXKVCache.replacingFullAttention(
            try model.newCache(parameters: nil), mode: .nsa(topBlocks: 1, window: 64))
        let tokens = Array(try TokenIterator(input: input, model: model, cache: caches, parameters: params))
        let nsaCaches = caches.compactMap { $0 as? SageMXKVCache }
        #expect(tokens.count == 4)
        #expect(!nsaCaches.isEmpty)
        #expect(nsaCaches.allSatisfy { $0.kernelSteps > 0 && $0.centroids != nil && $0.valueCentroids != nil })

        var prefix = prompt
        for emitted in tokens {
            let freshCaches = SageMXKVCache.replacingFullAttention(
                try model.newCache(parameters: nil), mode: .nsa(topBlocks: 1, window: 64))
            let logits = model(MLXArray(prefix.map(Int32.init))[.newAxis], cache: freshCaches)
            let expected = argMax(logits[0, -1, 0...]).item(Int.self)
            #expect(emitted == expected)
            prefix.append(emitted)
        }

        let first = try #require(nsaCaches.first)
        let originalOffset = first.offset
        let originalBlocks = first.indexedBlocks
        let fullK = first.state[0]
        let fullV = first.state[1]
        let qProbe = MLXArray(
            SparseAttentionCPUOracle.pattern(fullK.dim(0) * fullK.dim(1) * 1 * fullK.dim(3), mul: 13, mod: 29, scale: 0.1),
            [fullK.dim(0), fullK.dim(1), 1, fullK.dim(3)])
        let scale = 1.0 / sqrt(Float(fullK.dim(3)))
        let cachedOut = first.nsaAttention(
            queries: qProbe, keys: fullK, values: fullV,
            config: BlockSparseRoutingMetal.Config(
                blockSize: SageMXKVCache.blockTokens, topBlocks: 1, slidingWindow: 64, scale: scale, causal: false))
        let directOut = BlockSparseRoutingMetal.nsaThreeBranchAttention(
            q: qProbe, k: fullK, v: fullV,
            config: BlockSparseRoutingMetal.Config(
                blockSize: SageMXKVCache.blockTokens, topBlocks: 1, slidingWindow: 64, scale: scale, causal: false))
        eval(cachedOut, directOut)
        #expect(abs(cachedOut - directOut).max().item(Float.self) < 1e-4)

        let copied = try #require(first.copy() as? SageMXKVCache)
        #expect(copied.offset == originalOffset && copied.indexedBlocks == originalBlocks)

        let trimmed = copied.trim(100)
        #expect(trimmed == 100 && copied.offset == originalOffset - 100)
        #expect(copied.indexedBlocks == copied.offset / SageMXKVCache.blockTokens)
        #expect(first.offset == originalOffset && first.indexedBlocks == originalBlocks)
        #expect(copied.centroids?.dim(2) == copied.indexedBlocks)
        #expect(copied.valueCentroids?.dim(2) == copied.indexedBlocks)

        let dropAll = copied.trim(copied.offset)
        #expect(dropAll == originalOffset - 100 && copied.offset == 0)
        #expect(copied.centroids == nil && copied.valueCentroids == nil && copied.indexedBlocks == 0)
    }
}

extension SparseAttentionAndGatedMLATests {
    @Test("Gated MLA matches expanded reference across RoPE, NoPE (k_pe omitted from DSALatentKVCache), output/latent/value gates, and pre-gate RMSNorm")
    func gatedMLAGatingAndNoPEParity() throws {
        try withRandomState(MLXRandom.RandomState(seed: 8453)) {
            let hidden = MLXArray(SparseAttentionCPUOracle.pattern(1 * 12 * 64, mul: 19, mod: 37, scale: 0.08), [1, 12, 64])
            let cases: [(Int, GatedMLAOutputGate, GatedMLAValueGate, Bool, Bool)] = [
                (16, .headWise, .none, false, true),
                (0, .headWise, .none, true, true),
                (0, .elementWise, .latentGate, true, true),
                (16, .headWise, .valueHeadWise, true, true),
                (0, .none, .valueHeadWise, false, true),
                (16, .elementWise, .valueElementWise, true, false),
                (0, .headWise, .valueElementWise, false, false),
            ]
            for (ropeDim, outGate, valGate, useONorm, expectAbsorbed) in cases {
                var cfg = GatedMLAConfiguration(
                    hiddenSize: 64, numHeads: 4, kvLoraRank: 32,
                    qkNopeHeadDim: 16, qkRopeHeadDim: ropeDim, vHeadDim: 16)
                cfg.qLoraRank = 24
                cfg.outputGate = outGate
                cfg.valueGate = valGate
                cfg.useOutputNorm = useONorm
                let attn = GatedMLAAttention(cfg)
                #expect(attn.usesAbsorbedPath == expectAbsorbed)

                let cache = DSALatentKVCache()
                let refCache = DSALatentKVCache()
                try SparseAttentionCPUOracle.assertGatedMLAParity(
                    attn, hidden: hidden, cache: cache, refCache: refCache, tolerance: 1e-4)
                if ropeDim == 0 {
                    #expect(cache.cachedKPe == nil)
                } else {
                    #expect(cache.cachedKPe?.dim(3) == ropeDim)
                }
                let copyCache = try #require(cache.copy() as? DSALatentKVCache)
                let removed = copyCache.trim(4)
                #expect(removed == 4 && copyCache.offset == 8 && cache.offset == 12)
            }
        }
    }

    @Test("Gated MLA absorbed path matches expanded reference across FP32, BF16, 4-bit affine, SmallM, MXFP4, and NVFP4 globalScale")
    func gatedMLAParityAcrossQuantizationModes() throws {
        try withRandomState(MLXRandom.RandomState(seed: 8454)) {
            let hiddenF32 = MLXArray(SparseAttentionCPUOracle.pattern(1 * 12 * 64, mul: 17, mod: 31, scale: 0.07), [1, 12, 64])
            var cfg = GatedMLAConfiguration(
                hiddenSize: 64, numHeads: 4, kvLoraRank: 64, qkNopeHeadDim: 16, qkRopeHeadDim: 0, vHeadDim: 16)
            cfg.outputGate = .headWise
            cfg.valueGate = .valueHeadWise
            cfg.useOutputNorm = true

            let dense = GatedMLAAttention(cfg)
            try SparseAttentionCPUOracle.assertGatedMLAParity(
                dense, hidden: hiddenF32, cache: DSALatentKVCache(), refCache: DSALatentKVCache(), tolerance: 1e-4)

            let bf16 = GatedMLAAttention(cfg)
            bf16.update(parameters: bf16.parameters().mapValues { $0.asType(.bfloat16) })
            try SparseAttentionCPUOracle.assertGatedMLAParity(
                bf16, hidden: hiddenF32.asType(.bfloat16), cache: DSALatentKVCache(), refCache: DSALatentKVCache(), tolerance: 2e-2)

            let affine4 = GatedMLAAttention(cfg)
            quantize(model: affine4, groupSize: 32, bits: 4, mode: .affine) { path, mod in path == "kv_b_proj" && mod is Linear }
            #expect(affine4.kvBProj is QuantizedLinear && affine4.usesAbsorbedPath)
            try SparseAttentionCPUOracle.assertGatedMLAParity(
                affine4, hidden: hiddenF32, cache: DSALatentKVCache(), refCache: DSALatentKVCache(), tolerance: 1e-4)

            let smallM = GatedMLAAttention(cfg)
            quantize(model: smallM, groupSize: 32, bits: 4, mode: .affine) { path, mod in path == "kv_b_proj" && mod is Linear }
            #expect(SmallMQuantizedLinear.install(in: smallM) == 1 && smallM.kvBProj is SmallMQuantizedLinear)
            try SparseAttentionCPUOracle.assertGatedMLAParity(
                smallM, hidden: hiddenF32, cache: DSALatentKVCache(), refCache: DSALatentKVCache(), tolerance: 1e-4)

            let mxfp4 = GatedMLAAttention(cfg)
            quantize(model: mxfp4, groupSize: 32, bits: 4, mode: .mxfp4) { path, mod in path == "kv_b_proj" && mod is Linear }
            try SparseAttentionCPUOracle.assertGatedMLAParity(
                mxfp4, hidden: hiddenF32, cache: DSALatentKVCache(), refCache: DSALatentKVCache(), tolerance: 1e-3)

            let nvfp4 = GatedMLAAttention(cfg)
            let nvfp4Layer = QuantizedLinear(
                weight: nvfp4.kvBProj.weight, bias: nil, groupSize: 16, bits: 4, mode: .nvfp4,
                globalScale: MLXArray(Float(1344.0)))
            nvfp4.update(modules: ModuleChildren.unflattened([("kv_b_proj", nvfp4Layer)]))
            try SparseAttentionCPUOracle.assertGatedMLAParity(
                nvfp4, hidden: hiddenF32, cache: DSALatentKVCache(), refCache: DSALatentKVCache(), tolerance: 1e-3)
        }
    }

    @Test("Gated MLA with DSA indexer and valueHeadWise matches expanded sparse reference in RoPE and NoPE")
    func gatedMLASparseWithValueHeadGateParity() throws {
        try withRandomState(MLXRandom.RandomState(seed: 8455)) {
            let hidden = MLXArray(SparseAttentionCPUOracle.pattern(1 * 12 * 64, mul: 23, mod: 41, scale: 0.08), [1, 12, 64])
            for ropeDim in [0, 16] {
                var cfg = GatedMLAConfiguration(
                    hiddenSize: 64, numHeads: 4, kvLoraRank: 32,
                    qkNopeHeadDim: 16, qkRopeHeadDim: ropeDim, vHeadDim: 16)
                cfg.outputGate = .headWise
                cfg.valueGate = .valueHeadWise
                cfg.useOutputNorm = true
                cfg.indexer = DSALightningIndexer.Config(indexHeads: 4, indexHeadDim: 16, topK: 6, ropeHeadDim: 0)
                let attn = GatedMLAAttention(cfg)
                #expect(attn.usesAbsorbedPath)
                try SparseAttentionCPUOracle.assertGatedMLAParity(
                    attn, hidden: hidden, cache: DSALatentKVCache(), refCache: DSALatentKVCache(), tolerance: 1e-4)
            }
        }
    }
}

extension SparseAttentionAndGatedMLATests {
    @Test("BailingHybridMLA supports NoPE (qk_rope_head_dim == 0), optional pre-gate oNorm, and optional DSA indexer with full parity")
    func bailingHybridNoPEONormAndIndexerParity() throws {
        let nopeExtra = #""qk_rope_head_dim": 0, "use_mla_o_norm": true, "#
        let nopeModel = try #require(try TinyModel.make(.bailingHybrid, adding: nopeExtra) as? BailingHybridModel)
        let nopeMLA = try #require(nopeModel.model.layers[2].attention as? BailingHybridMLAAttention)
        #expect(nopeMLA.rope == nil && nopeMLA.oNorm != nil && nopeMLA.indexer == nil)
        let caches = try nopeModel.newCache(parameters: nil)
        #expect(caches[2] is DSALatentKVCache)

        let hidden = MLXArray(SparseAttentionCPUOracle.pattern(1 * 12 * 64, mul: 19, mod: 37, scale: 0.06), [1, 12, 64])
        let absorbed = nopeMLA(hidden, mask: .causal, cache: caches[2])
        let expanded = nopeMLA.forwardNonAbsorbed(hidden, mask: .causal, cache: nil)
        eval(absorbed, expanded)
        #expect(abs(absorbed - expanded).max().item(Float.self) < 1e-4)
        #expect((caches[2] as? DSALatentKVCache)?.cachedKPe == nil)

        let dsaExtra = #""qk_rope_head_dim": 0, "use_mla_o_norm": true, "dsa_index_heads": 4, "dsa_index_head_dim": 16, "dsa_top_k": 6, "#
        let dsaModel = try #require(try TinyModel.make(.bailingHybrid, adding: dsaExtra) as? BailingHybridModel)
        let dsaMLA = try #require(dsaModel.model.layers[2].attention as? BailingHybridMLAAttention)
        #expect(dsaMLA.indexer != nil)
        let dsaAbs = dsaMLA(hidden, mask: .causal, cache: DSALatentKVCache())
        let dsaExp = dsaMLA.forwardNonAbsorbed(hidden, mask: .causal, cache: DSALatentKVCache())
        eval(dsaAbs, dsaExp)
        #expect(abs(dsaAbs - dsaExp).max().item(Float.self) < 1e-4)

        let zeroBothJSON = """
            {"hidden_size": 64, "num_hidden_layers": 3, "intermediate_size": 128,
             "num_attention_heads": 4, "vocab_size": 256, "kv_lora_rank": 32,
             "qk_nope_head_dim": 0, "qk_rope_head_dim": 0, "v_head_dim": 16}
            """
        let zeroBoth = try JSONDecoder.json5().decode(BailingHybridConfiguration.self, from: Data(zeroBothJSON.utf8))
        #expect(throws: BasaltError.self) { _ = try zeroBoth.validated() }
    }

    @Test("CLI parses --attention-backend nsa --nsa-window, rejects --attention-backend dsa, and routes NSA through GenerationPlan and FreshCache")
    func cliParsingAndNSARouting() throws {
        let parsed = try KVCacheOptions.parse(["--attention-backend", "nsa", "--nsa-window", "128", "--moba-top-k", "6"])
        #expect(parsed.attentionBackend == .nsa && parsed.nsaWindow == 128 && parsed.mobaTopK == 6)
        #expect(throws: Never.self) { try parsed.validateCombination() }

        let defaults = try KVCacheOptions.parse(["--attention-backend", "nsa"])
        #expect(defaults.attentionBackend == .nsa && defaults.nsaWindow == 256)
        #expect(throws: Never.self) { try defaults.validateCombination() }

        let badWindow = try KVCacheOptions.parse(["--attention-backend", "nsa", "--nsa-window", "0"])
        #expect(throws: ValidationError.self) { try badWindow.validateCombination() }

        let conflict = try KVCacheOptions.parse(["--attention-backend", "nsa", "--kv-bits", "4"])
        #expect(throws: ValidationError.self) { try conflict.validateCombination() }

        #expect(throws: (any Error).self) { _ = try KVCacheOptions.parse(["--attention-backend", "dsa"]) }

        var engineCfg = EngineConfiguration()
        engineCfg.nsaWindow = 0
        #expect(engineCfg.normalized.nsaWindow == 1)

        let plan = GenerationPlan.make(
            GenerationRequest(prompt: "hi"),
            GenerationPlan.Settings(attentionBackend: .nsa, mobaTopK: 6, nsaWindow: 128, capabilities: .unrestricted),
            GenerationPlan.Facts())
        #expect(plan.attentionBackend == .nsa && plan.mobaTopK == 6 && plan.nsaWindow == 128)

        let fresh = FreshCache.plan(
            options: LaunchOptions(params: GenerateParameters(), attentionBackend: .nsa, mobaTopK: 6, nsaWindow: 128))
        #expect(fresh == .sageMX(.nsa(topBlocks: 6, window: 128)))
    }

    @Test("Retro-audit fixes: NSA centroid causal alignment, CPU fallback, DSA empty/training/neg-inf safety")
    func retroAuditNSAAndDSARegressionGuards() {
        let q = MLXArray(SparseAttentionCPUOracle.pattern(1 * 4 * 8 * 32, mul: 13, mod: 29, scale: 0.1), [1, 4, 8, 32])
        let k = MLXArray(SparseAttentionCPUOracle.pattern(1 * 2 * 16 * 32, mul: 17, mod: 31, scale: 0.1), [1, 2, 16, 32])
        let v = MLXArray(SparseAttentionCPUOracle.pattern(1 * 2 * 16 * 32, mul: 19, mod: 23, scale: 0.1), [1, 2, 16, 32])
        let k2 = concatenated([k[0..., 0..., 0 ..< 8, 0...], k[0..., 0..., 8 ..< 16, 0...] + Float(5.0)], axis: 2)
        let v2 = concatenated([v[0..., 0..., 0 ..< 8, 0...], v[0..., 0..., 8 ..< 16, 0...] + Float(5.0)], axis: 2)
        let cent1 = BlockSparseRoutingMetal.PrecomputedCentroids(
            keys: BlockSparseRoutingMetal.blockCentroids(k.reshaped(2, 16, 32), blockSize: 8),
            values: BlockSparseRoutingMetal.blockCentroids(v.reshaped(2, 16, 32), blockSize: 8),
            blockSize: 8, qOffset: 0)
        let cent2 = BlockSparseRoutingMetal.PrecomputedCentroids(
            keys: BlockSparseRoutingMetal.blockCentroids(k2.reshaped(2, 16, 32), blockSize: 8),
            values: BlockSparseRoutingMetal.blockCentroids(v2.reshaped(2, 16, 32), blockSize: 8),
            blockSize: 8, qOffset: 0)
        let cfgMismatch = BlockSparseRoutingMetal.Config(blockSize: 4, topBlocks: 2, slidingWindow: 4, scale: 1.0 / sqrt(32.0), causal: true)
        let out1 = BlockSparseRoutingMetal.nsaThreeBranchAttentionWithCentroids(q: q, k: k, v: v, centroids: cent1, config: cfgMismatch)
        let out2 = BlockSparseRoutingMetal.nsaThreeBranchAttentionWithCentroids(q: q, k: k2, v: v2, centroids: cent2, config: cfgMismatch)
        eval(out1, out2)
        #expect(abs(out1 - out2).max().item(Float.self) < 1e-5)

        let sel = MLXArray(Array(repeating: Int32(0), count: 2 * 8), [2, 8, 1])
        let opts = BlockSparseRoutingMetal.SparseOptions(blockSize: 8, scale: 1.0 / sqrt(32.0), causal: true, qOffset: 0)
        let gpuSel = BlockSparseRoutingMetal.sparseSelectedAttention(q: q, k: k, v: v, selectedBlocks: sel, options: opts)
        let cpuSel = Device.withDefaultDevice(.cpu) {
            BlockSparseRoutingMetal.sparseSelectedAttention(q: q, k: k, v: v, selectedBlocks: sel, options: opts)
        }
        eval(gpuSel, cpuSel)
        #expect(abs(gpuSel - cpuSel).max().item(Float.self) < 1e-4)

        let emptyDSA = DSALightningIndexerMetal.sparseLatentAttention(
            DSALightningIndexerMetal.SparseLatentInputs(
                qAbsorbed: MLXArray.zeros([0, 4, 4, 32]),
                cKV: MLXArray.zeros([0, 1, 8, 32]),
                topIndices: MLXArray.zeros([0, 4, 4], type: Int32.self)),
            options: DSALightningIndexerMetal.SparseLatentOptions(scale: 0.125))
        #expect(emptyDSA.shape == [0, 4, 4, 32])

        let bigQ = MLXArray(Array(repeating: Float(1e20), count: 32), [1, 1, 1, 32])
        let bigK = concatenated([
            MLXArray(Array(repeating: Float(-1e20), count: 32), [1, 1, 1, 32]),
            MLXArray.zeros([1, 1, 1, 32]),
        ], axis: 2)
        let bigV = concatenated([MLXArray.zeros([1, 1, 1, 32]), MLXArray.ones([1, 1, 1, 32])], axis: 2)
        let negInfOut = BlockSparseRoutingMetal.sparseSelectedAttention(
            q: bigQ, k: bigK, v: bigV, selectedBlocks: MLXArray([Int32(0)], [1, 1, 1]),
            options: BlockSparseRoutingMetal.SparseOptions(blockSize: 2, scale: 1.0, causal: false))
        eval(negInfOut)
        #expect(negInfOut.asArray(Float.self).allSatisfy { $0.isFinite })

        var dsaCfg = GatedMLAConfiguration(
            hiddenSize: 64, numHeads: 4, kvLoraRank: 32,
            qkNopeHeadDim: 16, qkRopeHeadDim: 0, vHeadDim: 16)
        dsaCfg.indexer = DSALightningIndexer.Config(indexHeads: 4, indexHeadDim: 16, topK: 4, ropeHeadDim: 0)
        let trainableMLA = GatedMLAAttention(dsaCfg)
        trainableMLA.train(true)
        let xTrain = MLXArray(SparseAttentionCPUOracle.pattern(1 * 6 * 64, mul: 11, mod: 23, scale: 0.05), [1, 6, 64])
        let (loss, _) = valueAndGrad(model: trainableMLA) { _, arrays in
            [trainableMLA(arrays[0], mask: .causal, cache: nil).sum()]
        }(trainableMLA, [xTrain])
        eval(loss)
        #expect(loss[0].item(Float.self).isFinite)
    }
}

extension SparseAttentionAndGatedMLATests {
    @Test("Gate 3d: moba_nsa_sparse_attn_metal compiles and matches CPU at real head dims d=128/256 bf16 and DSA latent at dc=512/dr=64 bf16 (#228)")
    func gate3dRealHeadDimsBF16Parity() {
        for d in [128, 256] {
            let q = MLXArray(SparseAttentionCPUOracle.pattern(1 * 4 * 4 * d, mul: 13, mod: 29, scale: 0.08), [1, 4, 4, d]).asType(.bfloat16)
            let k = MLXArray(SparseAttentionCPUOracle.pattern(1 * 2 * 16 * d, mul: 17, mod: 31, scale: 0.08), [1, 2, 16, d]).asType(.bfloat16)
            let v = MLXArray(SparseAttentionCPUOracle.pattern(1 * 2 * 16 * d, mul: 19, mod: 23, scale: 0.08), [1, 2, 16, d]).asType(.bfloat16)
            let sel = MLXArray([Int32(0), 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [2, 4, 2])
            let opts = BlockSparseRoutingMetal.SparseOptions(blockSize: 8, scale: 1.0 / sqrt(Float(d)), causal: true, qOffset: 12)
            let gpuOut = BlockSparseRoutingMetal.sparseSelectedAttention(q: q, k: k, v: v, selectedBlocks: sel, options: opts)
            let cpuOut = Device.withDefaultDevice(.cpu) {
                BlockSparseRoutingMetal.sparseSelectedAttention(q: q, k: k, v: v, selectedBlocks: sel, options: opts)
            }
            eval(gpuOut, cpuOut)
            #expect(gpuOut.dtype == .bfloat16 && gpuOut.shape == [1, 4, 4, d])
            #expect(abs(gpuOut.asType(.float32) - cpuOut.asType(.float32)).max().item(Float.self) < 5e-2)
        }

        let (dc, dr, lQ, lK, topK) = (512, 64, 2, 8, 4)
        let qAbs = MLXArray(SparseAttentionCPUOracle.pattern(1 * 2 * lQ * dc, mul: 11, mod: 23, scale: 0.05), [1, 2, lQ, dc]).asType(.bfloat16)
        let qPe = MLXArray(SparseAttentionCPUOracle.pattern(1 * 2 * lQ * dr, mul: 13, mod: 19, scale: 0.05), [1, 2, lQ, dr]).asType(.bfloat16)
        let cKV = MLXArray(SparseAttentionCPUOracle.pattern(1 * 1 * lK * dc, mul: 17, mod: 29, scale: 0.05), [1, 1, lK, dc]).asType(.bfloat16)
        let kPe = MLXArray(SparseAttentionCPUOracle.pattern(1 * 1 * lK * dr, mul: 19, mod: 31, scale: 0.05), [1, 1, lK, dr]).asType(.bfloat16)
        let topIdx = MLXArray([Int32(0), 1, 2, 3, 2, 3, 4, 5], [1, lQ, topK])
        let inputs = DSALightningIndexerMetal.SparseLatentInputs(qAbsorbed: qAbs, qPe: qPe, cKV: cKV, kPe: kPe, topIndices: topIdx)
        let scale = 1.0 / sqrt(Float(dc + dr))
        let dsaGpu = DSALightningIndexerMetal.sparseLatentAttention(
            inputs, options: DSALightningIndexerMetal.SparseLatentOptions(scale: scale, causal: true, qOffset: 6, useMetal: true))
        let dsaMlx = DSALightningIndexerMetal.sparseLatentAttention(
            inputs, options: DSALightningIndexerMetal.SparseLatentOptions(scale: scale, causal: true, qOffset: 6, useMetal: false))
        eval(dsaGpu, dsaMlx)
        #expect(dsaGpu.dtype == .bfloat16 && dsaGpu.shape == [1, 2, lQ, dc])
        #expect(abs(dsaGpu.asType(.float32) - dsaMlx.asType(.float32)).max().item(Float.self) < 5e-2)

        let unmaskedScores = MLXArray([Float(1), 2, 99, 100, 1, 2, 3, 100], [1, 2, 4])
        let causalPicked = DSALightningIndexerMetal.selectTopK(scores: unmaskedScores, topK: 2, causal: true, qOffset: 1)
        eval(causalPicked)
        #expect(causalPicked.asArray(Int32.self) == [1, 0, 2, 1])
    }

    @Test("NSA falls back to dense when indexedBlocks <= topBlocks + 2 and renormalizes branch gates at causal positions 0..<63")
    func nsaDenseFallbackAndPrefillGateRenormalization() {
        let dummyK = MLXArray.zeros([1, 2, 256, 32], dtype: .float32)
        #expect(!SageMXKVCache.canRunNSA(keys: dummyK, topBlocks: 2, indexedBlocks: 4))
        #expect(SageMXKVCache.canRunNSA(keys: dummyK, topBlocks: 2, indexedBlocks: 5))

        let (l, d) = (128, 32)
        let q = MLXArray(SparseAttentionCPUOracle.pattern(1 * 4 * l * d, mul: 13, mod: 29, scale: 0.1), [1, 4, l, d])
        let k = MLXArray(SparseAttentionCPUOracle.pattern(1 * 2 * l * d, mul: 17, mod: 31, scale: 0.1), [1, 2, l, d])
        let v = MLXArray(SparseAttentionCPUOracle.pattern(1 * 2 * l * d, mul: 19, mod: 23, scale: 0.1), [1, 2, l, d])
        let scale = 1.0 / sqrt(Float(d))
        let cfg = BlockSparseRoutingMetal.Config(
            blockSize: 64, topBlocks: 1, slidingWindow: 64,
            scale: scale, gates: [0.2, 0.5, 0.3], causal: true)
        let nsaOut = BlockSparseRoutingMetal.nsaThreeBranchAttention(q: q, k: k, v: v, config: cfg)
        let denseOut = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .causal)
        let earlyDiff = abs(nsaOut[0..., 0..., 0 ..< 63, 0...] - denseOut[0..., 0..., 0 ..< 63, 0...]).max()
        eval(earlyDiff)
        #expect(earlyDiff.item(Float.self) < 1e-4)
    }

    @Test("BailingHybridMLA NoPE and RoPE+DSA match uncached forward across multi-chunk prefill and single-token decode (#228)")
    func bailingNoPEAndDSAChunkedAndDecodeParity() throws {
        let configs = [
            #""qk_rope_head_dim": 0, "use_mla_o_norm": true, "#,
            #""qk_rope_head_dim": 16, "use_mla_o_norm": true, "dsa_index_heads": 4, "dsa_index_head_dim": 16, "dsa_top_k": 6, "#,
        ]
        let hidden = MLXArray(SparseAttentionCPUOracle.pattern(1 * 12 * 64, mul: 23, mod: 41, scale: 0.06), [1, 12, 64])
        for extra in configs {
            let model = try #require(try TinyModel.make(.bailingHybrid, adding: extra) as? BailingHybridModel)
            let mla = try #require(model.model.layers[2].attention as? BailingHybridMLAAttention)
            let full = mla(hidden, mask: .causal, cache: nil)
            let cache = DSALatentKVCache()
            let p1 = mla(hidden[0..., 0 ..< 5, 0...], mask: .causal, cache: cache)
            let p2 = mla(hidden[0..., 5 ..< 9, 0...], mask: .causal, cache: cache)
            var pieces: [MLXArray] = [p1, p2]
            for pos in 9 ..< 12 {
                pieces.append(mla(hidden[0..., pos ..< (pos + 1), 0...], mask: .none, cache: cache))
            }
            let streamed = concatenated(pieces, axis: 1)
            eval(full, streamed)
            #expect(cache.offset == 12)
            #expect(abs(full - streamed).max().item(Float.self) < 1e-4)
        }
    }
}

enum SparseAttentionCPUOracle {
    struct IndexerShape {
        let batch: Int
        let lQ: Int
        let lK: Int
        let hI: Int
        let dI: Int
    }

    struct TopKSpec {
        let batch: Int
        let lQ: Int
        let lK: Int
        let topK: Int
        let causal: Bool
        let qOffset: Int
    }

    struct SparseLatentHost {
        let qAbs: [Float]
        let qPe: [Float]?
        let cKV: [Float]
        let kPe: [Float]?
        let vGate: [Float]?
        let topIndices: [Int32]
    }

    struct SparseLatentSpec {
        let batch: Int
        let heads: Int
        let lQ: Int
        let lK: Int
        let dc: Int
        let dr: Int
        let topK: Int
        let scale: Float
        let causal: Bool
        let qOffset: Int
    }

    static func pattern(_ count: Int, mul: Int, mod: Int, scale: Float) -> [Float] {
        let half = mod / 2
        return (0 ..< count).map { Float(($0 * mul) % mod - half) * scale }
    }

    static func maxFiniteDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count)
        var worst: Float = 0
        for (a, b) in zip(lhs, rhs) {
            if a == -Float.infinity && b == -Float.infinity { continue }
            worst = max(worst, abs(a - b))
        }
        return worst
    }

    static func dsaIndexerScore(
        q: [Float],
        k: [Float],
        w: [Float],
        shape: IndexerShape,
        causal: Bool,
        qOffset: Int
    ) -> [Float] {
        var out = [Float](repeating: -Float.infinity, count: shape.batch * shape.lQ * shape.lK)
        for b in 0 ..< shape.batch {
            for t in 0 ..< shape.lQ {
                let bt = b * shape.lQ + t
                for s in 0 ..< shape.lK {
                    if causal && s > qOffset + t { continue }
                    var acc = 0.0
                    for j in 0 ..< shape.hI {
                        var dot = 0.0
                        let qOff = (bt * shape.hI + j) * shape.dI
                        let kOff = (b * shape.lK + s) * shape.dI
                        for d in 0 ..< shape.dI {
                            dot += Double(q[qOff + d]) * Double(k[kOff + d])
                        }
                        if dot > 0.0 {
                            acc += Double(w[bt * shape.hI + j]) * dot
                        }
                    }
                    out[bt * shape.lK + s] = Float(acc)
                }
            }
        }
        return out
    }

    static func selectTopK(scores: [Float], spec: TopKSpec) -> [Int32] {
        let kSel = min(spec.topK, spec.lK)
        var out = [Int32](repeating: -1, count: spec.batch * spec.lQ * kSel)
        for b in 0 ..< spec.batch {
            for t in 0 ..< spec.lQ {
                let rowBase = (b * spec.lQ + t) * spec.lK
                let pairs = (0 ..< spec.lK).map { (idx: $0, val: scores[rowBase + $0]) }.sorted { a, b in
                    a.val != b.val ? a.val > b.val : a.idx < b.idx
                }
                let outBase = (b * spec.lQ + t) * kSel
                for slot in 0 ..< kSel {
                    let chosen = pairs[slot].idx
                    out[outBase + slot] = (spec.causal && chosen > spec.qOffset + t) ? -1 : Int32(chosen)
                }
            }
        }
        return out
    }
}

extension SparseAttentionCPUOracle {
    private static func sparseRowEntries(
        host: SparseLatentHost,
        spec: SparseLatentSpec,
        batchIndex: Int,
        flatHead: Int,
        queryIndex: Int
    ) -> [(tok: Int, logit: Double)] {
        let maxTok = spec.causal ? min(spec.lK - 1, spec.qOffset + queryIndex) : (spec.lK - 1)
        let idxBase = (batchIndex * spec.lQ + queryIndex) * spec.topK
        var entries: [(tok: Int, logit: Double)] = []
        for slot in 0 ..< spec.topK {
            let tok = Int(host.topIndices[idxBase + slot])
            if tok < 0 || tok > maxTok { continue }
            var dot = 0.0
            let qOff = (flatHead * spec.lQ + queryIndex) * spec.dc
            let cOff = (batchIndex * spec.lK + tok) * spec.dc
            for d in 0 ..< spec.dc { dot += Double(host.qAbs[qOff + d]) * Double(host.cKV[cOff + d]) }
            if spec.dr > 0, let qPe = host.qPe, let kPe = host.kPe {
                let qpOff = (flatHead * spec.lQ + queryIndex) * spec.dr
                let kpOff = (batchIndex * spec.lK + tok) * spec.dr
                for d in 0 ..< spec.dr { dot += Double(qPe[qpOff + d]) * Double(kPe[kpOff + d]) }
            }
            entries.append((tok, dot * Double(spec.scale)))
        }
        return entries
    }

    static func dsaSparseLatentAttention(host: SparseLatentHost, spec: SparseLatentSpec) -> [Float] {
        var out = [Float](repeating: 0, count: spec.batch * spec.heads * spec.lQ * spec.dc)
        for b in 0 ..< spec.batch {
            for h in 0 ..< spec.heads {
                let bh = b * spec.heads + h
                for qi in 0 ..< spec.lQ {
                    let entries = sparseRowEntries(host: host, spec: spec, batchIndex: b, flatHead: bh, queryIndex: qi)
                    guard let maxL = entries.map(\.logit).max() else { continue }
                    let exps = entries.map { exp($0.logit - maxL) }
                    let sumExp = exps.reduce(0.0, +)
                    guard sumExp > 0 else { continue }
                    let outOff = (bh * spec.lQ + qi) * spec.dc
                    for (entry, eVal) in zip(entries, exps) {
                        let vg = host.vGate.map { Double($0[bh * spec.lK + entry.tok]) } ?? 1.0
                        let weight = (eVal / sumExp) * vg
                        let cOff = (b * spec.lK + entry.tok) * spec.dc
                        for d in 0 ..< spec.dc {
                            out[outOff + d] += Float(weight * Double(host.cKV[cOff + d]))
                        }
                    }
                }
            }
        }
        return out
    }

    static func selectNSABlocks(
        compWeights: [Float],
        kvHeads: Int,
        gqaGroup: Int,
        lQ: Int,
        numBlocks: Int,
        topBlocks: Int
    ) -> [Int32] {
        let kSel = min(topBlocks, numBlocks)
        var out = [Int32](repeating: 0, count: kvHeads * lQ * kSel)
        for kv in 0 ..< kvHeads {
            for qi in 0 ..< lQ {
                var groupSums = [Double](repeating: 0.0, count: numBlocks)
                for g in 0 ..< gqaGroup {
                    let base = ((kv * gqaGroup + g) * lQ + qi) * numBlocks
                    for blk in 0 ..< numBlocks {
                        groupSums[blk] += Double(compWeights[base + blk])
                    }
                }
                let ranked = (0 ..< numBlocks).map { (idx: $0, val: groupSums[$0]) }.sorted { a, b in
                    a.val != b.val ? a.val > b.val : a.idx < b.idx
                }
                let outBase = (kv * lQ + qi) * kSel
                for slot in 0 ..< kSel { out[outBase + slot] = Int32(ranked[slot].idx) }
            }
        }
        return out
    }

    static func assertGatedMLAParity(
        _ attn: GatedMLAAttention,
        hidden: MLXArray,
        cache: DSALatentKVCache,
        refCache: DSALatentKVCache,
        tolerance: Float
    ) throws {
        let fullGot = attn(hidden, mask: .causal, cache: nil).asType(.float32)
        let fullRef = attn.forwardExpandedReference(hidden, mask: .causal, cache: nil).asType(.float32)
        eval(fullGot, fullRef)
        #expect(abs(fullGot - fullRef).max().item(Float.self) < tolerance)

        var gotParts: [MLXArray] = []
        var refParts: [MLXArray] = []
        var cursor = 0
        let length = hidden.dim(1)
        for step in [5, 1, 3, 1, 2] where cursor < length {
            let end = min(length, cursor + step)
            let slice = hidden[0..., cursor ..< end, 0...]
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = (end - cursor) > 1 ? .causal : .none
            gotParts.append(attn(slice, mask: mask, cache: cache).asType(.float32))
            refParts.append(attn.forwardExpandedReference(slice, mask: mask, cache: refCache).asType(.float32))
            cursor = end
        }
        let chunkGot = concatenated(gotParts, axis: 1)
        let chunkRef = concatenated(refParts, axis: 1)
        eval(chunkGot, chunkRef)
        #expect(abs(chunkGot - chunkRef).max().item(Float.self) < tolerance)
        #expect(abs(chunkGot - fullRef).max().item(Float.self) < tolerance)
    }
}
