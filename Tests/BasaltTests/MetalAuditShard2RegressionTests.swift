import Foundation
import MLX
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Metal audit shard 2: PolarKeyRows constant-space buffers, empty guards, LUT parity and CPU fallback", .serialized)
struct MetalAuditShard2PolarTests {
    @Test("PolarKeyRows handles sigma.size < 8, all-constant buffers and rotated > 64 across float32, float16 and bfloat16")
    func polarSmallConstantAndLargeRotatedParity() throws {
        let rng = MLXRandom.RandomState(seed: 7201)
        let configs = [(1, 1, 1, 1, 1, 4, 4), (1, 1, 1, 1, 6, 8, 8), (1, 2, 2, 2, 12, 128, 128), (1, 2, 2, 1, 10, 96, 128)]
        for (b, hKV, gQ, lQ, tMid, rotated, headDim) in configs {
            let keys = MLXRandom.normal([b, hKV, tMid, headDim], dtype: .float32, key: rng)
            let q = MLXRandom.normal([b, hKV, gQ, lQ, headDim], dtype: .float32, key: rng)
            for dtype in [DType.float32, .float16, .bfloat16] {
                var rows = PolarKeyRows(rotated: rotated, restBits: 4, groupSize: 32, headDim: headDim)
                rows.append(keys.asType(dtype))
                let fused = try #require(rows.scores(q.asType(dtype))).asType(.float32)
                let deq = try #require(rows.dequantizedRows(dtype: dtype))
                let ref = matmul(q.asType(dtype), expandedDimensions(deq, axis: 2).swappedAxes(-1, -2)).asType(.float32)
                eval(fused, ref)
                #expect(fused.shape == ref.shape)
                let relErr = (abs(fused - ref).max() / maximum(abs(ref).max(), Float(1e-6))).item(Float.self)
                let tol: Float = dtype == .bfloat16 ? 2e-2 : (dtype == .float16 ? 4e-3 : 2e-4)
                #expect(relErr <= tol)
            }
        }
    }

    @Test("PolarKeyRows guards tMid == 0, tasks == 0 and CPU default device fallback cleanly")
    func polarEmptyAndCPUFallback() throws {
        var rows = PolarKeyRows(rotated: 32, restBits: 4, groupSize: 32, headDim: 64)
        rows.append(MLXArray.zeros([1, 2, 0, 64], type: Float.self))
        #expect(rows.count == 0 && rows.sigma == nil)
        let q = MLXRandom.normal([1, 2, 2, 3, 64], key: MLXRandom.key(7202))
        #expect(rows.scores(q) == nil)

        let keys = MLXRandom.normal([1, 2, 16, 64], key: MLXRandom.key(7203))
        rows.append(keys)
        rows.append(MLXArray.zeros([1, 2, 0, 64], type: Float.self))
        #expect(rows.count == 16)

        let emptyQ = MLXArray.zeros([1, 2, 2, 0, 64], type: Float.self)
        let emptyOut = try #require(rows.scores(emptyQ))
        eval(emptyOut)
        #expect(emptyOut.shape == [1, 2, 2, 0, 16])

        eval(rows.arrays + [q])
        let gpuOut = try #require(rows.scores(q)).asType(.float32)
        eval(gpuOut)
        let cpuOut = try Device.withDefaultDevice(.cpu) {
            let out = try #require(rows.scores(q)).asType(.float32)
            eval(out)
            return out
        }
        let maxDiff = abs(gpuOut - cpuOut).max().item(Float.self)
        #expect(maxDiff < 1e-4)
    }
}

@Suite("Metal audit shard 2: FusedLogitConfidence CPU fallback, V=1, empty shapes and unsupported dtypes", .serialized)
struct MetalAuditShard2ConfidenceTests {
    @Test("FusedLogitConfidence falls back to reference on CPU default device and unsupported dtypes")
    func confidenceCPUAndUnsupportedDTypeParity() throws {
        let logits = MLXRandom.normal([4, 64], key: MLXRandom.key(7204)) * 3.0
        eval(logits)
        let gpu = FusedLogitConfidence.compute(logits)
        eval(gpu.top1TokenID, gpu.top1Logprob, gpu.top2Margin, gpu.entropy)

        let cpu = Device.withDefaultDevice(.cpu) {
            let out = FusedLogitConfidence.compute(logits)
            eval(out.top1TokenID, out.top1Logprob, out.top2Margin, out.entropy)
            return out
        }
        #expect(cpu.top1TokenID.asArray(Int32.self) == gpu.top1TokenID.asArray(Int32.self))
        #expect(abs(cpu.top1Logprob - gpu.top1Logprob).max().item(Float.self) < 1e-4)
        #expect(abs(cpu.top2Margin - gpu.top2Margin).max().item(Float.self) < 1e-4)
        #expect(abs(cpu.entropy - gpu.entropy).max().item(Float.self) < 2e-4)

        let cpuSummary = Device.withDefaultDevice(.cpu) { FusedLogitConfidence.summarize(logits) }
        #expect(try #require(cpuSummary).meanEntropy != nil)

        let intLogits = MLXArray([Int32(1), 5, 2, 5], [1, 4])
        let intOut = FusedLogitConfidence.compute(intLogits)
        eval(intOut.top1TokenID, intOut.top2Margin)
        #expect(intOut.top1TokenID.asArray(Int32.self) == [1])
        #expect(intOut.top2Margin.asArray(Float.self) == [0])
    }

    @Test("FusedLogitConfidence handles V=1, size < 8 and empty shapes on both GPU and CPU")
    func confidenceSingleVocabAndEmptyShapes() {
        let single = MLXArray([Float(2.5), -1.25], [2, 1])
        eval(single)
        for onCPU in [false, true] {
            let out = onCPU
                ? Device.withDefaultDevice(.cpu) {
                    let c = FusedLogitConfidence.compute(single)
                    eval(c.top1TokenID, c.top1Logprob, c.top2Margin, c.entropy)
                    return c
                }
                : FusedLogitConfidence.compute(single)
            if !onCPU { eval(out.top1TokenID, out.top1Logprob, out.top2Margin, out.entropy) }
            #expect(out.top1TokenID.asArray(Int32.self) == [0, 0])
            #expect(out.top1Logprob.asArray(Float.self).allSatisfy { abs($0) < 1e-5 })
            #expect(out.top2Margin.asArray(Float.self).allSatisfy { $0.isInfinite && $0 > 0 })
            #expect(out.entropy.asArray(Float.self).allSatisfy { abs($0) < 1e-5 })
        }

        for shape in [[0, 4], [2, 0], [0]] {
            let empty = MLXArray.zeros(shape, type: Float.self)
            let out = FusedLogitConfidence.compute(empty)
            let expectedLeading = Array(shape.dropLast())
            #expect(out.top1TokenID.shape == expectedLeading && out.entropy.shape == expectedLeading)
        }
        let scalar = FusedLogitConfidence.compute(MLXArray(Float(1.0)))
        #expect(scalar.top1TokenID.shape == [] && scalar.entropy.shape == [])
    }
}

@Suite("Metal audit shard 2: SpecTrAndResidualRaceMetal residual acceptance, NaN/Inf, vocab=0 and CPU fallback", .serialized)
struct MetalAuditShard2SpecTrTests {
    @Test("SpecTrAndResidualRaceMetal reads updated p_res[tok] after subtraction so GPU matches CPU on residual collapse")
    func residualCollapseMatchesCPUExactly() {
        let tHost: [[Float]] = [[0.0, -30.0], [0.0, 0.0]]
        let dHost: [[[Float]]] = [[[0.0, -29.0]]]
        let tokHost: [[Int32]] = [[1]]
        let tArr = MLXArray(tHost.flatMap { $0 }, [2, 2])
        let dArr = MLXArray(dHost.flatMap { $0.flatMap { $0 } }, [1, 1, 2])
        let tokArr = MLXArray([Int32(1)], [1, 1])

        var rejectedAtLeastOnce = false
        for mode in [SpeculativeVerifyMode.residualRace, .specTrOT] {
            for seed: UInt32 in 0 ..< 24 {
                let gpu = SpecTrAndResidualRaceMetal.verifyGPU(
                    targetLogits: tArr, draftLogits: dArr, draftTokens: tokArr,
                    temperature: 1.0, mode: mode, seed: seed)
                let cpu = SpecTrAndResidualRaceMetal.verifyCPU(
                    targetLogits: tHost, draftLogits: dHost, draftTokens: tokHost,
                    temperature: 1.0, mode: mode, seed: seed)
                #expect(gpu == cpu)
                if cpu.acceptedCount == 0 { rejectedAtLeastOnce = true }
            }
        }
        #expect(rejectedAtLeastOnce)
    }

    @Test("SpecTrAndResidualRaceMetal aligns GPU and CPU on NaN and +-Inf logits at T=0 and T>0")
    func nanAndInfLogitsParity() {
        let rows: [[Float]] = [
            [.nan, 1.0, 0.5, -2.0],
            [.nan, -.infinity, -.infinity, .nan],
            [.infinity, 1.0, .infinity, .nan],
            [-.infinity, -.infinity, -.infinity, -.infinity],
            [.nan, .nan, .nan, .nan],
        ]
        for row in rows {
            let probs = SpecTrAndResidualRaceMetal.softmaxHost(row, temperature: 1.0)
            #expect(probs.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
            let total = probs.reduce(0, +)
            #expect(total == 0 || abs(total - 1) < 1e-5)

            let tHost = [row, [0.5, 1.5, -0.5, 0.0]]
            let dHost = [[row, [0.2, 0.8, -0.1, 0.0]]]
            let tokHost: [[Int32]] = [[0, 1]]
            let tArr = MLXArray(tHost.flatMap { $0 }, [2, 4])
            let dArr = MLXArray(dHost.flatMap { $0.flatMap { $0 } }, [1, 2, 4])
            let tokArr = MLXArray([Int32(0), 1], [1, 2])
            for temp: Float in [0.0, 1.0] {
                for mode in [SpeculativeVerifyMode.residualRace, .specTrOT] {
                    let gpu = SpecTrAndResidualRaceMetal.verifyGPU(
                        targetLogits: tArr, draftLogits: dArr, draftTokens: tokArr,
                        temperature: temp, mode: mode, seed: 77)
                    let cpu = SpecTrAndResidualRaceMetal.verifyCPU(
                        targetLogits: tHost, draftLogits: dHost, draftTokens: tokHost,
                        temperature: temp, mode: mode, seed: 77)
                    #expect(gpu == cpu)
                }
            }
        }
    }

    @Test("SpecTrAndResidualRaceMetal guards vocab == 0, falls back on CPU device and supports dynamic step/candidate shapes")
    func vocabZeroCPUFallbackAndDynamicShapes() {
        let emptyOut = SpecTrAndResidualRaceMetal.verifyGPU(
            targetLogits: MLXArray.zeros([2, 0], type: Float.self),
            draftLogits: MLXArray.zeros([1, 1, 0], type: Float.self),
            draftTokens: MLXArray([Int32(0)], [1, 1]),
            temperature: 0.7, mode: .specTrOT, seed: 1)
        #expect(emptyOut == SpeculativeVerifyOutput(acceptedCount: 0, emittedTokens: []))
        let emptyCPU = SpecTrAndResidualRaceMetal.verifyCPU(
            targetLogits: [[], []], draftLogits: [[[]]], draftTokens: [[0]],
            temperature: 0.7, mode: .specTrOT, seed: 1)
        #expect(emptyCPU == SpeculativeVerifyOutput(acceptedCount: 0, emittedTokens: []))

        let rng = MLXRandom.RandomState(seed: 7205)
        for (steps, cands) in [(1, 1), (3, 2), (4, 5)] {
            let vocab = 16
            let tLogits = MLXRandom.normal([steps + 1, vocab], dtype: .float32, key: rng)
            let dLogits = MLXRandom.normal([steps, cands, vocab], dtype: .float32, key: rng)
            let tokFlat = (0 ..< (steps * cands)).map { Int32($0 % vocab) }
            let dTokens = MLXArray(tokFlat, [steps, cands])
            eval(tLogits, dLogits, dTokens)
            for temp: Float in [0.0, 0.8] {
                let gpu = SpecTrAndResidualRaceMetal.verifyGPU(
                    targetLogits: tLogits, draftLogits: dLogits, draftTokens: dTokens,
                    temperature: temp, mode: .specTrOT, seed: 19)
                let cpuDevice = Device.withDefaultDevice(.cpu) {
                    SpecTrAndResidualRaceMetal.verifyGPU(
                        targetLogits: tLogits, draftLogits: dLogits, draftTokens: dTokens,
                        temperature: temp, mode: .specTrOT, seed: 19)
                }
                #expect(gpu == cpuDevice)
            }
        }
    }
}

@Suite("Metal audit shard 2: QuestKernel guards, >4096 fallback, constant buffers, distinct vh stride and CPU fallback", .serialized)
struct MetalAuditShard2QuestTests {
    @Test("QuestKernel.fits rejects zero dims, GQA mismatch, middleBlocks > 4096 and CPU default device")
    func fitsRejectsInvalidShapesAndCPU() {
        let validQ = MLXArray.zeros([1, 4, 1, 32], type: Float.self)
        let validK = MLXArray.zeros([1, 2, 512, 32], type: Float.self)
        let validV = MLXArray.zeros([1, 2, 512, 32], type: Float.self)
        #expect(QuestKernel.fits(queries: validQ, keys: validK, values: validV, middleBlocks: 8))
        #expect(!QuestKernel.fits(queries: validQ, keys: MLXArray.zeros([1, 0, 512, 32], type: Float.self), values: validV, middleBlocks: 8))
        #expect(!QuestKernel.fits(queries: MLXArray.zeros([1, 4, 1, 0], type: Float.self), keys: MLXArray.zeros([1, 2, 512, 0], type: Float.self), values: validV, middleBlocks: 8))
        #expect(!QuestKernel.fits(queries: MLXArray.zeros([1, 3, 1, 32], type: Float.self), keys: validK, values: validV, middleBlocks: 8))
        #expect(!QuestKernel.fits(queries: validQ, keys: validK, values: validV, middleBlocks: 0))
        #expect(!QuestKernel.fits(queries: validQ, keys: validK, values: validV, middleBlocks: QuestKernel.maxMiddleBlocks + 1))
        let cpuFits = Device.withDefaultDevice(.cpu) {
            QuestKernel.fits(queries: validQ, keys: validK, values: validV, middleBlocks: 8)
        }
        #expect(!cpuFits)
    }

    @Test("QuestKernel.selectedBlocks falls back cleanly for >4096 middle blocks, handles <8-element buffers and runs on CPU")
    func selectionLargeMiddleSmallConstantAndCPU() {
        let qSmall = MLXRandom.normal([1, 1, 1, 2], key: MLXRandom.key(7206))
        let aSmall = MLXRandom.normal([1, 1, 3, 2], key: MLXRandom.key(7207))
        let bSmall = MLXRandom.normal([1, 1, 3, 2], key: MLXRandom.key(7208))
        let (upSmall, loSmall) = (maximum(aSmall, bSmall), minimum(aSmall, bSmall))
        eval(qSmall, upSmall, loSmall)
        let smallGPU = QuestKernel.selectedBlocks(qSmall, upper: upSmall, lower: loSmall, first: 0, last: 3, topK: 2)
        let smallCPU = Device.withDefaultDevice(.cpu) {
            let out = QuestKernel.selectedBlocks(qSmall, upper: upSmall, lower: loSmall, first: 0, last: 3, topK: 2)
            eval(out)
            return out
        }
        eval(smallGPU)
        #expect((sorted(smallGPU, axis: -1) .== sorted(smallCPU, axis: -1)).all().item(Bool.self))

        let blocks = QuestKernel.maxMiddleBlocks + 24
        let qBig = MLXRandom.normal([1, 2, 1, 32], key: MLXRandom.key(7209))
        let aBig = MLXRandom.normal([1, 1, blocks, 32], key: MLXRandom.key(7210))
        let bBig = MLXRandom.normal([1, 1, blocks, 32], key: MLXRandom.key(7211))
        let (upBig, loBig) = (maximum(aBig, bBig), minimum(aBig, bBig))
        let middle = 2 ..< (blocks - 4)
        let picked = QuestKernel.selectedBlocks(qBig, upper: upBig, lower: loBig, first: middle.lowerBound, last: middle.upperBound, topK: 4)
        let reference = QuestKVCache.topBlocks(qBig, upper: upBig, lower: loBig, middle: middle, topK: 4)
        eval(picked, reference)
        #expect(picked.shape == [1, 1, 4])
        #expect((sorted(picked, axis: -1) .== sorted(reference, axis: -1)).all().item(Bool.self))
    }

    @Test("QuestKernel.attention uses values_shape[2] stride with distinct value capacity, B>1, E!=EV and CPU fallback")
    func attentionDistinctValueStrideAndCPUFallback() {
        let (batch, kvHeads, groups, dim, valueDim) = (2, 2, 2, 64, 32)
        let offset = 700
        let q = MLXRandom.normal([batch, kvHeads * groups, 1, dim], key: MLXRandom.key(7212))
        let k = MLXRandom.normal([batch, kvHeads, 768, dim], key: MLXRandom.key(7213))
        let v = MLXRandom.normal([batch, kvHeads, 1024, valueDim], key: MLXRandom.key(7214))
        let selected = MLXArray([Int32(2), 4, 3, 5, 2, 5, 3, 4], [batch, kvHeads, 2])
        let scale = 1.0 / Float(dim).squareRoot()
        eval(q, k, v, selected)

        let gpuOut = QuestKernel.attention(queries: q, keys: k, values: v, selected: selected, offset: offset, scale: scale)
        eval(gpuOut)
        let cpuOut = Device.withDefaultDevice(.cpu) {
            let out = QuestKernel.attention(queries: q, keys: k, values: v, selected: selected, offset: offset, scale: scale)
            eval(out)
            return out
        }
        #expect(gpuOut.shape == [batch, kvHeads * groups, 1, valueDim])
        #expect(allClose(gpuOut, cpuOut, atol: 1e-4).item(Bool.self))
    }
}

@Suite("Metal audit shard 2: KVarNFlashDecodeMetal empty/CPU/FP16-range/-Inf softmax and LSE parity", .serialized)
struct MetalAuditShard2KVarNTests {
    @Test("KVarNFlashDecodeMetal matches CPU reference on output and LSE including t=1 constant buffers and CPU device fallback")
    func kvarnLSEConstantBufferAndCPUFallback() {
        let rng = MLXRandom.RandomState(seed: 7215)
        let d = 32
        let scale = 1.0 / Float(d).squareRoot()
        let sJL = KVarNTileCodec.makeJLMatrix(dim: d, seed: 7216)

        for t in [1, 24] {
            let q = MLXRandom.normal([2, d], dtype: .float32, key: rng)
            let k = MLXRandom.normal([t, d], dtype: .float32, key: rng)
            let v = MLXRandom.normal([t, d], dtype: .float32, key: rng)
            let vTile = KVarNTileCodec.encodeTile(v, useQJL: false)
            for useQJL in [false, true] {
                let kTile = KVarNTileCodec.encodeTile(k, useQJL: useQJL, jlMatrix: useQJL ? sJL : nil)
                let inp = KVarNFlashDecodeMetal.Inputs(
                    queries: q, keyTile: kTile, valueTile: vTile, jlMatrix: useQJL ? sJL : nil, scale: scale)
                let (gpuOut, gpuLSE) = KVarNFlashDecodeMetal.flashDecodeWithLSE(inp)
                eval(gpuOut, gpuLSE)
                let (cpuDevOut, cpuDevLSE) = Device.withDefaultDevice(.cpu) {
                    let pair = KVarNFlashDecodeMetal.flashDecodeWithLSE(inp)
                    eval(pair.output, pair.lse)
                    return pair
                }
                let (refOut, refLSE) = KVarNFlashDecodeMetal.cpuFlashDecodeWithLSE(inp)
                let gpuOutArr = gpuOut.asArray(Float.self)
                let gpuLSEArr = gpuLSE.asArray(Float.self)
                #expect(!gpuOutArr.isEmpty && gpuOutArr.count == refOut.count)
                #expect(!gpuLSEArr.isEmpty && gpuLSEArr.count == refLSE.count)
                #expect(zip(gpuOutArr, refOut).allSatisfy { abs($0 - $1) <= 5e-4 })
                #expect(zip(gpuLSEArr, refLSE).allSatisfy { abs($0 - $1) <= 1e-3 })
                #expect(allClose(gpuOut, cpuDevOut, atol: 5e-4).item(Bool.self))
                #expect(allClose(gpuLSE, cpuDevLSE, atol: 1e-3).item(Bool.self))
            }
        }
    }

    @Test("KVarNFlashDecodeMetal guards empty heads and empty tiles without crashing")
    func kvarnEmptyHeadsAndEmptyTiles() {
        let d = 32
        let k = MLXRandom.normal([4, d], dtype: .float32, key: MLXRandom.key(7217))
        let v = MLXRandom.normal([4, d], dtype: .float32, key: MLXRandom.key(7218))
        let kTile = KVarNTileCodec.encodeTile(k, useQJL: false)
        let vTile = KVarNTileCodec.encodeTile(v, useQJL: false)

        let emptyHeadInp = KVarNFlashDecodeMetal.Inputs(
            queries: MLXArray.zeros([0, d], type: Float.self), keyTile: kTile, valueTile: vTile, scale: 0.125)
        let (ehOut, ehLSE) = KVarNFlashDecodeMetal.flashDecodeWithLSE(emptyHeadInp)
        eval(ehOut, ehLSE)
        #expect(ehOut.shape == [0, d] && ehLSE.shape == [0])

        var emptyK = kTile
        emptyK.tokens = 0
        emptyK.packedCodes = MLXArray.zeros([0, d / 16], type: UInt32.self)
        emptyK.tokenScale = MLXArray.zeros([0], type: Float.self)
        var emptyV = vTile
        emptyV.tokens = 0
        emptyV.packedCodes = MLXArray.zeros([0, d / 16], type: UInt32.self)
        emptyV.tokenScale = MLXArray.zeros([0], type: Float.self)

        let emptyTileInp = KVarNFlashDecodeMetal.Inputs(
            queries: MLXArray.ones([2, d], dtype: .float32), keyTile: emptyK, valueTile: emptyV, scale: 0.125)
        let (etOut, etLSE) = KVarNFlashDecodeMetal.flashDecodeWithLSE(emptyTileInp)
        eval(etOut, etLSE)
        #expect(etOut.shape == [2, d])
        #expect(etOut.asArray(Float.self).allSatisfy { $0 == 0 })
        #expect(etLSE.asArray(Float.self).allSatisfy { $0 == -.infinity })

        let intInp = KVarNFlashDecodeMetal.Inputs(
            queries: MLXArray.ones([1, d], dtype: .int32), keyTile: kTile, valueTile: vTile, scale: 0.125)
        let (intOut, intLSE) = KVarNFlashDecodeMetal.flashDecodeWithLSE(intInp)
        eval(intOut, intLSE)
        #expect(intOut.dtype == .int32 && intOut.shape == [1, d] && intLSE.shape == [1])
    }

    @Test("KVarNFlashDecodeMetal keeps qRot in FP32 to avoid FP16 WHT overflow and guards -Inf online softmax")
    func kvarnFP16RangeAndNegInfSoftmax() {
        let d = 32
        let k = MLXRandom.normal([8, d], dtype: .float32, key: MLXRandom.key(7219))
        let v = MLXRandom.normal([8, d], dtype: .float32, key: MLXRandom.key(7220))
        let kTile = KVarNTileCodec.encodeTile(k, useQJL: false)
        let vTile = KVarNTileCodec.encodeTile(v, useQJL: false)
        let largeFP16Q = MLXArray.full([2, d], values: MLXArray(Float(20_000.0))).asType(.float16)
        let fp16Inp = KVarNFlashDecodeMetal.Inputs(
            queries: largeFP16Q, keyTile: kTile, valueTile: vTile, scale: 1e-5)
        let (fp16Out, fp16LSE) = KVarNFlashDecodeMetal.flashDecodeWithLSE(fp16Inp)
        eval(fp16Out, fp16LSE)
        let (refOut, refLSE) = KVarNFlashDecodeMetal.cpuFlashDecodeWithLSE(fp16Inp)
        let outArr = fp16Out.asType(.float32).asArray(Float.self)
        let lseArr = fp16LSE.asArray(Float.self)
        #expect(!outArr.isEmpty && outArr.count == refOut.count)
        #expect(!lseArr.isEmpty && lseArr.count == refLSE.count)
        #expect(outArr.allSatisfy { $0.isFinite })
        #expect(lseArr.allSatisfy { $0.isFinite })
        #expect(zip(outArr, refOut).allSatisfy { abs($0 - $1) <= 1e-2 })
        #expect(zip(lseArr, refLSE).allSatisfy { abs($0 - $1) <= 5e-3 })

        var negInfK = KVarNTileCodec.encodeTile(MLXRandom.normal([4, d], dtype: .float32, key: MLXRandom.key(7221)), useQJL: false)
        negInfK.packedCodes = MLXArray.full([4, d / 16], values: MLXArray(UInt32.max))
        negInfK.channelScale = MLXArray.ones([d], dtype: .float32)
        negInfK.tokenScale = MLXArray([-Float.infinity, -Float.infinity, Float(1.0), Float(1.0)])
        let vTile4 = KVarNTileCodec.encodeTile(MLXRandom.normal([4, d], dtype: .float32, key: MLXRandom.key(7222)), useQJL: false)
        let onesQ = MLXArray.ones([1, d], dtype: .float32)
        let negInfInp = KVarNFlashDecodeMetal.Inputs(queries: onesQ, keyTile: negInfK, valueTile: vTile4, scale: 1.0)
        let (niOut, niLSE) = KVarNFlashDecodeMetal.flashDecodeWithLSE(negInfInp)
        eval(niOut, niLSE)
        let (niRefOut, niRefLSE) = KVarNFlashDecodeMetal.cpuFlashDecodeWithLSE(negInfInp)
        let niOutArr = niOut.asArray(Float.self)
        let niLSEArr = niLSE.asArray(Float.self)
        #expect(!niOutArr.isEmpty && niOutArr.count == niRefOut.count)
        #expect(!niLSEArr.isEmpty && niLSEArr.count == niRefLSE.count)
        #expect(niOutArr.allSatisfy { $0.isFinite })
        #expect(niLSEArr.allSatisfy { $0.isFinite })
        #expect(zip(niOutArr, niRefOut).allSatisfy { abs($0 - $1) <= 1e-3 })
        #expect(zip(niLSEArr, niRefLSE).allSatisfy { abs($0 - $1) <= 1e-3 })
    }
}
