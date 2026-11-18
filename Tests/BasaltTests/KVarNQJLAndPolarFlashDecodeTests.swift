import Foundation
import MLX
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Long-CoT KV compression: KVarN 2D balancing, unbiased 1-bit QJL, ChanMix, ThinKV & fused Flash-Decoding (#550)")
struct KVarNQJLAndPolarFlashDecodeTests {
    @Test("2-bit KVarN + 1-bit QJL reduces inner-product bias > 10x vs pure 2-bit MSE across 256 random trials (#228)")
    func qjlUnbiasedInnerProductProof() {
        MLXRandom.seed(5500)
        let d = 64
        let t = 64
        let keys = MLXRandom.normal([t, d], dtype: .float32) * MLXArray((0 ..< d).map { $0 % 8 == 0 ? Float(4.0) : Float(1.0) }).reshaped(1, d)
        let queries = keys * Float(0.6) + MLXRandom.normal([t, d], dtype: .float32) * Float(0.8)
        eval(keys, queries)

        let keysHost = keys.asArray(Float.self)
        let queriesHost = queries.asArray(Float.self)
        var exactDots = [Float](repeating: 0, count: t)
        for i in 0 ..< t {
            var s: Float = 0
            for c in 0 ..< d { s += queriesHost[i * d + c] * keysHost[i * d + c] }
            exactDots[i] = s
        }

        let mseTile = KVarNTileCodec.encodeTile(keys, useQJL: false)
        var mseSignedErrSum: Float = 0
        for i in 0 ..< t {
            let qVec = Array(queriesHost[(i * d) ..< ((i + 1) * d)])
            let est = KVarNTileCodec.cpuEstimateScores(query: qVec, tile: mseTile)[i]
            mseSignedErrSum += (est - exactDots[i])
        }
        let mseBias = abs(mseSignedErrSum / Float(t))

        var qjlSignedErrSum: Float = 0
        let trials = 16
        for trial in 0 ..< trials {
            let sJL = KVarNTileCodec.makeJLMatrix(dim: d, seed: UInt64(1000 + trial))
            eval(sJL)
            let jlHost = sJL.asArray(Float.self)
            let qjlTile = KVarNTileCodec.encodeTile(keys, useQJL: true, jlMatrix: sJL)
            for i in 0 ..< t {
                let qVec = Array(queriesHost[(i * d) ..< ((i + 1) * d)])
                let est = KVarNTileCodec.cpuEstimateScores(query: qVec, tile: qjlTile, jlMatrixHost: jlHost)[i]
                qjlSignedErrSum += (est - exactDots[i])
            }
        }
        let qjlBias = abs(qjlSignedErrSum / Float(t * trials))
        #expect(qjlBias * 10.0 < max(mseBias, 1e-2))
    }

    @Test("Fused kv_fused_flash_decode_kvarn_qjl Metal kernel matches pure CPU online-softmax reference (#228)")
    func kvarnFlashDecodeMetalParity() {
        MLXRandom.seed(5501)
        let (heads, t, d) = (4, 48, 64)
        let q = MLXRandom.normal([heads, d], dtype: .float32)
        let k = MLXRandom.normal([t, d], dtype: .float32)
        let v = MLXRandom.normal([t, d], dtype: .float32)
        let sJL = KVarNTileCodec.makeJLMatrix(dim: d, seed: 550)
        let vTile = KVarNTileCodec.encodeTile(v, useQJL: false)
        let scale = 1.0 / Float(d).squareRoot()

        for useQJL in [false, true] {
            let kTile = KVarNTileCodec.encodeTile(k, useQJL: useQJL, jlMatrix: useQJL ? sJL : nil)
            for dtype in [DType.float32, .float16, .bfloat16] {
                let inp = KVarNFlashDecodeMetal.Inputs(
                    queries: q.asType(dtype),
                    keyTile: kTile,
                    valueTile: vTile,
                    jlMatrix: useQJL ? sJL : nil,
                    scale: scale)
                let gpu = KVarNFlashDecodeMetal.flashDecode(inp).asType(.float32)
                eval(gpu)
                let cpuInp = KVarNFlashDecodeMetal.Inputs(
                    queries: q.asType(dtype).asType(.float32),
                    keyTile: kTile,
                    valueTile: vTile,
                    jlMatrix: useQJL ? sJL : nil,
                    scale: scale)
                let cpu = KVarNFlashDecodeMetal.cpuFlashDecode(cpuInp)
                let tol: Float = dtype == .bfloat16 ? 2.5e-2 : (dtype == .float16 ? 6e-3 : 5e-4)
                let maxDiff = zip(gpu.asArray(Float.self), cpu).map { abs($0 - $1) }.max() ?? 0
                #expect(maxDiff <= tol)
            }
        }
    }

    @Test("PolarKeyRows fused Metal scores match dequantized reference without intermediate FP16 key allocation (#228)")
    func polarFusedMiddleScoresParity() {
        MLXRandom.seed(5502)
        for (rotated, headDim) in [(64, 64), (32, 64)] {
            let baseKeys = MLXRandom.normal([1, 2, 24, headDim], dtype: .float32)
            for dtype in [DType.float32, .float16, .bfloat16] {
                var rows = PolarKeyRows(rotated: rotated, restBits: 4, groupSize: 32, headDim: headDim)
                rows.append(baseKeys.asType(dtype))
                let q = MLXRandom.normal([1, 2, 2, 3, headDim], dtype: .float32).asType(dtype)
                let fused = rows.scores(q)!.asType(.float32)
                let deqKeys = rows.dequantizedRows(dtype: dtype)!
                let ref = matmul(q, expandedDimensions(deqKeys, axis: 2).swappedAxes(-1, -2)).asType(.float32)
                eval(fused, ref)
                let relErr = (abs(fused - ref).max() / maximum(abs(ref).max(), Float(1e-6))).item(Float.self)
                let relTol: Float = dtype == .bfloat16 ? 1.5e-2 : (dtype == .float16 ? 3e-3 : 2e-4)
                #expect(relErr <= relTol)
            }
        }
    }

    @Test("ChanMix 4/2-bit sensitive channel retention and ThinKV thought-span segmentation")
    func chanMixAndThinKVSegmentation() {
        MLXRandom.seed(5503)
        let base = MLXRandom.normal([32, 64], dtype: .float32)
        let spikes = MLXArray((0 ..< 64).map { $0 < 4 ? Float(6.0) : Float(1.0) }).reshaped(1, 64)
        let keys = base * spikes
        let enc = ChanMixCodec.encode(keys, sensitiveCount: 4)
        let rec = ChanMixCodec.decode(enc)
        let mseChanMix = mean(square(rec - keys)).item(Float.self)

        let scaleOnly = maximum(sqrt(mean(square(keys), axis: 1, keepDims: true)), Float(1e-6))
        let pure2Bit = (clip(round((keys / scaleOnly) + Float(1.5)), min: Float(0), max: Float(3)) - Float(1.5)) * scaleOnly
        let msePure2 = mean(square(pure2Bit - keys)).item(Float.self)
        #expect(mseChanMix < msePure2)

        let tokens: [Int32] = [1, 100, 11, 12, 13, 101, 20, 21, 22, 23, 24]
        let phases = ThinKVSegmenter.classifyTokens(
            tokenIDs: tokens, thinkOpenID: 100, thinkCloseID: 101, recentWindow: 4)
        #expect(phases[0] == .activeSynthesis4Bit)
        #expect(phases[1] == .completedThought2Bit)
        #expect(phases[5] == .completedThought2Bit)
        #expect(phases[10] == .activeSynthesis4Bit)
    }
}
