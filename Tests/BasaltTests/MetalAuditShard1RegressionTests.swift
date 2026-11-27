import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

private enum Shard1TestHelpers {
    static func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let diff = MLX.abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
        eval(diff)
        return diff.item(Float.self)
    }

    static func maxAbsDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        var peak: Float = 0
        for (a, b) in zip(lhs, rhs) {
            guard a.isFinite, b.isFinite else { return .nan }
            peak = max(peak, abs(a - b))
        }
        return peak
    }

    static func stockDualBlockAttention(
        queries: MLXArray,
        segments: DualBlockFlashAttentionKernel.Segments,
        scale: Float
    ) -> MLXArray {
        let kParts = [segments.prefixKeys, segments.activeKeys, segments.suffixKeys]
            .compactMap { $0 }
            .filter { $0.dim(2) > 0 }
        let vParts = [segments.prefixValues, segments.activeValues, segments.suffixValues]
            .compactMap { $0 }
            .filter { $0.dim(2) > 0 }
        let keys = kParts.count == 1 ? kParts[0] : concatenated(kParts, axis: 2)
        let values = vParts.count == 1 ? vParts[0] : concatenated(vParts, axis: 2)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .none
        )
    }

    static func sageTolerance(_ dtype: DType) -> Float {
        switch dtype {
        case .float32: 5e-4
        case .float16: 8e-3
        case .bfloat16: 3e-2
        default: 1e-3
        }
    }

    static func dualBlockTolerance(_ dtype: DType) -> Float {
        switch dtype {
        case .float32: 1e-4
        case .float16: 3e-3
        case .bfloat16: 2e-3
        default: 1e-4
        }
    }
}

@Suite("DualBlockFlashAttentionKernel audit regressions and parity (#228)", .serialized)
struct DualBlockAuditRegressionTests {
    @Test(
        "DualBlockFlashAttentionKernel handles activeLen == 0 with non-empty prefix or suffix across dtypes without constant-space compile crash",
        arguments: [
            (DType.float32, 16, 0), (DType.float32, 0, 20), (DType.float32, 13, 19),
            (DType.float16, 16, 0), (DType.float16, 0, 20), (DType.float16, 13, 19),
            (DType.bfloat16, 16, 0), (DType.bfloat16, 0, 20), (DType.bfloat16, 13, 19),
        ]
    )
    func activeLenZeroWithPrefixAndSuffixMatchesStock(dtype: DType, prefixLen: Int, suffixLen: Int) {
        let (batch, qHeads, kvHeads, qLen, headDim) = (1, 4, 2, 10, 64)
        let seed = UInt64(prefixLen * 100 + suffixLen * 10 + (dtype == .float32 ? 1 : dtype == .float16 ? 2 : 3))
        let q = MLXRandom.uniform(low: -0.5, high: 0.5, [batch, qHeads, qLen, headDim], key: MLXRandom.key(seed)).asType(dtype)
        let kPre = prefixLen > 0 ? MLXRandom.uniform(low: -0.5, high: 0.5, [batch, kvHeads, prefixLen, headDim], key: MLXRandom.key(seed + 1)).asType(dtype) : nil
        let vPre = prefixLen > 0 ? MLXRandom.uniform(low: -0.5, high: 0.5, [batch, kvHeads, prefixLen, headDim], key: MLXRandom.key(seed + 2)).asType(dtype) : nil
        let kAct = MLXArray.zeros([batch, kvHeads, 0, headDim], dtype: dtype)
        let vAct = MLXArray.zeros([batch, kvHeads, 0, headDim], dtype: dtype)
        let kSuf = suffixLen > 0 ? MLXRandom.uniform(low: -0.5, high: 0.5, [batch, kvHeads, suffixLen, headDim], key: MLXRandom.key(seed + 3)).asType(dtype) : nil
        let vSuf = suffixLen > 0 ? MLXRandom.uniform(low: -0.5, high: 0.5, [batch, kvHeads, suffixLen, headDim], key: MLXRandom.key(seed + 4)).asType(dtype) : nil
        let seg = DualBlockFlashAttentionKernel.Segments(
            prefixKeys: kPre, prefixValues: vPre,
            activeKeys: kAct, activeValues: vAct,
            suffixKeys: kSuf, suffixValues: vSuf
        )
        let scale = 1.0 / Float(headDim).squareRoot()
        let fused = DualBlockFlashAttentionKernel.attention(queries: q, segments: seg, scale: scale)
        let stock = Shard1TestHelpers.stockDualBlockAttention(queries: q, segments: seg, scale: scale)
        eval(fused, stock)
        #expect(fused.shape == stock.shape && fused.dtype == dtype)
        #expect(Shard1TestHelpers.maxAbsDiff(fused, stock) < Shard1TestHelpers.dualBlockTolerance(dtype))
    }

    @Test("DualBlockFlashAttentionKernel falls back cleanly on CPU default device and reports ineligible")
    func cpuDefaultDeviceFallbackMatchesGPU() {
        let q = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 4, 16, 64], key: MLXRandom.key(7101))
        let kPre = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 2, 12, 64], key: MLXRandom.key(7102))
        let vPre = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 2, 12, 64], key: MLXRandom.key(7103))
        let kAct = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 2, 16, 64], key: MLXRandom.key(7104))
        let vAct = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 2, 16, 64], key: MLXRandom.key(7105))
        let kSuf = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 2, 9, 64], key: MLXRandom.key(7106))
        let vSuf = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 2, 9, 64], key: MLXRandom.key(7107))
        eval(q, kPre, vPre, kAct, vAct, kSuf, vSuf)
        let seg = DualBlockFlashAttentionKernel.Segments(
            prefixKeys: kPre, prefixValues: vPre,
            activeKeys: kAct, activeValues: vAct,
            suffixKeys: kSuf, suffixValues: vSuf
        )
        let emptyActSeg = DualBlockFlashAttentionKernel.Segments(
            prefixKeys: kPre, prefixValues: vPre,
            activeKeys: MLXArray.zeros([2, 2, 0, 64], dtype: .float32),
            activeValues: MLXArray.zeros([2, 2, 0, 64], dtype: .float32),
            suffixKeys: kSuf, suffixValues: vSuf
        )
        let scale = 1.0 / Float(64).squareRoot()
        let gpuOut = DualBlockFlashAttentionKernel.attention(queries: q, segments: seg, scale: scale)
        let gpuEmptyAct = DualBlockFlashAttentionKernel.attention(queries: q, segments: emptyActSeg, scale: scale)
        eval(gpuOut, gpuEmptyAct)

        let (cpuEligible, cpuOut, cpuEmptyAct) = Device.withDefaultDevice(.cpu) {
            let eligible = DualBlockFlashAttentionKernel.isEligible(queries: q, segments: seg, mask: .none, mode: .forced)
            let out = DualBlockFlashAttentionKernel.attention(queries: q, segments: seg, scale: scale)
            let outEmpty = DualBlockFlashAttentionKernel.attention(queries: q, segments: emptyActSeg, scale: scale)
            eval(out, outEmpty)
            return (eligible, out, outEmpty)
        }
        #expect(!cpuEligible)
        #expect(Shard1TestHelpers.maxAbsDiff(gpuOut, cpuOut) < 1e-4)
        #expect(Shard1TestHelpers.maxAbsDiff(gpuEmptyAct, cpuEmptyAct) < 1e-4)
    }

    @Test("DualBlockFlashAttentionKernel handles empty dimensions, unsupported headDim, and float64 fallback safely")
    func emptyDimensionsAndUnsupportedFallbacks() {
        let scale = 1.0 / Float(64).squareRoot()
        let qEmptyLen = MLXArray.zeros([1, 4, 0, 64], dtype: .float32)
        let kvNormal = MLXArray.ones([1, 2, 8, 64], dtype: .float32)
        let segNormal = DualBlockFlashAttentionKernel.Segments(
            prefixKeys: nil, prefixValues: nil, activeKeys: kvNormal, activeValues: kvNormal, suffixKeys: nil, suffixValues: nil
        )
        #expect(!DualBlockFlashAttentionKernel.isEligible(queries: qEmptyLen, segments: segNormal, mask: .none, mode: .forced))
        let outEmptyQ = DualBlockFlashAttentionKernel.attention(queries: qEmptyLen, segments: segNormal, scale: scale)
        eval(outEmptyQ)
        #expect(outEmptyQ.shape == [1, 4, 0, 64])

        let qNormal = MLXArray.ones([1, 4, 6, 64], dtype: .float32)
        let kvZero = MLXArray.zeros([1, 2, 0, 64], dtype: .float32)
        let segZeroK = DualBlockFlashAttentionKernel.Segments(
            prefixKeys: nil, prefixValues: nil, activeKeys: kvZero, activeValues: kvZero, suffixKeys: nil, suffixValues: nil
        )
        #expect(!DualBlockFlashAttentionKernel.isEligible(queries: qNormal, segments: segZeroK, mask: .none, mode: .forced))
        let outEmptyK = DualBlockFlashAttentionKernel.attention(queries: qNormal, segments: segZeroK, scale: scale)
        eval(outEmptyK)
        #expect(outEmptyK.shape == [1, 4, 6, 64] && outEmptyK.asArray(Float.self).allSatisfy { $0 == 0 })

        let qZeroBatch = MLXArray.zeros([0, 4, 6, 64], dtype: .float32)
        let outZeroBatch = DualBlockFlashAttentionKernel.attention(queries: qZeroBatch, segments: segNormal, scale: scale)
        eval(outZeroBatch)
        #expect(outZeroBatch.shape == [0, 4, 6, 64])

        let qSmallD = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 4, 8, 32], key: MLXRandom.key(7120))
        let kvSmallD = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 10, 32], key: MLXRandom.key(7121))
        let segSmallD = DualBlockFlashAttentionKernel.Segments(
            prefixKeys: nil, prefixValues: nil, activeKeys: kvSmallD, activeValues: kvSmallD, suffixKeys: nil, suffixValues: nil
        )
        let outSmallD = DualBlockFlashAttentionKernel.attention(queries: qSmallD, segments: segSmallD, scale: 1.0 / Float(32).squareRoot())
        let refSmallD = Shard1TestHelpers.stockDualBlockAttention(queries: qSmallD, segments: segSmallD, scale: 1.0 / Float(32).squareRoot())
        eval(outSmallD, refSmallD)
        #expect(Shard1TestHelpers.maxAbsDiff(outSmallD, refSmallD) < 1e-5)

        let (q64, kv64) = Device.withDefaultDevice(.cpu) {
            let qa = qNormal.asType(.float64)
            let kva = kvNormal.asType(.float64)
            eval(qa, kva)
            return (qa, kva)
        }
        let seg64 = DualBlockFlashAttentionKernel.Segments(
            prefixKeys: nil, prefixValues: nil, activeKeys: kv64, activeValues: kv64, suffixKeys: nil, suffixValues: nil
        )
        #expect(!DualBlockFlashAttentionKernel.isEligible(queries: q64, segments: seg64, mask: .none, mode: .forced))
        let out64 = DualBlockFlashAttentionKernel.attention(queries: q64, segments: seg64, scale: scale)
        let out64Vals = Device.withDefaultDevice(.cpu) {
            eval(out64)
            return out64.asArray(Double.self)
        }
        #expect(out64.shape == q64.shape && out64.dtype == .float64 && out64Vals.allSatisfy { $0.isFinite })
    }

    @Test("DualBlockFlashAttentionKernel matches stock SDPA for batch > 1, non-contiguous slices, and -Inf score underflow")
    func batchGreaterThanOneNonContiguousAndNegInfGuard() {
        let qRaw = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 18, 4, 128], key: MLXRandom.key(7130)).asType(.float16)
        let kPreRaw = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 20, 2, 128], key: MLXRandom.key(7131)).asType(.float16)
        let vPreRaw = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 20, 2, 128], key: MLXRandom.key(7132)).asType(.float16)
        let kActRaw = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 18, 2, 128], key: MLXRandom.key(7133)).asType(.float16)
        let vActRaw = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 18, 2, 128], key: MLXRandom.key(7134)).asType(.float16)
        let kSufRaw = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 15, 2, 128], key: MLXRandom.key(7135)).asType(.float16)
        let vSufRaw = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 15, 2, 128], key: MLXRandom.key(7136)).asType(.float16)
        let q = qRaw.transposed(0, 2, 1, 3)
        let seg = DualBlockFlashAttentionKernel.Segments(
            prefixKeys: kPreRaw.transposed(0, 2, 1, 3)[0..., 0..., 2 ..< 17, 0...],
            prefixValues: vPreRaw.transposed(0, 2, 1, 3)[0..., 0..., 2 ..< 17, 0...],
            activeKeys: kActRaw.transposed(0, 2, 1, 3),
            activeValues: vActRaw.transposed(0, 2, 1, 3),
            suffixKeys: kSufRaw.transposed(0, 2, 1, 3)[0..., 0..., ..<11, 0...],
            suffixValues: vSufRaw.transposed(0, 2, 1, 3)[0..., 0..., ..<11, 0...]
        )
        let scale = 1.0 / Float(128).squareRoot()
        let fused = DualBlockFlashAttentionKernel.attention(queries: q, segments: seg, scale: scale)
        let stock = Shard1TestHelpers.stockDualBlockAttention(queries: q, segments: seg, scale: scale)
        eval(fused, stock)
        #expect(Shard1TestHelpers.maxAbsDiff(fused, stock) < 3e-3)

        let qOnes = MLXArray.ones([1, 2, 8, 64], dtype: .float32)
        let kvOnes = MLXArray.ones([1, 2, 8, 64], dtype: .float32)
        let underflowSeg = DualBlockFlashAttentionKernel.Segments(
            prefixKeys: nil, prefixValues: nil, activeKeys: kvOnes, activeValues: kvOnes, suffixKeys: nil, suffixValues: nil
        )
        let underflowOut = DualBlockFlashAttentionKernel.attention(queries: qOnes, segments: underflowSeg, scale: -1e38)
        eval(underflowOut)
        let vals = underflowOut.asArray(Float.self)
        #expect(vals.allSatisfy { $0.isFinite && $0 == 0 })
    }
}

@Suite("SageMXAttentionMetal audit regressions and parity (#228)", .serialized)
struct SageMXAuditRegressionTests {
    @Test("SageMXAttentionMetal falls back cleanly on CPU default device, unsupported dtypes, and non-32 headDim")
    func cpuDefaultDeviceAndUnsupportedFallbacks() {
        let q = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 6, 64], key: MLXRandom.key(7201))
        let k = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 10, 64], key: MLXRandom.key(7202))
        let v = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 10, 64], key: MLXRandom.key(7203))
        eval(q, k, v)
        let scale = 1.0 / Float(64).squareRoot()

        for causal in [false, true] {
            let opts = SageMXAttentionMetal.Options(scale: scale, causal: causal, useUOS: true, usePNQ: true, meanCenter: true)
            let (cpuSupported, cpuOut, cpuRef) = Device.withDefaultDevice(.cpu) {
                let supported = SageMXAttentionMetal.supportsMetalKernel(q: q, k: k, v: v)
                let out = SageMXAttentionMetal.attention(q: q, k: k, v: v, options: opts)
                let ref = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: causal ? .causal : .none
                )
                eval(out, ref)
                return (supported, out, ref)
            }
            #expect(!cpuSupported)
            #expect(cpuOut.shape == q.shape && cpuOut.dtype == .float32)
            #expect(Shard1TestHelpers.maxAbsDiff(cpuOut, cpuRef) < 1e-5)
        }

        let q16 = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 4, 16], key: MLXRandom.key(7204))
        let k16 = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 6, 16], key: MLXRandom.key(7205))
        let v16 = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 6, 16], key: MLXRandom.key(7206))
        #expect(!SageMXAttentionMetal.supportsMetalKernel(q: q16, k: k16, v: v16))
        let out16 = SageMXAttentionMetal.attention(q: q16, k: k16, v: v16, options: SageMXAttentionMetal.Options(scale: 0.25))
        let ref16 = MLXFast.scaledDotProductAttention(queries: q16, keys: k16, values: v16, scale: 0.25, mask: .none)
        eval(out16, ref16)
        #expect(Shard1TestHelpers.maxAbsDiff(out16, ref16) < 1e-5)

        let q64 = Device.withDefaultDevice(.cpu) {
            let a = q.asType(.float64)
            eval(a)
            return a
        }
        let k64 = Device.withDefaultDevice(.cpu) {
            let a = k.asType(.float64)
            eval(a)
            return a
        }
        let v64 = Device.withDefaultDevice(.cpu) {
            let a = v.asType(.float64)
            eval(a)
            return a
        }
        #expect(!SageMXAttentionMetal.supportsMetalKernel(q: q64, k: k64, v: v64))
        let out64 = SageMXAttentionMetal.attention(q: q64, k: k64, v: v64, options: SageMXAttentionMetal.Options(scale: scale))
        let out64Vals = Device.withDefaultDevice(.cpu) {
            eval(out64)
            return out64.asArray(Double.self)
        }
        #expect(out64.shape == q64.shape && out64.dtype == .float64)
        #expect(out64Vals.allSatisfy { $0.isFinite })
    }

    @Test("SageMXAttentionMetal handles zero batch, heads, lQ, lK, and d dimensions without crashing")
    func emptyDimensionsReturnZeros() {
        let opts = SageMXAttentionMetal.Options(scale: 0.125, causal: false)
        let cases: [([Int], [Int], [Int])] = [
            ([0, 2, 4, 64], [0, 2, 8, 64], [0, 2, 8, 64]),
            ([1, 0, 4, 64], [1, 0, 8, 64], [1, 0, 8, 64]),
            ([1, 2, 0, 64], [1, 2, 8, 64], [1, 2, 8, 64]),
            ([1, 2, 4, 64], [1, 2, 0, 64], [1, 2, 0, 64]),
            ([1, 2, 4, 0], [1, 2, 8, 0], [1, 2, 8, 0]),
        ]
        for (qShape, kShape, vShape) in cases {
            let q = MLXArray.zeros(qShape, dtype: .float32)
            let k = MLXArray.zeros(kShape, dtype: .float32)
            let v = MLXArray.zeros(vShape, dtype: .float32)
            #expect(!SageMXAttentionMetal.supportsMetalKernel(q: q, k: k, v: v))
            let out = SageMXAttentionMetal.attention(q: q, k: k, v: v, options: opts)
            eval(out)
            #expect(out.shape == qShape && out.dtype == .float32)
            #expect(out.asArray(Float.self).allSatisfy { $0 == 0 })
        }
    }

    @Test("SageMXAttentionMetal compiles with < 8-element deltaS and matches cpuAttention for lQ == 1 meanCenter and meanCenter == false")
    func smallBuffersAndSingleQueryMeanCenterParity() {
        let scale = 1.0 / Float(64).squareRoot()
        let configs: [(Int, Int, Int, Bool, Bool)] = [
            (1, 2, 4, false, true),
            (1, 1, 4, false, true),
            (2, 1, 16, false, true),
            (2, 1, 16, true, true),
            (2, 1, 1, false, true),
            (2, 4, 8, false, false),
        ]
        for (idx, cfg) in configs.enumerated() {
            let (heads, lQ, lK, causal, meanCenter) = cfg
            let base = UInt64(7220 + idx * 10)
            let q = MLXRandom.normal([1, heads, lQ, 64], key: MLXRandom.key(base)) + Float(2.5)
            let k = MLXRandom.normal([1, heads, lK, 64], key: MLXRandom.key(base + 1)) - Float(3.0)
            let v = MLXRandom.normal([1, heads, lK, 64], key: MLXRandom.key(base + 2))
            eval(q, k, v)
            let opts = SageMXAttentionMetal.Options(scale: scale, causal: causal, useUOS: true, usePNQ: true, meanCenter: meanCenter)
            let gpu = SageMXAttentionMetal.attention(q: q, k: k, v: v, options: opts)
            eval(gpu)
            let cpu = SageMXAttentionMetal.cpuAttention(
                q: q.asArray(Float.self), k: k.asArray(Float.self), v: v.asArray(Float.self),
                shape: SageMXAttentionMetal.Shape(heads: heads, lQ: lQ, lK: lK, d: 64), options: opts
            )
            #expect(Shard1TestHelpers.maxAbsDiff(gpu.asArray(Float.self), cpu) <= 5e-4)
        }
    }

    @Test("SageMXAttentionMetal keeps all-masked causal rows and -Inf underflow rows finite without NaN poisoning")
    func allMaskedAndNegativeInfinityUnderflowStayFinite() {
        let qCausal = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 5, 32], key: MLXRandom.key(7240))
        let kCausal = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 2, 32], key: MLXRandom.key(7241))
        let vCausal = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 2, 32], key: MLXRandom.key(7242))
        eval(qCausal, kCausal, vCausal)
        let causalOpts = SageMXAttentionMetal.Options(scale: 1.0 / Float(32).squareRoot(), causal: true, useUOS: true, usePNQ: true, meanCenter: true)
        let gpuCausal = SageMXAttentionMetal.attention(q: qCausal, k: kCausal, v: vCausal, options: causalOpts)
        eval(gpuCausal)
        let cpuCausal = SageMXAttentionMetal.cpuAttention(
            q: qCausal.asArray(Float.self), k: kCausal.asArray(Float.self), v: vCausal.asArray(Float.self),
            shape: SageMXAttentionMetal.Shape(heads: 2, lQ: 5, lK: 2, d: 32), options: causalOpts
        )
        let cpuFallbackCausal = Device.withDefaultDevice(.cpu) {
            let out = SageMXAttentionMetal.attention(q: qCausal, k: kCausal, v: vCausal, options: causalOpts)
            eval(out)
            return out
        }
        #expect(Shard1TestHelpers.maxAbsDiff(gpuCausal.asArray(Float.self), cpuCausal) <= 5e-4)
        let maskedRowsGpu = gpuCausal[0, 0..., ..<3, 0...].asArray(Float.self)
        let maskedRowsCpu = cpuFallbackCausal[0, 0..., ..<3, 0...].asArray(Float.self)
        #expect(maskedRowsGpu.allSatisfy { $0 == 0 } && maskedRowsCpu.allSatisfy { $0 == 0 })

        let qOne = MLXArray.ones([1, 1, 2, 32], dtype: .float32)
        let kAllInf = MLXArray.ones([1, 1, 4, 32], dtype: .float32)
        let vAllInf = MLXArray.ones([1, 1, 4, 32], dtype: .float32) * Float(3.0)
        let underflowOpts = SageMXAttentionMetal.Options(scale: -1e38, causal: false, useUOS: true, usePNQ: false, meanCenter: false)
        let outAllInf = SageMXAttentionMetal.attention(q: qOne, k: kAllInf, v: vAllInf, options: underflowOpts)
        eval(outAllInf)
        #expect(outAllInf.asArray(Float.self).allSatisfy { $0.isFinite && $0 == 0 })

        let kFirstInf = concatenated([MLXArray.ones([1, 1, 1, 32], dtype: .float32), MLXArray.zeros([1, 1, 1, 32], dtype: .float32)], axis: 2)
        let vFirstInf = concatenated([MLXArray.ones([1, 1, 1, 32], dtype: .float32) * Float(99.0), MLXArray.ones([1, 1, 1, 32], dtype: .float32) * Float(7.0)], axis: 2)
        eval(qOne, kFirstInf, vFirstInf)
        let outRecover = SageMXAttentionMetal.attention(q: qOne, k: kFirstInf, v: vFirstInf, options: underflowOpts)
        eval(outRecover)
        let cpuRecover = SageMXAttentionMetal.cpuAttention(
            q: qOne.asArray(Float.self), k: kFirstInf.asArray(Float.self), v: vFirstInf.asArray(Float.self),
            shape: SageMXAttentionMetal.Shape(heads: 1, lQ: 2, lK: 2, d: 32), options: underflowOpts
        )
        #expect(outRecover.asArray(Float.self).allSatisfy { abs($0 - 7.0) < 1e-4 })
        #expect(cpuRecover.allSatisfy { abs($0 - 7.0) < 1e-4 })
    }

    @Test(
        "SageMXAttentionMetal matches cpuAttention across headDim 32, 128, 256, batch > 1, and floating dtypes (#228)",
        arguments: [32, 128, 256], [DType.float32, .float16, .bfloat16]
    )
    func multiBatchAndNon64HeadDimsParity(headDim: Int, dtype: DType) {
        let (batch, headsPerBatch, lQ, lK) = (2, 3, 6, 12)
        let seed = UInt64(7260 + headDim + (dtype == .float32 ? 1 : dtype == .float16 ? 2 : 3))
        let q = MLXRandom.normal([batch, headsPerBatch, lQ, headDim], key: MLXRandom.key(seed)).asType(dtype)
        let k = MLXRandom.normal([batch, headsPerBatch, lK, headDim], key: MLXRandom.key(seed + 1)).asType(dtype)
        let v = MLXRandom.normal([batch, headsPerBatch, lK, headDim], key: MLXRandom.key(seed + 2)).asType(dtype)
        eval(q, k, v)
        let opts = SageMXAttentionMetal.Options(
            scale: 1.0 / Float(headDim).squareRoot(), causal: true, useUOS: true, usePNQ: true, meanCenter: true
        )
        let gpu = SageMXAttentionMetal.attention(q: q, k: k, v: v, options: opts).asType(.float32)
        eval(gpu)
        let cpu = SageMXAttentionMetal.cpuAttention(
            q: q.asType(.float32).asArray(Float.self),
            k: k.asType(.float32).asArray(Float.self),
            v: v.asType(.float32).asArray(Float.self),
            shape: SageMXAttentionMetal.Shape(heads: batch * headsPerBatch, lQ: lQ, lK: lK, d: headDim),
            options: opts
        )
        #expect(gpu.shape == [batch, headsPerBatch, lQ, headDim])
        #expect(Shard1TestHelpers.maxAbsDiff(gpu.asArray(Float.self), cpu) <= Shard1TestHelpers.sageTolerance(dtype))
    }
}

@Suite("SlidingTileAttention3D audit regressions and parity (#228)", .serialized)
struct SlidingTileAuditRegressionTests {
    @Test("SlidingTileAttention3D supportsMetalKernel rejects CPU device, zero batch, and all mismatched Q/K/V/globalKV shapes")
    func supportsMetalKernelRejectsInvalidShapes() {
        let q = MLXArray.zeros([2, 4, 8, 16], dtype: .float32)
        let k = MLXArray.zeros([2, 2, 8, 16], dtype: .float32)
        let v = MLXArray.zeros([2, 2, 8, 16], dtype: .float32)
        let gOk = GlobalConditioningKV(keys: MLXArray.zeros([2, 2, 4, 16], dtype: .float32), values: MLXArray.zeros([2, 2, 4, 16], dtype: .float32))
        #expect(SlidingTileAttention3D.supportsMetalKernel(queries: q, keys: k, values: v, globalKV: gOk))

        let cpuSupported = Device.withDefaultDevice(.cpu) {
            SlidingTileAttention3D.supportsMetalKernel(queries: q, keys: k, values: v, globalKV: gOk)
        }
        #expect(!cpuSupported)
        #expect(!SlidingTileAttention3D.supportsMetalKernel(
            queries: MLXArray.zeros([0, 4, 8, 16], dtype: .float32),
            keys: MLXArray.zeros([0, 2, 8, 16], dtype: .float32),
            values: MLXArray.zeros([0, 2, 8, 16], dtype: .float32),
            globalKV: nil
        ))
        #expect(!SlidingTileAttention3D.supportsMetalKernel(queries: q, keys: MLXArray.zeros([2, 2, 8, 32], dtype: .float32), values: v, globalKV: nil))
        #expect(!SlidingTileAttention3D.supportsMetalKernel(queries: q, keys: k, values: MLXArray.zeros([2, 2, 8, 32], dtype: .float32), globalKV: nil))
        #expect(!SlidingTileAttention3D.supportsMetalKernel(queries: q, keys: MLXArray.zeros([1, 2, 8, 16], dtype: .float32), values: v, globalKV: nil))
        #expect(!SlidingTileAttention3D.supportsMetalKernel(queries: q, keys: k, values: MLXArray.zeros([1, 2, 8, 16], dtype: .float32), globalKV: nil))
        #expect(!SlidingTileAttention3D.supportsMetalKernel(queries: q, keys: k, values: MLXArray.zeros([2, 4, 8, 16], dtype: .float32), globalKV: nil))
        #expect(!SlidingTileAttention3D.supportsMetalKernel(
            queries: q, keys: k, values: v,
            globalKV: GlobalConditioningKV(keys: MLXArray.zeros([1, 2, 4, 16], dtype: .float32), values: MLXArray.zeros([1, 2, 4, 16], dtype: .float32))
        ))
        #expect(!SlidingTileAttention3D.supportsMetalKernel(
            queries: q, keys: k, values: v,
            globalKV: GlobalConditioningKV(keys: MLXArray.zeros([2, 2, 4, 16], dtype: .float32), values: MLXArray.zeros([2, 2, 3, 16], dtype: .float32))
        ))
    }

    @Test("SlidingTileAttention3D handles CFG batch == 2 with both per-branch and batch-1 shared global sinks safely")
    func cfgBatchTwoWithSharedAndPerBranchSinks() throws {
        let grid = try SpatioTemporalGrid(frames: 4, height: 2, width: 2)
        let sparsePlan = SlidingTileAttention3D.buildPlan(
            grid: grid, brick: try BrickShape(t: 2, h: 2, w: 2), window: WindowRadiusBricks(t: 0, h: 0, w: 0)
        )
        let fullPlan = SlidingTileAttention3D.buildPlan(
            grid: grid, brick: try BrickShape(t: 2, h: 2, w: 2), window: WindowRadiusBricks(t: 2, h: 2, w: 2)
        )
        let n = grid.tokenCount
        let q = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 4, n, 16], key: MLXRandom.key(7301))
        let k = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 2, n, 16], key: MLXRandom.key(7302))
        let v = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 2, n, 16], key: MLXRandom.key(7303))
        let sk1 = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 5, 16], key: MLXRandom.key(7304))
        let sv1 = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 5, 16], key: MLXRandom.key(7305))
        let sharedSink = GlobalConditioningKV(keys: sk1, values: sv1)
        let replicatedSink = GlobalConditioningKV(keys: concatenated([sk1, sk1], axis: 0), values: concatenated([sv1, sv1], axis: 0))
        eval(q, k, v, sk1, sv1)

        #expect(!SlidingTileAttention3D.supportsMetalKernel(queries: q, keys: k, values: v, globalKV: sharedSink))
        #expect(SlidingTileAttention3D.supportsMetalKernel(queries: q, keys: k, values: v, globalKV: replicatedSink))

        let cfgFallback = SlidingTileAttention3D.attention(queries: q, keys: k, values: v, plan: sparsePlan, globalKV: sharedSink)
        let cfgMetal = SlidingTileAttention3D.attention(queries: q, keys: k, values: v, plan: sparsePlan, globalKV: replicatedSink)
        let cfgDense = SlidingTileAttention3D.attention(queries: q, keys: k, values: v, plan: fullPlan, globalKV: sharedSink)
        let cfgDenseRep = SlidingTileAttention3D.attention(queries: q, keys: k, values: v, plan: fullPlan, globalKV: replicatedSink)
        eval(cfgFallback, cfgMetal, cfgDense, cfgDenseRep)

        #expect(Shard1TestHelpers.maxAbsDiff(cfgFallback, cfgMetal) < 1e-5)
        #expect(Shard1TestHelpers.maxAbsDiff(cfgDense, cfgDenseRep) < 1e-5)
    }

    @Test("SlidingTileAttention3D falls back cleanly on CPU default device, empty batch, and headDim > 256")
    func cpuDefaultDeviceEmptyBatchAndSparseFallbackParity() throws {
        let grid = try SpatioTemporalGrid(frames: 4, height: 2, width: 2)
        let plan = SlidingTileAttention3D.buildPlan(
            grid: grid, brick: try BrickShape(t: 2, h: 2, w: 2), window: WindowRadiusBricks(t: 0, h: 0, w: 0)
        )
        let n = grid.tokenCount
        let q = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 4, n, 16], key: MLXRandom.key(7320))
        let k = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, n, 16], key: MLXRandom.key(7321))
        let v = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, n, 16], key: MLXRandom.key(7322))
        eval(q, k, v)
        let gpuOut = SlidingTileAttention3D.attention(queries: q, keys: k, values: v, plan: plan)
        eval(gpuOut)
        let cpuOut = Device.withDefaultDevice(.cpu) {
            let out = SlidingTileAttention3D.attention(queries: q, keys: k, values: v, plan: plan)
            eval(out)
            return out
        }
        #expect(Shard1TestHelpers.maxAbsDiff(gpuOut, cpuOut) < 1e-5)

        let qEmpty = MLXArray.zeros([0, 4, n, 16], dtype: .float32)
        let kEmpty = MLXArray.zeros([0, 2, n, 16], dtype: .float32)
        let outEmpty = SlidingTileAttention3D.attention(queries: qEmpty, keys: kEmpty, values: kEmpty, plan: plan)
        eval(outEmpty)
        #expect(outEmpty.shape == [0, 4, n, 16])

        let qWide = MLXRandom.uniform(low: -0.3, high: 0.3, [1, 2, n, 288], key: MLXRandom.key(7325))
        let kWide = MLXRandom.uniform(low: -0.3, high: 0.3, [1, 2, n, 288], key: MLXRandom.key(7326))
        let vWide = MLXRandom.uniform(low: -0.3, high: 0.3, [1, 2, n, 288], key: MLXRandom.key(7327))
        eval(qWide, kWide, vWide)
        #expect(!SlidingTileAttention3D.supportsMetalKernel(queries: qWide, keys: kWide, values: vWide, globalKV: nil))
        let outWide = SlidingTileAttention3D.attention(queries: qWide, keys: kWide, values: vWide, plan: plan)
        eval(outWide)
        let refWide = SlidingTileAttention3D.cpuReferenceBrickAttention(
            queries: qWide.asArray(Float.self), keys: kWide.asArray(Float.self), values: vWide.asArray(Float.self),
            shape: CPUAttentionShape(batch: 1, qHeads: 2, kvHeads: 2, headDim: 288), plan: plan
        )
        #expect(Shard1TestHelpers.maxAbsDiff(outWide.asArray(Float.self), refWide) < 1e-5)
    }

    @Test("SlidingTileAttention3D matches cpuReferenceBrickAttention on < 8-element buffers, headDim > 128, brickVolume % 4 != 0, and sinkCount > TILE_KV")
    func smallConstantSpaceBuffersAndL2EdgeShapesParity() throws {
        let tinyGrid = try SpatioTemporalGrid(frames: 1, height: 1, width: 2)
        let tinyPlan = SlidingTileAttention3D.buildPlan(
            grid: tinyGrid, brick: try BrickShape(t: 1, h: 1, w: 1), window: WindowRadiusBricks(t: 0, h: 0, w: 0)
        )
        let qTiny = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 1, 2, 2], key: MLXRandom.key(7340))
        let kTiny = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 1, 2, 2], key: MLXRandom.key(7341))
        let vTiny = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 1, 2, 2], key: MLXRandom.key(7342))
        let sTinyK = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 1, 1, 2], key: MLXRandom.key(7343))
        let sTinyV = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 1, 1, 2], key: MLXRandom.key(7344))
        eval(qTiny, kTiny, vTiny, sTinyK, sTinyV)
        let tinyOut = SlidingTileAttention3D.attention(
            queries: qTiny, keys: kTiny, values: vTiny, plan: tinyPlan,
            globalKV: GlobalConditioningKV(keys: sTinyK, values: sTinyV)
        )
        eval(tinyOut)
        let tinyRef = SlidingTileAttention3D.cpuReferenceBrickAttention(
            queries: qTiny.asArray(Float.self), keys: kTiny.asArray(Float.self), values: vTiny.asArray(Float.self),
            shape: CPUAttentionShape(batch: 1, qHeads: 1, kvHeads: 1, headDim: 2, sinkTokens: 1),
            plan: tinyPlan, globalKV: CPUGlobalKV(keys: sTinyK.asArray(Float.self), values: sTinyV.asArray(Float.self))
        )
        #expect(Shard1TestHelpers.maxAbsDiff(tinyOut.asArray(Float.self), tinyRef) < 1e-5)

        let oddGrid = try SpatioTemporalGrid(frames: 2, height: 3, width: 2)
        let oddPlan = SlidingTileAttention3D.buildPlan(
            grid: oddGrid, brick: try BrickShape(t: 1, h: 3, w: 1), window: WindowRadiusBricks(t: 0, h: 0, w: 0)
        )
        #expect(oddPlan.brickVolume == 3)
        for (headDim, sinkLen) in [(160, 12), (64, 20)] {
            let base = UInt64(7350 + headDim)
            let q = MLXRandom.uniform(low: -0.4, high: 0.4, [1, 4, oddGrid.tokenCount, headDim], key: MLXRandom.key(base))
            let k = MLXRandom.uniform(low: -0.4, high: 0.4, [1, 2, oddGrid.tokenCount, headDim], key: MLXRandom.key(base + 1))
            let v = MLXRandom.uniform(low: -0.4, high: 0.4, [1, 2, oddGrid.tokenCount, headDim], key: MLXRandom.key(base + 2))
            let sk = MLXRandom.uniform(low: -0.4, high: 0.4, [1, 2, sinkLen, headDim], key: MLXRandom.key(base + 3))
            let sv = MLXRandom.uniform(low: -0.4, high: 0.4, [1, 2, sinkLen, headDim], key: MLXRandom.key(base + 4))
            eval(q, k, v, sk, sv)
            let out = SlidingTileAttention3D.attention(
                queries: q, keys: k, values: v, plan: oddPlan, globalKV: GlobalConditioningKV(keys: sk, values: sv)
            )
            eval(out)
            let ref = SlidingTileAttention3D.cpuReferenceBrickAttention(
                queries: q.asArray(Float.self), keys: k.asArray(Float.self), values: v.asArray(Float.self),
                shape: CPUAttentionShape(batch: 1, qHeads: 4, kvHeads: 2, headDim: headDim, sinkTokens: sinkLen),
                plan: oddPlan, globalKV: CPUGlobalKV(keys: sk.asArray(Float.self), values: sv.asArray(Float.self))
            )
            #expect(Shard1TestHelpers.maxAbsDiff(out.asArray(Float.self), ref) < 1e-5)
        }
    }
}
