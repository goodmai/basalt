import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Fast-weight schedule core, byte footprint, cache isolation, and closed-form math (#848)", .serialized)
struct FastWeightScheduleTests {
    @Test("FastWeightSchedule produces identical chunk boundary flush points across single-pass, multi-chunk, and token-by-token splits")
    func scheduleSplitsMatchAcrossChunkings() {
        var single = FastWeightSchedule(chunkSize: 4)
        let singleSlices = single.planSlices(for: 10)
        #expect(singleSlices.count == 3)
        #expect(singleSlices[0] == FastWeightChunkSlice(range: 0 ..< 4, stepStart: 0, stepEnd: 4, completesChunk: true, chunkSize: 4))
        #expect(singleSlices[1] == FastWeightChunkSlice(range: 4 ..< 8, stepStart: 0, stepEnd: 4, completesChunk: true, chunkSize: 4))
        #expect(singleSlices[2] == FastWeightChunkSlice(range: 8 ..< 10, stepStart: 0, stepEnd: 2, completesChunk: false, chunkSize: 4))
        #expect(single.offset == 10 && single.stepInChunk == 2)

        var chunked = FastWeightSchedule(chunkSize: 4)
        var flushPositions: [Int] = []
        for size in [5, 3, 2] {
            let base = chunked.offset
            for slice in chunked.planSlices(for: size) where slice.completesChunk {
                flushPositions.append(base + slice.range.upperBound)
            }
        }
        #expect(flushPositions == [4, 8])
        #expect(chunked.offset == 10 && chunked.stepInChunk == 2)

        var streaming = FastWeightSchedule(chunkSize: 4)
        var streamFlushes: [Int] = []
        for step in 0 ..< 10 {
            let planned = streaming.planSlices(for: 1)
            #expect(planned.count == 1)
            if planned[0].completesChunk { streamFlushes.append(step + 1) }
        }
        #expect(streamFlushes == [4, 8])
        #expect(streaming.offset == 10 && streaming.stepInChunk == 2)
        #expect(streaming.planSlices(for: 0).isEmpty)

        let huge = FastWeightSchedule.slices(chunkSize: 1_000_000, stepInChunk: 512, tokenCount: 512)
        #expect(huge.slices.count == 1 && !huge.slices[0].completesChunk && huge.nextStepInChunk == 1024)
    }

    @Test("FastWeightFootprint and KVCacheGeometry compute exact fixed state bytes for TTT, Titans, and LaCT")
    func footprintAndKVCacheGeometryMatchConfigurations() throws {
        let tttCfg = try JSONDecoder().decode(TTTConfiguration.self, from: Data(TinyModel.Family.tttLinear.config.utf8))
        let tttMlpCfg = try JSONDecoder().decode(TTTConfiguration.self, from: Data(TinyModel.Family.tttMlp.config.utf8))
        let titansCfg = try JSONDecoder().decode(TitansConfiguration.self, from: Data(TinyModel.Family.titans.config.utf8))
        let lactCfg = try JSONDecoder().decode(LaCTConfiguration.self, from: Data(TinyModel.Family.lact.config.utf8))

        let expectedTTT: Int64 = 2 * (2 * 4 * (16 * 16 + 16) * 4 + 3 * 2 * 64 * 2)
        #expect(FastWeightFootprint.tttStateBytes(config: tttCfg) == expectedTTT)
        #expect(FastWeightFootprint.tttStateBytes(config: tttMlpCfg) > expectedTTT)
        #expect(FastWeightFootprint.titansStateBytes(config: titansCfg) > 0)
        #expect(FastWeightFootprint.lactStateBytes(config: lactCfg) > 0)

        for (family, expectedBytes, expectedWindow) in [
            (TinyModel.Family.tttLinear, expectedTTT, 0),
            (TinyModel.Family.titans, FastWeightFootprint.titansStateBytes(config: titansCfg), TinyModel.slidingWindow),
            (TinyModel.Family.lact, FastWeightFootprint.lactStateBytes(config: lactCfg), TinyModel.slidingWindow),
        ] {
            let json = try #require(try JSONSerialization.jsonObject(with: Data(family.config.utf8)) as? [String: Any])
            let geom = try #require(KVCacheGeometry(config: json))
            #expect(geom.elementsPerToken == 0)
            #expect(geom.fixedStateBytes == expectedBytes)
            #expect(geom.window == expectedWindow)
        }

        let mamba2Config: [String: Any] = [
            "model_type": "mamba2", "num_hidden_layers": 64, "num_heads": 128,
            "head_dim": 64, "hidden_size": 4096,
        ]
        let rwkv7Config: [String: Any] = [
            "model_type": "rwkv7", "num_hidden_layers": 24, "num_heads": 32,
            "head_dim": 64, "hidden_size": 2048,
        ]
        #expect(KVCacheGeometry(config: mamba2Config) == nil)
        #expect(KVCacheGeometry(config: rwkv7Config) == nil)

        let lactAliasConfig: [String: Any] = [
            "model_type": "lact_swiglu", "vocab_size": 256, "num_hidden_layers": 2, "num_attn_heads": 4,
            "hidden_size": 64, "window_size": 16,
        ]
        let lactAliasGeom = try #require(KVCacheGeometry(config: lactAliasConfig))
        #expect(lactAliasGeom.elementsPerToken == 0 && lactAliasGeom.window == 16 && lactAliasGeom.fixedStateBytes > 0)
    }

    @Test("FastWeightLayerCache deep-copies without aliasing, round-trips sparse slots through state and metaState, and safely declines disk persistence")
    func cacheDeepCopyRoundTripAndDiskGuard() throws {
        let cache = FastWeightLayerCache(chunkSize: 8, slotCount: 6)
        #expect(!cache.isTrimmable && !cache.isTrimmable(after: 1) && cache.trim(3) == 0 && cache.maxSize == nil)
        let isNoneMask: Bool = if case .none = cache.makeMask(n: 4, windowSize: nil, returnArray: false) { true } else { false }
        #expect(isNoneMask)
        cache.commit(0, MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2))
        cache.commit(3, MLXArray([Float(9), 10]).reshaped(1, 2))
        cache.advance(tokens: 11, nextStepInChunk: 3)
        #expect(cache.offset == 11 && cache.stepInChunk == 3 && cache.innerState().count == 2)

        let cloned = try #require(cache.copy() as? FastWeightLayerCache)
        cache.commit(0, MLXArray([Float(99), 99, 99, 99]).reshaped(2, 2))
        let clonedSlot0 = try #require(cloned.slot(0))
        #expect(MLX.abs(clonedSlot0 - MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2)).max().item(Float.self) == 0)
        #expect(cloned.slot(1) == nil && cloned.slot(2) == nil && cloned.slot(3) != nil)
        #expect(cloned.offset == 11 && cloned.stepInChunk == 3)

        let restoredMetaFirst = FastWeightLayerCache(chunkSize: 8, slotCount: 6)
        restoredMetaFirst.metaState = cloned.metaState
        restoredMetaFirst.state = cloned.state
        #expect(restoredMetaFirst.offset == 11 && restoredMetaFirst.stepInChunk == 3)
        #expect(restoredMetaFirst.slot(1) == nil && restoredMetaFirst.slot(3) != nil)
        let restoredSlot3 = try #require(restoredMetaFirst.slot(3))
        let clonedSlot3 = try #require(cloned.slot(3))
        #expect(MLX.abs(restoredSlot3 - clonedSlot3).max().item(Float.self) == 0)

        let restoredStateFirst = FastWeightLayerCache(chunkSize: 8, slotCount: 6)
        restoredStateFirst.state = cloned.state
        restoredStateFirst.metaState = cloned.metaState
        #expect(restoredStateFirst.slot(1) == nil && restoredStateFirst.slot(3) != nil)

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("fw-disk-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        var store = PromptCacheDiskStore(directory: dir, byteQuota: 1 << 24, retentionDays: 7)
        let stored = store.store(fingerprint: "fp", model: "fw", namespace: "ns", tokens: [1, 2, 3], cache: PromptCache([cloned]))
        #expect(stored == nil)

        cloned.reset()
        #expect(cloned.offset == 0 && cloned.stepInChunk == 0 && cloned.innerState().isEmpty)
    }
}

extension FastWeightScheduleTests {
    @Test("Closed-form lnFusedL2Backward, geluTanhBackward, and siluBackward match central finite differences within 1e-4")
    func closedFormDerivativesMatchFiniteDifferences() {
        let eps: Float = 1e-3
        let (x, target, weight, bias) = withRandomState(MLXRandom.RandomState(seed: 4801)) {
            (
                MLXRandom.normal([1, 2, 3, 6]),
                MLXRandom.normal([1, 2, 3, 6]),
                MLXRandom.normal([1, 2, 1, 6]) * Float(0.5) + Float(1.0),
                MLXRandom.normal([1, 2, 1, 6]) * Float(0.2)
            )
        }
        let analytical = FastWeightMath.lnFusedL2Backward(x: x, target: target, weight: weight, bias: bias, eps: 1e-5)
        let xVals = x.asType(.float32).asArray(Float.self)
        let tVals = target.asType(.float32).asArray(Float.self)
        let wVals = weight.asType(.float32).asArray(Float.self)
        let bVals = bias.asType(.float32).asArray(Float.self)
        let anaVals = analytical.asType(.float32).asArray(Float.self)
        let hDouble = 1e-6
        var maxLnDiff: Float = 0
        for row in 0 ..< 6 {
            let head = row / 3
            let base = row * 6
            let wBase = head * 6
            for col in 0 ..< 6 {
                let idx = base + col
                var rowPlus = (0 ..< 6).map { Double(xVals[base + $0]) }
                var rowMinus = rowPlus
                rowPlus[col] += hDouble
                rowMinus[col] -= hDouble
                let lPlus = Self.rowLayerNormL2(rowPlus, target: tVals, weight: wVals, bias: bVals, base: base, wBase: wBase)
                let lMinus = Self.rowLayerNormL2(rowMinus, target: tVals, weight: wVals, bias: bVals, base: base, wBase: wBase)
                let num = Float((lPlus - lMinus) / (2.0 * hDouble))
                maxLnDiff = Swift.max(maxLnDiff, abs(num - anaVals[idx]))
            }
        }
        #expect(maxLnDiff < 1e-4)

        let z = MLXArray([Float(-2.1), -0.7, 0.0, 0.6, 1.8])
        let ones = MLXArray.ones(like: z)
        let gAna = FastWeightMath.geluTanhBackward(z: z, gradOut: ones).asArray(Float.self)
        let sAna = FastWeightMath.siluBackward(z: z, gradOut: ones).asArray(Float.self)
        let gNum = ((FastWeightMath.geluTanh(z + eps) - FastWeightMath.geluTanh(z - eps)) / (2 * eps)).asArray(Float.self)
        let sNum = ((silu(z + eps) - silu(z - eps)) / (2 * eps)).asArray(Float.self)
        for idx in 0 ..< gAna.count {
            #expect(abs(gAna[idx] - gNum[idx]) < 1e-3)
            #expect(abs(sAna[idx] - sNum[idx]) < 1e-3)
        }
    }

    private static func rowLayerNormL2(
        _ row: [Double],
        target: [Float],
        weight: [Float],
        bias: [Float],
        base: Int,
        wBase: Int
    ) -> Double {
        let meanVal = row.reduce(0.0, +) / Double(row.count)
        let variance = row.reduce(0.0) { $0 + ($1 - meanVal) * ($1 - meanVal) } / Double(row.count)
        let invStd = 1.0 / sqrt(variance + 1e-5)
        var loss = 0.0
        for col in 0 ..< row.count {
            let y = (row[col] - meanVal) * invStd * Double(weight[wBase + col]) + Double(bias[wBase + col])
            let d = y - Double(target[base + col])
            loss += 0.5 * d * d
        }
        return loss
    }

    @Test("Titans multiplication-only transition is exact and NaN-free at exact alpha = 1 (beta = 0) and eta = 0")
    func titansTransitionIsNaNFreeAtZeroBoundaries() {
        let betaVals: [Float] = [0.8, 0.0, 0.7, 0.0, 0.9]
        let etaVals: [Float] = [0.6, 0.5, 0.0, 0.8, 0.0]
        let zVals: [Float] = [0.3, -0.4, 0.7, -0.2, 0.5]
        let mInit: Float = 1.25
        let sInit: Float = -0.65

        var mRef = mInit
        var sRef = sInit
        for i in 0 ..< betaVals.count {
            sRef = etaVals[i] * sRef + zVals[i]
            mRef = betaVals[i] * mRef + sRef
        }

        let beta = MLXArray(betaVals).reshaped(1, 1, betaVals.count)
        let eta = MLXArray(etaVals).reshaped(1, 1, etaVals.count)
        let trans = FastWeightMath.titansTransition(beta: beta, eta: eta)
        let z = MLXArray(zVals).reshaped(1, 1, zVals.count, 1)
        let sCalc = (trans.pEta * sInit + (trans.eLast * z).sum(axis: 2, keepDims: true)).item(Float.self)
        let mCalc = (trans.pBeta * mInit + trans.cS * sInit + (trans.uLast * z).sum(axis: 2, keepDims: true)).item(Float.self)
        #expect(sCalc.isFinite && mCalc.isFinite)
        #expect(abs(sCalc - sRef) < 1e-5)
        #expect(abs(mCalc - mRef) < 1e-5)

        let mat = FastWeightCPUMath.transitionMatrix(factors: beta)
        #expect(!isNaN(mat).any().item(Bool.self))
        #expect(abs(mat[0, 0, 2, 1].item(Float.self) - 0.7) < 1e-6)
        #expect(abs(mat[0, 0, 2, 0].item(Float.self) - 0.0) < 1e-6)
    }

    @Test("Batched FP32 quintic Newton-Schulz-5 orthogonalizes wide and tall matrices, handles zero cleanly, and preserveRowNorm locks row norms")
    func newtonSchulz5AndRowNormBehavior() {
        let (wide, tall) = withRandomState(MLXRandom.RandomState(seed: 4802)) {
            (MLXRandom.normal([1, 2, 4, 8]), MLXRandom.normal([1, 2, 8, 4]))
        }
        let orthoWide = FastWeightMath.newtonSchulz5(wide, steps: 5)
        let gramWide = matmul(orthoWide, orthoWide.swappedAxes(-1, -2))
        let eye4 = MLXArray.eye(4, dtype: .float32).reshaped(1, 1, 4, 4)
        #expect(!isNaN(orthoWide).any().item(Bool.self))
        #expect(MLX.abs(gramWide - eye4).max().item(Float.self) < 0.45)

        let orthoTall = FastWeightMath.newtonSchulz5(tall, steps: 5)
        let gramTall = matmul(orthoTall.swappedAxes(-1, -2), orthoTall)
        #expect(!isNaN(orthoTall).any().item(Bool.self))
        #expect(MLX.abs(gramTall - eye4).max().item(Float.self) < 0.45)

        let eyeBatch = MLXArray.eye(4, dtype: .float32).reshaped(1, 1, 4, 4)
        let orthoEye = FastWeightMath.newtonSchulz5(eyeBatch, steps: 5)
        let gramEye = matmul(orthoEye, orthoEye.swappedAxes(-1, -2))
        let offDiagMask = MLXArray(Float(1.0)) - eye4
        #expect(MLX.abs(gramEye * offDiagMask).max().item(Float.self) < 1e-5)
        let diagEye = (gramEye * eye4).sum(axis: -1).asArray(Float.self)
        #expect(diagEye.allSatisfy { $0 >= 0.45 && $0 <= 1.35 })

        let zeroOut = FastWeightMath.newtonSchulz5(MLXArray.zeros([1, 2, 4, 4], dtype: .float32))
        #expect(!isNaN(zeroOut).any().item(Bool.self) && MLX.abs(zeroOut).max().item(Float.self) == 0)

        let target = FastWeightMath.rowL2Norm(wide)
        let scaled = wide * Float(3.7)
        let normed = FastWeightMath.preserveRowNorm(scaled, targetNorm: target)
        #expect(MLX.abs(FastWeightMath.rowL2Norm(normed) - target).max().item(Float.self) < 1e-4)
    }
}
