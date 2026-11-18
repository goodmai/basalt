import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

private enum VideoAttnTestHelpers {
    static func maxAbsDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        var peak: Float = 0.0
        for (a, b) in zip(lhs, rhs) {
            peak = max(peak, abs(a - b))
        }
        return peak
    }

    static func staTolerance(_ dtype: DType) -> Float {
        switch dtype {
        case .float32: 1e-5
        case .float16: 2e-3
        case .bfloat16: 5e-3
        default: 1e-4
        }
    }

    static func ropeTolerance(_ dtype: DType) -> Float {
        switch dtype {
        case .float32: 1e-5
        case .float16, .bfloat16: 2e-3
        default: 1e-4
        }
    }
}

@Suite("Fractional physical-second 3D Audio-Visual Multimodal RoPE (#582)", .serialized)
struct Fractional3DMultimodalRoPETests {
    @Test("Video frames at 24 fps (s_t = 4) and audio frames at 16 kHz / 44.1 kHz lock phase sub-millisecond at t = 1.0 s and 2.5 s")
    func subMillisecondAudioVideoPhaseAlignment() throws {
        let axes = try Fractional3DRoPEAxes(temporalDim: 16, heightDim: 8, widthDim: 8)
        let video = try VideoPatchTimeline(frames: 20, height: 3, width: 3, temporalStride: 4, fps: 24.0)
        let audio16k = try AudioLatentTimeline(frameCount: 300, hopSamples: 160, sampleRate: 16_000.0)
        let audio44kExact = try AudioLatentTimeline(frameCount: 300, hopSamples: 441, sampleRate: 44_100.0)
        let audio44kPow2 = try AudioLatentTimeline(frameCount: 2_000, hopSamples: 64, sampleRate: 44_100.0)

        for (videoFrame, targetSec) in [(6, 1.0), (15, 2.5)] {
            let vSec = video.timeSeconds(frame: videoFrame)
            #expect(abs(vSec - targetSec) < 1e-12)
            for audio in [audio16k, audio44kExact, audio44kPow2] {
                let m = audio.nearestFrame(atSeconds: targetSec)
                let aSec = audio.timeSeconds(frame: m)
                #expect(abs(vSec - aSec) < 1e-3)
                #expect(Fractional3DMultimodalRoPE.temporalPhaseDifferenceRadians(
                    videoTimeSeconds: vSec, audioTimeSeconds: aSec, axes: axes
                ) < 1e-3)
            }
        }

        let vFreq = Fractional3DMultimodalRoPE.videoFrequencies(axes: axes, timeline: video)
        let aFreq = Fractional3DMultimodalRoPE.audioFrequencies(axes: axes, timeline: audio44kExact)
        eval(vFreq.cos, vFreq.sin, aFreq.cos, aFreq.sin)
        for (vFrame, aFrame) in [(6, 100), (15, 250)] {
            let centerTok = vFrame * 9 + 4
            let vCos = vFreq.cos[0, 0, centerTok].asArray(Float.self)
            let vSin = vFreq.sin[0, 0, centerTok].asArray(Float.self)
            let aCos = aFreq.cos[0, 0, aFrame].asArray(Float.self)
            let aSin = aFreq.sin[0, 0, aFrame].asArray(Float.self)
            #expect(VideoAttnTestHelpers.maxAbsDiff(vCos, aCos) < 1e-6)
            #expect(VideoAttnTestHelpers.maxAbsDiff(vSin, aSin) < 1e-6)
        }
    }

    @Test("Cross-attention temporal inner product peaks at exact matching audio frame for t = 1.0 s and t = 2.5 s")
    func crossAttentionPhasePeakAlignment() throws {
        let axes = try Fractional3DRoPEAxes.balanced(headDim: 32, temporalTheta: 256.0, spatialTheta: 256.0)
        let video = try VideoPatchTimeline(frames: 20, height: 3, width: 3, temporalStride: 4, fps: 24.0)
        let audio = try AudioLatentTimeline(frameCount: 300, hopSamples: 160, sampleRate: 16_000.0)
        let probe = MLXArray((0 ..< 32).map { idx in idx.isMultiple(of: 2) ? Float(1.0) : Float(0.0) })
        let qVideoRaw = broadcast(probe.reshaped(1, 1, 1, 32), to: [1, 1, video.tokenCount, 32])
        let kAudioRaw = broadcast(probe.reshaped(1, 1, 1, 32), to: [1, 1, audio.frameCount, 32])
        let vFreq = Fractional3DMultimodalRoPE.videoFrequencies(axes: axes, timeline: video)
        let aFreq = Fractional3DMultimodalRoPE.audioFrequencies(axes: axes, timeline: audio)
        let qRot = Fractional3DMultimodalRoPE.apply(qVideoRaw, cos: vFreq.cos, sin: vFreq.sin)
        let kRot = Fractional3DMultimodalRoPE.apply(kAudioRaw, cos: aFreq.cos, sin: aFreq.sin)
        let scores = matmul(qRot, kRot.transposed(0, 1, 3, 2))
        eval(scores)
        let peakAt1Sec = scores[0, 0, 6 * 9 + 4].argMax().item(Int32.self)
        let peakAt2p5Sec = scores[0, 0, 15 * 9 + 4].argMax().item(Int32.self)
        #expect(peakAt1Sec == 100)
        #expect(peakAt2p5Sec == 250)
    }

    @Test(
        "Fractional3DMultimodalRoPE matches cpuReferenceApply on video, audio, and joint sequences across float32, float16, and bfloat16",
        arguments: [DType.float32, .float16, .bfloat16], [false, true]
    )
    func multimodalRoPECPUReferenceParity(dtype: DType, audioFirst: Bool) throws {
        let spatialMode: SpatialCoordinateMode = audioFirst ? .patchIndex : .unitInterval
        let axes = try Fractional3DRoPEAxes(temporalDim: 16, heightDim: 8, widthDim: 8)
        let video = try VideoPatchTimeline(frames: 4, height: 3, width: 4, temporalStride: 4, fps: 24.0, spatialMode: spatialMode)
        let audio = try AudioLatentTimeline(frameCount: 10, hopSamples: 441, sampleRate: 44_100.0)
        let jointTokens = video.tokenCount + audio.frameCount
        let x = MLXRandom.uniform(
            low: -0.45, high: 0.45, [2, 3, jointTokens, axes.headDim], key: MLXRandom.key(582)
        ).asType(dtype)
        let jointFreq = Fractional3DMultimodalRoPE.jointAudioVideoFrequencies(
            axes: axes, video: video, audio: audio, audioFirst: audioFirst
        )
        let actual = Fractional3DMultimodalRoPE.apply(x, cos: jointFreq.cos, sin: jointFreq.sin)
        let actual3D = Fractional3DMultimodalRoPE.apply(x[0], cos: jointFreq.cos, sin: jointFreq.sin)
        let actual2D = Fractional3DMultimodalRoPE.apply(x[0, 0], cos: jointFreq.cos, sin: jointFreq.sin)
        eval(x, actual, actual3D, actual2D)

        let vCoords = Fractional3DMultimodalRoPE.videoCoordinates(video)
        let aCoords = Fractional3DMultimodalRoPE.audioCoordinates(audio)
        let coords = audioFirst ? (aCoords + vCoords) : (vCoords + aCoords)
        let cpuRef = Fractional3DMultimodalRoPE.cpuReferenceApply(
            x.asType(.float32).asArray(Float.self),
            shape: CPU3DRotaryShape(batch: 2, heads: 3, tokens: jointTokens, headDim: axes.headDim),
            coordinates: coords,
            axes: axes
        )
        let actualVals = actual.asType(.float32).asArray(Float.self)
        #expect(actual.dtype == dtype)
        #expect(VideoAttnTestHelpers.maxAbsDiff(actualVals, cpuRef) < VideoAttnTestHelpers.ropeTolerance(dtype))
        #expect(VideoAttnTestHelpers.maxAbsDiff(
            actual3D.asType(.float32).asArray(Float.self),
            Array(actualVals.prefix(3 * jointTokens * axes.headDim))
        ) < 1e-6)
        #expect(VideoAttnTestHelpers.maxAbsDiff(
            actual2D.asType(.float32).asArray(Float.self),
            Array(actualVals.prefix(jointTokens * axes.headDim))
        ) < 1e-6)
    }

    @Test("Fractional3DRoPEAxes, timelines, and brick planners reject invalid dimensions with typed BasaltError")
    func parameterValidationErrors() throws {
        #expect(throws: BasaltError.self) {
            _ = try Fractional3DRoPEAxes(temporalDim: 3, heightDim: 4, widthDim: 4)
        }
        #expect(throws: BasaltError.self) {
            _ = try Fractional3DRoPEAxes.balanced(headDim: 5)
        }
        #expect(throws: BasaltError.self) {
            _ = try VideoPatchTimeline(frames: 0, height: 4, width: 4)
        }
        #expect(throws: BasaltError.self) {
            _ = try AudioLatentTimeline(frameCount: 4, hopSamples: 0, sampleRate: 16_000.0)
        }
        #expect(throws: BasaltError.self) {
            _ = try SpatioTemporalGrid(frames: 2, height: 0, width: 4)
        }
        #expect(throws: BasaltError.self) {
            _ = try BrickShape(t: 0, h: 4, w: 4)
        }
    }
}

@Suite("3D Brick-Tiled Sliding Tile and Radial Sparse Attention Metal Kernel (#582)", .serialized)
struct SlidingTileAttention3DTests {
    @Test("BrickSparsePlan, RadialDecaySchedule, and packIntoBricks / unpackFromBricks preserve exact and boundary-padded grids")
    func brickLayoutPackUnpackAndRadialSparsity() throws {
        let exactGrid = try SpatioTemporalGrid(frames: 4, height: 8, width: 8)
        let defaultPlan = SlidingTileAttention3D.buildPlan(
            grid: exactGrid, brick: .appleDefault, window: WindowRadiusBricks(t: 0, h: 0, w: 0)
        )
        #expect(defaultPlan.brickCount == 1 && defaultPlan.brickVolume == 256 && defaultPlan.coversFullGrid)

        let paddedGrid = try SpatioTemporalGrid(frames: 3, height: 5, width: 6)
        let smallBrick = try BrickShape(t: 2, h: 4, w: 4)
        let sparsePlan = SlidingTileAttention3D.buildPlan(
            grid: paddedGrid, brick: smallBrick, window: WindowRadiusBricks(t: 0, h: 1, w: 0)
        )
        #expect(!sparsePlan.isExactGrid && !sparsePlan.coversFullGrid && sparsePlan.sparsityRatio > 0.4)
        let x = MLXRandom.normal([2, 2, paddedGrid.tokenCount, 16], key: MLXRandom.key(91))
        let packed = SlidingTileAttention3D.packIntoBricks(x, plan: sparsePlan)
        let unpacked = SlidingTileAttention3D.unpackFromBricks(packed, plan: sparsePlan)
        eval(x, packed, unpacked)
        let xVals = x.asArray(Float.self)
        #expect(VideoAttnTestHelpers.maxAbsDiff(unpacked.asArray(Float.self), xVals) == 0.0)
        let cpuPacked = SlidingTileAttention3D.cpuTransferBricks(
            xVals, batch: 2, heads: 2, headDim: 16, plan: sparsePlan, pack: true
        )
        #expect(VideoAttnTestHelpers.maxAbsDiff(packed.reshaped(-1).asArray(Float.self), cpuPacked) == 0.0)

        let wideGrid = try SpatioTemporalGrid(frames: 8, height: 8, width: 8)
        let unitBrick = try BrickShape(t: 2, h: 2, w: 2)
        let radialManhattan = RadialDecaySchedule(
            metric: .manhattan, decayRate: 0.7, energyThreshold: 0.35, logarithmicShells: true
        )
        let radialEuclidean = RadialDecaySchedule(metric: .euclidean, decayRate: 0.9, energyThreshold: 0.45)
        let pMan = SlidingTileAttention3D.buildPlan(
            grid: wideGrid, brick: unitBrick, window: WindowRadiusBricks(t: 3, h: 3, w: 3), radial: radialManhattan
        )
        let pEuc = SlidingTileAttention3D.buildPlan(
            grid: wideGrid, brick: unitBrick, window: WindowRadiusBricks(t: 3, h: 3, w: 3), radial: radialEuclidean
        )
        #expect(pMan.sparsityRatio > 0.5 && pEuc.sparsityRatio > 0.5)
    }

    @Test(
        "SlidingTileAttention3D Metal kernel matches cpuReferenceBrickAttention across float32, float16, and bfloat16 with GQA and global sinks",
        arguments: [DType.float32, .float16, .bfloat16], [false, true]
    )
    func staMetalKernelMatchesCPUReference(dtype: DType, includeSinks: Bool) throws {
        let exactGrid = try SpatioTemporalGrid(frames: 4, height: 4, width: 4)
        let exactPlan = SlidingTileAttention3D.buildPlan(
            grid: exactGrid,
            brick: try BrickShape(t: 2, h: 2, w: 2),
            window: WindowRadiusBricks(t: 1, h: 0, w: 1),
            radial: RadialDecaySchedule(metric: .euclidean, decayRate: 0.4, energyThreshold: 0.5)
        )
        let paddedGrid = try SpatioTemporalGrid(frames: 3, height: 5, width: 5)
        let paddedPlan = SlidingTileAttention3D.buildPlan(
            grid: paddedGrid,
            brick: try BrickShape(t: 2, h: 4, w: 4),
            window: WindowRadiusBricks(t: 0, h: 1, w: 1)
        )
        for (plan, headDim) in [(exactPlan, 16), (paddedPlan, 64)] {
            Self.assertKernelCPUParity(plan: plan, dtype: dtype, headDim: headDim, includeSinks: includeSinks)
        }
    }

    private static func assertKernelCPUParity(
        plan: BrickSparsePlan, dtype: DType, headDim: Int, includeSinks: Bool
    ) {
        let batch = 2, qHeads = 4, kvHeads = 2, n = plan.grid.tokenCount, sinkLen = includeSinks ? 6 : 0
        let q = MLXRandom.uniform(low: -0.5, high: 0.5, [batch, qHeads, n, headDim], key: MLXRandom.key(101)).asType(dtype)
        let k = MLXRandom.uniform(low: -0.5, high: 0.5, [batch, kvHeads, n, headDim], key: MLXRandom.key(102)).asType(dtype)
        let v = MLXRandom.uniform(low: -0.5, high: 0.5, [batch, kvHeads, n, headDim], key: MLXRandom.key(103)).asType(dtype)
        let globalKV: GlobalConditioningKV? = includeSinks ? GlobalConditioningKV(
            keys: MLXRandom.uniform(low: -0.5, high: 0.5, [batch, kvHeads, sinkLen, headDim], key: MLXRandom.key(104)).asType(dtype),
            values: MLXRandom.uniform(low: -0.5, high: 0.5, [batch, kvHeads, sinkLen, headDim], key: MLXRandom.key(105)).asType(dtype)
        ) : nil
        if let globalKV { eval(globalKV.keys, globalKV.values) }
        let out = SlidingTileAttention3D.attention(queries: q, keys: k, values: v, plan: plan, globalKV: globalKV)
        eval(q, k, v, out)
        let cpuSinks = globalKV.map {
            CPUGlobalKV(keys: $0.keys.asType(.float32).asArray(Float.self), values: $0.values.asType(.float32).asArray(Float.self))
        }
        let cpuRef = SlidingTileAttention3D.cpuReferenceBrickAttention(
            queries: q.asType(.float32).asArray(Float.self),
            keys: k.asType(.float32).asArray(Float.self),
            values: v.asType(.float32).asArray(Float.self),
            shape: CPUAttentionShape(batch: batch, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, sinkTokens: sinkLen),
            plan: plan,
            globalKV: cpuSinks
        )
        #expect(out.shape == [batch, qHeads, n, headDim] && out.dtype == dtype)
        #expect(VideoAttnTestHelpers.maxAbsDiff(out.asType(.float32).asArray(Float.self), cpuRef) < VideoAttnTestHelpers.staTolerance(dtype))
    }

    @Test(
        "Full-grid window falls back cleanly to exact dense SDPA and matches forced Metal kernel across float32, float16, and bfloat16",
        arguments: [DType.float32, .float16, .bfloat16]
    )
    func fullWindowDenseFallbackAndForcedMetalEquivalence(dtype: DType) throws {
        let grid = try SpatioTemporalGrid(frames: 4, height: 4, width: 4)
        let fullPlan = SlidingTileAttention3D.buildPlan(
            grid: grid, brick: try BrickShape(t: 2, h: 2, w: 2), window: WindowRadiusBricks(t: 2, h: 2, w: 2)
        )
        #expect(fullPlan.coversFullGrid && fullPlan.sparsityRatio == 0.0)
        let n = grid.tokenCount, d = 32
        let q = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 4, n, d], key: MLXRandom.key(201)).asType(dtype)
        let k = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, n, d], key: MLXRandom.key(202)).asType(dtype)
        let v = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, n, d], key: MLXRandom.key(203)).asType(dtype)
        let sk = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 5, d], key: MLXRandom.key(204)).asType(dtype)
        let sv = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 2, 5, d], key: MLXRandom.key(205)).asType(dtype)
        let sinks = GlobalConditioningKV(keys: sk, values: sv)
        let scale = 1.0 / Float(d).squareRoot()

        let viaFallback = SlidingTileAttention3D.attention(queries: q, keys: k, values: v, plan: fullPlan, globalKV: sinks)
        let viaForcedMetal = SlidingTileAttention3D.attention(
            queries: q, keys: k, values: v, plan: fullPlan, globalKV: sinks, options: STAExecutionOptions(forceMetalKernel: true)
        )
        let stockDense = MLXFast.scaledDotProductAttention(
            queries: q, keys: concatenated([k, sk], axis: 2), values: concatenated([v, sv], axis: 2), scale: scale, mask: .none
        )
        eval(viaFallback, viaForcedMetal, stockDense)
        let fallbackVals = viaFallback.asType(.float32).asArray(Float.self)
        let metalVals = viaForcedMetal.asType(.float32).asArray(Float.self)
        let denseVals = stockDense.asType(.float32).asArray(Float.self)
        #expect(VideoAttnTestHelpers.maxAbsDiff(fallbackVals, denseVals) == 0.0)
        #expect(VideoAttnTestHelpers.maxAbsDiff(metalVals, denseVals) < VideoAttnTestHelpers.staTolerance(dtype))
    }

    @Test("TinyModel Qwen3 projections with 3D Multimodal RoPE, STA Metal kernel, and TeaCache step skipping match CPU reference (#228)")
    func tinyModelProjectionAndTeaCacheIntegrationParity() throws {
        let tiny = try TinyModel.make(.qwen3)
        tiny.update(parameters: tiny.parameters().mapValues { $0.asType(.float16) })
        eval(tiny)
        let modules = Dictionary(tiny.namedModules(), uniquingKeysWith: { a, _ in a })
        let qProj = try #require(modules["model.layers.0.self_attn.q_proj"] as? Linear)
        let kProj = try #require(modules["model.layers.0.self_attn.k_proj"] as? Linear)
        let vProj = try #require(modules["model.layers.0.self_attn.v_proj"] as? Linear)

        let video = try VideoPatchTimeline(frames: 4, height: 2, width: 2, temporalStride: 4, fps: 24.0)
        let audio = try AudioLatentTimeline(frameCount: 4, hopSamples: 400, sampleRate: 16_000.0)
        let axes = try Fractional3DRoPEAxes(temporalDim: 8, heightDim: 4, widthDim: 4)
        let plan = SlidingTileAttention3D.buildPlan(
            grid: try SpatioTemporalGrid(frames: 4, height: 2, width: 2),
            brick: try BrickShape(t: 2, h: 2, w: 2),
            window: WindowRadiusBricks(t: 0, h: 0, w: 0)
        )
        let vidHidden = MLXRandom.normal([1, video.tokenCount, 64], scale: 0.25, key: MLXRandom.key(301)).asType(.float16)
        let audHidden = MLXRandom.normal([1, audio.frameCount, 64], scale: 0.25, key: MLXRandom.key(302)).asType(.float16)
        let vFreq = Fractional3DMultimodalRoPE.videoFrequencies(axes: axes, timeline: video)
        let aFreq = Fractional3DMultimodalRoPE.audioFrequencies(axes: axes, timeline: audio)

        let qVid = Fractional3DMultimodalRoPE.apply(qProj(vidHidden).reshaped(1, 16, 4, 16).transposed(0, 2, 1, 3), cos: vFreq.cos, sin: vFreq.sin)
        let kVid = Fractional3DMultimodalRoPE.apply(kProj(vidHidden).reshaped(1, 16, 2, 16).transposed(0, 2, 1, 3), cos: vFreq.cos, sin: vFreq.sin)
        let vVid = vProj(vidHidden).reshaped(1, 16, 2, 16).transposed(0, 2, 1, 3)
        let kAud = Fractional3DMultimodalRoPE.apply(kProj(audHidden).reshaped(1, 4, 2, 16).transposed(0, 2, 1, 3), cos: aFreq.cos, sin: aFreq.sin)
        let vAud = vProj(audHidden).reshaped(1, 4, 2, 16).transposed(0, 2, 1, 3)

        var teaCache = TeaCacheState(totalSteps: 6, threshold: 0.5)
        let step0 = SlidingTileAttention3D.evaluateStepWithTeaCache(step: 0, modulatedInput: vidHidden, teaCache: &teaCache) {
            SlidingTileAttention3D.attention(queries: qVid, keys: kVid, values: vVid, plan: plan, globalKV: GlobalConditioningKV(keys: kAud, values: vAud))
        }
        let perturbed = (vidHidden * Float(1.001)).asType(.float16)
        let step1 = SlidingTileAttention3D.evaluateStepWithTeaCache(step: 1, modulatedInput: perturbed, teaCache: &teaCache) {
            MLXArray.zeros(step0.shape, dtype: .float16)
        }
        eval(qVid, kVid, vVid, kAud, vAud, step0, step1)
        #expect(teaCache.executedSteps == 1 && teaCache.skippedSteps == 1)

        let cpuRef = SlidingTileAttention3D.cpuReferenceBrickAttention(
            queries: qVid.asType(.float32).asArray(Float.self),
            keys: kVid.asType(.float32).asArray(Float.self),
            values: vVid.asType(.float32).asArray(Float.self),
            shape: CPUAttentionShape(batch: 1, qHeads: 4, kvHeads: 2, headDim: 16, sinkTokens: 4),
            plan: plan,
            globalKV: CPUGlobalKV(keys: kAud.asType(.float32).asArray(Float.self), values: vAud.asType(.float32).asArray(Float.self))
        )
        #expect(VideoAttnTestHelpers.maxAbsDiff(step0.asType(.float32).asArray(Float.self), cpuRef) < 2e-3)
        #expect(VideoAttnTestHelpers.maxAbsDiff(step1.asType(.float32).asArray(Float.self), step0.asType(.float32).asArray(Float.self)) == 0.0)
    }
}
