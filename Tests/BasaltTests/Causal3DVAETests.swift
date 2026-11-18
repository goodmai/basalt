import Foundation
import Testing
import MLX
import MLXNN
@testable import BasaltCore
@testable import BasaltInference

private enum Causal3DVAETestSupport {
    static func makeTinyVAE(_ variant: VideoVAEVariant, seed: UInt64 = 581) -> Causal3DVAE {
        let model = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Causal3DVAE(.tiny(variant))
        }
        eval(model)
        return model
    }

    static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}

@Suite("CausalConv3d temporal ring-buffer and OverlapTileBlender raised-cosine partition of unity (#581)", .serialized)
struct Causal3DVAEPrimitivesTests {

    @Test("CausalConv3d enforces strict temporal causality and matches chunked TemporalConvCache streaming across padding modes",
          arguments: CausalPaddingMode.allCases)
    func causalConv3dCausalityAndStreamingParity(mode: CausalPaddingMode) {
        let spec = CausalConv3dSpec(
            inputChannels: 8,
            outputChannels: 8,
            kernelSize: [3, 3, 3],
            stride: 1,
            paddingMode: mode
        ).withDilation([2, 1, 1])
        #expect(spec.causalPadT == 4)

        let layer = withRandomState(MLXRandom.RandomState(seed: 42)) {
            CausalConv3d(spec: spec)
        }
        eval(layer)

        let xA = MLXRandom.normal([1, 6, 8, 8, 8], key: MLXRandom.key(101))
        let xB = concatenated([
            xA[0..., 0 ..< 5, 0..., 0..., 0...],
            MLXArray.ones([1, 1, 8, 8, 8], dtype: .float32) * 9.0
        ], axis: 1)

        let monoA = layer(xA)
        let monoB = layer(xB)
        eval(monoA, monoB)
        #expect(Causal3DVAETestSupport.maxAbsDiff(monoA[0..., ..<5, 0..., 0..., 0...], monoB[0..., ..<5, 0..., 0..., 0...]) == 0)

        var cache = layer.makeCache()
        #expect(cache.isFirstChunk && cache.cachedFrameCount == 0)
        var streamed1: [MLXArray] = []
        for t in 0 ..< 6 {
            streamed1.append(layer(xA[0..., t ..< (t + 1), 0..., 0..., 0...], cache: &cache))
        }
        #expect(!cache.isFirstChunk && cache.cachedFrameCount == 4)
        let joined1 = concatenated(streamed1, axis: 1)
        eval(joined1)
        #expect(Causal3DVAETestSupport.maxAbsDiff(monoA, joined1) < 1e-5)

        cache.reset()
        #expect(cache.isFirstChunk && cache.cachedFrameCount == 0)
        let joined2 = concatenated([
            layer(xA[0..., 0 ..< 2, 0..., 0..., 0...], cache: &cache),
            layer(xA[0..., 2 ..< 5, 0..., 0..., 0...], cache: &cache),
            layer(xA[0..., 5 ..< 6, 0..., 0..., 0...], cache: &cache)
        ], axis: 1)
        eval(joined2)
        #expect(Causal3DVAETestSupport.maxAbsDiff(monoA, joined2) < 1e-5)
    }

    @Test("SpatialTileGrid and TileWindow separable raised-cosine weights sum to exact 1.0 across interior seams, corners, and exterior borders")
    func raisedCosinePartitionOfUnityAndBlender() throws {
        let cases: [(Int, Int, SpatialTileConfig)] = [
            (16, 16, SpatialTileConfig(tileLatentHeight: 8, tileLatentWidth: 8, overlapLatentHeight: 2, overlapLatentWidth: 2, haloLatents: 4)),
            (15, 19, SpatialTileConfig(tileLatentHeight: 8, tileLatentWidth: 8, overlapLatentHeight: 3, overlapLatentWidth: 2, haloLatents: 4)),
            (6, 6, SpatialTileConfig(tileLatentHeight: 8, tileLatentWidth: 8, overlapLatentHeight: 2, overlapLatentWidth: 2, haloLatents: 0)),
            (60, 108, .uma24GB)
        ]

        for (latH, latW, cfg) in cases {
            let grid = SpatialTileGrid(latentHeight: latH, latentWidth: latW, spatialScale: 8, config: cfg)
            let extrema = grid.accumulatedWeightExtrema()
            #expect(Swift.abs(extrema.min - 1.0) < 1e-6)
            #expect(Swift.abs(extrema.max - 1.0) < 1e-6)

            if latH <= 20 {
                let matrix = grid.weightSumMatrix()
                #expect(matrix.count == latH * 8 && matrix[0].count == latW * 8)
                for row in matrix {
                    for weight in row {
                        #expect(Swift.abs(weight - 1.0) < 1e-6)
                    }
                }
            }
        }

        let grid = SpatialTileGrid(
            latentHeight: 15, latentWidth: 19, spatialScale: 4,
            config: SpatialTileConfig(tileLatentHeight: 8, tileLatentWidth: 8, overlapLatentHeight: 2, overlapLatentWidth: 2, haloLatents: 0)
        )
        let topLeftTile = grid.tiles[0]
        let topLeftWindow = TileWindow(for: topLeftTile)
        #expect(topLeftWindow.weight(y: 0, x: 0) == 1.0)
        #expect(topLeftTile.topOverlapPixels == 0 && topLeftTile.leftOverlapPixels == 0)

        let latents = MLXRandom.normal([1, 2, 15, 19, 3], key: MLXRandom.key(77))
        let upsampler = Upsample(scaleFactor: [1.0, 4.0, 4.0], mode: .nearest)
        let expected = upsampler(latents)
        let blended = try OverlapTileBlender.decodeTiled(latents: latents, grid: grid, outputChannels: 3) { tile in
            upsampler(tile)
        }
        eval(expected, blended)
        #expect(Causal3DVAETestSupport.maxAbsDiff(expected, blended) < 1e-5)
    }
}

@Suite("Causal3DVAE streaming, tiling, precision, LTX-2.5 DiffVAE, and audio vocoder parity (#581)", .serialized)
struct Causal3DVAEParityTests {

    @Test("Causal3DVAE streaming decode with temporalChunkLatents 1 and spatial overlap-blend tiling matches monolithic decode within 1e-4 in FP32 across all VideoVAEVariant configs",
          arguments: VideoVAEVariant.allCases)
    func fp32StreamingAndTiledParityAcrossVariants(variant: VideoVAEVariant) throws {
        let vae = Causal3DVAETestSupport.makeTinyVAE(variant, seed: 581)
        let latents = MLXRandom.normal([1, 4, 16, 20, vae.config.latentChannels], key: MLXRandom.key(2026))
        let singleOut = try vae.decodeMonolithic(latents[0..., 0 ..< 1, 0..., 0..., 0...])
        eval(singleOut)
        #expect(singleOut.shape == [1, 1, 128, 160, 3])

        let mono = try vae.decodeMonolithic(latents)
        let streamUntiled = try vae.decodeStreaming(latents, temporalChunkLatents: 1, tileConfig: nil)
        let tileCfg = SpatialTileConfig(tileLatentHeight: 8, tileLatentWidth: 8, overlapLatentHeight: 2, overlapLatentWidth: 2)
        let streamTiled = try vae.decodeStreaming(latents, temporalChunkLatents: 1, tileConfig: tileCfg)
        eval(mono, streamUntiled, streamTiled)

        #expect(mono.shape == [1, 13, 128, 160, 3])
        #expect(streamUntiled.shape == mono.shape && streamTiled.shape == mono.shape)
        #expect(Causal3DVAETestSupport.maxAbsDiff(mono, streamUntiled) < 1e-4, "\(variant.rawValue) untiled temporal cache parity")
        #expect(Causal3DVAETestSupport.maxAbsDiff(mono, streamTiled) < 1e-4, "\(variant.rawValue) chunked + tiled parity")

        let boundaryFrames = mono[0..., 0 ..< 2, 48 ..< 80, 48 ..< 80, 0...]
        let tiledBoundary = streamTiled[0..., 0 ..< 2, 48 ..< 80, 48 ..< 80, 0...]
        #expect(Causal3DVAETestSupport.maxAbsDiff(boundaryFrames, tiledBoundary) < 1e-4, "\(variant.rawValue) seam + frame boundary parity")
    }

    @Test("Causal3DVAE chunked and tiled streaming decode matches monolithic decode within 2e-3 in FP16 and BF16",
          arguments: [DType.float16, DType.bfloat16])
    func reducedPrecisionStreamingAndTiledParity(dtype: DType) throws {
        for variant in [VideoVAEVariant.wan22VAE, VideoVAEVariant.ltx25DiffVAE] {
            let vae = Causal3DVAETestSupport.makeTinyVAE(variant, seed: 777)
            vae.update(parameters: vae.parameters().mapValues { $0.asType(dtype) })
            eval(vae)

            let latents = MLXRandom.normal([1, 4, 16, 20, vae.config.latentChannels], key: MLXRandom.key(888)).asType(dtype)
            let tileCfg = SpatialTileConfig(tileLatentHeight: 8, tileLatentWidth: 8, overlapLatentHeight: 2, overlapLatentWidth: 2)

            let mono = try vae.decodeMonolithic(latents)
            let streamed = try vae.decodeStreaming(latents, temporalChunkLatents: 1, tileConfig: tileCfg)
            eval(mono, streamed)

            #expect(mono.dtype == dtype && streamed.dtype == dtype)
            #expect(mono.shape == [1, 13, 128, 160, 3])
            let err = Causal3DVAETestSupport.maxAbsDiff(mono, streamed)
            #expect(err.isFinite && err < 2e-3, "\(variant.rawValue) dtype \(dtype) max abs diff \(err)")
        }
    }

    @Test("LTX-2.5 DiffVAE timestep conditioning and residual noise injection modulate output and preserve chunked tiled parity")
    func ltx25DiffVAETimestepAndNoiseParity() throws {
        let vae = Causal3DVAETestSupport.makeTinyVAE(.ltx25DiffVAE, seed: 909)
        let latents = MLXRandom.normal([1, 3, 16, 16, vae.config.latentChannels], key: MLXRandom.key(314))
        let noise = MLXRandom.normal(latents.shape, key: MLXRandom.key(271))

        let cleanZero = try vae.decodeMonolithic(latents, timestep: 0.0, residualNoise: nil)
        let steppedWithNoise = try vae.decodeMonolithic(latents, timestep: 0.25, residualNoise: noise)
        eval(cleanZero, steppedWithNoise)
        #expect(Causal3DVAETestSupport.maxAbsDiff(cleanZero, steppedWithNoise) > 1e-4)

        let tileCfg = SpatialTileConfig(tileLatentHeight: 8, tileLatentWidth: 8, overlapLatentHeight: 2, overlapLatentWidth: 2)
        let streamedWithNoise = try vae.decodeStreaming(
            latents, temporalChunkLatents: 1, tileConfig: tileCfg, timestep: 0.25, residualNoise: noise
        )
        eval(streamedWithNoise)
        #expect(Causal3DVAETestSupport.maxAbsDiff(steppedWithNoise, streamedWithNoise) < 1e-4)
    }

    @Test("Joint audio-video vocoder bridge synthesizes 44.1 kHz Kandinsky 6.0 and 24 kHz LTX-2.5 waveforms and rejects invalid shapes")
    func jointAudioVideoVocoderBridgeAndValidation() throws {
        for variant in [VideoVAEVariant.kandinsky6VAE, VideoVAEVariant.ltx25DiffVAE] {
            let vae = Causal3DVAETestSupport.makeTinyVAE(variant, seed: 600)
            let vocoder = withRandomState(MLXRandom.RandomState(seed: 601)) {
                CausalMelVocoderAdapter.tiny(for: variant)
            }
            eval(vocoder)

            let latents = MLXRandom.normal([1, 2, 8, 8, vae.config.latentChannels], key: MLXRandom.key(602))
            let mel = MLXRandom.normal([1, 10, vocoder.melBins], key: MLXRandom.key(603))
            let joint = try vae.decodeJointAudioVideo(
                videoLatents: latents, melSpectrogram: mel, vocoder: vocoder, temporalChunkLatents: 1
            )
            eval(joint.videoFrames, joint.audioWaveform)
            #expect(joint.videoFrames.shape == [1, 5, 64, 64, 3])
            #expect(joint.audioWaveform.shape == [1, 10 * vocoder.hopLength])
            #expect(joint.audioSampleRate == variant.defaultAudioSampleRate)
            #expect(MLX.abs(joint.audioWaveform).max().item(Float.self) <= 1.0)
        }

        let wanVAE = Causal3DVAETestSupport.makeTinyVAE(.wan22VAE, seed: 604)
        let vocoder = CausalMelVocoderAdapter.tiny(for: .kandinsky6VAE)
        let validLatents = MLXArray.zeros([1, 2, 8, 8, wanVAE.config.latentChannels], dtype: .float32)
        let validMel = MLXArray.zeros([1, 4, vocoder.melBins], dtype: .float32)

        #expect(throws: BasaltError.self) {
            _ = try wanVAE.decodeJointAudioVideo(videoLatents: validLatents, melSpectrogram: validMel, vocoder: vocoder)
        }
        #expect(throws: BasaltError.self) {
            _ = try wanVAE.decodeMonolithic(MLXArray.zeros([2, 8, 8, 8], dtype: .float32))
        }
        #expect(throws: BasaltError.self) {
            _ = try vocoder.synthesizeWaveform(fromMel: MLXArray.zeros([1, 4, vocoder.melBins + 1], dtype: .float32))
        }
    }
}
