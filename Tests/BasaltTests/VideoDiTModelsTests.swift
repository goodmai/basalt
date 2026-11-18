import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("LTX-2.5 AV-DiT, Kandinsky 6.0 CrossDiT, LingBot-Video MoE & Wan 2.2 (#584)")
struct VideoDiTModelsTests {
    @Test("TinyModel fixtures for ltxVideo, kandinskyCrossDiT and moeVideoDiT verify forward shape, A/V split, and Euler step parity")
    func tinyModelVideoFamiliesForwardAndEulerParity() throws {
        let ltx = try #require(try TinyModel.make(.ltxVideo) as? LTX25AVDiTModel)
        let kand = try #require(try TinyModel.make(.kandinskyCrossDiT) as? KandinskyCrossDiTModel)
        let moe = try #require(try TinyModel.make(.moeVideoDiT) as? MoEVideoDiTModel)

        let tokenBatch = MLXArray([Int32(4), 9, 15, 21], [1, 4])
        for mod: any LanguageModel in [ltx, kand, moe] {
            let logits = mod(tokenBatch, cache: nil)
            eval(logits)
            #expect(logits.shape == [1, 4, 256])
        }

        let vLat = MLXRandom.normal([1, 12, 16], dtype: .float32, key: MLXRandom.key(7))
        let aLat = MLXRandom.normal([1, 6, 8], dtype: .float32, key: MLXRandom.key(8))
        let txt = MLXRandom.normal([1, 5, 64], dtype: .float32, key: MLXRandom.key(9))

        let (ltxVV, ltxAV) = ltx.predictAudioVideoVelocity(videoLatents: vLat, audioLatents: aLat, textTokens: txt, timestep: 0.5)
        let (kandVV, kandAV) = kand.predictCrossVelocity(videoLatents: vLat, audioLatents: aLat, contextTokens: txt)
        let (moeVV, moeAV) = moe.predictMoEVelocity(videoLatents: vLat, audioLatents: aLat, textTokens: txt)
        eval(ltxVV, ltxAV, kandVV, kandAV, moeVV, moeAV)

        #expect(ltxVV.shape == [1, 12, 16] && ltxAV.shape == [1, 6, 8])
        #expect(kandVV.shape == [1, 12, 16] && kandAV.shape == [1, 6, 8])
        #expect(moeVV.shape == [1, 12, 16] && moeAV.shape == [1, 6, 8])

        let dt: Float = 0.125
        let eulerNext = vLat + dt * ltxVV
        let (ltxVVRepeat, _) = ltx.predictAudioVideoVelocity(videoLatents: vLat, audioLatents: aLat, textTokens: txt, timestep: 0.5)
        let refNext = vLat + dt * ltxVVRepeat
        let diff = abs(eulerNext - refNext).max()
        eval(diff)
        #expect(diff.item(Float.self) < 1e-6)
    }

    @Test("MoEVideoDiTModel supports Quantizable SwitchGLU conversion and VideoModelDescriptor routes 24 GB vs 64 GB hardware")
    func moeQuantizationAndDescriptorRouting() throws {
        let moe = try #require(try TinyModel.make(.moeVideoDiT) as? MoEVideoDiTModel)
        quantize(model: moe, groupSize: 32, bits: 4) { path, module in
            if path.hasSuffix(".router") { return false }
            if let lin = module as? Linear { return lin.weight.dim(-1).isMultiple(of: 32) }
            return true
        }
        let vLat = MLXRandom.normal([1, 8, 16], dtype: .float32, key: MLXRandom.key(31))
        let aLat = MLXRandom.normal([1, 4, 8], dtype: .float32, key: MLXRandom.key(32))
        let txt = MLXRandom.normal([1, 3, 64], dtype: .float32, key: MLXRandom.key(33))
        let (vv, av) = moe.predictMoEVelocity(videoLatents: vLat, audioLatents: aLat, textTokens: txt)
        eval(vv, av)
        #expect(vv.shape == [1, 8, 16] && av.shape == [1, 4, 8])

        #expect(VideoModelDescriptor.detect(identifier: "inclusionAI/LingBot-Video-30B-A3B").strategy24GB == .expertStreamingSSD)
        #expect(VideoModelDescriptor.detect(identifier: "Wan-Video/Wan2.2-A14B").supportsJointAudio == false)
        #expect(VideoModelDescriptor.detect(identifier: "Lightricks/LTX-2.5-22B-Distilled").family == .ltx25AV)
        #expect(VideoModelDescriptor.detect(identifier: "ai-forever/Kandinsky-6.0-29B-Pro").strategy24GB == .cascadeLiteToCloudPro)
        #expect(VideoModelDescriptor.selectKandinskyCascade(availableRAMBytes: 24_000_000_000).activeBillionParams == 3.0)
        #expect(VideoModelDescriptor.selectKandinskyCascade(availableRAMBytes: 64_000_000_000).activeBillionParams == 29.0)
    }
}
