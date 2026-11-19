import BasaltCore
import Foundation
import MLX
import MLXNN

package struct VideoDiTConfig: Sendable, Equatable {
    package var modelID: String
    package var modelType: String
    package var hiddenSize: Int
    package var numHeads: Int
    package var numLayers: Int
    package var latentChannels: Int
    package var temporalScale: Int
    package var spatialScale: Int
    package var footprint: VideoStageFootprint

    package init(
        modelID: String = "basalt/video-dit-tiny",
        modelType: String = "video_dit",
        hiddenSize: Int = 64,
        numHeads: Int = 4,
        numLayers: Int = 2,
        latentChannels: Int = 8
    ) {
        let alignedHidden = max(32, ((min(256, hiddenSize) + 31) / 32) * 32)
        self.modelID = modelID
        self.modelType = modelType
        self.hiddenSize = alignedHidden
        self.numHeads = max(1, min(alignedHidden / 16, numHeads))
        self.numLayers = max(1, min(6, numLayers))
        self.latentChannels = max(4, min(32, latentChannels))
        self.temporalScale = 4
        self.spatialScale = 8
        self.footprint = .standard24GB
    }

    package static func load(from directory: URL?, identifier: String) throws(BasaltError) -> (VideoModelDescriptor, VideoDiTConfig) {
        guard let directory else {
            return (VideoModelDescriptor.detect(identifier: identifier), VideoDiTConfig(modelID: identifier))
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDir), isDir.boolValue else {
            throw .modelNotFound(identifier: directory.path)
        }
        let raw = try readJSONConfig(in: directory)
        let modelType = (raw?["model_type"] as? String) ?? "video_dit"
        let desc = VideoModelDescriptor.detect(identifier: identifier, modelType: modelType)
        let hidden = (raw?["hidden_size"] as? Int) ?? (raw?["dim"] as? Int) ?? 64
        let heads = (raw?["num_attention_heads"] as? Int) ?? (raw?["num_heads"] as? Int) ?? 4
        let layers = (raw?["num_hidden_layers"] as? Int) ?? (raw?["num_layers"] as? Int) ?? 2
        let latCh = (raw?["latent_channels"] as? Int) ?? (raw?["in_channels"] as? Int) ?? 8
        var cfg = VideoDiTConfig(
            modelID: (raw?["model_id"] as? String) ?? identifier,
            modelType: modelType,
            hiddenSize: hidden,
            numHeads: heads,
            numLayers: layers,
            latentChannels: latCh
        )
        if let ditGB = (raw?["estimated_dit_gb"] as? NSNumber)?.doubleValue {
            let encGB = (raw?["estimated_encoder_gb"] as? NSNumber)?.doubleValue ?? 7.5
            let vaeGB = (raw?["estimated_vae_gb"] as? NSNumber)?.doubleValue ?? 2.2
            cfg.footprint = VideoStageFootprint(textEncoderGB: encGB, ditGB: ditGB, vaeAndVocoderGB: vaeGB)
        }
        return (desc, cfg)
    }

    private static func readJSONConfig(in directory: URL) throws(BasaltError) -> [String: Any]? {
        let candidates = [
            directory.appending(component: "config.json"),
            directory.appending(path: "transformer/config.json"),
        ]
        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            guard let data = try? Data(contentsOf: url),
                  let parsed = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
                throw .modelConfigurationInvalid(path: url.path, reason: "config.json must be a valid JSON object")
            }
            return parsed
        }
        return nil
    }
}

package struct MaterializedTextConditioning: Sendable {
    package let positivePooled: [Float]
    package let negativePooled: [Float]?
    package let tokenEmbeddings: [Float]
    package let tokenCount: Int
    package let hiddenDim: Int
}

package struct MaterializedVideoLatents: Sendable {
    package let values: [Float]
    package let latentFrames: Int
    package let latentHeight: Int
    package let latentWidth: Int
    package let channels: Int
    package let audioEnvelope: [Float]
    package let stepsExecuted: Int
    package let teacacheSkipped: Int
}

package final class VideoDiTBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "ada_ln") var adaLN: Linear
    @ModuleInfo(key: "mlp_in") var mlpIn: Linear
    @ModuleInfo(key: "mlp_out") var mlpOut: Linear
    let hiddenSize: Int
    let numHeads: Int

    package init(hiddenSize: Int, numHeads: Int) {
        self.hiddenSize = hiddenSize
        self.numHeads = max(1, numHeads)
        self._norm1.wrappedValue = RMSNorm(dimensions: hiddenSize)
        self._qkv.wrappedValue = Linear(hiddenSize, hiddenSize * 3)
        self._outProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._adaLN.wrappedValue = Linear(hiddenSize, hiddenSize * 2)
        self._mlpIn.wrappedValue = Linear(hiddenSize, hiddenSize * 2)
        self._mlpOut.wrappedValue = Linear(hiddenSize * 2, hiddenSize)
    }

    package func forward(_ x: MLXArray, conditioning: MLXArray, staWindow: Int?) -> MLXArray {
        let mod = adaLN(silu(conditioning))
        let shiftScale = split(mod, parts: 2, axis: -1)
        let h = norm1(x) * (1.0 + 0.15 * tanh(shiftScale[0][0..., .newAxis, 0...])) + 0.15 * shiftScale[1][0..., .newAxis, 0...]
        let attnOut = computeAttention(h, staWindow: staWindow)
        let mid = x + outProj(attnOut)
        return mid + mlpOut(gelu(mlpIn(mid)))
    }

    private func computeAttention(_ h: MLXArray, staWindow: Int?) -> MLXArray {
        let b = h.dim(0)
        let seq = h.dim(1)
        let headDim = max(1, hiddenSize / numHeads)
        let projected = qkv(h)
        let parts = split(projected, parts: 3, axis: -1)
        let q = parts[0].reshaped(b, seq, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = parts[1].reshaped(b, seq, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped(b, seq, numHeads, headDim).transposed(0, 2, 1, 3)
        let scale = 1.0 / Float(headDim).squareRoot()
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let staWindow, staWindow > 0, staWindow < seq {
            let idx = MLXArray(0 ..< seq)
            let dist = abs(idx.expandedDimensions(axis: 1) - idx.expandedDimensions(axis: 0))
            let allowed = dist .<= Int32(staWindow)
            let additive = MLX.where(allowed, MLXArray(Float(0)), MLXArray(Float(-1e4)))
            maskMode = .array(additive.reshaped(1, 1, seq, seq).asType(q.dtype))
        } else {
            maskMode = .none
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: maskMode
        )
        return attended.transposed(0, 2, 1, 3).reshaped(b, seq, hiddenSize)
    }
}

package final class CausalVideoDiTModule: Module {
    @ModuleInfo(key: "patch_in") var patchIn: Linear
    @ModuleInfo(key: "cond_in") var condIn: Linear
    @ModuleInfo(key: "blocks") var blocks: [VideoDiTBlock]
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "audio_head") var audioHead: Linear
    let config: VideoDiTConfig

    package init(config: VideoDiTConfig) {
        self.config = config
        let d = config.hiddenSize
        let c = max(32, ((config.latentChannels + 31) / 32) * 32)
        self._patchIn.wrappedValue = Linear(c, d)
        self._condIn.wrappedValue = Linear(d, d)
        self._blocks.wrappedValue = (0 ..< config.numLayers).map { _ in
            VideoDiTBlock(hiddenSize: d, numHeads: config.numHeads)
        }
        self._projOut.wrappedValue = Linear(d, config.latentChannels)
        self._audioHead.wrappedValue = Linear(d, 32)
    }

    package var paddedChannels: Int {
        max(32, ((config.latentChannels + 31) / 32) * 32)
    }

    package func modulatedInput(_ flatLatents: MLXArray, timestep: Float, pooledCond: MLXArray) -> MLXArray {
        let padded = padChannelsIfNeeded(flatLatents)
        let tBias = Float(sin(Double(timestep) * .pi)) * 0.25
        return patchIn(padded) + condIn(pooledCond)[0..., .newAxis, 0...] + tBias
    }

    package func predictVelocity(_ hidden: MLXArray, pooledCond: MLXArray, staWindow: Int?) -> (velocity: MLXArray, audio: MLXArray) {
        var cur = hidden
        let cond = condIn(pooledCond)
        for block in blocks {
            cur = block.forward(cur, conditioning: cond, staWindow: staWindow)
        }
        let vel = tanh(projOut(cur))
        let aud = tanh(audioHead(mean(cur, axis: 1)))
        return (vel, aud)
    }

    private func padChannelsIfNeeded(_ x: MLXArray) -> MLXArray {
        let curC = x.dim(2)
        let targetC = paddedChannels
        if curC == targetC { return x }
        if curC > targetC { return x[0..., 0..., ..<targetC] }
        let zeros = MLXArray.zeros([x.dim(0), x.dim(1), targetC - curC], dtype: x.dtype)
        return concatenated([x, zeros], axis: 2)
    }
}

package final class Causal3DVAEDecoder: Module {
    let latentChannels: Int
    let hiddenDim: Int
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "temporal_mix") var temporalMix: Linear
    @ModuleInfo(key: "rgb_out") var rgbOut: Linear

    package init(latentChannels: Int, hiddenDim: Int = 32) {
        let c = max(4, latentChannels)
        let h = max(16, hiddenDim)
        self.latentChannels = c
        self.hiddenDim = h
        self._inProj.wrappedValue = Linear(c, h)
        self._temporalMix.wrappedValue = Linear(h, h)
        self._rgbOut.wrappedValue = Linear(h, 3)
    }

    package func decodeTile(
        latents: MLXArray,
        targetHeight: Int,
        targetWidth: Int
    ) -> [MLXArray] {
        let tCount = latents.dim(0)
        var h = silu(inProj(latents))
        if tCount > 1 {
            let first = h[0 ..< 1]
            let shifted = concatenated([first, h[0 ..< (tCount - 1)]], axis: 0)
            h = h + 0.25 * temporalMix(shifted)
        }
        let lowResRGB = sigmoid(rgbOut(h))
        eval(lowResRGB)
        return (0 ..< tCount).map { idx in
            TinyAutoencoderPreview.upsampleBilinearGrid(lowResRGB[idx], height: targetHeight, width: targetWidth)
        }
    }
}
