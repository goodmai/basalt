import BasaltCore
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public struct VideoDiTConfiguration: Codable, Sendable {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var intermediateSize: Int
    public var videoChannels: Int
    public var audioChannels: Int
    public var numExperts: Int
    public var numExpertsPerTok: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case videoChannels = "video_channels"
        case audioChannels = "audio_channels"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "ltx_video"
        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 256
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 64
        self.numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 2
        self.numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 4
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 128
        self.videoChannels = try c.decodeIfPresent(Int.self, forKey: .videoChannels) ?? 16
        self.audioChannels = try c.decodeIfPresent(Int.self, forKey: .audioChannels) ?? 8
        self.numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 4
        self.numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 2
    }
}

final class JointAVAttentionBlock: Module {
    let dimensions: Int
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "mlp_in") var mlpIn: Linear
    @ModuleInfo(key: "mlp_out") var mlpOut: Linear

    init(dimensions: Int, heads: Int, intermediate: Int) {
        self.dimensions = dimensions
        self.heads = heads
        self.scale = 1.0 / Float(dimensions / heads).squareRoot()
        self._norm1.wrappedValue = RMSNorm(dimensions: dimensions)
        self._qkv.wrappedValue = Linear(dimensions, dimensions * 3, bias: false)
        self._outProj.wrappedValue = Linear(dimensions, dimensions, bias: false)
        self._norm2.wrappedValue = RMSNorm(dimensions: dimensions)
        self._mlpIn.wrappedValue = Linear(dimensions, intermediate, bias: false)
        self._mlpOut.wrappedValue = Linear(intermediate, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        forwardCausal(x, mask: .none, cache: nil)
    }

    func forwardCausal(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        let hd = dimensions / heads
        let parts = qkv(norm1(x)).reshaped(b, n, 3, heads, hd).transposed(2, 0, 3, 1, 4)
        let attn = attentionWithCacheUpdate(
            queries: parts[0], keys: parts[1], values: parts[2], cache: cache, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(b, n, dimensions)
        let h = x + outProj(attn)
        return h + mlpOut(geluApproximate(mlpIn(norm2(h))))
    }
}

final class CrossDiTBlock: Module {
    let dimensions: Int
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "self_norm") var selfNorm: RMSNorm
    @ModuleInfo(key: "self_qkv") var selfQKV: Linear
    @ModuleInfo(key: "self_out") var selfOut: Linear
    @ModuleInfo(key: "cross_norm") var crossNorm: RMSNorm
    @ModuleInfo(key: "cross_q") var crossQ: Linear
    @ModuleInfo(key: "cross_kv") var crossKV: Linear
    @ModuleInfo(key: "cross_out") var crossOut: Linear
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "ffn_in") var ffnIn: Linear
    @ModuleInfo(key: "ffn_out") var ffnOut: Linear

    init(dimensions: Int, heads: Int, intermediate: Int) {
        self.dimensions = dimensions
        self.heads = heads
        self.scale = 1.0 / Float(dimensions / heads).squareRoot()
        self._selfNorm.wrappedValue = RMSNorm(dimensions: dimensions)
        self._selfQKV.wrappedValue = Linear(dimensions, dimensions * 3, bias: false)
        self._selfOut.wrappedValue = Linear(dimensions, dimensions, bias: false)
        self._crossNorm.wrappedValue = RMSNorm(dimensions: dimensions)
        self._crossQ.wrappedValue = Linear(dimensions, dimensions, bias: false)
        self._crossKV.wrappedValue = Linear(dimensions, dimensions * 2, bias: false)
        self._crossOut.wrappedValue = Linear(dimensions, dimensions, bias: false)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: dimensions)
        self._ffnIn.wrappedValue = Linear(dimensions, intermediate, bias: false)
        self._ffnOut.wrappedValue = Linear(intermediate, dimensions, bias: false)
    }

    func callAsFunction(visual: MLXArray, context: MLXArray) -> MLXArray {
        let b = visual.dim(0)
        let nV = visual.dim(1)
        let nC = context.dim(1)
        let hd = dimensions / heads
        let sParts = selfQKV(selfNorm(visual)).reshaped(b, nV, 3, heads, hd).transposed(2, 0, 3, 1, 4)
        let sAttn = MLXFast.scaledDotProductAttention(
            queries: sParts[0], keys: sParts[1], values: sParts[2], scale: scale, mask: .none
        ).transposed(0, 2, 1, 3).reshaped(b, nV, dimensions)
        let h1 = visual + selfOut(sAttn)
        let q = crossQ(crossNorm(h1)).reshaped(b, nV, heads, hd).transposed(0, 2, 1, 3)
        let kv = crossKV(context).reshaped(b, nC, 2, heads, hd).transposed(2, 0, 3, 1, 4)
        let cAttn = MLXFast.scaledDotProductAttention(
            queries: q, keys: kv[0], values: kv[1], scale: scale, mask: .none
        ).transposed(0, 2, 1, 3).reshaped(b, nV, dimensions)
        let h2 = h1 + crossOut(cAttn)
        return h2 + ffnOut(MLXNN.silu(ffnIn(ffnNorm(h2))))
    }

    func forwardCausal(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        let hd = dimensions / heads
        let sParts = selfQKV(selfNorm(x)).reshaped(b, n, 3, heads, hd).transposed(2, 0, 3, 1, 4)
        let sAttn = attentionWithCacheUpdate(
            queries: sParts[0], keys: sParts[1], values: sParts[2], cache: cache, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(b, n, dimensions)
        let h1 = x + selfOut(sAttn)
        return h1 + ffnOut(MLXNN.silu(ffnIn(ffnNorm(h1))))
    }
}

final class MoEVideoDiTBlock: Module {
    let dimensions: Int
    let heads: Int
    let numExperts: Int
    let topK: Int
    let scale: Float
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "router") var router: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(dimensions: Int, heads: Int, intermediate: Int, numExperts: Int, topK: Int) {
        self.dimensions = dimensions
        self.heads = heads
        self.numExperts = max(2, numExperts)
        self.topK = max(1, min(numExperts, topK))
        self.scale = 1.0 / Float(dimensions / heads).squareRoot()
        self._norm1.wrappedValue = RMSNorm(dimensions: dimensions)
        self._qkv.wrappedValue = Linear(dimensions, dimensions * 3, bias: false)
        self._outProj.wrappedValue = Linear(dimensions, dimensions, bias: false)
        self._norm2.wrappedValue = RMSNorm(dimensions: dimensions)
        self._router.wrappedValue = Linear(dimensions, self.numExperts, bias: false)
        self._switchMLP.wrappedValue = SwitchGLU(inputDims: dimensions, hiddenDims: intermediate, numExperts: self.numExperts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        forwardCausal(x, mask: .none, cache: nil)
    }

    func forwardCausal(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        let hd = dimensions / heads
        let parts = qkv(norm1(x)).reshaped(b, n, 3, heads, hd).transposed(2, 0, 3, 1, 4)
        let attn = attentionWithCacheUpdate(
            queries: parts[0], keys: parts[1], values: parts[2], cache: cache, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(b, n, dimensions)
        let h = x + outProj(attn)
        let normed = norm2(h)
        let gates = softmax(router(normed), axis: -1)
        let topIdx = argPartition(-gates, kth: topK - 1, axis: -1)[0..., 0..., ..<topK]
        let topWeights = takeAlong(gates, topIdx, axis: -1)
        let normWeights = topWeights / maximum(sum(topWeights, axis: -1, keepDims: true), MLXArray(Float(1e-6)))
        let expertOut = switchMLP(normed, topIdx)
        let routed = sum(expertOut * normWeights.expandedDimensions(axis: -1), axis: -2)
        return h + routed
    }
}

public final class LTX25AVDiTModel: Module, LLMModel, KVCacheDimensionProvider {
    public let configuration: VideoDiTConfiguration
    public var vocabularySize: Int { configuration.vocabSize }
    public var kvHeads: [Int] { Array(repeating: configuration.numAttentionHeads, count: configuration.numHiddenLayers) }
    public var loraLayers: [Module] { blocks }
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "video_in") var videoIn: Linear
    @ModuleInfo(key: "audio_in") var audioIn: Linear
    @ModuleInfo(key: "time_proj") var timeProj: Linear
    @ModuleInfo(key: "blocks") var blocks: [JointAVAttentionBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm
    @ModuleInfo(key: "video_out") var videoOut: Linear
    @ModuleInfo(key: "audio_out") var audioOut: Linear
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: VideoDiTConfiguration) {
        self.configuration = configuration
        let d = configuration.hiddenSize
        self._embedTokens.wrappedValue = Embedding(embeddingCount: configuration.vocabSize, dimensions: d)
        self._videoIn.wrappedValue = Linear(configuration.videoChannels, d)
        self._audioIn.wrappedValue = Linear(configuration.audioChannels, d)
        self._timeProj.wrappedValue = Linear(8, d)
        self._blocks.wrappedValue = (0 ..< configuration.numHiddenLayers).map { _ in
            JointAVAttentionBlock(dimensions: d, heads: configuration.numAttentionHeads, intermediate: configuration.intermediateSize)
        }
        self._finalNorm.wrappedValue = RMSNorm(dimensions: d)
        self._videoOut.wrappedValue = Linear(d, configuration.videoChannels)
        self._audioOut.wrappedValue = Linear(d, configuration.audioChannels)
        self._lmHead.wrappedValue = Linear(d, configuration.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = makeAttentionMask(n: h.dim(1), cache: cache?.first, windowSize: nil)
        for (index, block) in blocks.enumerated() {
            let layerCache = cache.flatMap { index < $0.count ? $0[index] : nil }
            h = block.forwardCausal(h, mask: mask, cache: layerCache)
        }
        return lmHead(finalNorm(h))
    }

    public func predictAudioVideoVelocity(
        videoLatents: MLXArray,
        audioLatents: MLXArray,
        textTokens: MLXArray,
        timestep: Float
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray) {
        let tBias = timeProj(VideoTimestepEmbedder.embed8(timestep)).expandedDimensions(axis: 1)
        let vTok = videoIn(videoLatents) + tBias
        let aTok = audioIn(audioLatents) + tBias
        let nTxt = textTokens.dim(1)
        let nVid = vTok.dim(1)
        var seq = concatenated([textTokens, vTok, aTok], axis: 1)
        for block in blocks { seq = block(seq) }
        let normed = finalNorm(seq)
        let vSlice = normed[0..., nTxt ..< (nTxt + nVid), 0...]
        let aSlice = normed[0..., (nTxt + nVid)..., 0...]
        return (videoOut(vSlice), audioOut(aSlice))
    }
}

public final class KandinskyCrossDiTModel: Module, LLMModel, KVCacheDimensionProvider {
    public let configuration: VideoDiTConfiguration
    public var vocabularySize: Int { configuration.vocabSize }
    public var kvHeads: [Int] { Array(repeating: configuration.numAttentionHeads, count: configuration.numHiddenLayers) }
    public var loraLayers: [Module] { blocks }
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "video_in") var videoIn: Linear
    @ModuleInfo(key: "audio_in") var audioIn: Linear
    @ModuleInfo(key: "blocks") var blocks: [CrossDiTBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm
    @ModuleInfo(key: "video_out") var videoOut: Linear
    @ModuleInfo(key: "audio_out") var audioOut: Linear
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: VideoDiTConfiguration) {
        self.configuration = configuration
        let d = configuration.hiddenSize
        self._embedTokens.wrappedValue = Embedding(embeddingCount: configuration.vocabSize, dimensions: d)
        self._videoIn.wrappedValue = Linear(configuration.videoChannels, d)
        self._audioIn.wrappedValue = Linear(configuration.audioChannels, d)
        self._blocks.wrappedValue = (0 ..< configuration.numHiddenLayers).map { _ in
            CrossDiTBlock(dimensions: d, heads: configuration.numAttentionHeads, intermediate: configuration.intermediateSize)
        }
        self._finalNorm.wrappedValue = RMSNorm(dimensions: d)
        self._videoOut.wrappedValue = Linear(d, configuration.videoChannels)
        self._audioOut.wrappedValue = Linear(d, configuration.audioChannels)
        self._lmHead.wrappedValue = Linear(d, configuration.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = makeAttentionMask(n: h.dim(1), cache: cache?.first, windowSize: nil)
        for (index, block) in blocks.enumerated() {
            let layerCache = cache.flatMap { index < $0.count ? $0[index] : nil }
            h = block.forwardCausal(h, mask: mask, cache: layerCache)
        }
        return lmHead(finalNorm(h))
    }

    public func predictCrossVelocity(
        videoLatents: MLXArray,
        audioLatents: MLXArray,
        contextTokens: MLXArray
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray) {
        let vTok = videoIn(videoLatents)
        let aTok = audioIn(audioLatents)
        let nVid = vTok.dim(1)
        var joint = concatenated([vTok, aTok], axis: 1)
        for block in blocks { joint = block(visual: joint, context: contextTokens) }
        let normed = finalNorm(joint)
        return (videoOut(normed[0..., ..<nVid, 0...]), audioOut(normed[0..., nVid..., 0...]))
    }
}

public final class MoEVideoDiTModel: Module, LLMModel, KVCacheDimensionProvider {
    public let configuration: VideoDiTConfiguration
    public var vocabularySize: Int { configuration.vocabSize }
    public var kvHeads: [Int] { Array(repeating: configuration.numAttentionHeads, count: configuration.numHiddenLayers) }
    public var loraLayers: [Module] { blocks }
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "video_in") var videoIn: Linear
    @ModuleInfo(key: "audio_in") var audioIn: Linear
    @ModuleInfo(key: "blocks") var blocks: [MoEVideoDiTBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm
    @ModuleInfo(key: "video_out") var videoOut: Linear
    @ModuleInfo(key: "audio_out") var audioOut: Linear
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: VideoDiTConfiguration) {
        self.configuration = configuration
        let d = configuration.hiddenSize
        self._embedTokens.wrappedValue = Embedding(embeddingCount: configuration.vocabSize, dimensions: d)
        self._videoIn.wrappedValue = Linear(configuration.videoChannels, d)
        self._audioIn.wrappedValue = Linear(configuration.audioChannels, d)
        self._blocks.wrappedValue = (0 ..< configuration.numHiddenLayers).map { _ in
            MoEVideoDiTBlock(
                dimensions: d,
                heads: configuration.numAttentionHeads,
                intermediate: configuration.intermediateSize,
                numExperts: configuration.numExperts,
                topK: configuration.numExpertsPerTok
            )
        }
        self._finalNorm.wrappedValue = RMSNorm(dimensions: d)
        self._videoOut.wrappedValue = Linear(d, configuration.videoChannels)
        self._audioOut.wrappedValue = Linear(d, configuration.audioChannels)
        self._lmHead.wrappedValue = Linear(d, configuration.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = makeAttentionMask(n: h.dim(1), cache: cache?.first, windowSize: nil)
        for (index, block) in blocks.enumerated() {
            let layerCache = cache.flatMap { index < $0.count ? $0[index] : nil }
            h = block.forwardCausal(h, mask: mask, cache: layerCache)
        }
        return lmHead(finalNorm(h))
    }

    public func predictMoEVelocity(
        videoLatents: MLXArray,
        audioLatents: MLXArray,
        textTokens: MLXArray
    ) -> (videoVelocity: MLXArray, audioVelocity: MLXArray) {
        let vTok = videoIn(videoLatents)
        let aTok = audioIn(audioLatents)
        let nTxt = textTokens.dim(1)
        let nVid = vTok.dim(1)
        var seq = concatenated([textTokens, vTok, aTok], axis: 1)
        for block in blocks { seq = block(seq) }
        let normed = finalNorm(seq)
        return (
            videoOut(normed[0..., nTxt ..< (nTxt + nVid), 0...]),
            audioOut(normed[0..., (nTxt + nVid)..., 0...])
        )
    }
}

enum VideoTimestepEmbedder {
    static func embed8(_ timestep: Float) -> MLXArray {
        let vals = (0 ..< 4).flatMap { idx -> [Float] in
            let freq = Float(pow(2.0, Double(idx))) * Float.pi
            return [sin(timestep * freq), cos(timestep * freq)]
        }
        return MLXArray(vals).reshaped(1, 8)
    }
}
