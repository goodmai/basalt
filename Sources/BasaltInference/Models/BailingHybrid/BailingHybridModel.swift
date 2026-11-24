import BasaltCore
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM

package final class BailingHybridDecoderLayer: Module {
    let isMLA: Bool

    @ModuleInfo(key: "attention") var attention: Module
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    package init(_ config: BailingHybridConfiguration, layerIdx: Int) {
        self.isMLA = config.isMLALayer(layerIdx)
        if isMLA {
            self._attention.wrappedValue = BailingHybridMLAAttention(config, layerIdx: layerIdx)
        } else {
            self._attention.wrappedValue = BailingHybridLinearAttention(config, layerIdx: layerIdx)
        }
        if config.isMoELayer(layerIdx) {
            self._mlp.wrappedValue = BailingHybridSparseMoE(config, layerIdx: layerIdx)
        } else {
            self._mlp.wrappedValue = BailingHybridMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                bias: config.useBias
            )
        }
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    package func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        probe: (any ForwardProbe)? = nil
    ) -> MLXArray {
        let normedInput = inputLayerNorm(x)
        let attnOut: MLXArray
        if let mla = attention as? BailingHybridMLAAttention {
            attnOut = mla(normedInput, mask: mask, cache: cache, probe: probe)
        } else if let kda = attention as? BailingHybridLinearAttention {
            attnOut = kda(normedInput, cache: cache as? ArraysCache, probe: probe)
        } else {
            preconditionFailure("Unexpected attention module in BailingHybridDecoderLayer")
        }
        let hidden = x + attnOut
        let normedPost = postAttentionLayerNorm(hidden)
        let mlpOut: MLXArray
        if let moe = mlp as? BailingHybridSparseMoE {
            mlpOut = moe(normedPost, probe: probe)
        } else if let dense = mlp as? BailingHybridMLP {
            mlpOut = dense(normedPost)
        } else {
            preconditionFailure("Unexpected MLP module in BailingHybridDecoderLayer")
        }
        return hidden + mlpOut
    }
}

public final class BailingHybridModelInner: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [BailingHybridDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let firstMLAIndex: Int?

    public init(_ config: BailingHybridConfiguration) {
        self._wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        let builtLayers = (0 ..< config.numHiddenLayers).map { index in
            BailingHybridDecoderLayer(config, layerIdx: index)
        }
        self._layers.wrappedValue = builtLayers
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.firstMLAIndex = builtLayers.firstIndex(where: \.isMLA)
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        probe: (any ForwardProbe)? = nil
    ) -> MLXArray {
        var hidden = wordEmbeddings(inputs)
        let mlaCache = firstMLAIndex.flatMap { slot in
            cache.flatMap { slot < $0.count ? $0[slot] : nil }
        }
        let mlaMask = createAttentionMask(h: hidden, cache: mlaCache)
        for (index, layer) in layers.enumerated() {
            let layerCache = cache.flatMap { index < $0.count ? $0[index] : nil }
            let layerMask: MLXFast.ScaledDotProductAttentionMaskMode = layer.isMLA ? mlaMask : .none
            hidden = layer(hidden, mask: layerMask, cache: layerCache, probe: probe)
            if let probe {
                hidden = probe.residual(layer: index, hidden)
            }
        }
        return norm(hidden)
    }
}

public final class BailingHybridModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel, BasaltProbableModel {
    public static let modelType = "bailing_hybrid"

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: BailingHybridConfiguration

    @ModuleInfo(key: "model") public var model: BailingHybridModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: BailingHybridConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { config.isMLALayer($0) ? 1 : config.numAttentionHeads }
        self._model.wrappedValue = BailingHybridModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public convenience init(configurationData data: Data) throws {
        let config = try JSONDecoder.json5().decode(BailingHybridConfiguration.self, from: data).validated()
        self.init(config)
    }

    public var loraLayers: [Module] {
        model.layers
    }

    public var toolCallFormat: ToolCallFormat? {
        .glm4
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forwardWithProbe(inputs, cache: cache, probe: nil)
    }

    package func forwardWithProbe(_ inputs: MLXArray, cache: [KVCache]?, probe: (any ForwardProbe)?) -> MLXArray {
        let hidden = model(inputs, cache: cache, probe: probe)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.wordEmbeddings.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try model.layers.map { layer in
            layer.isMLA ? try makeAttentionKVCache(parameters: parameters) : ArraysCache(size: 4)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let mapped = BailingHybridCheckpoint.sanitize(weights, config: configuration)
        FlintModel.quantizeModulesMatchingPackedWeights(in: self, weights: mapped)
        return mapped
    }
}

enum BailingHybridCheckpoint {
    private static let projections = ["gate_proj", "up_proj", "down_proj"]
    private static let quantParts = ["weight", "scales", "biases"]

    static func sanitize(_ weights: [String: MLXArray], config: BailingHybridConfiguration) -> [String: MLXArray] {
        var out = weights.filter { key, _ in !shouldDrop(key, config: config) }
        stackExpertsIfUnmerged(&out, config: config)
        canonicalizeShortConvs(&out, kernelSize: config.convKernelSize)
        return out
    }

    private static func shouldDrop(_ key: String, config: BailingHybridConfiguration) -> Bool {
        if config.tieWordEmbeddings && key.hasPrefix("lm_head.") {
            return true
        }
        if !config.moeRouterEnableExpertBias && key.hasSuffix(".mlp.gate.expert_bias") {
            return true
        }
        if key.contains("rotary_emb.inv_freq")
            || key.hasPrefix("mtp_")
            || key.contains(".mtp_")
            || key.hasPrefix("mtp.")
            || key.contains(".mtp.")
        {
            return true
        }
        let parts = key.split(separator: ".", maxSplits: 3)
        if parts.count >= 3, parts[0] == "model", parts[1] == "layers",
           let layerIndex = Int(parts[2]), layerIndex >= config.numHiddenLayers
        {
            return true
        }
        return false
    }

    private static func stackExpertsIfUnmerged(
        _ out: inout [String: MLXArray],
        config: BailingHybridConfiguration
    ) {
        for layerIdx in 0 ..< config.numHiddenLayers where config.isMoELayer(layerIdx) {
            let prefix = "model.layers.\(layerIdx).mlp.experts"
            for projection in projections {
                for part in quantParts {
                    stackExpertPart(&out, prefix: prefix, projection: projection, part: part, count: config.numExperts)
                }
            }
        }
    }

    private static func stackExpertPart(
        _ out: inout [String: MLXArray],
        prefix: String,
        projection: String,
        part: String,
        count: Int
    ) {
        guard count > 0, out["\(prefix).0.\(projection).\(part)"] != nil else { return }
        let keys = (0 ..< count).map { "\(prefix).\($0).\(projection).\(part)" }
        guard keys.allSatisfy({ out[$0] != nil }) else { return }
        let slices = keys.compactMap { out.removeValue(forKey: $0) }
        out["\(prefix).\(projection).\(part)"] = MLX.stacked(slices)
    }

    private static func canonicalizeShortConvs(_ out: inout [String: MLXArray], kernelSize: Int) {
        for key in Array(out.keys) {
            if key.hasSuffix("_conv1d.weight"), let weight = out.removeValue(forKey: key) {
                let target = String(key.dropLast(".weight".count)) + ".conv.weight"
                out[target] = normalizeConvWeight(weight, kernelSize: kernelSize)
            } else if key.hasSuffix("_conv1d.conv.weight"), let weight = out[key] {
                out[key] = normalizeConvWeight(weight, kernelSize: kernelSize)
            }
        }
    }

    private static func normalizeConvWeight(_ weight: MLXArray, kernelSize: Int) -> MLXArray {
        if weight.ndim == 2 {
            return weight.expandedDimensions(axis: -1)
        }
        if weight.ndim == 3 && weight.dim(1) == 1 && weight.dim(2) == kernelSize && kernelSize != 1 {
            return weight.swappedAxes(1, 2)
        }
        return weight
    }
}

extension MLXInferenceEngine {
    static func registerBailingHybridAliases() async {
        await LLMTypeRegistry.shared.registerModelType(BailingHybridModel.modelType) { data in
            try BailingHybridModel(configurationData: data)
        }
        await VLMTypeRegistry.shared.registerModelType(BailingHybridModel.modelType) { data in
            try BailingHybridModel(configurationData: data)
        }
    }
}
