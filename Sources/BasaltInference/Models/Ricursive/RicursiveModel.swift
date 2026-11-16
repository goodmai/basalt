import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

package final class RicursiveDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: RicursiveAttention
    @ModuleInfo(key: "mlp") var mlp: RicursiveMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    package init(_ args: RicursiveConfiguration, layerIdx: Int) {
        self._selfAttn.wrappedValue = RicursiveAttention(args, layerIdx: layerIdx)
        self._mlp.wrappedValue = RicursiveMLP(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    package func callAsFunction(
        _ x: MLXArray,
        valueResidual: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attnOut = selfAttn(
            inputLayerNorm(x),
            valueResidual: valueResidual,
            mask: mask,
            cache: cache
        )
        let afterAttn = x + postAttentionLayerNorm(attnOut)
        let ffOut = mlp(preFeedforwardLayerNorm(afterAttn))
        return afterAttn + postFeedforwardLayerNorm(ffOut)
    }
}

public final class RicursiveModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [RicursiveDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let args: RicursiveConfiguration
    private let firstFullAttentionLayer: Int?
    private let firstSlidingAttentionLayer: Int?

    public init(_ args: RicursiveConfiguration) {
        self.args = args
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< args.hiddenLayers).map { idx in
            RicursiveDecoderLayer(args, layerIdx: idx)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.firstFullAttentionLayer = (0 ..< args.hiddenLayers).first { !args.usesSlidingWindow(layer: $0) }
        self.firstSlidingAttentionLayer = (0 ..< args.hiddenLayers).first { args.usesSlidingWindow(layer: $0) }
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)
        if let embedScale = args.embedScale {
            h = (h.asType(.float32) * embedScale).asType(h.dtype)
        }
        let valueResidual = args.useValres ? h : nil
        let seqLen = h.dim(1)
        let windowSize = args.effectiveSlidingWindow
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?

        for (idx, layer) in layers.enumerated() {
            let layerMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                layerMask = mask
            } else if args.usesSlidingWindow(layer: idx) {
                if slidingMask == nil {
                    let repCache = firstSlidingAttentionLayer.flatMap { slot in
                        cache.flatMap { slot < $0.count ? $0[slot] : nil }
                    }
                    slidingMask = makeAttentionMask(n: seqLen, cache: repCache, windowSize: windowSize)
                }
                layerMask = slidingMask!
            } else {
                if fullMask == nil {
                    let repCache = firstFullAttentionLayer.flatMap { slot in
                        cache.flatMap { slot < $0.count ? $0[slot] : nil }
                    }
                    fullMask = makeAttentionMask(n: seqLen, cache: repCache, windowSize: nil)
                }
                layerMask = fullMask!
            }
            let layerCache = cache.flatMap { idx < $0.count ? $0[idx] : nil }
            h = layer(h, valueResidual: valueResidual, mask: layerMask, cache: layerCache)
        }
        return norm(h)
    }
}

public final class RicursiveModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public static let modelType = "ricursive"
    public static let aliasModelType = "ripple"
    private static let defaultPrefillChunkSize = 128

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: RicursiveConfiguration

    @ModuleInfo(key: "model") public var model: RicursiveModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: RicursiveConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self._model.wrappedValue = RicursiveModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }

    public var toolCallFormat: ToolCallFormat? {
        .json
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try (0 ..< configuration.hiddenLayers).map { idx in
            if configuration.usesSlidingWindow(layer: idx) {
                return try makeHybridAttentionKVCache(
                    parameters: parameters,
                    slidingWindow: configuration.effectiveSlidingWindow,
                    usesSlidingWindow: true
                )
            }
            let cache = try makeAttentionKVCache(parameters: parameters)
            if let standard = cache as? StandardKVCache {
                standard.step = 1024
            }
            return cache
        }
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        state _: LMOutput.State? = nil,
        prefill: PrefillParameters = .init()
    ) throws -> PrepareResult {
        let tokens = input.text
        guard tokens.tokens.dim(0) > 0 else {
            return .tokens(.init(tokens: MLXArray(Int32(0))[0 ..< 0]))
        }
        let maxChunk = configuration.effectiveSlidingWindow ?? Self.defaultPrefillChunkSize
        let processed = try prefill.forEachChunk(
            total: tokens.tokens.size,
            defaultStepSize: min(Self.defaultPrefillChunkSize, maxChunk),
            maximumStepSize: maxChunk
        ) { range in
            _ = model(tokens[.newAxis, range].tokens, mask: nil, cache: cache)
            asyncEval(cache)
        }
        return .tokens(tokens[processed...])
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let hasLangPrefix = weights.keys.contains { $0.hasPrefix("language_model.") }
        var out: [String: MLXArray]
        if hasLangPrefix {
            out = Dictionary(minimumCapacity: weights.count)
            for (key, value) in weights {
                let stripped = key.hasPrefix("language_model.")
                    ? String(key.dropFirst("language_model.".count))
                    : key
                out[stripped] = value
            }
        } else {
            out = weights
        }
        if configuration.tieWordEmbeddings {
            for key in Array(out.keys) where key.hasPrefix("lm_head.") {
                out.removeValue(forKey: key)
            }
        }
        for key in Array(out.keys) where key.contains("rotary_emb.inv_freq") || key.hasPrefix("mtp.") {
            out.removeValue(forKey: key)
        }
        if hasLangPrefix {
            FlintModel.quantizeModulesMatchingPackedWeights(in: self, weights: out)
        }
        return out
    }
}
