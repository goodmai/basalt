import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

package final class FlintDecoderLayer: Module {
    let layerIdx: Int
    @ModuleInfo(key: "self_attn") var selfAttn: FlintAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer

    package init(_ args: FlintConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self._selfAttn.wrappedValue = FlintAttention(args, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        if args.isSparseMoE(layer: layerIdx) {
            self._mlp.wrappedValue = FlintSparseMoEBlock(args)
        } else {
            self._mlp.wrappedValue = FlintMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }
    }

    package func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public final class FlintModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [FlintDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let args: FlintConfiguration
    private let firstFullAttentionLayer: Int?
    private let firstSlidingAttentionLayer: Int?

    public init(_ args: FlintConfiguration) {
        self.args = args
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< args.hiddenLayers).map { idx in
            FlintDecoderLayer(args, layerIdx: idx)
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
            h = layer(h, mask: layerMask, cache: layerCache)
        }
        return norm(h)
    }
}

public final class FlintModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public static let modelType = "flint"
    private static let defaultPrefillChunkSize = 128

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: FlintConfiguration

    @ModuleInfo(key: "model") public var model: FlintModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: FlintConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self._model.wrappedValue = FlintModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        } else {
            self._lmHead.wrappedValue = nil
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }

    public var toolCallFormat: ToolCallFormat? {
        .glm4
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
        dropIgnoredTensors(&out)
        stackUnmergedExpertsIfPresent(&out)
        canonicalizeExpertBias(&out)
        if hasLangPrefix {
            Self.quantizeModulesMatchingPackedWeights(in: self, weights: out)
        }
        return out
    }
}

extension FlintModel {
    private func dropIgnoredTensors(_ out: inout [String: MLXArray]) {
        if configuration.tieWordEmbeddings {
            for key in Array(out.keys) where key.hasPrefix("lm_head.") {
                out.removeValue(forKey: key)
            }
        }
        for key in Array(out.keys)
            where key.contains("rotary_emb.inv_freq")
                || key.hasPrefix("mtp.")
                || key.contains(".mtp.")
                || key.hasPrefix("vision_tower.")
        {
            out.removeValue(forKey: key)
        }
    }

    private func stackUnmergedExpertsIfPresent(_ out: inout [String: MLXArray]) {
        for layerIdx in 0 ..< configuration.hiddenLayers where configuration.isSparseMoE(layer: layerIdx) {
            let prefix = "model.layers.\(layerIdx)"
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    guard out["\(prefix).mlp.experts.0.\(proj).\(suffix)"] != nil else { continue }
                    var slices: [MLXArray] = []
                    slices.reserveCapacity(configuration.numExperts)
                    var complete = true
                    for expertIdx in 0 ..< configuration.numExperts {
                        guard let tensor = out.removeValue(forKey: "\(prefix).mlp.experts.\(expertIdx).\(proj).\(suffix)") else {
                            complete = false
                            break
                        }
                        slices.append(tensor)
                    }
                    if complete, slices.count == configuration.numExperts {
                        out["\(prefix).mlp.switch_mlp.\(proj).\(suffix)"] = MLX.stacked(slices)
                    }
                }
            }
        }
    }

    private func canonicalizeExpertBias(_ out: inout [String: MLXArray]) {
        for layerIdx in 0 ..< configuration.hiddenLayers where configuration.isSparseMoE(layer: layerIdx) {
            let prefix = "model.layers.\(layerIdx)"
            if let gateBias = out.removeValue(forKey: "\(prefix).mlp.gate.expert_bias") {
                out["\(prefix).mlp.expert_bias"] = gateBias
            } else if out["\(prefix).mlp.expert_bias"] == nil {
                out["\(prefix).mlp.expert_bias"] = MLXArray.zeros([configuration.numExperts], dtype: .float32)
            }
        }
    }

    static func quantizeModulesMatchingPackedWeights(in root: Module, weights: [String: MLXArray]) {
        quantize(model: root) { path, module in
            guard !(module is any Quantized),
                  let packedWeight = weights["\(path).weight"],
                  let scales = weights["\(path).scales"],
                  packedWeight.dtype == .uint32,
                  case .value(let unquantizedWeight) = module.parameters()["weight"]
            else {
                return nil
            }
            let origInCols = unquantizedWeight.dim(-1)
            let groups = scales.dim(-1)
            let packedCols = packedWeight.dim(-1)
            guard origInCols > 0, groups > 0, origInCols.isMultiple(of: groups) else { return nil }
            let groupSize = origInCols / groups
            let totalBits = packedCols * 32
            guard totalBits.isMultiple(of: origInCols) else { return nil }
            let bits = totalBits / origInCols
            guard [2, 3, 4, 5, 6, 8].contains(bits) else { return nil }
            let mode: QuantizationMode = weights["\(path).biases"] != nil ? .affine : .mxfp4
            return (groupSize, bits, mode)
        }
    }
}
