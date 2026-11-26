import BasaltCore
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public final class HyperConnectionAttention: Module {
    let layerIdx: Int
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    public init(_ config: HyperConnectionConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Float(config.headDim).squareRoot()
        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        self._wq.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * config.headDim, bias: false)
        self._wk.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        self._wv.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        self._wo.wrappedValue = Linear(config.numAttentionHeads * config.headDim, config.hiddenSize, bias: false)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        probe: (any ForwardProbe)? = nil
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let offset = cache?.ropeOffset
        let queries = applyRotaryPosition(
            rope,
            to: wq(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3),
            offset: offset
        )
        let keys = applyRotaryPosition(
            rope,
            to: wk(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3),
            offset: offset
        )
        let values = wv(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        var attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        if let probe {
            if let hMask = probe.headMask(layer: layerIdx) {
                let perHead = hMask.ndim == 2
                    ? hMask.transposed().expandedDimensions(axes: [0, 3])
                    : hMask.reshaped(1, -1, 1, 1)
                attended = attended * perHead.asType(attended.dtype)
            }
            probe.headOutputs(layer: layerIdx, attended)
        }
        return wo(attended.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

public final class HyperConnectionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    public init(_ config: HyperConnectionConfiguration) {
        self._gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

public final class HyperConnectionDecoderLayer: Module {
    public let layerIdx: Int

    @ModuleInfo(key: "self_attn") public var selfAttn: HyperConnectionAttention
    @ModuleInfo(key: "mlp") public var mlp: HyperConnectionMLP
    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "hc_attn") public var hcAttn: HyperConnectionSublayer
    @ModuleInfo(key: "hc_mlp") public var hcMlp: HyperConnectionSublayer

    public init(_ config: HyperConnectionConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self._selfAttn.wrappedValue = HyperConnectionAttention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = HyperConnectionMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._hcAttn.wrappedValue = HyperConnectionSublayer(
            hiddenSize: config.hiddenSize,
            spec: config.hyperConnection,
            rmsNormEps: config.rmsNormEps
        )
        self._hcMlp.wrappedValue = HyperConnectionSublayer(
            hiddenSize: config.hiddenSize,
            spec: config.hyperConnection,
            rmsNormEps: config.rmsNormEps
        )
    }

    public func callAsFunction(
        _ streams: MLXArray,
        computeDType: DType,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        probe: (any ForwardProbe)? = nil
    ) -> MLXArray {
        let afterAttn = hcAttn(streams, computeDType: computeDType) { hIn in
            selfAttn(inputLayerNorm(hIn), mask: mask, cache: cache, probe: probe)
        }
        return hcMlp(afterAttn, computeDType: computeDType) { hIn in
            mlp(postAttentionLayerNorm(hIn))
        }
    }
}

public final class HyperConnectionModelInner: Module {
    public let configuration: HyperConnectionConfiguration

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [HyperConnectionDecoderLayer]
    @ModuleInfo(key: "norm") public var norm: RMSNorm

    public init(_ config: HyperConnectionConfiguration) {
        self.configuration = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { idx in
            HyperConnectionDecoderLayer(config, layerIdx: idx)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        probe: (any ForwardProbe)? = nil
    ) -> MLXArray {
        let embedded = embedTokens(inputs)
        let computeDType = embedded.dtype
        let mask = createAttentionMask(h: embedded, cache: cache?.first)
        let hc = configuration.hyperConnection
        var streams = HyperConnectionSublayer.expandStreams(embedded, rate: hc.expansionRate)
        for (idx, layer) in layers.enumerated() {
            let layerCache = cache.flatMap { idx < $0.count ? $0[idx] : nil }
            streams = layer(streams, computeDType: computeDType, mask: mask, cache: layerCache, probe: probe)
            if let probe {
                let collapsed = HyperConnectionSublayer.collapseStreams(streams, mode: hc.collapseMode)
                let probeInput = collapsed.asType(computeDType)
                let edited = probe.residual(layer: idx, probeInput)
                streams = HyperConnectionSublayer.distributeProbeDelta(
                    streams: streams,
                    collapsed: probeInput,
                    edited: edited,
                    mode: hc.collapseMode
                )
            }
        }
        let collapsed = HyperConnectionSublayer.collapseStreams(streams, mode: hc.collapseMode)
        return norm(collapsed.asType(computeDType))
    }
}

public final class HyperConnectionModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel, BasaltProbableModel {
    public static let modelType = "mhc"
    public static let aliasModelTypes = ["hyper_connection", "ihc"]
    public static let modelTypes = [modelType] + aliasModelTypes

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: HyperConnectionConfiguration

    @ModuleInfo(key: "model") public var model: HyperConnectionModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: HyperConnectionConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self._model.wrappedValue = HyperConnectionModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()
        train(false)
    }

    public convenience init(configurationData data: Data) throws {
        let config = try JSONDecoder.json5().decode(HyperConnectionConfiguration.self, from: data).validated()
        self.init(config)
    }

    public var loraLayers: [Module] {
        model.layers
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forwardWithProbe(inputs, cache: cache, probe: nil)
    }

    package func forwardWithProbe(_ inputs: MLXArray, cache: [KVCache]?, probe: (any ForwardProbe)?) -> MLXArray {
        let hidden = model(inputs, cache: cache, probe: probe)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try (0 ..< configuration.numHiddenLayers).map { _ in
            try makeAttentionKVCache(parameters: parameters)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let filtered = weights.filter { key, _ in
            !key.contains("rotary_emb.inv_freq")
                && !(configuration.tieWordEmbeddings && key.hasPrefix("lm_head."))
        }
        FlintModel.quantizeModulesMatchingPackedWeights(in: self, weights: filtered)
        return filtered
    }

    public func setForcePureMLX(_ force: Bool) {
        for layer in model.layers {
            layer.hcAttn.forcePureMLX = force
            layer.hcMlp.forcePureMLX = force
        }
    }
}

extension MLXInferenceEngine {
    static func registerHyperConnectionAliases() async {
        for type in HyperConnectionModel.modelTypes {
            await LLMTypeRegistry.shared.registerModelType(type) { data in
                try HyperConnectionModel(configurationData: data)
            }
        }
    }
}
