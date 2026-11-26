import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public struct RWKV7Configuration: Decodable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let headDim: Int
    public let numHeads: Int
    public let decayLowRankDim: Int
    public let aLowRankDim: Int
    public let vLowRankDim: Int
    public let gLowRankDim: Int
    public let intermediateSize: Int
    public let normEps: Float
    public let groupNormEps: Float
    public let normFirst: Bool
    public let normBias: Bool
    public let tieWordEmbeddings: Bool
    public let fuseCrossEntropy: Bool

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size", hiddenSize = "hidden_size", nEmbd = "n_embd"
        case numHiddenLayers = "num_hidden_layers", nLayer = "n_layer"
        case headDim = "head_dim", headSize = "head_size", numHeads = "num_heads", nHead = "n_head"
        case decayLowRankDim = "decay_low_rank_dim", wLoraDim = "w_lora_dim"
        case aLowRankDim = "a_low_rank_dim", aLoraDim = "a_lora_dim"
        case vLowRankDim = "v_low_rank_dim", vLoraDim = "v_lora_dim"
        case gLowRankDim = "g_low_rank_dim", gateLowRankDim = "gate_low_rank_dim", gLoraDim = "g_lora_dim"
        case intermediateSize = "intermediate_size", dimFFN = "dim_ffn", hiddenRatio = "hidden_ratio"
        case normEps = "norm_eps", layerNormEpsilon = "layer_norm_epsilon", rmsNormEps = "rms_norm_eps"
        case groupNormEps = "group_norm_eps", normFirst = "norm_first", normBias = "norm_bias"
        case tieWordEmbeddings = "tie_word_embeddings", fuseCrossEntropy = "fuse_cross_entropy"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try Self.decodeRequired(c, keys: [.hiddenSize, .nEmbd])
        numHiddenLayers = try Self.decodeRequired(c, keys: [.numHiddenLayers, .nLayer])
        (headDim, numHeads) = try Self.resolveHeads(c, hiddenSize: hiddenSize)
        let defaultDecay = max(1, min(64, hiddenSize))
        decayLowRankDim = try Self.decodeFirst(c, keys: [.decayLowRankDim, .wLoraDim]) ?? defaultDecay
        aLowRankDim = try Self.decodeFirst(c, keys: [.aLowRankDim, .aLoraDim]) ?? defaultDecay
        vLowRankDim = try Self.decodeFirst(c, keys: [.vLowRankDim, .vLoraDim]) ?? max(1, min(32, hiddenSize))
        gLowRankDim = try Self.decodeFirst(c, keys: [.gLowRankDim, .gateLowRankDim, .gLoraDim]) ?? max(1, min(128, hiddenSize))
        let ratio: Float = try Self.decodeFirst(c, keys: [.hiddenRatio]) ?? 4.0
        intermediateSize = try Self.decodeFirst(c, keys: [.intermediateSize, .dimFFN]) ?? Int((Float(hiddenSize) * ratio).rounded())
        normEps = try Self.decodeFirst(c, keys: [.normEps, .layerNormEpsilon, .rmsNormEps]) ?? 1e-5
        groupNormEps = try Self.decodeFirst(c, keys: [.groupNormEps]) ?? normEps * Float(headDim)
        normFirst = try c.decodeIfPresent(Bool.self, forKey: .normFirst) ?? true
        normBias = try c.decodeIfPresent(Bool.self, forKey: .normBias) ?? true
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        fuseCrossEntropy = try c.decodeIfPresent(Bool.self, forKey: .fuseCrossEntropy) ?? true
        try validate(in: c)
    }

    private func validate(in c: KeyedDecodingContainer<CodingKeys>) throws {
        let product = numHeads.multipliedReportingOverflow(by: headDim)
        let headsValid = numHeads >= 1 && headDim >= 1 && !product.overflow && hiddenSize == product.partialValue
        let checks: [(Bool, CodingKeys, String)] = [
            (vocabSize >= 1, .vocabSize, "rwkv7 requires vocab_size >= 1"),
            (hiddenSize >= 1, .hiddenSize, "rwkv7 requires hidden_size >= 1"),
            (numHiddenLayers >= 1, .numHiddenLayers, "rwkv7 requires num_hidden_layers >= 1"),
            (headsValid, .headDim, "rwkv7 requires num_heads >= 1, head_dim >= 1 and hidden_size == num_heads * head_dim"),
            (decayLowRankDim >= 1 && aLowRankDim >= 1 && vLowRankDim >= 1 && gLowRankDim >= 1, .decayLowRankDim,
             "rwkv7 requires positive low-rank projection dimensions"),
            (intermediateSize >= 1, .intermediateSize, "rwkv7 requires intermediate_size >= 1"),
            (normEps > 0 && groupNormEps > 0, .normEps, "rwkv7 requires positive normalization epsilons"),
        ]
        for (valid, key, message) in checks where !valid {
            throw DecodingError.dataCorruptedError(forKey: key, in: c, debugDescription: message)
        }
    }

    private static func decodeFirst<T: Decodable>(_ c: KeyedDecodingContainer<CodingKeys>, keys: [CodingKeys]) throws -> T? {
        for key in keys {
            if let value = try c.decodeIfPresent(T.self, forKey: key) { return value }
        }
        return nil
    }

    private static func decodeRequired(_ c: KeyedDecodingContainer<CodingKeys>, keys: [CodingKeys]) throws -> Int {
        if let value: Int = try decodeFirst(c, keys: keys) { return value }
        return try c.decode(Int.self, forKey: keys[0])
    }

    private static func resolveHeads(_ c: KeyedDecodingContainer<CodingKeys>, hiddenSize: Int) throws -> (Int, Int) {
        let explicitHeadDim: Int? = try decodeFirst(c, keys: [.headDim, .headSize])
        let explicitNumHeads: Int? = try decodeFirst(c, keys: [.numHeads, .nHead])
        if let h = explicitNumHeads, h < 1 {
            throw DecodingError.dataCorruptedError(forKey: .numHeads, in: c, debugDescription: "rwkv7 requires num_heads >= 1")
        }
        if let d = explicitHeadDim {
            guard d >= 1 else { throw DecodingError.dataCorruptedError(forKey: .headDim, in: c, debugDescription: "rwkv7 requires head_dim >= 1") }
            return (d, hiddenSize / d)
        }
        if let h = explicitNumHeads {
            return (hiddenSize / h, h)
        }
        return (64, hiddenSize / 64)
    }
}

struct RWKV7RecurrenceInputs {
    let r, v, w, a, kk, kEff: MLXArray
}

enum RWKV7Ops {
    static let decayScale: Float = -0.6065306597126334

    static func tokenShift(_ x: MLXArray, previous: MLXArray?) -> (delta: MLXArray, last: MLXArray) {
        let (b, l, c) = (x.dim(0), x.dim(1), x.dim(2))
        let prev = previous?.asType(x.dtype) ?? MLXArray.zeros([b, 1, c], dtype: x.dtype)
        let shifted = l == 1 ? prev : concatenated([prev, x[0..., ..<(l - 1), 0...]], axis: 1)
        let last = l == 1 ? x : (x[0..., (l - 1)..., 0...] + MLXArray(Float(0)).asType(x.dtype))
        return (shifted - x, last)
    }

    static func l2Normalize(_ x: MLXArray, eps: Float = 1e-12) -> MLXArray {
        let xf = x.asType(.float32)
        return xf / maximum(sqrt((xf * xf).sum(axis: -1, keepDims: true)), MLXArray(eps, dtype: .float32))
    }

    static func wkv7Recurrence(_ inputs: RWKV7RecurrenceInputs, initialState: MLXArray?) -> (output: MLXArray, state: MLXArray) {
        let (b, l, h, d) = (inputs.r.dim(0), inputs.r.dim(1), inputs.r.dim(2), inputs.r.dim(3))
        var state = initialState?.asType(.float32) ?? MLXArray.zeros([b, h, d, d], dtype: .float32)
        guard l > 0 else { return (MLXArray.zeros([b, 0, h, d], dtype: .float32), state) }
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(l)
        for step in 0 ..< l {
            let (rt, vt) = (inputs.r[0..., step, 0..., 0...], inputs.v[0..., step, 0..., 0...])
            let (wt, at) = (inputs.w[0..., step, 0..., 0...], inputs.a[0..., step, 0..., 0...])
            let (kkt, kefft) = (inputs.kk[0..., step, 0..., 0...], inputs.kEff[0..., step, 0..., 0...])
            let sa = matmul(state, kkt.expandedDimensions(axis: -1))
            state = state * wt.expandedDimensions(axis: -2)
                - sa * (at * kkt).expandedDimensions(axis: -2)
                + vt.expandedDimensions(axis: -1) * kefft.expandedDimensions(axis: -2)
            outputs.append(matmul(state, rt.expandedDimensions(axis: -1)).squeezed(axis: -1))
        }
        return (MLX.stacked(outputs, axis: 1), state)
    }
}

enum RWKV7LoRAActivation {
    case none, tanh, sigmoid
}

final class RWKV7LoRA: Module {
    let activation: RWKV7LoRAActivation
    @ModuleInfo(key: "lora_a") var loraA: Linear
    @ModuleInfo(key: "lora_b") var loraB: Linear

    init(dims: Int, lowRankDim: Int, activation: RWKV7LoRAActivation, bias: Bool) {
        self.activation = activation
        _loraA.wrappedValue = Linear(dims, lowRankDim, bias: false)
        _loraB.wrappedValue = Linear(lowRankDim, dims, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = loraA(x)
        let activated = switch activation {
        case .none: hidden
        case .tanh: tanh(hidden)
        case .sigmoid: sigmoid(hidden)
        }
        return loraB(activated)
    }
}

final class RWKV7TimeMix: Module {
    let layerIdx: Int
    let numHeads: Int
    let headDim: Int

    @ParameterInfo(key: "x_r") var xR: MLXArray
    @ParameterInfo(key: "x_w") var xW: MLXArray
    @ParameterInfo(key: "x_k") var xK: MLXArray
    @ParameterInfo(key: "x_v") var xV: MLXArray
    @ParameterInfo(key: "x_a") var xA: MLXArray
    @ParameterInfo(key: "x_g") var xG: MLXArray
    @ParameterInfo(key: "k_k") var kK: MLXArray
    @ParameterInfo(key: "k_a") var kA: MLXArray
    @ParameterInfo(key: "r_k") var rK: MLXArray

    @ModuleInfo(key: "r_proj") var rProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "w_lora") var wLora: RWKV7LoRA
    @ModuleInfo(key: "a_lora") var aLora: RWKV7LoRA
    @ModuleInfo(key: "v_lora") var vLora: RWKV7LoRA?
    @ModuleInfo(key: "g_lora") var gLora: RWKV7LoRA
    @ModuleInfo(key: "g_norm") var gNorm: GroupNorm

    init(_ config: RWKV7Configuration, layerIdx: Int) {
        (self.layerIdx, self.numHeads, self.headDim) = (layerIdx, config.numHeads, config.headDim)
        let c = config.hiddenSize
        _xR.wrappedValue = MLXArray.zeros([1, 1, c], dtype: .float32)
        _xW.wrappedValue = MLXArray.zeros([1, 1, c], dtype: .float32)
        _xK.wrappedValue = MLXArray.zeros([1, 1, c], dtype: .float32)
        _xV.wrappedValue = MLXArray.zeros([1, 1, c], dtype: .float32)
        _xA.wrappedValue = MLXArray.zeros([1, 1, c], dtype: .float32)
        _xG.wrappedValue = MLXArray.zeros([1, 1, c], dtype: .float32)
        _kK.wrappedValue = MLXArray.ones([1, 1, c], dtype: .float32)
        _kA.wrappedValue = MLXArray.ones([1, 1, c], dtype: .float32)
        _rK.wrappedValue = MLXArray.zeros([config.numHeads, config.headDim], dtype: .float32)
        _rProj.wrappedValue = Linear(c, c, bias: false)
        _kProj.wrappedValue = Linear(c, c, bias: false)
        _vProj.wrappedValue = Linear(c, c, bias: false)
        _oProj.wrappedValue = Linear(c, c, bias: false)
        _wLora.wrappedValue = RWKV7LoRA(dims: c, lowRankDim: config.decayLowRankDim, activation: .tanh, bias: true)
        _aLora.wrappedValue = RWKV7LoRA(dims: c, lowRankDim: config.aLowRankDim, activation: .none, bias: true)
        if layerIdx > 0 {
            _vLora.wrappedValue = RWKV7LoRA(dims: c, lowRankDim: config.vLowRankDim, activation: .none, bias: true)
        }
        _gLora.wrappedValue = RWKV7LoRA(dims: c, lowRankDim: config.gLowRankDim, activation: .sigmoid, bias: false)
        _gNorm.wrappedValue = GroupNorm(groupCount: config.numHeads, dimensions: c, eps: config.groupNormEps, affine: true, pytorchCompatible: true)
    }

    func callAsFunction(_ x: MLXArray, vFirst: MLXArray?, cache: RWKV7LayerCache?) -> (output: MLXArray, vFirst: MLXArray) {
        let (b, l, c) = (x.dim(0), x.dim(1), x.dim(2))
        guard l > 0 else {
            let empty = MLXArray.zeros([b, 0, c], dtype: x.dtype)
            return (empty, vFirst ?? empty)
        }
        let (dx, lastX) = RWKV7Ops.tokenShift(x, previous: cache?.prevTimeX)
        cache?.prevTimeX = lastX
        let r = rProj(x + dx * xR.asType(x.dtype))
        let k = kProj(x + dx * xK.asType(x.dtype))
        let xv = x + dx * xV.asType(x.dtype)
        let (v, resolvedVFirst) = blendValue(vProj(xv), xv: xv, vFirst: vFirst)
        let w = exp(RWKV7Ops.decayScale * sigmoid(wLora(x + dx * xW.asType(x.dtype)).asType(.float32)))
        let a = sigmoid(aLora(x + dx * xA.asType(x.dtype)).asType(.float32))
        let g = gLora(x + dx * xG.asType(x.dtype))
        let kk = RWKV7Ops.l2Normalize((k * kK.asType(k.dtype)).reshaped(b, l, numHeads, headDim))
        let kEff = (k.asType(.float32) * (1 + (a - 1) * kA.asType(.float32))).reshaped(b, l, numHeads, headDim)
        let rF = r.asType(.float32).reshaped(b, l, numHeads, headDim)
        let vF = v.asType(.float32).reshaped(b, l, numHeads, headDim)
        let inputs = RWKV7RecurrenceInputs(
            r: rF, v: vF, w: w.reshaped(b, l, numHeads, headDim),
            a: a.reshaped(b, l, numHeads, headDim), kk: kk, kEff: kEff
        )
        let (y, nextState) = RWKV7Ops.wkv7Recurrence(inputs, initialState: cache?.stateMatrix)
        cache?.stateMatrix = nextState
        let normed = gNorm(y.reshaped(b * l, c).asType(x.dtype)).reshaped(b, l, c)
        let bonus = ((rF * kEff * rK.asType(.float32)).sum(axis: -1, keepDims: true) * vF).reshaped(b, l, c).asType(x.dtype)
        return (oProj((normed + bonus) * g), resolvedVFirst)
    }

    private func blendValue(_ vRaw: MLXArray, xv: MLXArray, vFirst: MLXArray?) -> (MLXArray, MLXArray) {
        guard layerIdx > 0, let vLora, let first = vFirst else { return (vRaw, vFirst ?? vRaw) }
        let gate = sigmoid(vLora(xv).asType(.float32)).asType(vRaw.dtype)
        return (vRaw + (first - vRaw) * gate, first)
    }
}

final class RWKV7FeedForward: Module {
    @ParameterInfo(key: "x_k") var xK: MLXArray
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    init(_ config: RWKV7Configuration) {
        _xK.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize], dtype: .float32)
        _key.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _value.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: RWKV7LayerCache?) -> MLXArray {
        guard x.dim(1) > 0 else { return MLXArray.zeros(like: x) }
        let (dx, lastX) = RWKV7Ops.tokenShift(x, previous: cache?.prevChannelX)
        cache?.prevChannelX = lastX
        return value(reluSquared(key(x + dx * xK.asType(x.dtype))))
    }
}

final class RWKV7Block: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm?
    @ModuleInfo(key: "attn_norm") var attnNorm: LayerNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: LayerNorm
    @ModuleInfo(key: "attn") var attn: RWKV7TimeMix
    @ModuleInfo(key: "ffn") var ffn: RWKV7FeedForward

    init(_ config: RWKV7Configuration, layerIdx: Int) {
        if config.normFirst && layerIdx == 0 {
            _preNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias)
        }
        _attnNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias)
        _ffnNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias)
        _attn.wrappedValue = RWKV7TimeMix(config, layerIdx: layerIdx)
        _ffn.wrappedValue = RWKV7FeedForward(config)
    }

    func callAsFunction(_ x: MLXArray, vFirst: MLXArray?, cache: RWKV7LayerCache?) -> (hidden: MLXArray, vFirst: MLXArray) {
        let base = preNorm.map { $0(x) } ?? x
        let (timeOut, nextVFirst) = attn(attnNorm(base), vFirst: vFirst, cache: cache)
        let h = base + timeOut
        let out = h + ffn(ffnNorm(h), cache: cache)
        cache?.advance(by: x.dim(1))
        return (out, nextVFirst)
    }
}

final class RWKV7ModelInner: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [RWKV7Block]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ config: RWKV7Configuration) {
        _embeddings.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { RWKV7Block(config, layerIdx: $0) }
        _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings(inputs)
        var vFirst: MLXArray?
        for (index, layer) in layers.enumerated() {
            let layerCache = cache.flatMap { index < $0.count ? ($0[index] as? RWKV7LayerCache) : nil }
            (h, vFirst) = layer(h, vFirst: vFirst, cache: layerCache)
        }
        return norm(h)
    }
}

public final class RWKV7Model: Module, LLMModel {
    public static let modelType = "rwkv7"
    public let vocabularySize: Int
    public let configuration: RWKV7Configuration
    @ModuleInfo(key: "model") var model: RWKV7ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: RWKV7Configuration) {
        (self.configuration, self.vocabularySize) = (config, config.vocabSize)
        _model.wrappedValue = RWKV7ModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public convenience init(configurationData data: Data) throws {
        self.init(try JSONDecoder.json5().decode(RWKV7Configuration.self, from: data))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead.map { $0(hidden) } ?? model.embeddings.asLinear(hidden)
    }

    public var loraLayers: [Module] { model.layers }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        (0 ..< configuration.numHiddenLayers).map { _ in RWKV7LayerCache() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        RWKV7Checkpoint.sanitize(weights, config: configuration)
    }
}

enum RWKV7Checkpoint {
    private static let shiftNames = ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]
    private static let vectorSuffixes = [
        ".attn.x_r", ".attn.x_w", ".attn.x_k", ".attn.x_v", ".attn.x_a", ".attn.x_g",
        ".attn.k_k", ".attn.k_a", ".ffn.x_k",
    ]
    private static let canonicalLoraMap: [(suffix: String, replacement: String, transpose: Bool)] = [
        (".w0", ".w_lora.lora_b.bias", false), (".w1", ".w_lora.lora_a.weight", true), (".w2", ".w_lora.lora_b.weight", true),
        (".a0", ".a_lora.lora_b.bias", false), (".a1", ".a_lora.lora_a.weight", true), (".a2", ".a_lora.lora_b.weight", true),
        (".v0", ".v_lora.lora_b.bias", false), (".v1", ".v_lora.lora_a.weight", true), (".v2", ".v_lora.lora_b.weight", true),
        (".g1", ".g_lora.lora_a.weight", true), (".g2", ".g_lora.lora_b.weight", true),
    ]

    static func sanitize(_ weights: [String: MLXArray], config: RWKV7Configuration) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (rawKey, rawValue) in weights {
            let (key, transpose) = canonicalKey(rawKey)
            guard !shouldDrop(key, config: config) else { continue }
            if key.hasSuffix(".attn.x_x"), rawValue.ndim >= 1, rawValue.dim(0) == 6 {
                let prefix = String(key.dropLast("x_x".count))
                for (idx, name) in shiftNames.enumerated() {
                    out[prefix + name] = rawValue[idx].reshaped(1, 1, config.hiddenSize)
                }
                continue
            }
            let oriented = transpose && rawValue.ndim == 2 ? rawValue.T : rawValue
            out[key] = normalizeShape(oriented, key: key, config: config)
        }
        return out
    }

    private static func canonicalKey(_ raw: String) -> (String, Bool) {
        if raw == "emb.weight" || raw == "model.embed_tokens.weight" || raw == "rwkv.embeddings.weight" {
            return ("model.embeddings.weight", false)
        }
        var key = raw
        for (from, to) in [
            ("rwkv.blocks.", "model.layers."), ("model.blocks.", "model.layers."),
            ("ln_out.", "model.norm."), ("head.", "lm_head."), ("rwkv.", "model."), ("blocks.", "model.layers."),
        ] where key.hasPrefix(from) {
            key = to + key.dropFirst(from.count)
            break
        }
        for (from, to) in [
            (".ln0.", ".pre_norm."), (".ln1.", ".attn_norm."), (".ln2.", ".ffn_norm."), (".att.", ".attn."),
            (".attn.receptance.", ".attn.r_proj."), (".attn.key.", ".attn.k_proj."),
            (".attn.value.", ".attn.v_proj."), (".attn.output.", ".attn.o_proj."), (".attn.ln_x.", ".attn.g_norm."),
            (".lora.0.", ".lora_a."), (".lora.1.", ".lora_b."), (".lora.2.", ".lora_b."),
        ] {
            key = key.replacingOccurrences(of: from, with: to)
        }
        if key.contains(".attn.") {
            for entry in canonicalLoraMap where key.hasSuffix(entry.suffix) {
                return (String(key.dropLast(entry.suffix.count)) + entry.replacement, entry.transpose)
            }
        }
        return (key, false)
    }

    private static func shouldDrop(_ key: String, config: RWKV7Configuration) -> Bool {
        if config.tieWordEmbeddings && key.hasPrefix("lm_head.") { return true }
        if key.hasPrefix("model.layers.0.attn.v_lora.") { return true }
        if !config.normFirst && key.hasPrefix("model.layers.0.pre_norm.") { return true }
        let parts = key.split(separator: ".", maxSplits: 3)
        return parts.count >= 3 && parts[0] == "model" && parts[1] == "layers"
            && (Int(parts[2]).map { $0 >= config.numHiddenLayers } ?? false)
    }

    private static func normalizeShape(_ value: MLXArray, key: String, config: RWKV7Configuration) -> MLXArray {
        if vectorSuffixes.contains(where: { key.hasSuffix($0) }), value.size == config.hiddenSize {
            return value.reshaped(1, 1, config.hiddenSize)
        }
        if key.hasSuffix(".attn.r_k"), value.size == config.hiddenSize {
            return value.reshaped(config.numHeads, config.headDim)
        }
        if key.hasSuffix(".lora_b.bias"), value.size == config.hiddenSize, value.ndim != 1 {
            return value.reshaped(config.hiddenSize)
        }
        return value
    }
}
