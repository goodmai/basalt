import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

struct PLTConfiguration: Decodable, Sendable {
    let hiddenSize: Int
    let layers: Int
    let intermediateSize: Int
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let ropeTheta: Float
    let loops: Int
    let window: Int
    let normalizePerLoop: Bool
    let embeddingScale: Float
    let hiddenScale: Float
    let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", layers = "num_hidden_layers", intermediateSize = "intermediate_size"
        case heads = "num_attention_heads", kvHeads = "num_key_value_heads", headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps", vocabularySize = "vocab_size", ropeTheta = "rope_theta"
        case loops = "plt_num_loops", window = "plt_window_size", normalizePerLoop = "plt_normalize_per_loop"
        case embeddingScale = "plt_emb_scale", hiddenScale = "plt_hidden_scale", gateUsesHidden = "plt_gate_use_hidden_states"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        layers = try c.decode(Int.self, forKey: .layers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        heads = try c.decode(Int.self, forKey: .heads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (heads > 0 ? hiddenSize / heads : 0)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        loops = try c.decodeIfPresent(Int.self, forKey: .loops) ?? 2
        let windows = try c.decodeIfPresent([Int].self, forKey: .window) ?? [64, 0]
        window = windows.first ?? 64
        normalizePerLoop = try c.decodeIfPresent(Bool.self, forKey: .normalizePerLoop) ?? true
        embeddingScale = try c.decodeIfPresent(Float.self, forKey: .embeddingScale) ?? 1
        hiddenScale = try c.decodeIfPresent(Float.self, forKey: .hiddenScale) ?? 1
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        let gateUsesHidden = try c.decodeIfPresent(Bool.self, forKey: .gateUsesHidden) ?? false
        guard loops >= 1, layers >= 1, heads > 0, kvHeads > 0, headDim > 0, heads % kvHeads == 0, window >= 1, windows.dropFirst().allSatisfy({ $0 == 0 }), gateUsesHidden || loops == 1 else {
            throw DecodingError.dataCorruptedError(
                forKey: .loops, in: c,
                debugDescription: "iquestpltcoder needs plt_num_loops ≥ 1, heads divisible by key-value heads, a causal window [left ≥ 1, 0] and plt_gate_use_hidden_states true (the query-einsum gate is not implemented)")
        }
    }
}

final class PLTGate: Module {
    @ModuleInfo(key: "gate_norm") var norm: RMSNorm
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ config: PLTConfiguration) {
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _proj.wrappedValue = Linear(config.hiddenSize, config.heads, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { expandedDimensions(sigmoid(proj(norm(x))).transposed(0, 2, 1), axis: -1) }
}

final class PLTAttention: Module {
    let config: PLTConfiguration
    let scale: Float
    let rope: RoPE
    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "plt_gate") var gate: PLTGate

    init(_ config: PLTConfiguration) {
        self.config = config
        scale = pow(Float(config.headDim), -0.5)
        rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        _wq.wrappedValue = Linear(config.hiddenSize, config.heads * config.headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _wv.wrappedValue = Linear(config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _wo.wrappedValue = Linear(config.heads * config.headDim, config.hiddenSize, bias: false)
        _gate.wrappedValue = PLTGate(config)
    }

    func project(_ x: MLXArray, cache: KVCache?) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (b, l) = (x.dim(0), x.dim(1))
        let queries = applyRotaryPosition(rope, to: wq(x).reshaped(b, l, config.heads, -1).transposed(0, 2, 1, 3), offset: cache?.ropeOffset)
        let keys = applyRotaryPosition(rope, to: wk(x).reshaped(b, l, config.kvHeads, -1).transposed(0, 2, 1, 3), offset: cache?.ropeOffset)
        let values = wv(x).reshaped(b, l, config.kvHeads, -1).transposed(0, 2, 1, 3)
        guard let cache else { return (queries, keys, values) }
        let (allKeys, allValues) = cache.update(keys: keys, values: values)
        return (queries, allKeys, allValues)
    }

    func output(_ attended: MLXArray) -> MLXArray {
        wo(attended.transposed(0, 2, 1, 3).reshaped(attended.dim(0), attended.dim(2), -1))
    }

    func first(_ x: MLXArray, cache: KVCache?) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let p = project(x, cache: cache)
        let attended = MLXFast.scaledDotProductAttention(queries: p.queries, keys: p.keys, values: p.values, scale: scale, mask: PLTMasks.causal(x.dim(1), p.keys.dim(2)))
        return (output(attended), p.keys, p.values)
    }

    func looped(_ x: MLXArray, shared: (keys: MLXArray, values: MLXArray), cache: KVCache?) -> MLXArray {
        let p = project(x, cache: cache)
        let global = MLXFast.scaledDotProductAttention(queries: p.queries, keys: shared.keys, values: shared.values, scale: scale, mask: PLTMasks.causal(x.dim(1), shared.keys.dim(2)))
        let local = PLTMasks.window(p.queries, p.keys, p.values, left: config.window, scale: scale)
        let g = gate(x)
        return output(g * global + (1 - g) * local)
    }
}

enum PLTMasks {
    static func causal(_ queries: Int, _ keys: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        queries == 1 ? .none : .array(allowed(queries, keys, left: nil))
    }

    static func allowed(_ queries: Int, _ keys: Int, left: Int?) -> MLXArray {
        let row = MLXArray((0 ..< queries).map { Int32(keys - queries + $0) })[0..., .newAxis]
        let column = MLXArray((0 ..< keys).map(Int32.init))[.newAxis]
        let causal = column .<= row
        return left.map { causal .&& (column .>= row - Int32($0)) } ?? causal
    }

    static func window(_ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, left: Int, scale: Float) -> MLXArray {
        let (lq, lk) = (queries.dim(2), keys.dim(2))
        if lq == 1 {
            let start = max(0, lk - left - 1)
            return MLXFast.scaledDotProductAttention(queries: queries, keys: keys[0..., 0..., start...], values: values[0..., 0..., start...], scale: scale, mask: .none)
        }
        return MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: .array(allowed(lq, lk, left: left)))
    }
}

final class PLTMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: PLTConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class PLTDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: PLTAttention
    @ModuleInfo(key: "mlp") var mlp: PLTMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    init(_ config: PLTConfiguration) {
        _attention.wrappedValue = PLTAttention(config)
        _mlp.wrappedValue = PLTMLP(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func first(_ x: MLXArray, cache: KVCache?) -> (hidden: MLXArray, keys: MLXArray, values: MLXArray) {
        let a = attention.first(inputNorm(x), cache: cache)
        let h = x + a.output
        return (h + mlp(postNorm(h)), a.keys, a.values)
    }

    func looped(_ x: MLXArray, shared: (keys: MLXArray, values: MLXArray), cache: KVCache?) -> MLXArray {
        let h = x + attention.looped(inputNorm(x), shared: shared, cache: cache)
        return h + mlp(postNorm(h))
    }
}
