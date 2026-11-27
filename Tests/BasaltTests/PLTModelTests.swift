import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("Parallel Loop Transformer (iquestpltcoder): loop 0 causal, later loops fuse global attention over loop-0 KV with a sliding window through a per-head gate, inputs a·E + b·shift(norm(H)) (#820)", .serialized)
struct PLTModelTests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 9, 31, 200, 3, 77, 101, 11, 58]

    static func model() throws -> PLTModel { try #require(try TinyModel.make(.plt) as? PLTModel) }

    static func reference(_ ids: [Int32], _ w: [String: MLXArray], _ c: PLTConfiguration) -> MLXArray {
        let t = ids.count
        func rms(_ x: MLXArray, _ key: String) -> MLXArray { x * rsqrt(mean(x * x, axis: -1, keepDims: true) + c.rmsNormEps) * w[key]! }
        func linear(_ x: MLXArray, _ key: String) -> MLXArray { matmul(x, w[key + ".weight"]!.T) + (w[key + ".bias"] ?? MLXArray(Float(0))) }
        let inverse = pow(MLXArray(c.ropeTheta), -MLXArray(stride(from: 0, to: c.headDim, by: 2).map(Float.init)) / Float(c.headDim))
        let angles = MLXArray((0 ..< t).map(Float.init))[0..., .newAxis] * inverse[.newAxis]
        func rope(_ x: MLXArray) -> MLXArray {
            let half = c.headDim / 2
            let (a, b) = (x[.ellipsis, ..<half], x[.ellipsis, half...])
            return concatenated([a * cos(angles) - b * sin(angles), b * cos(angles) + a * sin(angles)], axis: -1)
        }
        func heads(_ x: MLXArray, _ n: Int) -> MLXArray { x.reshaped(t, n, c.headDim).transposed(1, 0, 2) }
        func mask(window: Int?) -> MLXArray {
            MLXArray((0 ..< t * t).map { index -> Float in
                let (i, j) = (index / t, index % t)
                return j <= i && (window.map { i - j <= $0 } ?? true) ? 0 : -Float.infinity
            }).reshaped(t, t)
        }
        func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ m: MLXArray) -> MLXArray {
            let group = c.heads / c.kvHeads
            let (kk, vv) = (repeated(k, count: group, axis: 0), repeated(v, count: group, axis: 0))
            return matmul(softmax(matmul(q, kk.transposed(0, 2, 1)) / Float(c.headDim).squareRoot() + m, axis: -1), vv)
        }
        func mlp(_ x: MLXArray, _ p: String) -> MLXArray { linear(silu(linear(x, p + "mlp.gate_proj")) * linear(x, p + "mlp.up_proj"), p + "mlp.down_proj") }
        let embedded = w["model.embed_tokens.weight"]![MLXArray(ids)]
        var h = embedded
        var shared: [(MLXArray, MLXArray)] = []
        for layer in 0 ..< c.layers {
            let p = "model.layers.\(layer)."
            let x = rms(h, p + "input_layernorm.weight")
            let (q, k, v) = (rope(heads(linear(x, p + "self_attn.q_proj"), c.heads)), rope(heads(linear(x, p + "self_attn.k_proj"), c.kvHeads)), heads(linear(x, p + "self_attn.v_proj"), c.kvHeads))
            shared.append((k, v))
            h = h + linear(attend(q, k, v, mask(window: nil)).transposed(1, 0, 2).reshaped(t, -1), p + "self_attn.o_proj")
            h = h + mlp(rms(h, p + "post_attention_layernorm.weight"), p)
        }
        for _ in 1 ..< c.loops {
            let previous = rms(h, "model.norm.weight")
            h = c.embeddingScale * embedded + c.hiddenScale * concatenated([MLXArray.zeros([1, c.hiddenSize]), previous[..<(t - 1)]], axis: 0)
            for layer in 0 ..< c.layers {
                let p = "model.layers.\(layer)."
                let x = rms(h, p + "input_layernorm.weight")
                let (q, k, v) = (rope(heads(linear(x, p + "self_attn.q_proj"), c.heads)), rope(heads(linear(x, p + "self_attn.k_proj"), c.kvHeads)), heads(linear(x, p + "self_attn.v_proj"), c.kvHeads))
                let g = sigmoid(linear(rms(x, p + "self_attn.plt_gate.gate_norm.weight"), p + "self_attn.plt_gate.proj")).T[0..., 0..., .newAxis]
                let mixed = g * attend(q, shared[layer].0, shared[layer].1, mask(window: nil)) + (1 - g) * attend(q, k, v, mask(window: c.window))
                h = h + linear(mixed.transposed(1, 0, 2).reshaped(t, -1), p + "self_attn.o_proj")
                h = h + mlp(rms(h, p + "post_attention_layernorm.weight"), p)
            }
        }
        return linear(rms(h, "model.norm.weight"), "lm_head")
    }

    @Test("Uncached logits equal an independent plain-math reference of both loops, longer than the sliding window")
    func matchesReference() throws {
        let model = try Self.model()
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let expected = Self.reference(Self.tokens, weights, model.config)
        let logits = model(MLXArray(Self.tokens)[.newAxis], cache: nil)[0]
        #expect(abs(logits - expected).max().item(Float.self) < 1e-3)
    }

    @Test("Chunked prefill and one-token decode through the caches give the uncached logits at every position")
    func cachedDecodeMatchesFullForward() throws {
        let model = try Self.model()
        let full = model(MLXArray(Self.tokens)[.newAxis], cache: nil)[0]
        let cache = model.newCache(parameters: nil)
        var rows = [model(MLXArray(Array(Self.tokens[..<5]))[.newAxis], cache: cache)[0]]
        rows.append(model(MLXArray(Array(Self.tokens[5 ..< 9]))[.newAxis], cache: cache)[0])
        for token in Self.tokens[9...] { rows.append(model(MLXArray([token])[.newAxis], cache: cache)[0]) }
        #expect(abs(concatenated(rows, axis: 0) - full).max().item(Float.self) < 1e-3)
        #expect(cache.count == 2 * 2 + 1 && cache.allSatisfy { $0.offset == Self.tokens.count })
    }

    @Test("Trimming the caches back and decoding again gives the same logits as never having gone forward")
    func trimRewinds() throws {
        let model = try Self.model()
        let cache = model.newCache(parameters: nil)
        _ = model(MLXArray(Array(Self.tokens[..<10]))[.newAxis], cache: cache)
        let next = model(MLXArray(Array(Self.tokens[10 ..< 13]))[.newAxis], cache: cache)[0]
        for entry in cache { _ = entry.trim(3) }
        let again = model(MLXArray(Array(Self.tokens[10 ..< 13]))[.newAxis], cache: cache)[0]
        #expect(abs(next - again).max().item(Float.self) == 0)
    }

    @Test("Checkpoint gate keys map onto the gate's projection, the gate norm keeps its name, and a query-einsum gate is refused")
    func gateKeysAndRefusal() {
        #expect(PLTModel.gateKey("model.layers.3.self_attn.plt_gate.weight") == "model.layers.3.self_attn.plt_gate.proj.weight")
        #expect(PLTModel.gateKey("model.layers.3.self_attn.plt_gate.scales") == "model.layers.3.self_attn.plt_gate.proj.scales")
        #expect(PLTModel.gateKey("model.layers.3.self_attn.plt_gate.gate_norm.weight") == "model.layers.3.self_attn.plt_gate.gate_norm.weight")
        let config = TinyModel.Family.plt.config.replacingOccurrences(of: "\"plt_gate_use_hidden_states\": true", with: "\"plt_gate_use_hidden_states\": false")
        #expect(throws: (any Error).self) { try PLTModel(configurationData: Data(config.utf8)) }
    }
}
