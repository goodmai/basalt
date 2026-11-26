import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("Ouro looped LM: shared layers run total_ut_steps times with sandwich norms, and --experimental-loop-exit uses the trained exit gate (#820)", .serialized)
struct OuroModelTests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 9, 31, 200, 3]

    static func model(threshold: Float? = nil) throws -> OuroModel {
        let model = try #require(try TinyModel.make(.ouro) as? OuroModel)
        model.model.loopExitThreshold = threshold
        return model
    }

    static func reference(_ model: OuroModel, _ tokens: [Int32]) -> MLXArray {
        let w = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) })
        let c = model.config
        let (t, half) = (tokens.count, c.headDim / 2)
        func rms(_ x: MLXArray, _ key: String) -> MLXArray { x * rsqrt(mean(x * x, axis: -1, keepDims: true) + c.rmsNormEps) * w[key]! }
        func linear(_ x: MLXArray, _ key: String) -> MLXArray { matmul(x, w[key + ".weight"]!.T) }
        let inverse = pow(MLXArray(c.ropeTheta), -MLXArray(stride(from: 0, to: c.headDim, by: 2).map(Float.init)) / Float(c.headDim))
        let angles = MLXArray((0 ..< t).map(Float.init))[0..., .newAxis] * inverse[.newAxis]
        func rope(_ x: MLXArray) -> MLXArray {
            let (x1, x2) = (x[.ellipsis, ..<half], x[.ellipsis, half...])
            return concatenated([x1 * cos(angles) - x2 * sin(angles), x2 * cos(angles) + x1 * sin(angles)], axis: -1)
        }
        let causal = MLXArray((0 ..< t * t).map { $0 % t > $0 / t ? -Float.infinity : 0 }).reshaped(t, t)
        let group = c.heads / c.kvHeads
        var h = w["model.embed_tokens.weight"]![MLXArray(tokens)]
        for _ in 0 ..< c.steps {
            for layer in 0 ..< c.layers {
                let p = "model.layers.\(layer)."
                let x = rms(h, p + "input_layernorm.weight")
                let q = rope(linear(x, p + "self_attn.q_proj").reshaped(t, c.heads, c.headDim).transposed(1, 0, 2))
                let k = repeated(rope(linear(x, p + "self_attn.k_proj").reshaped(t, c.kvHeads, c.headDim).transposed(1, 0, 2)), count: group, axis: 0)
                let v = repeated(linear(x, p + "self_attn.v_proj").reshaped(t, c.kvHeads, c.headDim).transposed(1, 0, 2), count: group, axis: 0)
                let scores = softmax(matmul(q, k.transposed(0, 2, 1)) / Float(c.headDim).squareRoot() + causal, axis: -1)
                let attention = linear(matmul(scores, v).transposed(1, 0, 2).reshaped(t, -1), p + "self_attn.o_proj")
                h = h + rms(attention, p + "input_layernorm_2.weight")
                let m = rms(h, p + "post_attention_layernorm.weight")
                let mlp = linear(silu(linear(m, p + "mlp.gate_proj")) * linear(m, p + "mlp.up_proj"), p + "mlp.down_proj")
                h = h + rms(mlp, p + "post_attention_layernorm_2.weight")
            }
            h = rms(h, "model.norm.weight")
        }
        return matmul(h, w["lm_head.weight"]!.T)[.newAxis]
    }

    @Test("The full forward equals a plain-math reference of the Ouro recurrence: logits within 1e-4 and identical argmax ids (#228)")
    func matchesReference() throws {
        let model = try Self.model()
        let logits = model(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        let reference = Self.reference(model, Self.tokens)
        #expect(abs(logits - reference).max().item(Float.self) < 1e-4)
        #expect(argMax(logits, axis: -1).asArray(Int32.self) == argMax(reference, axis: -1).asArray(Int32.self))
    }

    @Test("One KV cache per (step, layer), and token-by-token decode gives the full-sequence logits at every position")
    func decodeMatchesFullForward() throws {
        let model = try Self.model()
        let cache = model.newCache(parameters: nil)
        #expect(cache.count == model.config.steps * model.config.layers)
        let full = model(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        let stepped = concatenated(Self.tokens.map { model(MLXArray([$0])[.newAxis], cache: cache).asType(.float32) }, axis: 1)
        #expect(abs(stepped - full).max().item(Float.self) < 1e-4)
    }

    @Test("An exit threshold the cumulative gate never reaches decodes exactly like the stock model with every step counted")
    func unreachedThresholdIsStock() throws {
        let stock = try Self.model()
        let gated = try Self.model(threshold: 2)
        let (a, b) = (stock.newCache(parameters: nil), gated.newCache(parameters: nil))
        _ = stock(MLXArray(Self.tokens)[.newAxis], cache: a)
        _ = gated(MLXArray(Self.tokens)[.newAxis], cache: b)
        for token in [Int32(150), 77, 12] {
            let (x, y) = (stock(MLXArray([token])[.newAxis], cache: a), gated(MLXArray([token])[.newAxis], cache: b))
            #expect(abs(x - y).max().item(Float.self) == 0)
        }
        #expect(gated.model.exitSteps == 3 && gated.model.exitPasses == 3 * gated.model.config.steps)
    }

    @Test("A threshold the first gate always reaches ends each one-token step after step 1, fills every later step's cache from it, and keeps decoding finite")
    func reachedThresholdExitsAfterFirstStep() throws {
        let model = try Self.model(threshold: -1)
        let cache = model.newCache(parameters: nil)
        _ = model(MLXArray(Self.tokens)[.newAxis], cache: cache)
        #expect(model.model.exitSteps == 0)
        let logits = concatenated([Int32(150), 77, 12].map { model(MLXArray([$0])[.newAxis], cache: cache).asType(.float32) }, axis: 1)
        #expect(model.model.exitSteps == 3 && model.model.exitPasses == 3)
        #expect(!isNaN(logits).any().item(Bool.self))
        #expect(cache.allSatisfy { $0.offset == Self.tokens.count + 3 })
        let layers = model.config.layers
        for layer in 0 ..< layers {
            let first = cache[layer].state[0][0..., 0..., (-1)..., 0...]
            for step in 1 ..< model.config.steps {
                #expect(abs(cache[step * layers + layer].state[0][0..., 0..., (-1)..., 0...] - first).max().item(Float.self) == 0)
            }
        }
    }
}
