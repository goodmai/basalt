import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("Huginn-0125 recurrent depth: prelude, a sandwich core re-injected through the adapter num_steps times from a zero state, coda, and a latent-change exit under --experimental-loop-exit (#820)", .serialized)
struct HuginnModelTests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 9, 31, 200, 3]

    static func model(threshold: Float? = nil) throws -> HuginnModel {
        let model = try #require(try TinyModel.make(.huginn) as? HuginnModel)
        model.transformer.loopExitThreshold = threshold
        return model
    }

    static func block(_ h: MLXArray, _ p: String, _ w: [String: MLXArray], _ c: HuginnConfiguration) -> MLXArray {
        let (t, half) = (h.dim(0), c.headDim / 2)
        func rms(_ x: MLXArray, _ key: String) -> MLXArray { x * rsqrt(mean(x * x, axis: -1, keepDims: true) + c.normEps) * w[p + key]! }
        func linear(_ x: MLXArray, _ key: String) -> MLXArray { matmul(x, w[p + key + ".weight"]!.T) }
        let inverse = pow(MLXArray(c.ropeBase), -MLXArray(stride(from: 0, to: c.headDim, by: 2).map(Float.init)) / Float(c.headDim))
        let angles = MLXArray((0 ..< t).map(Float.init))[0..., .newAxis] * inverse[.newAxis]
        func rope(_ x: MLXArray) -> MLXArray {
            let pairs = x.reshaped(c.heads, t, half, 2)
            let (a, b) = (pairs[.ellipsis, 0], pairs[.ellipsis, 1])
            return stacked([a * cos(angles) - b * sin(angles), b * cos(angles) + a * sin(angles)], axis: -1).reshaped(c.heads, t, c.headDim)
        }
        let causal = MLXArray((0 ..< t * t).map { $0 % t > $0 / t ? -Float.infinity : 0 }).reshaped(t, t)
        let x = rms(h, "norm_1.weight")
        let qkv = linear(x, "attn.Wqkv").split(parts: 3, axis: -1).map { $0.reshaped(t, c.heads, c.headDim) }
        let bias = w[p + "attn.qk_bias"]!
        let q = rope((qkv[0] + bias[0]).transposed(1, 0, 2))
        let k = rope((qkv[1] + bias[1]).transposed(1, 0, 2))
        let scores = softmax(matmul(q, k.transposed(0, 2, 1)) / Float(c.headDim).squareRoot() + causal, axis: -1)
        let attention = linear(matmul(scores, qkv[2].transposed(1, 0, 2)).transposed(1, 0, 2).reshaped(t, -1), "attn.proj")
        let y = rms(attention + h, "norm_2.weight")
        let (gate, up) = linear(rms(y, "norm_3.weight"), "mlp.fc").split(axis: -1)
        return rms(linear(silu(gate) * up, "mlp.proj") + y, "norm_4.weight")
    }

    static func reference(_ model: HuginnModel, _ tokens: [Int32]) -> MLXArray {
        let w = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) })
        let c = model.config
        func lnF(_ x: MLXArray) -> MLXArray { x * rsqrt(mean(x * x, axis: -1, keepDims: true) + c.normEps) * w["transformer.ln_f.weight"]! }
        let wte = w["transformer.wte.weight"]!
        var e = wte[MLXArray(tokens)] * Float(c.hiddenSize).squareRoot()
        for layer in 0 ..< c.preludeLayers { e = block(e, "transformer.prelude.\(layer).", w, c) }
        var x = MLXArray.zeros(like: e)
        for _ in 0 ..< c.steps {
            x = matmul(concatenated([x, e], axis: -1), w["transformer.adapter.weight"]!.T)
            for layer in 0 ..< c.coreLayers { x = block(x, "transformer.core_block.\(layer).", w, c) }
        }
        x = lnF(x)
        for layer in 0 ..< c.codaLayers { x = block(x, "transformer.coda.\(layer).", w, c) }
        return matmul(lnF(x), wte.T)[.newAxis]
    }

    @Test("The full forward equals a plain-math reference of the Huginn recurrence: logits within 1e-4 and identical argmax ids (#228)")
    func matchesReference() throws {
        let model = try Self.model()
        #expect(abs(model.transformer.prelude[0].attention.qkBias!).max().item(Float.self) > 0)
        let logits = model(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        let reference = Self.reference(model, Self.tokens)
        #expect(abs(logits - reference).max().item(Float.self) < 1e-4)
        #expect(argMax(logits, axis: -1).asArray(Int32.self) == argMax(reference, axis: -1).asArray(Int32.self))
    }

    @Test("Caches run prelude, then core step-major, then coda, and token-by-token decode gives the full-sequence logits at every position")
    func decodeMatchesFullForward() throws {
        let model = try Self.model()
        let c = model.config
        let cache = model.newCache(parameters: nil)
        #expect(cache.count == c.preludeLayers + c.steps * c.coreLayers + c.codaLayers)
        let full = model(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        let stepped = concatenated(Self.tokens.map { model(MLXArray([$0])[.newAxis], cache: cache).asType(.float32) }, axis: 1)
        #expect(abs(stepped - full).max().item(Float.self) < 1e-4)
    }

    @Test("A zero threshold the latent change never falls below decodes exactly like the stock model with every step counted")
    func unreachedThresholdIsStock() throws {
        let stock = try Self.model()
        let gated = try Self.model(threshold: 0)
        let (a, b) = (stock.newCache(parameters: nil), gated.newCache(parameters: nil))
        _ = stock(MLXArray(Self.tokens)[.newAxis], cache: a)
        _ = gated(MLXArray(Self.tokens)[.newAxis], cache: b)
        for token in [Int32(150), 77, 12] {
            let (x, y) = (stock(MLXArray([token])[.newAxis], cache: a), gated(MLXArray([token])[.newAxis], cache: b))
            #expect(abs(x - y).max().item(Float.self) == 0)
        }
        #expect(gated.transformer.exitSteps == 3 && gated.transformer.exitPasses == 3 * gated.config.steps)
    }

    @Test("A threshold above the first step's relative change ends each one-token step after core step 1, fills every later step's cache from it, and keeps decoding finite")
    func reachedThresholdExitsAfterFirstStep() throws {
        let model = try Self.model(threshold: 2)
        let c = model.config
        let cache = model.newCache(parameters: nil)
        _ = model(MLXArray(Self.tokens)[.newAxis], cache: cache)
        #expect(model.transformer.exitSteps == 0)
        let logits = concatenated([Int32(150), 77, 12].map { model(MLXArray([$0])[.newAxis], cache: cache).asType(.float32) }, axis: 1)
        #expect(model.transformer.exitSteps == 3 && model.transformer.exitPasses == 3)
        #expect(!isNaN(logits).any().item(Bool.self))
        #expect(cache.allSatisfy { $0.offset == Self.tokens.count + 3 })
        for layer in 0 ..< c.coreLayers {
            let first = cache[c.preludeLayers + layer].state[0][0..., 0..., (-1)..., 0...]
            for step in 1 ..< c.steps {
                #expect(abs(cache[c.preludeLayers + step * c.coreLayers + layer].state[0][0..., 0..., (-1)..., 0...] - first).max().item(Float.self) == 0)
            }
        }
    }
}
