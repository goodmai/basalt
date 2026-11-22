import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Inkling 2 P1: looped text stack on the inkling2Loop TinyModel (#650)", .serialized)
struct Inkling2ModelTests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 9, 31, 200, 3, 3, 150, 77, 12, 44, 101, 6, 250]

    static func model() throws -> Inkling2Model {
        try #require(try TinyModel.make(.inkling2Loop) as? Inkling2Model)
    }

    @Test("Two cache slots per layer visit: prelude 2 + core 2 × 3 passes + coda 2 = 10 visits, windowed caches on local layers")
    func cacheLayout() throws {
        let model = try Self.model()
        let caches = try model.newCache(parameters: nil)
        #expect(caches.count == 20)
        #expect(model.config.visitOrder == [0, 1, 2, 3, 2, 3, 2, 3, 4, 5])
        #expect(caches.enumerated().filter { $0.offset.isMultiple(of: 2) }.map { $0.element is Inkling2WindowKVCache } == [true, false, true, false, true, false, true, false, true, false])
        #expect(caches.enumerated().filter { !$0.offset.isMultiple(of: 2) }.allSatisfy { $0.element is ArraysCache })
    }

    @Test("Token-by-token decode through the caches gives the full-sequence logits at every position, past the 8-token window", arguments: [1, 4, 7])
    func decodeMatchesFullForward(chunk: Int) throws {
        let model = try Self.model()
        let full = model(MLXArray(Self.tokens)[.newAxis], cache: nil).asType(.float32)
        let caches = try model.newCache(parameters: nil)
        var steps: [MLXArray] = []
        var start = 0
        while start < Self.tokens.count {
            let end = min(Self.tokens.count, start + chunk)
            steps.append(model(MLXArray(Array(Self.tokens[start ..< end]))[.newAxis], cache: caches).asType(.float32))
            start = end
        }
        let stepped = concatenated(steps, axis: 1)
        let gap = abs(stepped - full).max().item(Float.self)
        #expect(stepped.shape == [1, Self.tokens.count, 256])
        #expect(gap <= 2e-3, "max logit gap \(gap)")
    }

    @Test("The loop matters: one pass gives different logits than three, and passes 2 and 3 read messages")
    func loopChangesOutput() throws {
        let model = try Self.model()
        let three = model(MLXArray(Self.tokens)[.newAxis], cache: nil)
        var single = model.config
        single.loop = Inkling2Configuration.Loop(coreStart: 2, coreEnd: 4, numPasses: 1, gateRank: 8)
        let onePass = Inkling2Model(single)
        onePass.update(parameters: model.parameters())
        let one = onePass(MLXArray(Self.tokens)[.newAxis], cache: nil)
        #expect(abs(three - one).max().item(Float.self) > 1e-3)
    }

    @Test("The oQ pack layout (fused projections, packed loop controllers) maps back to the same parameters")
    func oqLayoutRoundTrip() throws {
        let model = try Self.model()
        let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let packed = Inkling2OQFixture.pack(params, config: model.config)
        #expect(packed.keys.contains("model.layers.0.attn.in_proj.weight") && packed.keys.contains("model.layers.2.loop_down"))
        let restored = Inkling2Checkpoint.sanitized(packed, config: model.config)
        #expect(Set(restored.keys) == Set(params.keys))
        for (key, value) in params {
            let back = try #require(restored[key], "\(key)")
            #expect(back.shape == value.shape && abs(back - value).max().item(Float.self) == 0, "\(key)")
        }
    }

    @Test("The window cache keeps at least window − 1 earlier tokens, reports where its keys start, and trims")
    func windowCache() {
        let cache = Inkling2WindowKVCache(window: 4)
        var lastStart = 0
        for step in 0 ..< 12 {
            let k = MLXArray.ones([1, 1, 1, 2]) * Float(step)
            let (keys, _) = cache.update(keys: k, values: k)
            lastStart = cache.lastKeyStart
            #expect(keys.dim(2) == min(step + 1, 4))
            #expect(keys[0, 0, -1, 0].item(Float.self) == Float(step))
            #expect(keys[0, 0, 0, 0].item(Float.self) == Float(lastStart))
        }
        #expect(cache.offset == 12 && lastStart == 8)
        #expect(cache.trim(2) == 2 && cache.offset == 10)
    }
}

enum Inkling2OQFixture {
    static func pack(_ params: [String: MLXArray], config: Inkling2Configuration) -> [String: MLXArray] {
        let text = Inkling2Checkpoint.text
        var out: [String: MLXArray] = [:]
        var used = Set<String>()
        func take(_ key: String) -> MLXArray { used.insert(key); return params[key]! }
        for index in 0 ..< config.layers {
            let pre = text + "layers.\(index)."
            let attention = pre + "self_attn."
            let oq = "model.layers.\(index)."
            let global = !config.localLayers.contains(index)
            if global {
                out[oq + "attn.q_proj.weight"] = concatenated(["q_proj", "r_proj", "gate_proj", "sink_proj"].map { take(attention + $0 + ".weight") })
                out[oq + "attn.k_proj.weight"] = concatenated([take(attention + "k_proj.weight"), take(attention + "k_sconv.weight_proj.weight")])
                out[oq + "attn.v_proj.weight"] = concatenated([take(attention + "v_proj.weight"), take(attention + "v_sconv.weight_proj.weight")])
            } else {
                out[oq + "attn.in_proj.weight"] = concatenated(["q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj"].map { take(attention + $0 + ".weight") })
                out[oq + "attn.k_conv"] = take(attention + "k_sconv.conv1d.weight").squeezed(axis: -1)
                out[oq + "attn.v_conv"] = take(attention + "v_sconv.conv1d.weight").squeezed(axis: -1)
            }
            if config.loop.map({ ($0.coreStart ..< $0.coreEnd).contains(index) }) == true {
                let gates = Inkling2Checkpoint.messageGates
                out[oq + "loop_down"] = stacked(gates.map { gate in
                    let down = take(pre + gate + "_message_gate.down_proj.weight")
                    return down.reshaped(down.dim(0), 2, -1).transposed(1, 2, 0)
                })
                out[oq + "loop_gain_forget"] = stacked(gates.map { gate in
                    concatenated([take(pre + gate + "_message_gate.message_gate_proj.weight").transposed(),
                                  take(pre + gate + "_message_gate.forget_gate_proj.weight").transposed()], axis: 1)
                })
                out[oq + "loop_message.query"] = take(text + "message_input.query")[index - config.loop!.coreStart]
            }
            for (from, to) in [("post_attention_convolution.conv1d.weight", "attn_conv"), ("post_mlp_convolution.conv1d.weight", "mlp_conv")] {
                out[oq + to] = take(pre + from).squeezed(axis: -1)
            }
        }
        let renames: [(String, String)] = [
            ("attention_input.query", "attn_input.query"), ("input_layernorm.weight", "attn_norm.weight"),
            ("post_attention_layernorm.weight", "mlp_norm.weight"), ("self_attn.q_l2_clip.scale", "attn.q_scale"),
            ("self_attn.rel_l2_clip.scale", "attn.r_scale"), ("self_attn.rel_logits_proj.proj", "attn.rel_proj"),
            ("self_attn.k_sconv.static_weight", "attn.k_static"), ("self_attn.v_sconv.static_weight", "attn.v_static"),
            ("mlp.gate.weight", "mlp.router_weight"), ("mlp.gate.e_score_correction_bias", "mlp.router_bias"),
            ("mlp.experts.gate_up_proj.weight", "mlp.experts.w13.weight"), ("mlp.experts.down_proj.weight", "mlp.experts.w2.weight"),
            ("self_attn.o_proj.weight", "attn.o_proj.weight"), ("q_input_layernorm.weight", "q_input_norm.weight"),
            ("k_input_layernorm.weight", "k_input_norm.weight"), ("v_input_layernorm.weight", "v_input_norm.weight"),
        ]
        for (key, value) in params where !used.contains(key) {
            if key == text + "embed_tokens.weight" { out["model.embed.weight"] = value; continue }
            if key == "lm_head.weight" { out[key] = value; continue }
            if key.hasPrefix(text + "layers.") {
                let rest = key.dropFirst((text + "layers.").count)
                let index = rest.prefix { $0 != "." }
                let name = String(rest.dropFirst(index.count + 1))
                let mapped = renames.first { $0.0 == name }?.1 ?? name
                out["model.layers.\(index).\(mapped)"] = value
            } else {
                out["model." + key.dropFirst(text.count)] = value
            }
        }
        return out
    }
}
