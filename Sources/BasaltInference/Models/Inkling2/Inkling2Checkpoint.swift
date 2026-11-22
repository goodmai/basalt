import Foundation
import MLX

enum Inkling2Checkpoint {
    static let text = "model.language_model."
    static let quantParts = ["weight", "scales", "biases"]
    static let messageGates = ["q", "k", "v", "mlp"]

    static func sanitized(_ weights: [String: MLXArray], config: Inkling2Configuration) -> [String: MLXArray] {
        let dropped = weights.filter { key, _ in !(key.hasPrefix("vision_tower.") || key.hasPrefix("model.vision_tower.") || key.hasPrefix("mtp.")) }
        guard dropped.keys.contains(where: { $0.hasPrefix("model.layers.") }) else { return dropped }
        return fromOQ(dropped, config: config)
    }

    static func fromOQ(_ weights: [String: MLXArray], config: Inkling2Configuration) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var messageQueries: [Int: MLXArray] = [:]
        let renames = ["model.embed.": text + "embed_tokens.", "model.embed_norm.": text + "embed_norm.", "model.norm.": text + "norm.", "model.final_input.": text + "final_input."]
        for (key, value) in weights {
            if let (from, to) = renames.first(where: { key.hasPrefix($0.key) }) {
                out[to + key.dropFirst(from.count)] = value
            } else if key.hasPrefix("model.layers.") {
                let parts = key.split(separator: ".", maxSplits: 3).map(String.init)
                guard parts.count == 4, let index = Int(parts[2]) else { continue }
                if parts[3] == "loop_message.query" { messageQueries[index] = value; continue }
                layer(parts[3], value, index: index, config: config, into: &out)
            } else {
                out[key] = value
            }
        }
        if let loop = config.loop, !messageQueries.isEmpty {
            let rows = (loop.coreStart ..< loop.coreEnd).compactMap { messageQueries[$0] }
            if rows.count == loop.coreEnd - loop.coreStart { out[text + "message_input.query"] = stacked(rows) }
        }
        return out
    }

    static func layer(_ name: String, _ value: MLXArray, index: Int, config: Inkling2Configuration, into out: inout [String: MLXArray]) {
        let pre = text + "layers.\(index)."
        let attention = pre + "self_attn."
        let (q, kv, r) = (config.heads * config.headDim, config.kvHeads * config.headDim, config.heads * config.relativeDim)
        let dynamicTaps = config.kvHeads * config.shortConvKernel
        let simple: [String: String] = [
            "attn_input.query": pre + "attention_input.query", "mlp_input.query": pre + "mlp_input.query",
            "attn_norm.weight": pre + "input_layernorm.weight", "mlp_norm.weight": pre + "post_attention_layernorm.weight",
            "q_input_norm.weight": pre + "q_input_layernorm.weight", "k_input_norm.weight": pre + "k_input_layernorm.weight",
            "v_input_norm.weight": pre + "v_input_layernorm.weight", "attn.q_scale": attention + "q_l2_clip.scale",
            "attn.r_scale": attention + "rel_l2_clip.scale", "attn.rel_proj": attention + "rel_logits_proj.proj",
            "attn.k_static": attention + "k_sconv.static_weight", "attn.v_static": attention + "v_sconv.static_weight",
            "mlp.router_weight": pre + "mlp.gate.weight", "mlp.router_bias": pre + "mlp.gate.e_score_correction_bias",
        ]
        if let target = simple[name] { out[target] = value; return }
        let convs = ["attn_conv": pre + "post_attention_convolution.conv1d.weight", "mlp_conv": pre + "post_mlp_convolution.conv1d.weight",
                     "attn.k_conv": attention + "k_sconv.conv1d.weight", "attn.v_conv": attention + "v_sconv.conv1d.weight"]
        if let target = convs[name] { out[target] = value.expandedDimensions(axis: -1); return }
        let (stem, part) = splitPart(name)
        switch stem {
        case "attn.o_proj": out[attention + "o_proj." + part] = value
        case "mlp.experts.w13": out[pre + "mlp.experts.gate_up_proj." + part] = value
        case "mlp.experts.w2": out[pre + "mlp.experts.down_proj." + part] = value
        case "attn.in_proj": rows(value, part, [("q_proj", q), ("k_proj", kv), ("v_proj", kv), ("r_proj", r), ("gate_proj", q), ("sink_proj", config.heads)], prefix: attention, into: &out)
        case "attn.q_proj": rows(value, part, [("q_proj", q), ("r_proj", r), ("gate_proj", q), ("sink_proj", config.heads)], prefix: attention, into: &out)
        case "attn.k_proj": rows(value, part, [("k_proj", kv), ("k_sconv.weight_proj", dynamicTaps)], prefix: attention, into: &out)
        case "attn.v_proj": rows(value, part, [("v_proj", kv), ("v_sconv.weight_proj", dynamicTaps)], prefix: attention, into: &out)
        case "loop_down": loopDown(value, prefix: pre, into: &out)
        case "loop_gain_forget": loopGainForget(value, prefix: pre, into: &out)
        default: out[pre + name] = value
        }
    }

    static func splitPart(_ name: String) -> (String, String) {
        guard let dot = name.lastIndex(of: "."), quantParts.contains(String(name[name.index(after: dot)...])) else { return (name, "weight") }
        return (String(name[..<dot]), String(name[name.index(after: dot)...]))
    }

    static func rows(_ value: MLXArray, _ part: String, _ pieces: [(String, Int)], prefix: String, into out: inout [String: MLXArray]) {
        var start = 0
        for (name, count) in pieces {
            out[prefix + name + "." + part] = value[start ..< (start + count)]
            start += count
        }
    }

    static func loopDown(_ value: MLXArray, prefix: String, into out: inout [String: MLXArray]) {
        let rank = value.dim(-1)
        for (slot, gate) in messageGates.enumerated() {
            out[prefix + gate + "_message_gate.down_proj.weight"] = value[slot].transposed(2, 0, 1).reshaped(rank, -1)
        }
    }

    static func loopGainForget(_ value: MLXArray, prefix: String, into out: inout [String: MLXArray]) {
        let dims = value.dim(-1) / 2
        for (slot, gate) in messageGates.enumerated() {
            out[prefix + gate + "_message_gate.message_gate_proj.weight"] = value[slot][0..., ..<dims].transposed()
            out[prefix + gate + "_message_gate.forget_gate_proj.weight"] = value[slot][0..., dims...].transposed()
        }
    }
}
