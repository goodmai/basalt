import Foundation

public struct GGUFTensorPlan: Sendable, Equatable {
    public enum Transform: Sendable, Equatable {
        case none
        case logOfNegated
        case vHeadRows(from: Int)
        case vHeadColumns
        case convChannels(from: Int)
    }

    public let source: String
    public let target: String
    public let transform: Transform
    public let bits: Int?

    public static func bits(for type: GGMLType, rank: Int) -> Int? {
        guard rank == 2 else { return nil }
        switch type {
        case .iq1M, .iq2XXS, .iq2XS, .iq2S, .q2K: return 2
        case .iq3XXS, .iq3S: return 3
        case .iq4XS, .iq4NL, .q4K: return 4
        case .q6K: return 6
        case .q8_0, .f16, .bf16, .f32: return 8
        }
    }
}

public struct GGUFQwen35Layout: Sendable, Equatable {
    public let keyHeads: Int
    public let valueHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let prefix: String

    public init(keyHeads: Int, valueHeads: Int, keyHeadDim: Int, valueHeadDim: Int, prefix: String = "language_model.model.") {
        (self.keyHeads, self.valueHeads, self.keyHeadDim, self.valueHeadDim, self.prefix) = (keyHeads, valueHeads, keyHeadDim, valueHeadDim, prefix)
    }

    public var qkChannels: Int { 2 * keyHeads * keyHeadDim }

    public func tiledToGrouped(_ headDim: Int) -> [Int] {
        let perKey = valueHeads / keyHeads
        var grouped = [Int](repeating: 0, count: valueHeads * headDim)
        for tiled in 0 ..< valueHeads * headDim {
            let (head, offset) = (tiled / headDim, tiled % headDim)
            let (copy, key) = (head / keyHeads, head % keyHeads)
            grouped[(key * perKey + copy) * headDim + offset] = tiled
        }
        return grouped
    }

    static let names: [String: (String, GGUFTensorPlan.Transform?)] = [
        "attn_norm.weight": ("input_layernorm.weight", nil), "post_attention_norm.weight": ("post_attention_layernorm.weight", nil),
        "ffn_gate.weight": ("mlp.gate_proj.weight", nil), "ffn_up.weight": ("mlp.up_proj.weight", nil), "ffn_down.weight": ("mlp.down_proj.weight", nil),
        "attn_q.weight": ("self_attn.q_proj.weight", nil), "attn_k.weight": ("self_attn.k_proj.weight", nil), "attn_v.weight": ("self_attn.v_proj.weight", nil),
        "attn_output.weight": ("self_attn.o_proj.weight", nil), "attn_q_norm.weight": ("self_attn.q_norm.weight", nil), "attn_k_norm.weight": ("self_attn.k_norm.weight", nil),
        "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", .vHeadRows(from: -1)), "attn_gate.weight": ("linear_attn.in_proj_z.weight", .vHeadRows(from: 0)),
        "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", .vHeadRows(from: 0)), "ssm_beta.weight": ("linear_attn.in_proj_b.weight", .vHeadRows(from: 0)),
        "ssm_a": ("linear_attn.A_log", .logOfNegated), "ssm_dt.bias": ("linear_attn.dt_bias", .vHeadRows(from: 0)),
        "ssm_conv1d.weight": ("linear_attn.conv1d.weight", .convChannels(from: -1)), "ssm_norm.weight": ("linear_attn.norm.weight", nil),
        "ssm_out.weight": ("linear_attn.out_proj.weight", .vHeadColumns),
    ]

    public func plan(_ tensor: GGUFTensor) -> GGUFTensorPlan? {
        let bits = GGUFTensorPlan.bits(for: tensor.type, rank: tensor.dims.count)
        switch tensor.name {
        case "token_embd.weight": return GGUFTensorPlan(source: tensor.name, target: prefix + "embed_tokens.weight", transform: .none, bits: bits)
        case "output_norm.weight": return GGUFTensorPlan(source: tensor.name, target: prefix + "norm.weight", transform: .none, bits: nil)
        case "output.weight": return GGUFTensorPlan(source: tensor.name, target: (prefix.hasSuffix("model.") ? String(prefix.dropLast("model.".count)) : prefix) + "lm_head.weight", transform: .none, bits: bits)
        default: break
        }
        let parts = tensor.name.split(separator: ".", maxSplits: 2)
        guard parts.count == 3, parts[0] == "blk", let block = Int(parts[1]), let (name, transform) = Self.names[String(parts[2])] else { return nil }
        let resolved: GGUFTensorPlan.Transform = switch transform {
        case .vHeadRows(from: -1)?: .vHeadRows(from: qkChannels)
        case .convChannels(from: -1)?: .convChannels(from: qkChannels)
        case let other?: other
        case nil: .none
        }
        let isOneD = tensor.dims.count == 1 || name.hasSuffix("conv1d.weight")
        return GGUFTensorPlan(source: tensor.name, target: "\(prefix)layers.\(block).\(name)", transform: resolved, bits: isOneD ? nil : bits)
    }
}
