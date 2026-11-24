import Foundation

extension KVCacheGeometry {
    static func localLayerKinds(_ c: [String: Any], layers: Int) -> [LayerKind]? {
        if let local = c["local_layer_ids"] as? [Int], c["sliding_window_size"] is Int {
            let windowed = Set(local)
            return (0 ..< layers).map { windowed.contains($0) ? .windowed : .full }
        }
        if let group = c["layer_group_size"] as? Int, group > 0, c["kv_lora_rank"] is Int {
            let full = (layers / group) * group
            return (0 ..< layers).map { ($0 + 1).isMultiple(of: group) || $0 >= full ? .full : .linear }
        }
        return nil
    }

    static func bailingLinearStateBytes(_ c: [String: Any]) -> Int64 {
        guard c["layer_group_size"] is Int, c["kv_lora_rank"] is Int,
              let heads = c["num_attention_heads"] as? Int, heads > 0,
              let headDim = c["head_dim"] as? Int, headDim > 0 else { return 0 }
        let kernel = c["short_conv_kernel_size"] as? Int ?? c["conv_kernel_size"] as? Int ?? 4
        let conv = Double(3 * max(0, kernel - 1) * heads * headDim) * activationBytes
        let recurrent = Double(heads * headDim * headDim) * recurrentStateBytes
        return Int64(conv + recurrent)
    }

    static func visits(_ kinds: [LayerKind], _ c: [String: Any]) -> [LayerKind] {
        guard let loop = c["loop_config"] as? [String: Any], let start = loop["core_start"] as? Int, let end = loop["core_end"] as? Int,
              let passes = loop["num_passes"] as? Int, passes > 1, start >= 0, start < end, end <= kinds.count else { return kinds }
        return Array(kinds[..<start]) + Array(repeating: kinds[start ..< end], count: passes).flatMap { $0 } + Array(kinds[end...])
    }

    static func shortConvStateBytes(_ c: [String: Any], visits: Int, kvHeads: Int, headDim: Int) -> Int64 {
        guard let kernel = c["sconv_kernel_size"] as? Int, kernel > 1, let hidden = c["hidden_size"] as? Int else { return 0 }
        let channels = 2 * kvHeads * headDim + 2 * hidden
        return Int64(visits * (kernel - 1) * channels) * Int64(activationBytes)
    }
}
