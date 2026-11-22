import Foundation

extension KVCacheGeometry {
    static func localLayerKinds(_ c: [String: Any], layers: Int) -> [LayerKind]? {
        guard let local = c["local_layer_ids"] as? [Int], c["sliding_window_size"] is Int else { return nil }
        let windowed = Set(local)
        return (0 ..< layers).map { windowed.contains($0) ? .windowed : .full }
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
