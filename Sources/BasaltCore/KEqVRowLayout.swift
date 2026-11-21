import Foundation

public struct KEqVRowLayout: Sendable, Equatable {
    public let headDim: Int
    public let rotatedHalf: Int
    public let kvHeads: Int
    public let globalLayers: Int

    public init?(headDim: Int, partialRotaryFactor: Double, kvHeads: Int, globalLayers: Int) {
        let rotatedHalf = Int(partialRotaryFactor * Double(headDim) / 2)
        guard headDim > 0, headDim.isMultiple(of: 2), rotatedHalf > 0, 2 * rotatedHalf < headDim,
              kvHeads > 0, globalLayers > 0 else { return nil }
        self.headDim = headDim
        self.rotatedHalf = rotatedHalf
        self.kvHeads = kvHeads
        self.globalLayers = globalLayers
    }

    public init?(config: [String: Any]) {
        let text = config["text_config"] as? [String: Any] ?? config
        guard text["attention_k_eq_v"] as? Bool == true,
              let headDim = text["global_head_dim"] as? Int ?? text["head_dim"] as? Int else { return nil }
        let rope = (text["rope_parameters"] as? [String: Any])?["full_attention"] as? [String: Any]
        let factor = Self.number(rope?["partial_rotary_factor"]) ?? Self.number(text["global_partial_rotary_factor"]) ?? 1
        let kvHeads = text["num_global_key_value_heads"] as? Int ?? text["num_key_value_heads"] as? Int ?? 0
        self.init(headDim: headDim, partialRotaryFactor: factor, kvHeads: kvHeads, globalLayers: Self.globalLayerCount(text))
    }

    public var rotated: [Int] { Array(0 ..< rotatedHalf) + Array(headDim / 2 ..< headDim / 2 + rotatedHalf) }
    public var unrotated: [Int] { Array(rotatedHalf ..< headDim / 2) + Array(headDim / 2 + rotatedHalf ..< headDim) }
    public var rowWidth: Int { headDim + 2 * rotatedHalf }
    public var keyOrder: [Int] { rotated + unrotated }
    public var valueOrder: [Int] { unrotated + rotated }

    public var outputRestore: [Int] {
        var inverse = [Int](repeating: 0, count: headDim)
        for (position, dim) in valueOrder.enumerated() { inverse[dim] = position }
        return inverse
    }

    public func globalBytesPerToken(elementBytes: Int = 2) -> (stock: Int, row: Int) {
        let perHead = globalLayers * kvHeads * elementBytes
        return (perHead * 2 * headDim, perHead * rowWidth)
    }

    static func globalLayerCount(_ text: [String: Any]) -> Int {
        let layers = text["num_hidden_layers"] as? Int ?? 0
        let shared = text["num_kv_shared_layers"] as? Int ?? 0
        let owning = max(0, layers - shared)
        if let types = text["layer_types"] as? [String] {
            return types.prefix(owning).filter { $0 == "full_attention" }.count
        }
        let pattern = text["sliding_window_pattern"] as? Int ?? 5
        return pattern > 0 ? (0 ..< owning).filter { ($0 + 1).isMultiple(of: pattern) }.count : 0
    }

    static func number(_ value: Any?) -> Double? {
        (value as? Double) ?? (value as? Int).map(Double.init) ?? (value as? NSNumber)?.doubleValue
    }
}

extension KVCacheGeometry {
    init(_ base: KVCacheGeometry, elementsPerToken: Double, keyElementsPerToken: Double) {
        self.elementsPerToken = elementsPerToken
        self.keyElementsPerToken = keyElementsPerToken
        windowedElementsPerToken = base.windowedElementsPerToken
        window = base.window
        fixedStateBytes = base.fixedStateBytes
        quantizedGroupSize = base.quantizedGroupSize
        isFromConfig = base.isFromConfig
    }

    public func sharingKEqVRows(_ layout: KEqVRowLayout) -> KVCacheGeometry {
        let heads = Double(layout.globalLayers * layout.kvHeads)
        let saved = heads * Double(2 * layout.headDim - layout.rowWidth)
        guard isFromConfig, elementsPerToken >= saved else { return self }
        return KVCacheGeometry(self, elementsPerToken: elementsPerToken - saved, keyElementsPerToken: keyElementsPerToken)
    }
}
