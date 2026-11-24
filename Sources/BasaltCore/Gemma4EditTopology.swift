import Foundation

public enum EditTargetScope: String, Sendable, Codable, Equatable {
    case textBackboneOnly = "text_backbone_only"
    case includeMultimodal = "include_multimodal"
}

public struct Gemma4EditTopology: Sendable, Codable, Equatable {
    public let numLayers: Int
    public let kvDonorForLayers: [Int: [Int]]
    public let kEqVLayers: [Int]
    public let hasParallelDenseAndMoE: Bool
    public let numExperts: Int?
    public let topKExperts: Int?
    public let hasPerLayerEmbeddings: Bool
    public let finalLogitSoftcapping: Float?

    private static let supportedModelTypes: Set<String> = [
        "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text",
    ]

    public init?(config root: [String: Any]) {
        let text = root["text_config"] as? [String: Any] ?? root
        let rootType = (root["model_type"] as? String)?.lowercased()
        let textType = (text["model_type"] as? String)?.lowercased()
        guard rootType.map(Self.supportedModelTypes.contains) == true
            || textType.map(Self.supportedModelTypes.contains) == true
        else {
            return nil
        }
        guard let layers = text["num_hidden_layers"] as? Int, layers > 0 else {
            return nil
        }
        let layerTypes = Self.resolvedLayerTypes(text, count: layers)
        let sharedCount = min(layers, max(0, text["num_kv_shared_layers"] as? Int ?? 0))
        let firstShared = layers - sharedCount
        let kEqV = text["attention_k_eq_v"] as? Bool ?? false
        let enableMoE = text["enable_moe_block"] as? Bool ?? false
        let pleDim = text["hidden_size_per_layer_input"] as? Int ?? 0

        self.numLayers = layers
        self.kvDonorForLayers = Self.donorMap(layerTypes: layerTypes, firstShared: firstShared)
        self.kEqVLayers = kEqV
            ? layerTypes.enumerated().compactMap { $0.element == "full_attention" ? $0.offset : nil }
            : []
        self.hasParallelDenseAndMoE = enableMoE
        self.numExperts = enableMoE ? text["num_experts"] as? Int : nil
        self.topKExperts = enableMoE ? text["top_k_experts"] as? Int : nil
        self.hasPerLayerEmbeddings = pleDim > 0
        self.finalLogitSoftcapping = Self.floatValue(text["final_logit_softcapping"])
    }

    public func blastRadiusWarning(for targetPath: String) -> String? {
        let parts = targetPath.split(separator: ".").map(String.init)
        guard let layersIndex = parts.firstIndex(of: "layers"),
              layersIndex + 1 < parts.count,
              let layer = Int(parts[layersIndex + 1])
        else {
            return nil
        }
        let isKProj = parts.contains("k_proj")
        let isVProj = parts.contains("v_proj")
        var warnings: [String] = []
        if (isKProj || isVProj), let recipients = kvDonorForLayers[layer], !recipients.isEmpty {
            let list = recipients.map(String.init).joined(separator: ",")
            warnings.append("layer \(layer) is a KV donor for shared layers [\(list)]")
        }
        if isKProj, kEqVLayers.contains(layer) {
            warnings.append("layer \(layer) uses attention_k_eq_v (k_proj defines both keys and values)")
        }
        return warnings.isEmpty ? nil : warnings.joined(separator: "; ")
    }

    private static func resolvedLayerTypes(_ text: [String: Any], count: Int) -> [String] {
        if let explicit = text["layer_types"] as? [String], explicit.count == count {
            return explicit
        }
        let pattern = max(1, text["sliding_window_pattern"] as? Int ?? 5)
        return (0 ..< count).map { ($0 + 1).isMultiple(of: pattern) ? "full_attention" : "sliding_attention" }
    }

    private static func donorMap(layerTypes: [String], firstShared: Int) -> [Int: [Int]] {
        guard firstShared > 0, firstShared < layerTypes.count else { return [:] }
        var lastByType: [String: Int] = [:]
        for idx in 0 ..< firstShared {
            lastByType[layerTypes[idx]] = idx
        }
        var donors: [Int: [Int]] = [:]
        for idx in firstShared ..< layerTypes.count {
            if let donor = lastByType[layerTypes[idx]] {
                donors[donor, default: []].append(idx)
            }
        }
        return donors
    }

    private static func floatValue(_ raw: Any?) -> Float? {
        if let f = raw as? Float { return f }
        if let d = raw as? Double { return Float(d) }
        if let i = raw as? Int { return Float(i) }
        return nil
    }
}
