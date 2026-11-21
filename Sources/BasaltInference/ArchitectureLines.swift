import Foundation

func architectureLines(config json: [String: Any]) -> [String] {
    let text = json["text_config"] as? [String: Any]
    let experts = (json["num_experts"] as? Int
        ?? text?["num_experts"] as? Int
        ?? json["n_routed_experts"] as? Int
        ?? text?["n_routed_experts"] as? Int
        ?? json["num_local_experts"] as? Int
        ?? text?["num_local_experts"] as? Int
        ?? 0)
    let activeExperts = (json["num_experts_per_tok"] as? Int ?? text?["num_experts_per_tok"] as? Int)
    let layerTypes = (json["layer_types"] as? [String] ?? text?["layer_types"] as? [String])
    let slidingCount = layerTypes?.filter { $0.contains("sliding") }.count ?? 0
    let topology: String
    if experts > 0 {
        let activePart = activeExperts.map { ", top-\($0)" } ?? ""
        let attnPart = slidingCount > 0 ? ", \(slidingCount)/\(layerTypes?.count ?? 0) sliding" : ""
        topology = "Topology: Sparse MoE (Experts: \(experts)\(activePart)\(attnPart))"
    } else {
        topology = "Topology: Dense"
    }
    return [
        "--- Model Architecture ---",
        "Type: \(json["model_type"] as? String ?? "Unknown") (\((json["architectures"] as? [String])?.first ?? "Unknown"))",
        topology,
        "Quantization: \(quantizationDescription(config: json))",
        "--------------------------",
    ]
}

private func quantizationDescription(config json: [String: Any]) -> String {
    let quantization = json["quantization_config"] as? [String: Any] ?? json["quantization"] as? [String: Any]
    let dynamic = json["quantization"] as? [String: Any] ?? [:]
    let base: String? = if let bits = quantization?["bits"] as? Int {
        "\(bits)-bit (Group Size: \(quantization?["group_size"] as? Int ?? -1))\((quantization?["mode"] as? String).map { " [\($0)]" } ?? "")"
    } else if !dynamic.isEmpty {
        "Dynamic Quantization (DQ - per-layer, \(dynamic.count) layers)"
    } else {
        nil
    }
    if let manifest = moduleManifestDescription(config: json) {
        return base.map { "\(manifest) + \($0)" } ?? manifest
    }
    return base ?? "None (FP16/BF16 Base)"
}

private func moduleManifestDescription(config json: [String: Any]) -> String? {
    let candidates = [
        ("macaq_quantization", "MACaQ"),
        ("flatquant_quantization", "FlatQuant"),
        ("lirquant_quantization", "LIRQuant"),
    ]
    for (key, label) in candidates {
        guard let manifest = json[key] as? [String: Any],
              let modules = manifest["modules"] as? [String: Any],
              !modules.isEmpty else { continue }
        let specs = modules.values.compactMap { $0 as? [String: Any] }
        let bits = Set(specs.compactMap { $0["bits"] as? Int }).sorted()
        let groups = Set(specs.compactMap { $0["group_size"] as? Int }).sorted()
        let modes = Set(specs.compactMap { $0["mode"] as? String }).sorted()
        let bitsPart = bits.isEmpty ? "" : " \(bits.map(String.init).joined(separator: "/"))-bit"
        let groupPart = groups.isEmpty ? "" : " (Group Size: \(groups.map(String.init).joined(separator: "/")))"
        let modePart = modes.isEmpty ? "" : " [\(modes.joined(separator: "/"))]"
        let suffix = modules.count == 1 ? "module" : "modules"
        return "\(label)\(bitsPart)\(groupPart)\(modePart) (\(modules.count) \(suffix))"
    }
    return nil
}
