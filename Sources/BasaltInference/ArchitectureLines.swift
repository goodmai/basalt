import Foundation

func architectureLines(config json: [String: Any]) -> [String] {
    let text = json["text_config"] as? [String: Any]
    let experts = (json["num_experts"] as? Int ?? text?["num_experts"] as? Int ?? 0)
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
    let quantization = json["quantization_config"] as? [String: Any] ?? json["quantization"] as? [String: Any]
    let dynamic = json["quantization"] as? [String: Any] ?? [:]
    let precision = if let bits = quantization?["bits"] as? Int {
        "Quantization: \(bits)-bit (Group Size: \(quantization?["group_size"] as? Int ?? -1))\((quantization?["mode"] as? String).map { " [\($0)]" } ?? "")"
    } else if !dynamic.isEmpty {
        "Quantization: Dynamic Quantization (DQ - per-layer, \(dynamic.count) layers)"
    } else {
        "Quantization: None (FP16/BF16 Base)"
    }
    return [
        "--- Model Architecture ---",
        "Type: \(json["model_type"] as? String ?? "Unknown") (\((json["architectures"] as? [String])?.first ?? "Unknown"))",
        topology,
        precision,
        "--------------------------",
    ]
}
