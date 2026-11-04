import Foundation

func architectureLines(config json: [String: Any]) -> [String] {
    let text = json["text_config"] as? [String: Any]
    let experts = (json["num_experts"] as? Int ?? text?["num_experts"] as? Int ?? 0)
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
        experts > 0 ? "Topology: Sparse MoE (Experts: \(experts))" : "Topology: Dense",
        precision,
        "--------------------------",
    ]
}
