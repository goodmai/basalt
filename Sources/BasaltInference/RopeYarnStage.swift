import CryptoKit
import Foundation

enum RopeYarnStage {
    struct Refusal: Error, Equatable {
        let reason: String
    }

    struct Staged: Equatable {
        let directory: URL
        let nativeContext: Int
        let effectiveContext: Int
    }

    static let modelTypes: Set<String> = ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"]

    static func refusal(modelType: String?) -> Refusal? {
        guard let modelType else { return Refusal(reason: "config.json names no model_type") }
        if modelType.hasPrefix("gemma4") {
            return Refusal(reason: "Gemma 4 uses proportional RoPE (p-RoPE) on its global layers and takes no RoPE scaling (#729)")
        }
        return modelTypes.contains(modelType) ? nil : Refusal(reason: "YaRN is wired for Qwen 3.5, 3.6 and 3.8 (qwen3_5) only, not \(modelType)")
    }

    static func rewritten(_ root: [String: Any], factor: Double) throws(Refusal) -> (config: [String: Any], native: Int, effective: Int) {
        let nested = root["text_config"] as? [String: Any]
        var text = nested ?? root
        guard let native = text["max_position_embeddings"] as? Int, native > 0 else {
            throw Refusal(reason: "config.json has no max_position_embeddings to scale")
        }
        if let scaling = text["rope_scaling"], !(scaling is NSNull) { throw Refusal(reason: "config.json already sets rope_scaling") }
        var rope = text["rope_parameters"] as? [String: Any] ?? [:]
        if let type = (rope["rope_type"] ?? rope["type"]) as? String, type != "default" {
            throw Refusal(reason: "rope_parameters already uses rope_type \(type)")
        }
        for key in ["rope_theta", "partial_rotary_factor"] where rope[key] == nil { rope[key] = text[key] }
        rope["rope_type"] = "yarn"
        rope["factor"] = factor
        rope["original_max_position_embeddings"] = native
        let effective = Int((Double(native) * factor).rounded())
        text["rope_parameters"] = rope
        text["max_position_embeddings"] = effective
        var config = root
        if nested != nil { config["text_config"] = text } else { config = text }
        return (config, native, effective)
    }

    static func stagingDirectory(for directory: URL, factor: Double, root: URL) -> URL {
        let digest = SHA256.hash(data: Data(directory.standardizedFileURL.path.utf8)).map { String(format: "%02x", $0) }.joined()
        return root.appending(component: "\(digest.prefix(16))-yarn-x\(factor)")
    }

    static func stage(directory: URL, modelType: String?, factor: Double, root: URL) throws(Refusal) -> Staged {
        if let refusal = refusal(modelType: modelType) { throw refusal }
        guard factor > 1, factor <= 64 else { throw Refusal(reason: "the factor must lie in (1, 64], got \(factor)") }
        guard let data = try? Data(contentsOf: directory.appending(component: "config.json")),
              let root0 = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Refusal(reason: "config.json is unreadable")
        }
        let (config, native, effective) = try rewritten(root0, factor: factor)
        let staging = stagingDirectory(for: directory, factor: factor, root: root)
        do {
            let fm = FileManager.default
            try? fm.removeItem(at: staging)
            try fm.createDirectory(at: staging, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]).write(to: staging.appending(component: "config.json"))
            for name in try fm.contentsOfDirectory(atPath: directory.path) where name != "config.json" && !name.hasPrefix(".") {
                try fm.createSymbolicLink(at: staging.appending(component: name), withDestinationURL: directory.appending(component: name).resolvingSymlinksInPath())
            }
        } catch {
            throw Refusal(reason: "the staged config could not be written to \(staging.path): \(error.localizedDescription)")
        }
        return Staged(directory: staging, nativeContext: native, effectiveContext: effective)
    }
}
