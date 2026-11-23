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
            try publish(config: JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]), source: directory, at: staging)
        } catch {
            throw Refusal(reason: "the staged config could not be written to \(staging.path): \(error.localizedDescription)")
        }
        return Staged(directory: staging, nativeContext: native, effectiveContext: effective)
    }

    static func publish(config: Data, source: URL, at staging: URL) throws {
        let fm = FileManager.default
        let names = try fm.contentsOfDirectory(atPath: source.path).filter { $0 != "config.json" && !$0.hasPrefix(".") }
        if matches(staging, config: config, names: names, source: source) { return }
        try fm.createDirectory(at: staging.deletingLastPathComponent(), withIntermediateDirectories: true)
        sweepAbandonedTemporaries(beside: staging, olderThan: Date().addingTimeInterval(-3600))
        let temporary = staging.deletingLastPathComponent().appending(component: "\(staging.lastPathComponent).\(UUID().uuidString)")
        try fm.createDirectory(at: temporary, withIntermediateDirectories: false)
        defer { try? fm.removeItem(at: temporary) }
        try config.write(to: temporary.appending(component: "config.json"))
        for name in names {
            try fm.createSymbolicLink(at: temporary.appending(component: name), withDestinationURL: source.appending(component: name).resolvingSymlinksInPath())
        }
        guard rename(temporary.path, staging.path) != 0 else { return }
        if !matches(staging, config: config, names: names, source: source) {
            _ = try fm.replaceItemAt(staging, withItemAt: temporary)
        }
    }

    static func matches(_ staging: URL, config: Data, names: [String], source: URL) -> Bool {
        guard (try? Data(contentsOf: staging.appending(component: "config.json"))) == config,
              let present = try? FileManager.default.contentsOfDirectory(atPath: staging.path).filter({ !$0.hasPrefix(".") }),
              Set(present) == Set(names + ["config.json"]) else { return false }
        return names.allSatisfy { name in
            staging.appending(component: name).resolvingSymlinksInPath() == source.appending(component: name).resolvingSymlinksInPath()
        }
    }

    static func sweepAbandonedTemporaries(beside staging: URL, olderThan cutoff: Date) {
        let fm = FileManager.default
        let parent = staging.deletingLastPathComponent()
        for name in (try? fm.contentsOfDirectory(atPath: parent.path)) ?? [] where name.hasPrefix("\(staging.lastPathComponent).") {
            let url = parent.appending(component: name)
            guard let modified = (try? fm.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date, modified < cutoff else { continue }
            try? fm.removeItem(at: url)
        }
    }
}
