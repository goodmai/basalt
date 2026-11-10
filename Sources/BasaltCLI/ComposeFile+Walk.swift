import Foundation
import BasaltCore
import Yams

extension ComposeFile.Walk {
    mutating func fields(_ node: Node, path: String, allowed: Set<String>) -> [String: Node]? {
        if path.isEmpty == false { lines[path] = lines[path] ?? node.mark?.line }
        guard let mapping = node.mapping else {
            issue(path.isEmpty ? "(root)" : path, node, "must be a mapping")
            return nil
        }
        var fields: [String: Node] = [:]
        for (key, value) in mapping {
            let name = key.string ?? ""
            let child = path.isEmpty ? name : "\(path).\(name)"
            if ComposeFile.secretKeys.contains(name.lowercased()) {
                issue(child, key, "secrets never go in a compose file; name the environment variable that holds it with api_key_env")
            } else if !allowed.contains(name) {
                issue(child, key, "unknown key; expected one of \(allowed.sorted().joined(separator: ", "))")
            } else {
                fields[name] = value
                lines[child] = key.mark?.line
            }
        }
        return fields
    }

    mutating func scan(_ node: Node, path: String) {
        if let value = node.scalar?.string {
            if value.contains("${") { issue(path, node, "environment interpolation applies only to secrets, by name through api_key_env") }
            if SecretRedactor.redact(value) != value { issue(path, node, "looks like a secret; name the environment variable that holds it with api_key_env") }
        }
        for (key, value) in node.mapping ?? [:] { scan(value, path: path.isEmpty ? (key.string ?? "") : "\(path).\(key.string ?? "")") }
        for (index, item) in (node.sequence.map { Array($0) } ?? []).enumerated() { scan(item, path: "\(path)[\(index)]") }
    }

    mutating func text(_ node: Node?, _ path: String) -> String? {
        guard let node else { return nil }
        guard let value = node.scalar?.string, node.mapping == nil, node.sequence == nil else {
            issue(path, node, "must be a string")
            return nil
        }
        return value
    }

    mutating func list(_ node: Node?, _ path: String) -> [String] {
        guard let node else { return [] }
        let items = node.sequence?.compactMap { $0.scalar?.string }
        guard let items, items.count == node.sequence?.count else {
            issue(path, node, "must be a list of names")
            return []
        }
        return items
    }

    mutating func number(_ node: Node?, _ key: String) -> Double? {
        guard let node else { return nil }
        guard let value = node.int.map(Double.init) ?? node.float else {
            issue("pipeline.decide.\(key)", node, "must be a number")
            return nil
        }
        return value
    }

    mutating func issue(_ path: String, _ node: Node, _ message: String) {
        issues.append(.init(path: path, line: node.mark?.line, column: node.mark?.column, message: message))
    }

    static func seconds(_ node: Node) -> Int? {
        if let seconds = node.int { return seconds }
        guard let text = node.scalar?.string, let unit = text.last, let value = Int(text.dropLast()) else { return nil }
        switch unit {
        case "s": return value
        case "m": return value * 60
        case "h": return value * 3600
        default: return nil
        }
    }
}

extension ComposeFile {
    static let bundled: [String: String] = [
        "document-qa": """
            version: 1
            name: document-qa
            memory: auto
            models:
              chat:
                source: mlx-community/gemma-4-12B-it-qat-4bit
              ocr:
                source: mlx-community/PaddleOCR-VL-1.5-8bit
                keep: on-demand
                idle_unload: 120s
            pipeline:
              perceive:
                documents: [ocr, apple-vision]
              decide:
                cascade: [chat]
                abstain_below: 0.75
                min_margin: 0.1
              answer: chat
            """,
        "voice-agent": """
            version: 1
            name: voice-agent
            memory: sequential
            models:
              chat:
                source: mlx-community/gemma-4-12B-it-qat-4bit
              asr:
                source: divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX
                keep: sequential
            pipeline:
              perceive:
                audio: [asr, speech-analyzer]
              answer: chat
            """,
    ]
}
