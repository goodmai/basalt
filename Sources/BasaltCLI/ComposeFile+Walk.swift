import Foundation
import BasaltCore
import Yams

extension ComposeFile.Walk {
    mutating func fields(_ node: Node, path: String, section: String) -> [String: Node]? {
        if path.isEmpty == false { lines[path] = lines[path] ?? node.mark?.line }
        guard let mapping = node.mapping else {
            issue(path.isEmpty ? "(root)" : path, node, "must be a mapping")
            return nil
        }
        let (known, allowed) = (ComposeSchema.keys(section), ComposeSchema.keys(section, version: version))
        var fields: [String: Node] = [:]
        for (key, value) in mapping {
            let name = key.string ?? ""
            let child = path.isEmpty ? name : "\(path).\(name)"
            if key.scalar == nil {
                issue(path.isEmpty ? "(root)" : path, key, "keys are plain names")
            } else if ComposeFile.secretKeys.contains(name.lowercased()) {
                issue(child, key, "secrets never go in a compose file; name the environment variable that holds it with api_key_env")
            } else if known.contains(name), !allowed.contains(name) {
                issue(child, key, "needs version: \(ComposeSchema.since(name, in: section))")
            } else if !allowed.contains(name) {
                issue(child, key, "unknown key; expected one of \(allowed.sorted().joined(separator: ", "))")
            } else {
                fields[name] = value
                lines[child] = key.mark?.line
            }
        }
        return fields
    }

    mutating func named(_ node: Node?, path: String) -> [(String, Node)] {
        var pairs: [(String, Node)] = []
        for (key, value) in node?.mapping ?? [:] {
            guard key.scalar != nil, let name = key.string, !name.isEmpty else {
                issue(path, key, "keys are plain names")
                continue
            }
            lines["\(path).\(name)"] = key.mark?.line
            pairs.append((name, value))
        }
        return pairs
    }

    mutating func included(_ node: Node?, path: String) -> Bool {
        guard let node else { return true }
        let names = list(node, path)
        for name in names where !declared.contains(name) { issue(path, node, "\(name) is not declared under profiles") }
        return names.contains(where: active.contains)
    }

    mutating func profile(_ node: Node, path: String) {
        guard let field = fields(node, path: path, section: "profile"), let when = field["when"] else { return }
        guard let machine = fields(when, path: "\(path).when", section: "machine") else { return }
        _ = text(machine["chip"], "\(path).when.chip")
        if let range = machine["memory_gb"], let bounds = fields(range, path: "\(path).when.memory_gb", section: "range") {
            for (name, value) in bounds { _ = number(value, "\(path).when.memory_gb.\(name)") }
        }
    }

    mutating func scan(_ node: Node, path: String) {
        if let value = node.scalar?.string {
            if let match = value.firstMatch(of: ComposeFile.reference) {
                issue(path, node, "${\(match.1)} is not a variable declared under variables (version 2); environment interpolation applies only to secrets, by name through api_key_env")
            }
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

    mutating func number(_ node: Node?, _ path: String) -> Double? {
        guard let node else { return nil }
        guard let value = node.int.map(Double.init) ?? node.float else {
            issue(path, node, "must be a number")
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

extension ComposeFile.Walk {
    mutating func routes(_ node: Node) -> [AnswerRoute] {
        guard let items = node.sequence, !items.isEmpty else {
            issue("\(prefix).answer.routes", node, "a list of routes, each with when, model and an optional fallback")
            return []
        }
        return items.enumerated().compactMap { index, item in
            let path = "\(prefix).answer.routes[\(index)]"
            guard let field = fields(item, path: path, section: "route") else { return nil }
            var route = AnswerRoute(model: text(field["model"], "\(path).model") ?? "", fallback: text(field["fallback"], "\(path).fallback"))
            if let when = field["when"] { route.when = self.when(when, path: "\(path).when") }
            return route
        }
    }

    mutating func when(_ node: Node, path: String) -> AnswerRoute.When {
        guard let field = fields(node, path: path, section: "when") else { return .init() }
        var when = AnswerRoute.When()
        if let modality = text(field["modality"], "\(path).modality") {
            when.modality = AnswerRoute.Modality(rawValue: modality)
            if when.modality == nil { issue("\(path).modality", field["modality"]!, "must be documents, audio or text") }
        }
        if let decide = field["decide"] {
            var labels: [String: String] = [:]
            for (key, value) in decide.mapping ?? [:] {
                if let name = key.string, let label = value.scalar?.string { labels[name] = label }
            }
            if decide.mapping == nil || labels.count != decide.mapping?.count { issue("\(path).decide", decide, "a mapping of question ID to the label that selects this route") }
            when.decide = labels
        }
        if let tokens = field["prompt_tokens"] {
            when.promptTokensOver = tokens.mapping?["over"]?.int
            if when.promptTokensOver == nil { issue("\(path).prompt_tokens", tokens, "{ over: <tokens> }") }
        }
        (when.escalated, when.costLeft) = (flag(field["escalated"], "\(path).escalated"), flag(field["cost_left"], "\(path).cost_left"))
        return when
    }

    mutating func flag(_ node: Node?, _ path: String) -> Bool? {
        guard let node else { return nil }
        guard let value = node.bool else {
            issue(path, node, "must be true or false")
            return nil
        }
        return value
    }

    mutating func budget(_ node: Node?, _ path: String) -> Double? {
        guard let node, let field = fields(node, path: path, section: "budget") else { return nil }
        guard let value = field["usd_per_day"].flatMap({ $0.int.map(Double.init) ?? $0.float }) else {
            issue(path, node, "{ usd_per_day: <amount> }")
            return nil
        }
        return value
    }
}
