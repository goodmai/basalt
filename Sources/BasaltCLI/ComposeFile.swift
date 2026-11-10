import Foundation
import BasaltCore
import Yams

enum ComposeFile {
    static let secretKeys: Set<String> = ["api_key", "apikey", "key", "token", "secret", "password"]

    static func load(_ reference: String) throws(ServeConfigError) -> ComposeSpec {
        if let bundled = bundled[reference] { return try parse(bundled, file: "\(reference) (bundled)") }
        guard let data = FileManager.default.contents(atPath: reference), data.count <= ServeConfigFile.maxBytes else {
            throw .unreadable(file: reference, reason: "not a readable compose file of at most \(ServeConfigFile.maxBytes) bytes, nor a bundled one (\(bundled.keys.sorted().joined(separator: ", ")))")
        }
        return try parse(String(decoding: data, as: UTF8.self), file: reference)
    }

    static func parse(_ yaml: String, file: String) throws(ServeConfigError) -> ComposeSpec {
        let root: Node?
        do {
            root = try Yams.compose(yaml: yaml)
        } catch {
            throw .unreadable(file: file, reason: String(describing: error))
        }
        var walk = Walk()
        let spec = root.flatMap { walk.root($0) }
        if let spec, walk.issues.isEmpty {
            walk.issues += spec.issues.map { ServeConfigError.Issue(path: $0.path, line: walk.lines[$0.path.split(separator: ".").prefix(2).joined(separator: ".")], message: $0.message) }
        }
        guard let spec, walk.issues.isEmpty else { throw .invalid(file: file, issues: walk.issues.isEmpty ? [.init(path: "", message: "the file is empty")] : walk.issues) }
        return spec
    }

    struct Walk {
        var issues: [ServeConfigError.Issue] = []
        var lines: [String: Int] = [:]

        mutating func root(_ node: Node) -> ComposeSpec? {
            scan(node, path: "")
            guard let top = fields(node, path: "", allowed: ["version", "name", "memory", "models", "pipeline", "budget"]) else { return nil }
            let models = (top["models"]?.mapping.map { mapping in
                Dictionary(mapping.compactMap { key, value in key.string.flatMap { name in model(value, path: "models.\(name)").map { (name, $0) } } }) { first, _ in first }
            }) ?? [:]
            if top["models"]?.mapping == nil { issue("models", top["models"] ?? node, "a compose file declares its models as a mapping of name to model") }
            let pipeline = fields(top["pipeline"] ?? node, path: "pipeline", allowed: ["perceive", "decide", "answer"]) ?? [:]
            var spec = ComposeSpec(name: text(top["name"], "name") ?? "", models: models, answer: answer(pipeline["answer"]) ?? "")
            spec.version = top["version"]?.int ?? 0
            spec.budget = budget(top["budget"], "budget")
            if let routes = pipeline["answer"]?.mapping?["routes"] { spec.routes = self.routes(routes) }
            if let memory = text(top["memory"], "memory") {
                if let parsed = PipelineMemory(rawValue: memory) { spec.memory = parsed } else { issue("memory", top["memory"]!, "must be resident, sequential or auto") }
            }
            perceive(pipeline["perceive"], into: &spec)
            decide(pipeline["decide"], into: &spec)
            return spec
        }

        mutating func model(_ node: Node, path: String) -> ComposeSpec.Model? {
            lines[path] = node.mark?.line
            guard let field = fields(node, path: path, allowed: ["source", "keep", "idle_unload", "api_key_env", "data", "budget"]) else { return nil }
            var model = ComposeSpec.Model(source: .init(text(field["source"], "\(path).source") ?? ""))
            if let keep = text(field["keep"], "\(path).keep") {
                model.keep = ComposeSpec.Keep(rawValue: keep)
                if model.keep == nil { issue("\(path).keep", field["keep"]!, "must be resident, on-demand or sequential") }
            }
            if let idle = field["idle_unload"] {
                model.idleUnload = Self.seconds(idle)
                if model.idleUnload == nil { issue("\(path).idle_unload", idle, "a duration such as 120s, 5m or 1h") }
            }
            (model.apiKeyEnv, model.data) = (text(field["api_key_env"], "\(path).api_key_env"), text(field["data"], "\(path).data"))
            model.budget = budget(field["budget"], "\(path).budget")
            return model
        }

        mutating func perceive(_ node: Node?, into spec: inout ComposeSpec) {
            guard let node, let field = fields(node, path: "pipeline.perceive", allowed: ["documents", "audio"]) else { return }
            spec.documents = list(field["documents"], "pipeline.perceive.documents")
            spec.audio = list(field["audio"], "pipeline.perceive.audio")
        }

        mutating func decide(_ node: Node?, into spec: inout ComposeSpec) {
            guard let node, let field = fields(node, path: "pipeline.decide", allowed: ["cascade", "escalate_below", "abstain_below", "min_margin"]) else { return }
            spec.cascade = list(field["cascade"], "pipeline.decide.cascade")
            (spec.escalateBelow, spec.abstainBelow, spec.minMargin) = (number(field["escalate_below"], "escalate_below"), number(field["abstain_below"], "abstain_below"), number(field["min_margin"], "min_margin"))
        }

        mutating func answer(_ node: Node?) -> String? {
            guard let node else { return nil }
            if let name = node.scalar?.string { return name }
            guard let field = fields(node, path: "pipeline.answer", allowed: ["model", "routes"]) else { return nil }
            if let model = field["model"] { return text(model, "pipeline.answer.model") }
            return field["routes"]?.sequence?.last.flatMap { $0.mapping?["model"]?.scalar?.string }
        }
    }
}
