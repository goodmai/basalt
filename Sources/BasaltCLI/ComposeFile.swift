import Foundation
import BasaltCore
import Yams

enum ComposeFile {
    static let secretKeys: Set<String> = ["api_key", "apikey", "key", "token", "secret", "password"]

    static func read(_ reference: String) throws(ServeConfigError) -> (text: String, file: String, isFile: Bool) {
        if let bundled = bundled[reference] { return (bundled, "\(reference) (bundled)", false) }
        guard let data = FileManager.default.contents(atPath: reference), data.count <= ServeConfigFile.maxBytes else {
            throw .unreadable(file: reference, reason: "not a readable compose file of at most \(ServeConfigFile.maxBytes) bytes, nor a bundled one (\(bundled.keys.sorted().joined(separator: ", ")))")
        }
        return (String(decoding: data, as: UTF8.self), reference, true)
    }

    static func load(_ reference: String, context: ComposeContext = .machine) throws(ServeConfigError) -> ComposeSpec {
        let source = try read(reference)
        return try parse(source.text, file: source.file, context: source.isFile ? context.with(base: reference) : context)
    }

    static func compose(_ yaml: String, file: String) throws(ServeConfigError) -> Node? {
        do {
            return try Yams.compose(yaml: yaml)
        } catch {
            throw .unreadable(file: file, reason: String(describing: error))
        }
    }

    static func parse(_ yaml: String, file: String, context: ComposeContext = ComposeContext()) throws(ServeConfigError) -> ComposeSpec {
        let resolved = resolve(yaml, root: try compose(yaml, file: file), context: context)
        let root = try compose(resolved.text, file: file)
        var walk = Walk(declared: resolved.declared, active: resolved.active)
        walk.issues = resolved.issues
        let spec = root.flatMap { walk.root($0) }.map { relative($0, base: context.base) }
        if let spec, walk.issues.isEmpty {
            walk.issues += spec.issues.map { ServeConfigError.Issue(path: $0.path, line: walk.line($0.path), message: $0.message) }
        }
        guard let spec, walk.issues.isEmpty else { throw .invalid(file: file, issues: walk.issues.isEmpty ? [.init(path: "", message: "the file is empty")] : walk.issues) }
        return spec
    }

    struct Walk {
        var issues: [ServeConfigError.Issue] = []
        var lines: [String: Int] = [:]
        var prefix = "pipeline"
        var version = ComposeSchema.versions.upperBound
        var declared: Set<String> = []
        var active: Set<String> = []

        init(declared: Set<String> = [], active: Set<String> = []) {
            (self.declared, self.active) = (declared, active)
        }

        func line(_ path: String) -> Int? {
            var path = Substring(path)
            while !path.isEmpty {
                if let line = lines[String(path)] { return line }
                guard let cut = path.lastIndex(where: { $0 == "." || $0 == "[" }) else { return nil }
                path = path[..<cut]
            }
            return nil
        }

        mutating func root(_ node: Node) -> ComposeSpec? {
            scan(node, path: "")
            let declared = node.mapping?["version"]?.int ?? 0
            version = ComposeSchema.versions.contains(declared) ? declared : ComposeSchema.versions.upperBound
            guard let top = fields(node, path: "", section: "root") else { return nil }
            let models = named(top["models"], path: "models").compactMap { name, value in model(value, path: "models.\(name)").map { (name, $0) } }
            if top["models"]?.mapping == nil { issue("models", top["models"] ?? node, "a compose file declares its models as a mapping of name to model") }
            for (name, value) in named(top["profiles"], path: "profiles") { profile(value, path: "profiles.\(name)") }
            guard let pipelineNode = top["pipeline"] else {
                issue("pipeline", node, "a compose file needs a pipeline, at least { answer: <model> }")
                return nil
            }
            let pipeline = fields(pipelineNode, path: "pipeline", section: "pipeline") ?? [:]
            var spec = ComposeSpec(name: text(top["name"], "name") ?? "", models: Dictionary(models) { first, _ in first }, answer: answer(pipeline["answer"]) ?? "")
            spec.version = declared
            spec.budget = budget(top["budget"], "budget")
            if let routes = pipeline["answer"]?.mapping?["routes"] { spec.routes = self.routes(routes) }
            if let memory = text(top["memory"], "memory") {
                if let parsed = PipelineMemory(rawValue: memory) { spec.memory = parsed } else { issue("memory", top["memory"]!, "must be resident, sequential or auto") }
            }
            perceive(pipeline["perceive"], into: &spec)
            decide(pipeline["decide"], into: &spec)
            experimental(top["experimental"], into: &spec)
            mcp(top["mcp"], into: &spec)
            for (name, value) in named(top["pipelines"], path: "pipelines") {
                prefix = "pipelines.\(name)"
                if let named = self.pipeline(value, models: spec.models) { spec.pipelines[name] = named }
            }
            prefix = "pipeline"
            if let named = top["pipelines"], named.mapping == nil { issue("pipelines", named, "a mapping of pipeline name to perceive, decide and answer") }
            return spec
        }

        mutating func pipeline(_ node: Node, models: [String: ComposeSpec.Model]) -> PipelineSpec? {
            guard let field = fields(node, path: prefix, section: "pipeline"), included(field["profiles"], path: "\(prefix).profiles") else { return nil }
            var view = ComposeSpec(name: "", models: models, answer: answer(field["answer"]) ?? "")
            if let routes = field["answer"]?.mapping?["routes"] { view.routes = self.routes(routes) }
            perceive(field["perceive"], into: &view)
            decide(field["decide"], into: &view)
            return view.defaultSpec
        }

        mutating func model(_ node: Node, path: String) -> ComposeSpec.Model? {
            lines[path] = node.mark?.line
            guard let field = fields(node, path: path, section: "model"), included(field["profiles"], path: "\(path).profiles") else { return nil }
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
            guard let node, let field = fields(node, path: "\(prefix).perceive", section: "perceive") else { return }
            spec.documents = list(field["documents"], "\(prefix).perceive.documents")
            spec.audio = list(field["audio"], "\(prefix).perceive.audio")
        }

        mutating func decide(_ node: Node?, into spec: inout ComposeSpec) {
            let path = "\(prefix).decide"
            guard let node, let field = fields(node, path: path, section: "decide") else { return }
            spec.cascade = list(field["cascade"], "\(path).cascade")
            spec.escalateBelow = number(field["escalate_below"], "\(path).escalate_below")
            spec.abstainBelow = number(field["abstain_below"], "\(path).abstain_below")
            spec.minMargin = number(field["min_margin"], "\(path).min_margin")
        }

        mutating func answer(_ node: Node?) -> String? {
            guard let node else { return nil }
            if let name = node.scalar?.string { return name }
            guard let field = fields(node, path: "\(prefix).answer", section: "answer") else { return nil }
            if let model = field["model"] { return text(model, "\(prefix).answer.model") }
            return field["routes"]?.sequence?.last.flatMap { $0.mapping?["model"]?.scalar?.string }
        }
    }
}
