import Foundation
import BasaltCore
import Yams

struct ComposeContext: Sendable, Equatable {
    var memoryGB: Double?
    var chip: String?
    var profiles: [String]?
    var variables: [String: String] = [:]
    var base: URL?

    static var machine: ComposeContext {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var brand = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("machdep.cpu.brand_string", &brand, &size, nil, 0)
        let chip = String(decoding: brand.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        return ComposeContext(memoryGB: Double(ProcessInfo.processInfo.physicalMemory) / 1e9, chip: chip.isEmpty ? nil : chip)
    }

    func overriding(assignments: [String], profiles: [String]) -> ComposeContext {
        var context = self
        context.profiles = profiles.isEmpty ? nil : profiles
        for assignment in assignments {
            let parts = assignment.split(separator: "=", maxSplits: 1).map(String.init)
            if let name = parts.first { context.variables[name] = parts.count > 1 ? parts[1] : "" }
        }
        return context
    }

    func with(base file: String) -> ComposeContext {
        var context = self
        context.base = URL(fileURLWithPath: (file as NSString).expandingTildeInPath).deletingLastPathComponent()
        return context
    }
}

extension ComposeFile {
    struct Resolved {
        var text: String
        var declared: Set<String> = []
        var active: Set<String> = []
        var issues: [ServeConfigError.Issue] = []
    }

    static var variableName: Regex<Substring> { /^[a-z][a-z0-9_]*$/ }
    static var reference: Regex<(Substring, Substring)> { /\$\{([^}]*)\}/ }

    static func resolve(_ yaml: String, root: Node?, context: ComposeContext) -> Resolved {
        var resolved = Resolved(text: yaml)
        guard let top = root?.mapping else { return resolved }
        var values = variables(top["variables"], into: &resolved)
        let profiles = top["profiles"]?.mapping.map { mapping in mapping.compactMap { key, value in key.string.map { ($0, value) } } } ?? []
        resolved.declared = Set(profiles.map(\.0))
        for name in context.profiles ?? [] where !resolved.declared.contains(name) {
            resolved.issues.append(.init(path: "profiles", message: "--profile \(name) is not declared under profiles (\(resolved.declared.sorted().joined(separator: ", ")))"))
        }
        for (name, profile) in profiles where context.profiles.map({ $0.contains(name) }) ?? matches(profile.mapping?["when"], context) {
            resolved.active.insert(name)
            values.merge(variables(profile.mapping?["variables"], into: &resolved)) { $1 }
        }
        for (name, value) in context.variables {
            guard values[name] != nil else {
                resolved.issues.append(.init(path: "variables", message: "--set \(name) is not declared under variables"))
                continue
            }
            values[name] = value
        }
        resolved.text = yaml.replacing(reference) { match in values[String(match.1)] ?? String(match.0) }
        return resolved
    }

    static func variables(_ node: Node?, into resolved: inout Resolved) -> [String: String] {
        var values: [String: String] = [:]
        for (key, value) in node?.mapping ?? [:] {
            let name = key.string ?? ""
            guard name.wholeMatch(of: variableName) != nil, let text = value.scalar?.string, !text.contains("\n"), !text.contains("${") else {
                resolved.issues.append(.init(path: "variables.\(name)", line: key.mark?.line, column: key.mark?.column,
                                             message: "a variable is a lowercase name with a one-line plain value, such as chat_repo: mlx-community/gemma-4-12B-it-qat-4bit"))
                continue
            }
            values[name] = text
        }
        return values
    }

    static func matches(_ when: Node?, _ context: ComposeContext) -> Bool {
        guard let when = when?.mapping else { return false }
        if let range = when["memory_gb"]?.mapping {
            guard let memory = context.memoryGB else { return false }
            if let under = range["under"].flatMap(number), !(memory < under) { return false }
            if let over = range["over"].flatMap(number), !(memory > over) { return false }
        }
        if let chip = when["chip"]?.scalar?.string, context.chip?.localizedCaseInsensitiveContains(chip) != true { return false }
        return true
    }

    static func number(_ node: Node) -> Double? { node.int.map(Double.init) ?? node.float }

    static func relative(_ spec: ComposeSpec, base: URL?) -> ComposeSpec {
        guard let base else { return spec }
        var spec = spec
        for (name, model) in spec.models {
            guard case .local(let source) = model.source, source.hasPrefix("./") || source.hasPrefix("../") else { continue }
            spec.models[name]?.source = .local(base.appending(path: source).standardizedFileURL.path)
        }
        return spec
    }
}
