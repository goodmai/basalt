import Foundation
import BasaltCore

public struct HarnessValidationReport: Codable, Sendable, Equatable {
    public let isValid: Bool
    public let errors: [String]
    public var warnings: [String]
    public let surface: String
    public let alwaysOnFiles: [String]
    public let alwaysOnChars: Int
    public let alwaysOnMaxChars: Int
    public let skillCount: Int
    public let subagentCount: Int
    public let scriptCount: Int
    public let untriggeredSkills: [String]
    public let diversityScore: Double?
}

public enum HarnessDiversity {
    static func lineJaccardDistance(_ first: HarnessTree, _ second: HarnessTree) -> Double {
        let setA = first.serializedLines()
        let setB = second.serializedLines()
        let union = setA.union(setB).count
        guard union > 0 else { return 0 }
        return 1 - Double(setA.intersection(setB).count) / Double(union)
    }

    public static func pairwiseDiversity(_ trees: [HarnessTree]) -> Double {
        guard trees.count >= 2 else { return 1 }
        var total = 0.0
        var pairs = 0
        for i in trees.indices {
            for j in trees.indices where j > i {
                total += lineJaccardDistance(trees[i], trees[j])
                pairs += 1
            }
        }
        return total / Double(pairs)
    }
}

public enum HarnessValidator {
    public static let claudeHookEvents: Set<String> = [
        "PreToolUse", "PostToolUse", "PostToolUseFailure", "Notification", "UserPromptSubmit",
        "Stop", "SubagentStart", "SubagentStop", "PreCompact", "SessionStart", "SessionEnd", "PermissionRequest"
    ]
    static let opencodeAgentModes: Set<String> = ["subagent", "primary", "all"]

    public static func validate(
        _ tree: HarnessTree,
        spec: HarnessSurfaceSpec = .claude,
        requireDiverseFrom siblings: [HarnessTree] = []
    ) -> HarnessValidationReport {
        var errors: [String] = []
        let alwaysOnPaths = tree.alwaysOnPaths(spec: spec)
        let alwaysOnChars = tree.alwaysOnCharCount(spec: spec)
        if alwaysOnChars > spec.alwaysOnMaxChars {
            errors.append("always-on files exceed budget (\(alwaysOnChars) > \(spec.alwaysOnMaxChars) chars)")
        }

        var skills: [(path: String, name: String)] = []
        var subagentCount = 0
        var scriptCount = 0
        for path in tree.files.keys.sorted() {
            let content = tree.files[path] ?? ""
            if let directoryName = spec.skillName(at: path) {
                if let name = validateSkill(path: path, directoryName: directoryName, content: content, errors: &errors) {
                    skills.append((path, name))
                }
            } else if spec.isAgentFile(path) {
                subagentCount += 1
                validateAgent(path: path, content: content, format: spec.agentFormat, errors: &errors)
            } else if path == spec.hookSettingsFile {
                validateHookSettings(content: content, path: path, errors: &errors)
            } else if spec.classify(path: path) == .scripts || spec.isUnder(spec.pluginDirectories, path) {
                scriptCount += 1
                if content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    errors.append("\(path): file is empty")
                }
            }
        }

        for (path, content) in tree.files {
            for reference in HarnessPath.scriptReferences(in: content)
            where !HarnessPath.resolve(reference, from: path).contains(where: { tree.files[$0] != nil }) {
                errors.append("\(path): references missing script '\(reference)'")
            }
        }
        errors = Array(Set(errors)).sorted()

        let untriggered = untriggeredSkills(skills, tree: tree, spec: spec)
        var diversity: Double?
        if !siblings.isEmpty {
            let score = HarnessDiversity.pairwiseDiversity(siblings + [tree])
            diversity = score
            if score < spec.minDiversity {
                errors.append(String(format: "candidate diversity %.4f is below minimum %.4f", score, spec.minDiversity))
            }
        }

        return HarnessValidationReport(
            isValid: errors.isEmpty,
            errors: errors,
            warnings: untriggered.map { "\($0): skill is never referenced by the always-on files, agents or other skills" },
            surface: spec.name,
            alwaysOnFiles: alwaysOnPaths,
            alwaysOnChars: alwaysOnChars,
            alwaysOnMaxChars: spec.alwaysOnMaxChars,
            skillCount: skills.count,
            subagentCount: subagentCount,
            scriptCount: scriptCount,
            untriggeredSkills: untriggered,
            diversityScore: diversity
        )
    }

    private static func validateSkill(path: String, directoryName: String, content: String, errors: inout [String]) -> String? {
        guard let frontmatter = Frontmatter.parse(content) else {
            errors.append("\(path): missing YAML frontmatter")
            return nil
        }
        guard let name = frontmatter.fields["name"], !name.isEmpty else {
            errors.append("\(path): frontmatter missing 'name'")
            return nil
        }
        if name != directoryName {
            errors.append("\(path): skill name '\(name)' does not match directory '\(directoryName)'")
        }
        if (frontmatter.fields["description"] ?? "").isEmpty {
            errors.append("\(path): frontmatter missing non-empty 'description'")
        }
        return name
    }

    private static func validateAgent(path: String, content: String, format: HarnessSurfaceSpec.AgentFormat, errors: inout [String]) {
        guard let frontmatter = Frontmatter.parse(content) else {
            errors.append("\(path): missing YAML frontmatter")
            return
        }
        if (frontmatter.fields["description"] ?? "").isEmpty {
            errors.append("\(path): agent frontmatter missing 'description'")
        }
        switch format {
        case .claude:
            if (frontmatter.fields["name"] ?? "").isEmpty {
                errors.append("\(path): agent frontmatter missing 'name'")
            }
        case .opencode:
            if let mode = frontmatter.fields["mode"], !opencodeAgentModes.contains(mode) {
                errors.append("\(path): agent mode '\(mode)' is not one of subagent, primary, all")
            }
        }
    }

    private static func validateHookSettings(content: String, path: String, errors: inout [String]) {
        guard let raw = try? JSONSerialization.jsonObject(with: Data(content.utf8)) as? [String: Any] else {
            errors.append("\(path): invalid JSON object")
            return
        }
        guard let hooksValue = raw["hooks"] else { return }
        guard let hooks = hooksValue as? [String: Any] else {
            errors.append("\(path): 'hooks' must be an object keyed by event")
            return
        }
        for (event, groupsValue) in hooks.sorted(by: { $0.key < $1.key }) {
            guard claudeHookEvents.contains(event) else {
                errors.append("\(path): unsupported hook event '\(event)'")
                continue
            }
            guard let groups = groupsValue as? [[String: Any]], !groups.isEmpty else {
                errors.append("\(path): hook event '\(event)' must map to a non-empty array of matcher groups")
                continue
            }
            for group in groups {
                guard let handlers = group["hooks"] as? [[String: Any]], !handlers.isEmpty else {
                    errors.append("\(path): hook event '\(event)' has a matcher group without a non-empty 'hooks' array")
                    continue
                }
                for handler in handlers {
                    validateHookHandler(handler, event: event, path: path, errors: &errors)
                }
            }
        }
    }

    private static func validateHookHandler(_ handler: [String: Any], event: String, path: String, errors: inout [String]) {
        let type = handler["type"] as? String ?? ""
        let field: String
        switch type {
        case "command": field = "command"
        case "prompt": field = "prompt"
        default:
            errors.append("\(path): hook event '\(event)' has a handler of unsupported type '\(type)'")
            return
        }
        if ((handler[field] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            errors.append("\(path): hook event '\(event)' has a \(type) handler with an empty '\(field)'")
        }
    }

    private static func untriggeredSkills(_ skills: [(path: String, name: String)], tree: HarnessTree, spec: HarnessSurfaceSpec) -> [String] {
        let referencing = tree.files.filter { path, _ in
            spec.instructionFiles.contains(path) || tree.alwaysOnPaths(spec: spec).contains(path)
                || spec.isAgentFile(path) || spec.classify(path: path) == .skills
        }
        return skills.filter { skill in
            !referencing.contains { path, content in path != skill.path && content.contains(skill.name) }
        }.map(\.path)
    }
}
