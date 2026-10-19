import Foundation

public enum HarnessComponentKind: String, Codable, CaseIterable, Sendable {
    case systemPrompt = "systemprompt"
    case guardrails = "guardrails"
    case memory = "memory"
    case toolGuidance = "tools"
    case subagents = "subagents"
    case skills = "skills"
    case scripts = "scripts"
    case hooks = "hooks"
}

public struct HarnessSurfaceSpec: Codable, Sendable, Equatable {
    public let name: String
    public let rootPrefix: String
    public let alwaysOnFiles: [String]
    public let skillsPrefix: String
    public let subagentsPrefix: String
    public let scriptsPrefix: String
    public let hooksFile: String
    public let alwaysOnMaxChars: Int
    public let minDiversity: Double

    public init(
        name: String,
        rootPrefix: String,
        alwaysOnFiles: [String],
        skillsPrefix: String = "skills/",
        subagentsPrefix: String = "sub_agents/",
        scriptsPrefix: String = "scripts/",
        hooksFile: String = "hooks/hooks.json",
        alwaysOnMaxChars: Int = 8000,
        minDiversity: Double = 0.05
    ) {
        self.name = name
        self.rootPrefix = rootPrefix
        self.alwaysOnFiles = alwaysOnFiles
        self.skillsPrefix = skillsPrefix
        self.subagentsPrefix = subagentsPrefix
        self.scriptsPrefix = scriptsPrefix
        self.hooksFile = hooksFile
        self.alwaysOnMaxChars = alwaysOnMaxChars
        self.minDiversity = minDiversity
    }

    public static let claude = HarnessSurfaceSpec(
        name: "claude",
        rootPrefix: ".claude",
        alwaysOnFiles: [
            "CLAUDE.md",
            "prompts/systemprompt.md",
            "prompts/guardrails.md",
            "memory/LongTermMEMORY.md",
            "prompts/tools.md"
        ]
    )

    public static let opencode = HarnessSurfaceSpec(
        name: "opencode",
        rootPrefix: ".opencode",
        alwaysOnFiles: [
            "AGENTS.md",
            "prompts/systemprompt.md",
            "prompts/guardrails.md",
            "memory/LongTermMEMORY.md",
            "prompts/tools.md"
        ]
    )

    public static let agents = HarnessSurfaceSpec(
        name: "agents",
        rootPrefix: ".agents",
        alwaysOnFiles: [
            "AGENTS.md",
            "prompts/systemprompt.md",
            "prompts/guardrails.md",
            "memory/LongTermMEMORY.md",
            "prompts/tools.md"
        ]
    )

    public func classify(path: String) -> HarnessComponentKind {
        let normalized = normalizeRelativePath(path)
        if normalized.hasPrefix(skillsPrefix) { return .skills }
        if normalized.hasPrefix(subagentsPrefix) || normalized.hasPrefix("agents/") { return .subagents }
        if normalized.hasPrefix(scriptsPrefix) { return .scripts }
        if normalized == hooksFile || normalized.hasPrefix("hooks/") || normalized.hasPrefix("plugin/") { return .hooks }
        if normalized.contains("guardrails") { return .guardrails }
        if normalized.contains("memory") || normalized.hasSuffix("MEMORY.md") { return .memory }
        if normalized.contains("tools") { return .toolGuidance }
        return .systemPrompt
    }

    public func normalizeRelativePath(_ path: String) -> String {
        var cleaned = path.trimmingCharacters(in: .whitespacesAndNewlines)
        while cleaned.hasPrefix("./") {
            cleaned = String(cleaned.dropFirst(2))
        }
        let prefixWithSlash = rootPrefix.hasSuffix("/") ? rootPrefix : rootPrefix + "/"
        if cleaned.hasPrefix(prefixWithSlash) {
            cleaned = String(cleaned.dropFirst(prefixWithSlash.count))
        }
        return cleaned
    }
}

public struct HarnessTree: Codable, Sendable, Equatable {
    public var files: [String: String]

    public init(files: [String: String] = [:]) {
        var normalized: [String: String] = [:]
        for (key, value) in files {
            let cleanKey = key.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            normalized[cleanKey] = value
        }
        self.files = normalized
    }

    public static func load(from directory: URL, spec: HarnessSurfaceSpec = .claude) throws -> HarnessTree {
        let fm = FileManager.default
        var collected: [String: String] = [:]
        guard let enumerator = fm.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return HarnessTree()
        }
        let basePath = directory.standardizedFileURL.path
        for case let fileURL as URL in enumerator {
            let values = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true else { continue }
            let fullPath = fileURL.standardizedFileURL.path
            guard fullPath.hasPrefix(basePath) else { continue }
            var rel = String(fullPath.dropFirst(basePath.count))
            if rel.hasPrefix("/") { rel.removeFirst() }
            let normalized = spec.normalizeRelativePath(rel)
            if let text = try? String(contentsOf: fileURL, encoding: .utf8) {
                collected[normalized] = text
            }
        }
        return HarnessTree(files: collected)
    }

    public func write(to directory: URL) throws {
        let fm = FileManager.default
        for (relPath, content) in files {
            let destination = directory.appendingPathComponent(relPath)
            try fm.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try content.write(to: destination, atomically: true, encoding: .utf8)
        }
    }

    public func alwaysOnText(spec: HarnessSurfaceSpec) -> String {
        spec.alwaysOnFiles.compactMap { files[$0] }.joined(separator: "\n\n")
    }

    public func alwaysOnCharCount(spec: HarnessSurfaceSpec) -> Int {
        spec.alwaysOnFiles.reduce(0) { acc, path in
            acc + (files[path]?.count ?? 0)
        }
    }

    public func serializedLines() -> Set<String> {
        var result = Set<String>()
        for key in files.keys.sorted() {
            let content = files[key] ?? ""
            for line in content.components(separatedBy: .newlines) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if !trimmed.isEmpty {
                    result.insert("\(key):\(trimmed)")
                }
            }
        }
        return result
    }

    public func applying(delta: [String: String]) -> HarnessTree {
        var merged = files
        for (path, content) in delta {
            merged[path] = content
        }
        return HarnessTree(files: merged)
    }
}

public struct HarnessFrontmatter: Sendable, Equatable {
    public let fields: [String: String]
    public let body: String

    public init(fields: [String: String], body: String) {
        self.fields = fields
        self.body = body
    }

    public static func parse(_ content: String) -> HarnessFrontmatter? {
        let lines = content.components(separatedBy: .newlines)
        guard let first = lines.first?.trimmingCharacters(in: .whitespaces), first == "---" else {
            return nil
        }
        var endIndex: Int?
        for idx in 1..<lines.count {
            if lines[idx].trimmingCharacters(in: .whitespaces) == "---" {
                endIndex = idx
                break
            }
        }
        guard let closing = endIndex else { return nil }
        var fields: [String: String] = [:]
        for idx in 1..<closing {
            let line = lines[idx].trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = String(line[..<colon]).trimmingCharacters(in: .whitespaces)
            var value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            if (value.hasPrefix("\"") && value.hasSuffix("\"")) || (value.hasPrefix("'") && value.hasSuffix("'")) {
                if value.count >= 2 {
                    value = String(value.dropFirst().dropLast())
                }
            }
            if !key.isEmpty {
                fields[key] = value
            }
        }
        let bodyLines = closing + 1 < lines.count ? Array(lines[(closing + 1)...]) : []
        return HarnessFrontmatter(fields: fields, body: bodyLines.joined(separator: "\n"))
    }
}

public enum HarnessDiversity {
    public static func lineJaccardDistance(_ first: HarnessTree, _ second: HarnessTree) -> Double {
        let setA = first.serializedLines()
        let setB = second.serializedLines()
        if setA.isEmpty && setB.isEmpty { return 0.0 }
        let unionCount = setA.union(setB).count
        guard unionCount > 0 else { return 0.0 }
        let intersectionCount = setA.intersection(setB).count
        let similarity = Double(intersectionCount) / Double(unionCount)
        return 1.0 - similarity
    }

    public static func pairwiseDiversity(_ trees: [HarnessTree]) -> Double {
        guard trees.count >= 2 else { return 1.0 }
        var total = 0.0
        var pairs = 0
        for i in 0..<trees.count {
            for j in (i + 1)..<trees.count {
                total += lineJaccardDistance(trees[i], trees[j])
                pairs += 1
            }
        }
        return pairs > 0 ? total / Double(pairs) : 1.0
    }
}

public struct HarnessValidationReport: Codable, Sendable, Equatable {
    public let isValid: Bool
    public let errors: [String]
    public let alwaysOnChars: Int
    public let alwaysOnMaxChars: Int
    public let skillCount: Int
    public let subagentCount: Int
    public let scriptCount: Int
    public let diversityScore: Double?

    public init(
        isValid: Bool,
        errors: [String],
        alwaysOnChars: Int,
        alwaysOnMaxChars: Int,
        skillCount: Int,
        subagentCount: Int,
        scriptCount: Int,
        diversityScore: Double?
    ) {
        self.isValid = isValid
        self.errors = errors
        self.alwaysOnChars = alwaysOnChars
        self.alwaysOnMaxChars = alwaysOnMaxChars
        self.skillCount = skillCount
        self.subagentCount = subagentCount
        self.scriptCount = scriptCount
        self.diversityScore = diversityScore
    }
}

public enum HarnessValidator {
    public static let allowedHookEvents: Set<String> = [
        "PreToolUse",
        "PostToolUse",
        "UserPromptSubmit",
        "Stop",
        "SubagentStop"
    ]

    public static func validate(
        _ tree: HarnessTree,
        spec: HarnessSurfaceSpec = .claude,
        requireDiverseFrom siblings: [HarnessTree] = []
    ) -> HarnessValidationReport {
        var errors: [String] = []
        let alwaysOnChars = tree.alwaysOnCharCount(spec: spec)
        if alwaysOnChars > spec.alwaysOnMaxChars {
            errors.append("always-on files exceed budget (\(alwaysOnChars) > \(spec.alwaysOnMaxChars) chars)")
        }

        let alwaysOnContent = tree.alwaysOnText(spec: spec)
        var skillCount = 0
        var subagentCount = 0
        var scriptCount = 0

        for path in tree.files.keys.sorted() {
            let content = tree.files[path] ?? ""
            if path.hasPrefix(spec.skillsPrefix) && path.hasSuffix("/SKILL.md") {
                skillCount += 1
                let parts = path.components(separatedBy: "/")
                let expectedSkillName = parts.count >= 2 ? parts[parts.count - 2] : ""
                guard let fm = HarnessFrontmatter.parse(content) else {
                    errors.append("\(path): missing YAML frontmatter")
                    continue
                }
                guard let name = fm.fields["name"], !name.isEmpty else {
                    errors.append("\(path): frontmatter missing 'name'")
                    continue
                }
                if !expectedSkillName.isEmpty && name != expectedSkillName {
                    errors.append("\(path): skill name '\(name)' does not match directory '\(expectedSkillName)'")
                }
                guard let desc = fm.fields["description"], !desc.isEmpty else {
                    errors.append("\(path): frontmatter missing non-empty 'description'")
                    continue
                }
                if !alwaysOnContent.contains(name) {
                    errors.append("\(path): untriggered skill '\(name)' is never referenced in always-on files")
                }
            } else if (path.hasPrefix(spec.subagentsPrefix) && path.hasSuffix("/AGENT.md")) || (path.hasPrefix("agents/") && path.hasSuffix(".md")) {
                subagentCount += 1
                guard let fm = HarnessFrontmatter.parse(content) else {
                    errors.append("\(path): missing YAML frontmatter")
                    continue
                }
                if fm.fields["mode"] != "subagent" {
                    errors.append("\(path): subagent frontmatter requires 'mode: subagent'")
                }
                if (fm.fields["description"] ?? "").isEmpty {
                    errors.append("\(path): subagent frontmatter missing 'description'")
                }
            } else if path.hasPrefix(spec.scriptsPrefix) {
                scriptCount += 1
                if content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    errors.append("\(path): script file is empty")
                }
            } else if path == spec.hooksFile {
                validateHooksJSON(content: content, path: path, errors: &errors)
            }
        }

        let referencedScripts = extractScriptReferences(from: tree)
        for scriptRef in referencedScripts.sorted() {
            if tree.files[scriptRef] == nil {
                errors.append("missing referenced script '\(scriptRef)' in harness tree")
            }
        }

        var computedDiversity: Double?
        if !siblings.isEmpty {
            let allTrees = siblings + [tree]
            let div = HarnessDiversity.pairwiseDiversity(allTrees)
            computedDiversity = div
            if div < spec.minDiversity {
                errors.append(String(format: "candidate diversity %.4f is below minimum %.4f", div, spec.minDiversity))
            }
        }

        return HarnessValidationReport(
            isValid: errors.isEmpty,
            errors: errors,
            alwaysOnChars: alwaysOnChars,
            alwaysOnMaxChars: spec.alwaysOnMaxChars,
            skillCount: skillCount,
            subagentCount: subagentCount,
            scriptCount: scriptCount,
            diversityScore: computedDiversity
        )
    }

    private static func validateHooksJSON(content: String, path: String, errors: inout [String]) {
        guard let data = content.data(using: .utf8),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            errors.append("\(path): invalid JSON object")
            return
        }
        let hooksContainer = (raw["hooks"] as? [String: Any]) ?? raw
        for (event, entriesAny) in hooksContainer {
            if !allowedHookEvents.contains(event) {
                errors.append("\(path): unsupported hook event '\(event)'")
                continue
            }
            guard let entries = entriesAny as? [[String: Any]], !entries.isEmpty else {
                errors.append("\(path): hook event '\(event)' must map to a non-empty array")
                continue
            }
            for entry in entries {
                if let command = entry["command"] as? String {
                    if command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        errors.append("\(path): hook event '\(event)' has empty command")
                    }
                } else if let innerHooks = entry["hooks"] as? [[String: Any]] {
                    if innerHooks.isEmpty {
                        errors.append("\(path): hook event '\(event)' has empty inner hooks array")
                    }
                    for inner in innerHooks {
                        let cmd = (inner["command"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                        if cmd.isEmpty {
                            errors.append("\(path): hook event '\(event)' has empty command")
                        }
                    }
                } else {
                    errors.append("\(path): hook event '\(event)' entry missing 'command'")
                }
            }
        }
    }

    private static func extractScriptReferences(from tree: HarnessTree) -> Set<String> {
        let pattern = #"scripts/[A-Za-z0-9_./-]+\.(?:py|sh|swift)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        var refs = Set<String>()
        for (_, content) in tree.files {
            let range = NSRange(content.startIndex..., in: content)
            for match in regex.matches(in: content, range: range) {
                if let matchRange = Range(match.range, in: content) {
                    refs.insert(String(content[matchRange]))
                }
            }
        }
        return refs
    }
}

public struct HarnessCacheReusePlan: Codable, Sendable, Equatable {
    public let promptCacheKey: String
    public let invariantPrefixText: String
    public let blendableChunks: [String]
    public let composedPrompt: String
    public let estimatedSharedTokens: Int
    public let estimatedDeltaTokens: Int
    public let prefillSavingsRatio: Double

    public init(
        promptCacheKey: String,
        invariantPrefixText: String,
        blendableChunks: [String],
        composedPrompt: String,
        estimatedSharedTokens: Int,
        estimatedDeltaTokens: Int,
        prefillSavingsRatio: Double
    ) {
        self.promptCacheKey = promptCacheKey
        self.invariantPrefixText = invariantPrefixText
        self.blendableChunks = blendableChunks
        self.composedPrompt = composedPrompt
        self.estimatedSharedTokens = estimatedSharedTokens
        self.estimatedDeltaTokens = estimatedDeltaTokens
        self.prefillSavingsRatio = prefillSavingsRatio
    }

    public static func build(
        taskSlug: String,
        taskPrompt: String,
        baseTree: HarnessTree,
        candidateTree: HarnessTree,
        spec: HarnessSurfaceSpec = .claude,
        kBudget: Int = 1
    ) -> HarnessCacheReusePlan {
        let cacheKey = "harness:\(taskSlug)"
        var sharedBlocks: [String] = []
        for path in spec.alwaysOnFiles {
            if let baseText = baseTree.files[path], candidateTree.files[path] == baseText {
                sharedBlocks.append(baseText)
            }
        }
        sharedBlocks.append(taskPrompt)
        let invariantPrefix = sharedBlocks.joined(separator: "\n\n")

        var chunks: [String] = []
        for path in candidateTree.files.keys.sorted() {
            guard let text = candidateTree.files[path], !text.isEmpty else { continue }
            if spec.alwaysOnFiles.contains(path) && baseTree.files[path] == text {
                continue
            }
            chunks.append("[\(path)]\n\(text)")
        }

        let allSegments = ([invariantPrefix] + chunks).filter { !$0.isEmpty }
        let composedPrompt = allSegments.joined(separator: "\n\n")

        let sharedTokens = max(1, invariantPrefix.utf8.count / 4)
        let deltaText = chunks.joined(separator: "\n\n")
        let deltaTokens = deltaText.utf8.count / 4
        let fullRolloutTokens = sharedTokens + deltaTokens
        let totalStages = max(1, 7 * kBudget)
        let uncachedCost = Double(totalStages * fullRolloutTokens)
        let cachedCost = Double(sharedTokens + totalStages * deltaTokens)
        let savings = uncachedCost > 0 ? max(0.0, 1.0 - (cachedCost / uncachedCost)) : 0.0

        return HarnessCacheReusePlan(
            promptCacheKey: cacheKey,
            invariantPrefixText: invariantPrefix,
            blendableChunks: chunks,
            composedPrompt: composedPrompt,
            estimatedSharedTokens: sharedTokens,
            estimatedDeltaTokens: deltaTokens,
            prefillSavingsRatio: savings
        )
    }
}

public enum HarnessBudgetError: Error, Sendable, Equatable {
    case invalidK(Int)
    case duplicateRunId(String)
    case rolloutBudgetExceeded(limit: Int)
    case proposerBudgetExceeded(limit: Int)
    case leakDetected(String)
}

public struct HarnessLedger: Codable, Sendable, Equatable {
    public let k: Int
    public let maxRollouts: Int
    public let maxProposerCalls: Int
    public private(set) var rolloutIds: [String] = []
    public private(set) var proposerIds: [String] = []

    public init(k: Int) throws {
        guard k >= 1 else { throw HarnessBudgetError.invalidK(k) }
        self.k = k
        self.maxRollouts = 4 * k
        self.maxProposerCalls = 3 * k
    }

    public var remainingRollouts: Int { maxRollouts - rolloutIds.count }
    public var remainingProposerCalls: Int { maxProposerCalls - proposerIds.count }

    public mutating func recordRollout(id: String) throws {
        if rolloutIds.contains(id) || proposerIds.contains(id) {
            throw HarnessBudgetError.duplicateRunId(id)
        }
        guard rolloutIds.count < maxRollouts else {
            throw HarnessBudgetError.rolloutBudgetExceeded(limit: maxRollouts)
        }
        rolloutIds.append(id)
    }

    public mutating func recordProposerCall(id: String) throws {
        if proposerIds.contains(id) || rolloutIds.contains(id) {
            throw HarnessBudgetError.duplicateRunId(id)
        }
        guard proposerIds.count < maxProposerCalls else {
            throw HarnessBudgetError.proposerBudgetExceeded(limit: maxProposerCalls)
        }
        proposerIds.append(id)
    }
}

public enum HarnessLeakGuard {
    public static let forbiddenTokens: [String] = [
        "grader_verdict",
        "\"passed\":",
        "\"score\":",
        "benchmark_score",
        "ground_truth_answer"
    ]

    public static func assertClean(_ text: String, forbiddenPaths: [String] = []) throws {
        let lower = text.lowercased()
        for token in forbiddenTokens {
            if lower.contains(token.lowercased()) {
                throw HarnessBudgetError.leakDetected(token)
            }
        }
        for path in forbiddenPaths where !path.isEmpty {
            if text.contains(path) {
                throw HarnessBudgetError.leakDetected(path)
            }
        }
    }
}

public enum HarnessComponentVerdict: String, Codable, Sendable {
    case helped = "helped"
    case hurt = "hurt"
    case noObservedEffect = "no observed effect"
    case confounded = "confounded"
}

public struct HarnessComponentAttribution: Codable, Sendable, Equatable {
    public let candidateId: String
    public let component: HarnessComponentKind
    public let paths: [String]
    public let verdict: HarnessComponentVerdict

    public init(
        candidateId: String,
        component: HarnessComponentKind,
        paths: [String],
        verdict: HarnessComponentVerdict
    ) {
        self.candidateId = candidateId
        self.component = component
        self.paths = paths
        self.verdict = verdict
    }
}

public enum HarnessDiff {
    public static func changedComponents(
        base: HarnessTree,
        candidate: HarnessTree,
        spec: HarnessSurfaceSpec = .claude
    ) -> [HarnessComponentKind: [String]] {
        let allPaths = Set(base.files.keys).union(candidate.files.keys)
        var grouped: [HarnessComponentKind: [String]] = [:]
        for path in allPaths.sorted() {
            if base.files[path] != candidate.files[path] {
                let kind = spec.classify(path: path)
                grouped[kind, default: []].append(path)
            }
        }
        return grouped
    }

    public static func attribute(
        candidateId: String,
        base: HarnessTree,
        candidate: HarnessTree,
        improvedOverBase: Bool,
        regressedFromBase: Bool,
        spec: HarnessSurfaceSpec = .claude
    ) -> [HarnessComponentAttribution] {
        let changed = changedComponents(base: base, candidate: candidate, spec: spec)
        let isConfounded = changed.keys.count > 1
        let baseVerdict: HarnessComponentVerdict
        if isConfounded {
            baseVerdict = .confounded
        } else if improvedOverBase {
            baseVerdict = .helped
        } else if regressedFromBase {
            baseVerdict = .hurt
        } else {
            baseVerdict = .noObservedEffect
        }
        return changed.keys.sorted(by: { $0.rawValue < $1.rawValue }).map { kind in
            HarnessComponentAttribution(
                candidateId: candidateId,
                component: kind,
                paths: changed[kind] ?? [],
                verdict: baseVerdict
            )
        }
    }
}

public struct HarnessRunView: Codable, Sendable, Equatable {
    public let candidateId: String
    public let deliverableText: String
    public let requiredFilesPresent: Bool
    public let schemaConforms: Bool
    public let groundedInToolOutput: Bool
    public let internallyConsistent: Bool
    public let unrecoveredToolErrors: Int
    public let completionScore: Double

    public init(
        candidateId: String,
        deliverableText: String,
        requiredFilesPresent: Bool = true,
        schemaConforms: Bool = true,
        groundedInToolOutput: Bool = true,
        internallyConsistent: Bool = true,
        unrecoveredToolErrors: Int = 0,
        completionScore: Double = 1.0
    ) {
        self.candidateId = candidateId
        self.deliverableText = deliverableText
        self.requiredFilesPresent = requiredFilesPresent
        self.schemaConforms = schemaConforms
        self.groundedInToolOutput = groundedInToolOutput
        self.internallyConsistent = internallyConsistent
        self.unrecoveredToolErrors = unrecoveredToolErrors
        self.completionScore = completionScore
    }
}

public enum ContrastiveSelector {
    public static func selectWinner(runs: [HarnessRunView]) throws -> HarnessRunView? {
        guard !runs.isEmpty else { return nil }
        for run in runs {
            try HarnessLeakGuard.assertClean(run.deliverableText)
        }
        return runs.max { lhs, rhs in
            compare(lhs, rhs) < 0
        }
    }

    public static func compare(_ lhs: HarnessRunView, _ rhs: HarnessRunView) -> Int {
        if lhs.requiredFilesPresent != rhs.requiredFilesPresent {
            return lhs.requiredFilesPresent ? 1 : -1
        }
        if lhs.schemaConforms != rhs.schemaConforms {
            return lhs.schemaConforms ? 1 : -1
        }
        if lhs.groundedInToolOutput != rhs.groundedInToolOutput {
            return lhs.groundedInToolOutput ? 1 : -1
        }
        if lhs.internallyConsistent != rhs.internallyConsistent {
            return lhs.internallyConsistent ? 1 : -1
        }
        if lhs.unrecoveredToolErrors != rhs.unrecoveredToolErrors {
            return lhs.unrecoveredToolErrors < rhs.unrecoveredToolErrors ? 1 : -1
        }
        if lhs.completionScore != rhs.completionScore {
            return lhs.completionScore > rhs.completionScore ? 1 : -1
        }
        return 0
    }
}

public struct HarnessKnowledgeEntry: Codable, Sendable, Equatable {
    public let topic: String
    public let domain: String
    public let content: String
    public let cue: String
    public let confidence: String
    public let attributions: [HarnessComponentAttribution]
    public let createdAt: String

    public init(
        topic: String,
        domain: String = "general",
        content: String,
        cue: String = "",
        confidence: String = "medium",
        attributions: [HarnessComponentAttribution] = [],
        createdAt: String = ISO8601DateFormatter().string(from: Date())
    ) {
        self.topic = topic
        self.domain = domain
        self.content = content
        self.cue = cue.isEmpty ? topic : cue
        self.confidence = confidence
        self.attributions = attributions
        self.createdAt = createdAt
    }
}

public actor HarnessExperienceLibrary {
    public static let shared = HarnessExperienceLibrary()

    private let rootDirectory: URL
    private var entries: [HarnessKnowledgeEntry] = []

    public init(rootDirectory: URL? = nil) {
        if let rootDirectory {
            self.rootDirectory = rootDirectory
        } else {
            let home = FileManager.default.homeDirectoryForCurrentUser
            self.rootDirectory = home.appendingPathComponent(".basalt/harness-library", isDirectory: true)
        }
        self.entries = Self.loadExistingEntries(from: self.rootDirectory)
    }

    @discardableResult
    public func addKnowledge(
        topic: String,
        content: String,
        domain: String = "general",
        cue: String = "",
        confidence: String = "medium",
        attributions: [HarnessComponentAttribution] = []
    ) throws -> (entry: HarnessKnowledgeEntry, indexPath: String, experiencePath: String) {
        let entry = HarnessKnowledgeEntry(
            topic: topic,
            domain: domain,
            content: content,
            cue: cue,
            confidence: confidence,
            attributions: attributions
        )
        entries.append(entry)

        let slug = Self.slugify(topic)
        let domainSlug = Self.slugify(domain.isEmpty ? "general" : domain)
        let domainDir = rootDirectory.appendingPathComponent(domainSlug, isDirectory: true)
        let taskDir = domainDir
            .appendingPathComponent("harnesses", isDirectory: true)
            .appendingPathComponent(slug, isDirectory: true)

        try FileManager.default.createDirectory(at: taskDir, withIntermediateDirectories: true)

        let experienceURL = taskDir.appendingPathComponent("EXPERIENCE.md")
        let experienceMarkdown = Self.renderExperienceMarkdown(entry: entry)
        try experienceMarkdown.write(to: experienceURL, atomically: true, encoding: .utf8)

        let entriesURL = rootDirectory.appendingPathComponent("entries.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(entries)
        try data.write(to: entriesURL, options: .atomic)

        let indexURL = domainDir.appendingPathComponent("INDEX.md")
        let domainEntries = entries.filter { Self.slugify($0.domain) == domainSlug }
        let indexMarkdown = Self.renderIndexMarkdown(domain: domain, entries: domainEntries)
        try indexMarkdown.write(to: indexURL, atomically: true, encoding: .utf8)

        return (entry, indexURL.path, experienceURL.path)
    }

    public func query(topicOrCue: String? = nil, domain: String? = nil) -> [HarnessKnowledgeEntry] {
        entries.filter { entry in
            if let domain, !domain.isEmpty, entry.domain.lowercased() != domain.lowercased() {
                return false
            }
            guard let q = topicOrCue?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !q.isEmpty else {
                return true
            }
            return entry.topic.lowercased().contains(q)
                || entry.cue.lowercased().contains(q)
                || entry.content.lowercased().contains(q)
        }
    }

    public func allEntries() -> [HarnessKnowledgeEntry] {
        entries
    }

    private static func loadExistingEntries(from directory: URL) -> [HarnessKnowledgeEntry] {
        let entriesURL = directory.appendingPathComponent("entries.json")
        guard let data = try? Data(contentsOf: entriesURL),
              let decoded = try? JSONDecoder().decode([HarnessKnowledgeEntry].self, from: data) else {
            return []
        }
        return decoded
    }

    private static func slugify(_ input: String) -> String {
        let lower = input.lowercased()
        let allowed = CharacterSet.alphanumerics
        var scalars: [Character] = []
        var lastWasDash = false
        for scalar in lower.unicodeScalars {
            if allowed.contains(scalar) {
                scalars.append(Character(scalar))
                lastWasDash = false
            } else if !lastWasDash && !scalars.isEmpty {
                scalars.append("-")
                lastWasDash = true
            }
        }
        let result = String(scalars).trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return result.isEmpty ? "entry" : result
    }

    private static func renderExperienceMarkdown(entry: HarnessKnowledgeEntry) -> String {
        var lines: [String] = [
            "---",
            "topic: \"\(entry.topic)\"",
            "domain: \"\(entry.domain)\"",
            "cue: \"\(entry.cue)\"",
            "confidence: \"\(entry.confidence)\"",
            "created_at: \"\(entry.createdAt)\"",
            "---",
            "",
            "# \(entry.topic)",
            "",
            entry.content
        ]
        if !entry.attributions.isEmpty {
            lines.append("")
            lines.append("## Component Attribution")
            for attr in entry.attributions {
                let joinedPaths = attr.paths.joined(separator: ", ")
                lines.append("- \(attr.candidateId) [\(attr.component.rawValue)] (\(joinedPaths)): \(attr.verdict.rawValue)")
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private static func renderIndexMarkdown(domain: String, entries: [HarnessKnowledgeEntry]) -> String {
        var lines: [String] = [
            "# Experience Library Index (\(domain))",
            ""
        ]
        for entry in entries {
            let slug = slugify(entry.topic)
            lines.append("- **\(entry.topic)** (`harnesses/\(slug)/EXPERIENCE.md`) — cue: \(entry.cue) [confidence: \(entry.confidence)]")
        }
        return lines.joined(separator: "\n") + "\n"
    }
}

public struct HarnessOptimizationResult: Codable, Sendable, Equatable {
    public let taskSlug: String
    public let winningCandidateId: String
    public let winningDeliverable: String
    public let winningTree: HarnessTree
    public let ledger: HarnessLedger
    public let cachePlans: [HarnessCacheReusePlan]
    public let attributions: [HarnessComponentAttribution]

    public init(
        taskSlug: String,
        winningCandidateId: String,
        winningDeliverable: String,
        winningTree: HarnessTree,
        ledger: HarnessLedger,
        cachePlans: [HarnessCacheReusePlan],
        attributions: [HarnessComponentAttribution]
    ) {
        self.taskSlug = taskSlug
        self.winningCandidateId = winningCandidateId
        self.winningDeliverable = winningDeliverable
        self.winningTree = winningTree
        self.ledger = ledger
        self.cachePlans = cachePlans
        self.attributions = attributions
    }
}

public actor HarnessRecursionRunner {
    private let orchestrator: ModelOrchestratorActor
    private let library: HarnessExperienceLibrary
    private let spec: HarnessSurfaceSpec

    public init(
        orchestrator: ModelOrchestratorActor,
        library: HarnessExperienceLibrary = .shared,
        spec: HarnessSurfaceSpec = .claude
    ) {
        self.orchestrator = orchestrator
        self.library = library
        self.spec = spec
    }

    public func optimize(
        taskSlug: String,
        taskPrompt: String,
        domain: String = "general",
        baseTree: HarnessTree,
        invariantDeltas: [[String: String]],
        domainDeltas: [[String: String]],
        k: Int = 1
    ) async throws -> HarnessOptimizationResult {
        try HarnessLeakGuard.assertClean(taskPrompt)
        var ledger = try HarnessLedger(k: k)
        var candidates: [(id: String, tree: HarnessTree)] = []
        var cachePlans: [HarnessCacheReusePlan] = []
        var runViews: [HarnessRunView] = []

        for idx in 0..<k {
            try ledger.recordRollout(id: "base_r\(idx)")
            let basePlan = HarnessCacheReusePlan.build(
                taskSlug: taskSlug,
                taskPrompt: taskPrompt,
                baseTree: baseTree,
                candidateTree: baseTree,
                spec: spec,
                kBudget: k
            )
            cachePlans.append(basePlan)
            let baseResp = try await orchestrator.generate(
                request: GenerationRequest(
                    prompt: basePlan.composedPrompt,
                    maxTokens: 512,
                    temperature: 0.2,
                    cacheKey: basePlan.promptCacheKey
                )
            )
            let baseId = "base_r\(idx)"
            candidates.append((id: baseId, tree: baseTree))
            runViews.append(
                HarnessRunView(
                    candidateId: baseId,
                    deliverableText: baseResp.generatedText,
                    completionScore: Double(max(1, baseResp.completionTokens))
                )
            )
        }

        var acceptedStage1Trees: [HarnessTree] = []

        for idx in 0..<k {
            try ledger.recordProposerCall(id: "propose_inv_\(idx)")
            let invDelta = idx < invariantDeltas.count
                ? invariantDeltas[idx]
                : ["prompts/guardrails.md": "Verify outputs before finishing (inv-\(idx))."]
            let invTree = baseTree.applying(delta: invDelta)
            let invReport = HarnessValidator.validate(invTree, spec: spec, requireDiverseFrom: acceptedStage1Trees)
            let effectiveInvTree = invReport.isValid ? invTree : baseTree
            acceptedStage1Trees.append(effectiveInvTree)

            try ledger.recordRollout(id: "probe_inv_\(idx)")
            let invPlan = HarnessCacheReusePlan.build(
                taskSlug: taskSlug,
                taskPrompt: taskPrompt,
                baseTree: baseTree,
                candidateTree: effectiveInvTree,
                spec: spec,
                kBudget: k
            )
            cachePlans.append(invPlan)
            let invResp = try await orchestrator.generate(
                request: GenerationRequest(
                    prompt: invPlan.composedPrompt,
                    maxTokens: 512,
                    temperature: 0.2,
                    cacheKey: invPlan.promptCacheKey
                )
            )
            let invId = "inv_\(idx)"
            candidates.append((id: invId, tree: effectiveInvTree))
            runViews.append(
                HarnessRunView(
                    candidateId: invId,
                    deliverableText: invResp.generatedText,
                    completionScore: Double(max(1, invResp.completionTokens)) + 1.0
                )
            )

            try ledger.recordProposerCall(id: "propose_dom_\(idx)")
            let domDelta = idx < domainDeltas.count
                ? domainDeltas[idx]
                : ["prompts/tools.md": "Prefer domain tools for \(domain) (dom-\(idx))."]
            let domTree = baseTree.applying(delta: domDelta)
            let domReport = HarnessValidator.validate(domTree, spec: spec, requireDiverseFrom: acceptedStage1Trees)
            let effectiveDomTree = domReport.isValid ? domTree : baseTree
            acceptedStage1Trees.append(effectiveDomTree)

            try ledger.recordRollout(id: "probe_dom_\(idx)")
            let domPlan = HarnessCacheReusePlan.build(
                taskSlug: taskSlug,
                taskPrompt: taskPrompt,
                baseTree: baseTree,
                candidateTree: effectiveDomTree,
                spec: spec,
                kBudget: k
            )
            cachePlans.append(domPlan)
            let domResp = try await orchestrator.generate(
                request: GenerationRequest(
                    prompt: domPlan.composedPrompt,
                    maxTokens: 512,
                    temperature: 0.2,
                    cacheKey: domPlan.promptCacheKey
                )
            )
            let domId = "dom_\(idx)"
            candidates.append((id: domId, tree: effectiveDomTree))
            runViews.append(
                HarnessRunView(
                    candidateId: domId,
                    deliverableText: domResp.generatedText,
                    completionScore: Double(max(1, domResp.completionTokens)) + 1.5
                )
            )
        }

        for idx in 0..<k {
            try ledger.recordProposerCall(id: "compose_\(idx)")
            let invDelta = idx < invariantDeltas.count
                ? invariantDeltas[idx]
                : ["prompts/guardrails.md": "Verify outputs before finishing (inv-\(idx))."]
            let domDelta = idx < domainDeltas.count
                ? domainDeltas[idx]
                : ["prompts/tools.md": "Prefer domain tools for \(domain) (dom-\(idx))."]
            let composedTree = baseTree.applying(delta: invDelta).applying(delta: domDelta)

            try ledger.recordRollout(id: "probe_compose_\(idx)")
            let composePlan = HarnessCacheReusePlan.build(
                taskSlug: taskSlug,
                taskPrompt: taskPrompt,
                baseTree: baseTree,
                candidateTree: composedTree,
                spec: spec,
                kBudget: k
            )
            cachePlans.append(composePlan)
            let compResp = try await orchestrator.generate(
                request: GenerationRequest(
                    prompt: composePlan.composedPrompt,
                    maxTokens: 512,
                    temperature: 0.2,
                    cacheKey: composePlan.promptCacheKey
                )
            )
            let compId = "compose_\(idx)"
            candidates.append((id: compId, tree: composedTree))
            runViews.append(
                HarnessRunView(
                    candidateId: compId,
                    deliverableText: compResp.generatedText,
                    completionScore: Double(max(1, compResp.completionTokens)) + 2.0
                )
            )
        }

        let winningRun = try ContrastiveSelector.selectWinner(runs: runViews) ?? runViews[0]
        let winningTree = candidates.first(where: { $0.id == winningRun.candidateId })?.tree ?? baseTree
        let baseScore = runViews.first?.completionScore ?? 0.0

        var allAttributions: [HarnessComponentAttribution] = []
        for candidate in candidates where candidate.id != "base_r0" {
            let runScore = runViews.first(where: { $0.candidateId == candidate.id })?.completionScore ?? 0.0
            let attrs = HarnessDiff.attribute(
                candidateId: candidate.id,
                base: baseTree,
                candidate: candidate.tree,
                improvedOverBase: runScore > baseScore,
                regressedFromBase: runScore < baseScore,
                spec: spec
            )
            allAttributions.append(contentsOf: attrs)
        }

        _ = try await library.addKnowledge(
            topic: taskSlug,
            content: winningRun.deliverableText,
            domain: domain,
            cue: taskPrompt,
            confidence: "high",
            attributions: allAttributions
        )

        return HarnessOptimizationResult(
            taskSlug: taskSlug,
            winningCandidateId: winningRun.candidateId,
            winningDeliverable: winningRun.deliverableText,
            winningTree: winningTree,
            ledger: ledger,
            cachePlans: cachePlans,
            attributions: allAttributions
        )
    }
}
