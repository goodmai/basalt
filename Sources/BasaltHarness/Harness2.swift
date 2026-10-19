import Foundation
import BasaltCore

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
        for path in baseTree.alwaysOnPaths(spec: spec) {
            if let baseText = baseTree.files[path], candidateTree.files[path] == baseText {
                sharedBlocks.append(baseText)
            }
        }
        sharedBlocks.append(taskPrompt)
        let invariantPrefix = sharedBlocks.joined(separator: "\n\n")

        var chunks: [String] = []
        for path in candidateTree.files.keys.sorted() {
            guard let text = candidateTree.files[path], !text.isEmpty else { continue }
            if baseTree.alwaysOnPaths(spec: spec).contains(path) && baseTree.files[path] == text {
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
    private let rootDirectory: URL
    private var entries: [HarnessKnowledgeEntry] = []

    public static var defaultRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".basalt/harness-library", isDirectory: true)
    }

    public init(rootDirectory: URL) {
        self.rootDirectory = rootDirectory
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
        library: HarnessExperienceLibrary,
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
