import Foundation

public struct ComplexityAssessment: Sendable, Codable, Equatable {
    public var isComplex: Bool
    public var complexityScore: Double
    public var estimatedSubtasks: Int
    public var signals: [String]

    public init(isComplex: Bool, complexityScore: Double, estimatedSubtasks: Int, signals: [String]) {
        self.isComplex = isComplex
        self.complexityScore = complexityScore
        self.estimatedSubtasks = estimatedSubtasks
        self.signals = signals
    }
}

public struct DecomposedSubtask: Sendable, Codable, Equatable {
    public var id: String
    public var prompt: String
    public var dependsOn: [String]
    public var domainHint: RouterDomainClass

    public init(id: String, prompt: String, dependsOn: [String] = [], domainHint: RouterDomainClass = .general) {
        self.id = id
        self.prompt = prompt
        self.dependsOn = dependsOn
        self.domainHint = domainHint
    }
}

public struct ExecutionDAGPlan: Sendable, Codable, Equatable {
    public var waves: [[DecomposedSubtask]]

    public var totalTasks: Int {
        waves.reduce(0) { $0 + $1.count }
    }

    public init(waves: [[DecomposedSubtask]]) {
        self.waves = waves
    }
}

public struct BusinessTierModels: Sendable, Codable, Equatable {
    public var fastLocalModel: String
    public var reasoningModel: String
    public var symbolicVerifierModel: String

    public init(
        fastLocalModel: String = "fast-local",
        reasoningModel: String = "reasoning-local",
        symbolicVerifierModel: String = "symbolic-verifier"
    ) {
        self.fastLocalModel = fastLocalModel
        self.reasoningModel = reasoningModel
        self.symbolicVerifierModel = symbolicVerifierModel
    }

    public func model(for tier: RouterExecutionTier) -> String {
        switch tier {
        case .fastLocal: fastLocalModel
        case .reasoning: reasoningModel
        case .symbolicVerified: symbolicVerifierModel
        }
    }
}

public struct RoutedSubtask: Sendable, Codable, Equatable {
    public var subtask: DecomposedSubtask
    public var decision: CalibratedRouterDecision
    public var selectedModel: String

    public init(subtask: DecomposedSubtask, decision: CalibratedRouterDecision, selectedModel: String) {
        self.subtask = subtask
        self.decision = decision
        self.selectedModel = selectedModel
    }
}

public struct SubtaskExecutionOutput: Sendable, Codable, Equatable {
    public var subtaskId: String
    public var text: String
    public var tier: RouterExecutionTier
    public var cacheHit: Bool
    public var ttftSeconds: Double

    public init(subtaskId: String, text: String, tier: RouterExecutionTier, cacheHit: Bool, ttftSeconds: Double) {
        self.subtaskId = subtaskId
        self.text = text
        self.tier = tier
        self.cacheHit = cacheHit
        self.ttftSeconds = ttftSeconds
    }
}

public struct AggregatedBusinessResponse: Sendable, Codable, Equatable {
    public var combinedText: String
    public var tiersUsed: [RouterExecutionTier]
    public var cacheHits: Int
    public var maxTTFTSeconds: Double

    public init(combinedText: String, tiersUsed: [RouterExecutionTier], cacheHits: Int, maxTTFTSeconds: Double) {
        self.combinedText = combinedText
        self.tiersUsed = tiersUsed
        self.cacheHits = cacheHits
        self.maxTTFTSeconds = maxTTFTSeconds
    }
}

public enum BusinessOrchestratorCore {
    private static let compoundMarkers = [" and then ", " then ", "; ", " step 2", " compare ", " versus "]

    public static func detectComplexity(query: String) -> ComplexityAssessment {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        var score = 0.0
        var signals: [String] = []
        let words = trimmed.split(whereSeparator: \.isWhitespace).count
        if words >= 28 {
            score += 0.35
            signals.append("long_prompt")
        }
        if compoundMarkers.contains(where: { lower.contains($0) }) {
            score += 0.40
            signals.append("compound_clause")
        }
        let sentences = splitClauses(trimmed)
        if sentences.count >= 2 {
            score += 0.25
            signals.append("multi_clause")
        }
        let clamped = min(1.0, score)
        return ComplexityAssessment(
            isComplex: clamped >= 0.50 || sentences.count >= 2,
            complexityScore: clamped,
            estimatedSubtasks: max(1, sentences.count),
            signals: signals
        )
    }

    public static func decompose(query: String) -> [DecomposedSubtask] {
        let clauses = splitClauses(query)
        guard !clauses.isEmpty else { return [] }
        let sequential = query.lowercased().contains(" then ")
        return clauses.enumerated().map { idx, clause in
            let id = "t\(idx + 1)"
            let deps = (sequential && idx > 0) ? ["t\(idx)"] : []
            return DecomposedSubtask(id: id, prompt: clause, dependsOn: deps, domainHint: inferDomainHint(clause))
        }
    }

    public static func buildDAG(_ subtasks: [DecomposedSubtask]) throws(BasaltError) -> ExecutionDAGPlan {
        guard !subtasks.isEmpty else { return ExecutionDAGPlan(waves: []) }
        let validIds = Set(subtasks.map(\.id))
        for task in subtasks {
            for dep in task.dependsOn where !validIds.contains(dep) {
                throw .invalidRequestStructure(details: "orchestrator DAG: unknown dependency '\(dep)' for '\(task.id)'")
            }
        }
        var completed: Set<String> = []
        var remaining = subtasks
        var waves: [[DecomposedSubtask]] = []
        while !remaining.isEmpty {
            let ready = remaining.filter { $0.dependsOn.allSatisfy(completed.contains) }
            guard !ready.isEmpty else {
                throw .invalidRequestStructure(details: "orchestrator DAG: dependency cycle detected")
            }
            waves.append(ready)
            let readyIds = Set(ready.map(\.id))
            completed.formUnion(readyIds)
            remaining.removeAll { readyIds.contains($0.id) }
        }
        return ExecutionDAGPlan(waves: waves)
    }

    public static func routePlan(
        _ plan: ExecutionDAGPlan,
        gate: CalibratedRouterGate,
        models: BusinessTierModels = .init(),
        estimatedFastTTFTSeconds: Double? = nil,
        logitProvider: (DecomposedSubtask) -> [Double]
    ) throws(BasaltError) -> [[RoutedSubtask]] {
        var routedWaves: [[RoutedSubtask]] = []
        for wave in plan.waves {
            var routed: [RoutedSubtask] = []
            for task in wave {
                let logits = logitProvider(task)
                let decision = try gate.route(
                    rawDomainLogits: logits,
                    domains: RouterDomainClass.allCases,
                    estimatedFastTTFTSeconds: estimatedFastTTFTSeconds
                )
                routed.append(RoutedSubtask(
                    subtask: task,
                    decision: decision,
                    selectedModel: models.model(for: decision.tier)
                ))
            }
            routedWaves.append(routed)
        }
        return routedWaves
    }

    public static func aggregate(_ outputs: [SubtaskExecutionOutput]) -> AggregatedBusinessResponse {
        let text = outputs.map(\.text).filter { !$0.isEmpty }.joined(separator: "\n\n")
        let tiers = Array(Set(outputs.map(\.tier))).sorted { $0.rawValue < $1.rawValue }
        let hits = outputs.filter(\.cacheHit).count
        let maxTTFT = outputs.map(\.ttftSeconds).max() ?? 0.0
        return AggregatedBusinessResponse(
            combinedText: text,
            tiersUsed: tiers,
            cacheHits: hits,
            maxTTFTSeconds: maxTTFT
        )
    }

    private static func splitClauses(_ text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let normalized = trimmed
            .replacingOccurrences(of: " and then ", with: "; ", options: .caseInsensitive)
            .replacingOccurrences(of: " then ", with: "; ", options: .caseInsensitive)
        return normalized
            .split(separator: ";")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static let domainKeywords: [(RouterDomainClass, [String])] = [
        (.medical, ["patient", "icd-10", "diagnos"]),
        (.legal, ["contract", "statute", "law"]),
        (.finance, ["invoice", "erp", "audit"]),
        (.security, ["injection", "vulnerability", "cve"]),
        (.coding, ["swift", "python", "function"]),
    ]

    private static func inferDomainHint(_ text: String) -> RouterDomainClass {
        let lower = text.lowercased()
        for (domain, keywords) in domainKeywords where keywords.contains(where: lower.contains) {
            return domain
        }
        return .general
    }
}
