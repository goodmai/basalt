import Foundation

public enum CascadeCore {
    public typealias Part = (response: SystemOneResponse, stage: SystemOneStage)

    public static func confidence(_ answer: SystemOneAnswer) -> Double {
        answer.confidence ?? answer.noul.map { abs(2 * $0 - 1) } ?? 0
    }

    public static func split(_ request: SystemOneRequest, _ first: SystemOneResponse, below threshold: Double) -> (kept: [String], escalated: [String]) {
        let ids = request.questions.keys.sorted()
        let escalated = ids.filter { id in first.answers[id].map { confidence($0) < threshold } ?? true }
        return (ids.filter { !escalated.contains($0) }, escalated)
    }

    public static func subrequest(_ request: SystemOneRequest, _ ids: [String]) -> SystemOneRequest {
        var sub = request
        sub.questions = request.questions.filter { ids.contains($0.key) }
        return sub
    }

    public static func merged(_ request: SystemOneRequest, _ parts: [Part], failed: SystemOneStage?) -> SystemOneResponse {
        merged(request, parts, skipped: [failed].compactMap { $0 })
    }

    public static func merged(_ request: SystemOneRequest, _ parts: [Part], skipped: [SystemOneStage] = []) -> SystemOneResponse {
        var answers: [String: SystemOneAnswer] = [:]
        for part in parts {
            answers.merge(part.response.answers.filter { part.stage.questions.contains($0.key) && request.questions[$0.key] != nil }) { $1 }
        }
        let stages = skipped + parts.map(\.stage)
        let costs = stages.compactMap(\.cost)
        var usage = parts.last?.response.usage ?? SystemOneUsage(prefixTokens: 0, candidateBranches: 0, prefixEvaluations: 0, branchForwardCalls: 0, suffixTokens: 0)
        (usage.cost, usage.stages) = (costs.isEmpty ? nil : costs.reduce(0, +), stages)
        return SystemOneResponse(
            model: parts.map(\.response.model).joined(separator: "+"), answers: answers, usage: usage,
            calibrated: parts.allSatisfy(\.response.calibrated))
    }
}

public struct CascadeDecisionEngine: DecisionEngine {
    public let stages: [any DecisionEngine]
    public let escalateBelow: Double
    public var runsBesideDecode: Bool { stages.allSatisfy(\.runsBesideDecode) }

    public var remote: any DecisionEngine { stages[0] }
    public var local: any DecisionEngine { stages[stages.count - 1] }
    public var modelID: String { stages.map(\.modelID).joined(separator: "+") }
    public var acceptsMedia: Bool { stages.contains(where: \.acceptsMedia) }
    public var sharesPerceived: Bool { stages.contains(where: \.sharesPerceived) }

    public var onDevice: (any DecisionEngine)? {
        let local = stages.compactMap(\.onDevice)
        return local.count > 1 ? CascadeDecisionEngine(stages: local, escalateBelow: escalateBelow) : local.first
    }

    public init(stages: [any DecisionEngine], escalateBelow: Double) {
        precondition(!stages.isEmpty, "a cascade needs at least one stage")
        (self.stages, self.escalateBelow) = (stages, escalateBelow)
    }

    public init(remote: any DecisionEngine, local: any DecisionEngine, escalateBelow: Double, gate: GenerationGate, besideDecode: Bool = true) {
        self.init(stages: [remote, GatedDecisionEngine.serving(local, gate: gate, besideDecode: besideDecode) ?? local], escalateBelow: escalateBelow)
    }

    public func unload() async {
        for stage in stages { await stage.unload() }
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try SystemOneCore.validate(request)
        var (remaining, parts, skipped) = (request.questions.keys.sorted(), [CascadeCore.Part](), [SystemOneStage]())
        for (index, stage) in stages.enumerated() {
            let last = index == stages.count - 1
            if let reason = Self.ineligibleReason(stage, for: request) {
                skipped.append(SystemOneStage(model: stage.modelID, questions: [], latencyMs: 0, fallback: reason))
                if last, parts.isEmpty { throw .invalidRequestStructure(details: "systemone: \(reason)") }
                if last { break }
                continue
            }
            let started = ContinuousClock.now
            var sub = CascadeCore.subrequest(request, remaining)
            if request.hasPerceivedContent == true, !stage.acceptsMedia { sub.media = nil }
            let response: SystemOneResponse
            do {
                response = try await stage.decide(sub)
            } catch {
                if Task.isCancelled { throw .cancelled }
                if last, parts.isEmpty { throw error }
                skipped.append(SystemOneStage(model: stage.modelID, questions: [], latencyMs: Self.milliseconds(since: started), fallback: error.localizedDescription))
                continue
            }
            let split = last ? (kept: remaining, escalated: []) : CascadeCore.split(sub, response, below: escalateBelow)
            parts.append((response, SystemOneStage(model: response.model, questions: split.kept, latencyMs: Self.milliseconds(since: started), cost: response.usage.cost)))
            remaining = split.escalated
            if remaining.isEmpty { break }
        }
        var merged = CascadeCore.merged(request, parts, skipped: skipped)
        if merged.answers.count < request.questions.count { merged.escalate = true }
        return merged
    }

    private static func ineligibleReason(_ stage: any DecisionEngine, for request: SystemOneRequest) -> String? {
        if request.hasPerceivedContent == true, !stage.sharesPerceived {
            return "stage does not accept perceived state"
        }
        if request.media?.isEmpty == false, request.hasPerceivedContent != true, !stage.acceptsMedia {
            return "stage does not accept media"
        }
        return nil
    }

    static func milliseconds(since start: ContinuousClock.Instant) -> Int {
        PipelineCore.milliseconds(ContinuousClock.now - start)
    }
}
