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

    public static func merged(_ request: SystemOneRequest, _ parts: [Part], failed: SystemOneStage? = nil) -> SystemOneResponse {
        var answers: [String: SystemOneAnswer] = [:]
        for part in parts {
            answers.merge(part.response.answers.filter { part.stage.questions.contains($0.key) && request.questions[$0.key] != nil }) { $1 }
        }
        let stages = [failed].compactMap { $0 } + parts.map(\.stage)
        let costs = stages.compactMap(\.cost)
        var usage = parts.last?.response.usage ?? SystemOneUsage(prefixTokens: 0, candidateBranches: 0, prefixEvaluations: 0, branchForwardCalls: 0, suffixTokens: 0)
        (usage.cost, usage.stages) = (costs.isEmpty ? nil : costs.reduce(0, +), stages)
        return SystemOneResponse(
            model: parts.map(\.response.model).joined(separator: "+"), answers: answers, usage: usage,
            calibrated: parts.allSatisfy(\.response.calibrated))
    }
}

public struct CascadeDecisionEngine: DecisionEngine {
    public let remote: any DecisionEngine
    public let local: any DecisionEngine
    public let escalateBelow: Double
    public let runsBesideDecode = true

    public var modelID: String { remote.modelID + "+" + local.modelID }
    public var onDevice: (any DecisionEngine)? { local }

    public init(remote: any DecisionEngine, local: any DecisionEngine, escalateBelow: Double, gate: GenerationGate, besideDecode: Bool = true) {
        (self.remote, self.escalateBelow) = (remote, escalateBelow)
        self.local = GatedDecisionEngine.serving(local, gate: gate, besideDecode: besideDecode) ?? local
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try SystemOneCore.validate(request)
        let started = ContinuousClock.now
        let first: SystemOneResponse
        do {
            first = try await remote.decide(request)
        } catch {
            if Task.isCancelled { throw .cancelled }
            let failed = SystemOneStage(model: remote.modelID, questions: [], latencyMs: Self.milliseconds(since: started), fallback: error.localizedDescription)
            return CascadeCore.merged(request, [try await locally(request)], failed: failed)
        }
        let split = CascadeCore.split(request, first, below: escalateBelow)
        let answered = (first, SystemOneStage(model: first.model, questions: split.kept, latencyMs: Self.milliseconds(since: started), cost: first.usage.cost))
        guard !split.escalated.isEmpty else { return CascadeCore.merged(request, [answered]) }
        return CascadeCore.merged(request, [answered, try await locally(CascadeCore.subrequest(request, split.escalated))])
    }

    func locally(_ request: SystemOneRequest) async throws(BasaltError) -> CascadeCore.Part {
        let started = ContinuousClock.now
        let response = try await local.decide(request)
        return (response, SystemOneStage(model: response.model, questions: request.questions.keys.sorted(), latencyMs: Self.milliseconds(since: started)))
    }

    static func milliseconds(since start: ContinuousClock.Instant) -> Int {
        PipelineCore.milliseconds(ContinuousClock.now - start)
    }
}
