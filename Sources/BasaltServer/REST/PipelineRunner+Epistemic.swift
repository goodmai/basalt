import Foundation
import BasaltCore

extension PipelineRunner {
    struct DecidedStage: Sendable {
        let decision: SystemOneResponse?
        let receipt: PipelineStageReceipt?
        let epistemic: EpistemicGateDecision?
    }

    struct ExecutedRoute: Sendable {
        let answers: [String: PipelineAnswer]
        let system2: PipelineSystem2?
        let replies: [PipelineCore.System2Reply]
    }

    func decideWithEpistemic(
        _ request: PipelineRequest,
        context: DecideContext,
        by engine: (any DecisionEngine)?,
        origin: RequestOrigin,
        sessionId: String?
    ) async throws(BasaltError) -> DecidedStage {
        guard let spec = stages.epistemic, spec.enabled else {
            let (decision, deciding) = try await decide(
                request.questions, context: context, by: engine, origin: origin, sessionId: sessionId)
            return DecidedStage(decision: decision, receipt: deciding, epistemic: nil)
        }
        let merged = EpistemicGateCore.mergedQuestions(request.questions, spec: spec, includeTools: true)
        let (raw, deciding) = try await decide(
            merged, context: context, by: engine, origin: origin, sessionId: sessionId)
        guard let raw else { return DecidedStage(decision: nil, receipt: deciding, epistemic: nil) }
        let gate = EpistemicGateCore.reduce(raw, spec: spec, hasTools: stages.hasTools)
        let userDecision = EpistemicGateCore.userResponse(raw, keeping: Set(request.questions.keys))
        let receipt: PipelineStageReceipt? = deciding.map { base in
            var copy = PipelineStageReceipt(
                stage: base.stage,
                model: base.model,
                ms: base.ms,
                tokens: base.tokens,
                confidence: base.confidence,
                escalated: (userDecision?.escalate ?? false) || gate.action == .escalate
            )
            (copy.route, copy.cost, copy.skipped) = (base.route, base.cost, base.skipped)
            return copy
        }
        return DecidedStage(decision: userDecision, receipt: receipt, epistemic: gate)
    }

    func executeRoute(
        _ request: PipelineRequest,
        state: String,
        typed: Bool,
        decided: DecidedStage,
        origin: RequestOrigin,
        sessionId: String?
    ) async throws(BasaltError) -> ExecutedRoute {
        if let gate = decided.epistemic, gate.shortCircuitsAnswer {
            return shortCircuitedRoute(request, decided: decided, gate: gate)
        }
        let route = PipelineCore.route(questions: request.questions, decision: decided.decision, task: request.task, state: state)
        var replies: [PipelineCore.System2Reply] = []
        let answering = typed ? origin : RequestOrigin(source: PipelineCore.perceivedSource, endpoint: origin.endpoint, stream: false)
        let routeSituation = await situation(request, decision: decided.decision, route: route, epistemic: decided.epistemic)
        for call in route.calls {
            replies.append(try await answer(call, origin: answering, sessionId: sessionId, situation: routeSituation))
        }
        let (answers, system2) = PipelineCore.answers(route, replies: replies, questions: request.questions, decision: decided.decision)
        return ExecutedRoute(answers: answers, system2: system2, replies: replies)
    }

    func shortCircuitedRoute(
        _ request: PipelineRequest,
        decided: DecidedStage,
        gate: EpistemicGateDecision
    ) -> ExecutedRoute {
        var answers: [String: PipelineAnswer] = [:]
        for (id, question) in request.questions {
            if let existing = decided.decision?.answers[id], existing.abstain != true {
                answers[id] = PipelineAnswer(answeredBy: .decide, label: PipelineCore.label(of: existing, question), decision: existing)
            } else {
                answers[id] = PipelineAnswer(answeredBy: .decide, label: nil, decision: decided.decision?.answers[id], text: gate.message)
            }
        }
        let model = decided.decision?.model ?? decisions?.modelID ?? "epistemic-gate"
        let system2 = request.task == nil ? nil : PipelineSystem2(model: model, text: gate.message ?? "")
        return ExecutedRoute(answers: answers, system2: system2, replies: [])
    }
}
