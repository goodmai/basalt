import Foundation
import BasaltCore

extension PipelineRunner {
    func situation(_ request: PipelineRequest, decision: SystemOneResponse?, route: PipelineCore.Route) async -> AnswerSituation? {
        guard let router = stages.answers else { return nil }
        var modalities: Set<AnswerRoute.Modality> = request.text == nil ? [] : [.text]
        for attachment in request.attachments {
            switch attachment {
            case .document: modalities.insert(.documents)
            case .audio: modalities.insert(.audio)
            }
        }
        if modalities.isEmpty { modalities.insert(.text) }
        let labels = request.questions.reduce(into: [String: String]()) { labels, entry in
            if let answer = decision?.answers[entry.key], answer.abstain != true, let label = PipelineCore.label(of: answer, entry.value) { labels[entry.key] = label }
        }
        let escalated = (decision?.escalate ?? false) || route.calls.contains { $0.question != nil }
        return AnswerSituation(modalities: modalities, labels: labels, promptTokens: 0, escalated: escalated, costLeft: await router.costLeft())
    }

    func answer(_ call: PipelineCore.System2Call, origin: RequestOrigin, sessionId: String?, situation: AnswerSituation?) async throws(BasaltError) -> PipelineCore.System2Reply {
        let started = ContinuousClock.now
        let request = PipelineCore.request(for: call, origin: origin).withRouting(sessionId: sessionId, budgetKey: nil)
        guard let router = stages.answers, var situation else {
            let response = try await chat.generate(request: request)
            return PipelineCore.System2Reply(
                call: call, model: await chat.modelInfo, text: response.generatedText,
                tokens: response.promptTokens + response.completionTokens, ms: Self.elapsed(since: started))
        }
        situation.promptTokens = AnswerSituation.estimatedTokens(call.prompt)
        let routed = try await router.answer(request, situation: situation)
        var reply = PipelineCore.System2Reply(call: call, model: routed.reply.model, text: routed.reply.text, tokens: routed.reply.tokens, ms: Self.elapsed(since: started))
        (reply.route, reply.cost, reply.skipped) = (routed.rule, routed.reply.cost, routed.skipped)
        return reply
    }
}
