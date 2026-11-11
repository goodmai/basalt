import Foundation
import BasaltCore
import BasaltCloud
import BasaltServer
import BasaltSpeech

extension ComposeHost {
    func systems(for spec: ComposeSpec, held: [String: ComposeHeld], defaultDecisions: (any DecisionEngine)?) async throws -> [String: PipelineSystem] {
        var systems: [String: PipelineSystem] = [:]
        for (name, view) in spec.views {
            let first = name == ComposeSpec.defaultPipeline
            let decisions = try (first ? defaultDecisions : nil)
                ?? serve.composedDecisions(view, local: decider, gate: chat.gate, besideDecode: besideDecode, cloudGate: cloudGate)
            systems[name] = PipelineSystem(decisions: decisions, stages: try stages(view, held: held, first: first))
        }
        return systems
    }

    func stages(_ view: ComposeSpec, held: [String: ComposeHeld], first: Bool) throws -> PipelineStages {
        func orchestrator(_ name: String?) -> ModelOrchestratorActor? {
            switch name.flatMap({ held[$0] }) {
            case .resident(let model): model
            case .onDemand(let model): model.model
            case .cloud, .decision, nil: nil
            }
        }
        let speech: (any AudioTranscriber)? = view.audio.contains(ComposeSpec.speechAnalyzer) ? (first ? startup.speech : SpeechAnalyzerTranscriber()) : nil
        var stages = PipelineStages(ocr: orchestrator(view.ocr), audio: orchestrator(view.speech), speech: speech, abstainBelow: view.abstainBelow, minMargin: view.minMargin)
        for (perceiver, name) in [(PipelineCore.Perceiver.ocrModel, view.ocr), (.speechModel, view.speech)] {
            if case .onDemand(let model)? = name.flatMap({ held[$0] }) { stages.onDemand[perceiver] = model }
        }
        if first { stages.swap = startup.swap }
        stages.answers = try router(view, held: held)
        return stages
    }

    func router(_ view: ComposeSpec, held: [String: ComposeHeld]) throws -> AnswerRouter? {
        guard !view.routes.isEmpty || view.answer != spec.answer else { return nil }
        var targets: [String: any AnswerTarget] = [:]
        for name in view.answerModels { targets[name] = try target(name, held: held) }
        return AnswerRouter(routes: view.routes.isEmpty ? [AnswerRoute(model: view.answer)] : view.routes, targets: targets, fallback: try target(view.answer, held: held))
    }

    func target(_ name: String, held: [String: ComposeHeld]) throws -> any AnswerTarget {
        switch held[name] {
        case .resident(let model): return LocalAnswer(model)
        case .onDemand(let model): return OnDemandAnswer(model)
        case .cloud(let id):
            let client = try OpenRouterClient(config: .init(apiKey: serve.key(spec, name), baseURL: ServeCommand.openRouterAnswers, timeout: 120, maxRetries: 1))
            return CloudAnswer(client: client, model: id, gate: cloudGate)
        case .decision, nil: throw BasaltError.invalidRequestStructure(details: "compose: \(name) is not loaded as an answer model")
        }
    }

    func status() async -> ComposeStatus {
        var models: [ComposeModelStatus] = []
        for (name, model) in spec.models.sorted(by: { $0.key < $1.key }) {
            var entry = ComposeModelStatus(name: name, source: Self.describe(model.source), keep: model.keep?.rawValue ?? (model.isCloud ? "cloud" : "resident"), state: "unknown")
            switch held[name] {
            case .resident(let orchestrator): entry.state = await orchestrator.isLoaded ? "loaded" : "unloaded"
            case .onDemand(let onDemand):
                (entry.state, entry.lastUsed, entry.idleUnload) = ("\(await onDemand.phase)", await onDemand.lastUsed, model.idleUnload ?? ComposeSpec.defaultIdleUnload)
            case .cloud(let id):
                (entry.state, entry.spentToday, entry.limit, entry.refusal) = ("cloud", await cloudGate.spent(id), model.budget, await cloudGate.refusal(id)?.reason)
            case .decision: entry.state = "decision"
            case nil: break
            }
            models.append(entry)
        }
        return ComposeStatus(name: spec.name, pipelines: spec.pipelineNames, models: models, spentToday: await cloudGate.spent(CloudGateCore.total))
    }

    static func describe(_ source: ComposeSpec.Source) -> String {
        switch source {
        case .local(let source): source
        case .cloud(let provider, let id): "cloud:\(provider)/\(id)"
        }
    }
}
