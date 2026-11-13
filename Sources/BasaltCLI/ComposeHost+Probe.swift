import Foundation
import BasaltCore
import BasaltInference
import BasaltServer

extension ComposeHost {
    func probe(_ names: [String], in spec: ComposeSpec) async {
        let answering = Set(spec.views.flatMap { $0.spec.answerModels })
        for name in names {
            failures[name] = nil
            guard let model = held[name], Self.probes(model, answering: answering.contains(name)) else { continue }
            failures[name] = await services.probe(model)
            if let failure = failures[name] { await events.emit("failed", model: name, failure) }
        }
    }

    static func probes(_ model: ComposeHeld, answering: Bool) -> Bool {
        switch model {
        case .resident: answering
        case .decision, .onDemand, .onDemandDecision: true
        case .cloud: false
        }
    }
}

enum ComposeProbe {
    static let request = SystemOneRequest(state: "The door is open.", questions: ["open": SystemOneQuestion(type: .noul, instructions: "The door is open.")])

    @Sendable static func live(_ model: ComposeHeld) async -> String? {
        do {
            switch model {
            case .resident(let orchestrator):
                _ = try await orchestrator.generate(request: GenerationRequest(prompt: "Reply with OK.", maxTokens: 1))
            case .decision(let engine):
                _ = try await engine.decide(request)
            case .onDemand(let onDemand):
                let report = await ModelInspector().inspect(modelIdOrPath: onDemand.path)
                if !report.intact { return "its files at \(onDemand.path) are incomplete; run basalt models inspect \(onDemand.path)" }
            case .onDemandDecision(let onDemand):
                let url = URL(fileURLWithPath: onDemand.path)
                if CoreAIDecisionEngine.isBundle(url) {
                    try CoreAIDecisionEngine.validateBundle(url)
                } else if JevEvaluator.isBundle(url) {
                    _ = try JevEvaluator.baseModel(of: url)
                } else {
                    let report = await ModelInspector().inspect(modelIdOrPath: onDemand.path)
                    if !report.intact { return "its files at \(onDemand.path) are incomplete; run basalt models inspect \(onDemand.path)" }
                }
            case .cloud:
                break
            }
            return nil
        } catch {
            return "startup probe failed: \(error.localizedDescription)"
        }
    }
}

extension ComposeHost {
    nonisolated var control: ComposeControl {
        var control = ComposeControl(status: { await self.status() }, apply: { request throws(BasaltError) in try await self.apply(request) })
        (control.preview, control.events) = ({ request throws(BasaltError) in try await self.preview(request) }, events)
        return control
    }
}
