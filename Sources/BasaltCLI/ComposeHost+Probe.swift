import Foundation
import BasaltCore
import BasaltServer

extension ComposeHost {
    func probe(_ names: [String], in spec: ComposeSpec) async {
        let answering = Set(spec.views.flatMap { $0.spec.answerModels })
        for name in names {
            failures[name] = nil
            guard let model = held[name], Self.probes(model, answering: answering.contains(name)) else { continue }
            failures[name] = await services.probe(model)
        }
    }

    static func probes(_ model: ComposeHeld, answering: Bool) -> Bool {
        switch model {
        case .resident: answering
        case .decision, .onDemand: true
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
            case .cloud:
                break
            }
            return nil
        } catch {
            return "startup probe failed: \(error.localizedDescription)"
        }
    }
}
