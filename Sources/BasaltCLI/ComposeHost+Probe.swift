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
        case .decision, .onDemand, .onDemandDecision, .speaker, .onDemandSpeaker, .imageGenerator, .onDemandImageGenerator, .imageDecoder: true
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
                return await inspectIntact(onDemand.path)
            case .onDemandDecision(let onDemand):
                return try await probeOnDemandDecision(onDemand.path)
            case .speaker(let loaded):
                _ = try await SpeechSynthesizerLoader.probeSpeechSynthesizer(loaded.synthesizer)
            case .onDemandSpeaker(let onDemand):
                return try probeOnDemandSpeaker(onDemand.descriptor)
            case .onDemandImageGenerator(let onDemand):
                if !FileManager.default.fileExists(atPath: onDemand.path) {
                    return "its files at \(onDemand.path) are incomplete; run basalt models inspect \(onDemand.path)"
                }
            case .imageDecoder(_, let path):
                if !FileManager.default.fileExists(atPath: path) {
                    return "its files at \(path) are incomplete; run basalt models inspect \(path)"
                }
            case .imageGenerator, .cloud:
                break
            }
            return nil
        } catch {
            return "startup probe failed: \(error.localizedDescription)"
        }
    }

    private static func inspectIntact(_ path: String) async -> String? {
        let report = await ModelInspector().inspect(modelIdOrPath: path)
        return report.intact ? nil : "its files at \(path) are incomplete; run basalt models inspect \(path)"
    }

    private static func probeOnDemandDecision(_ path: String) async throws -> String? {
        let url = URL(fileURLWithPath: path)
        if CoreAIDecisionEngine.isBundle(url) {
            try CoreAIDecisionEngine.validateBundle(url)
            return nil
        }
        if MultiHeadDecisionEngine.isBundle(url) {
            _ = try MultiHeadDecisionEngine.baseModel(of: url)
            return nil
        }
        if JevEvaluator.isBundle(url) {
            _ = try JevEvaluator.baseModel(of: url)
            return nil
        }
        return await inspectIntact(path)
    }

    private static func probeOnDemandSpeaker(_ descriptor: OnDemandSpeechSynthesizer.Descriptor) throws -> String? {
        let url = URL(fileURLWithPath: descriptor.path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return "its files at \(descriptor.path) are incomplete; run basalt models inspect \(descriptor.path)"
        }
        _ = try SpeechSynthesizerLoader.inspectOnDemand(identifier: descriptor.modelID, directory: url)
        return nil
    }
}

extension ComposeHost {
    nonisolated var control: ComposeControl {
        var control = ComposeControl(status: { await self.status() }, apply: { request throws(BasaltError) in try await self.apply(request) })
        (control.preview, control.events) = ({ request throws(BasaltError) in try await self.preview(request) }, events)
        return control
    }
}
