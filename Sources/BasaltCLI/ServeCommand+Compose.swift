import Foundation
import ArgumentParser
import BasaltCore
import BasaltCloud
import BasaltInference

extension ServeCommand {
    static let openRouterAnswers = URL(string: "https://openrouter.ai/api/v1")!

    func composeSpec() throws -> ComposeSpec? {
        guard let reference = pipeline.compose else { return nil }
        do {
            return try ComposeFile.load(reference)
        } catch {
            log("\(red("[error]")) \(error.description)")
            throw ExitCode.failure
        }
    }

    func cloudGate(_ spec: ComposeSpec?) -> CloudGate {
        var limits: [String: Double] = [:]
        for model in spec?.models.values ?? [:].values {
            if case .cloud(_, let id) = model.source, let budget = model.budget { limits[id] = budget }
        }
        if let total = spec?.budget { limits[CloudGateCore.total] = total }
        if let cloud = decision.cloudModel, let budget = decision.budgetPerDay { limits[cloud] = budget }
        return CloudGate(limits: limits, storage: DecisionOptions.spendFile)
    }

    func key(_ spec: ComposeSpec, _ name: String) -> String {
        ProcessInfo.processInfo.environment[spec.models[name]?.apiKeyEnv ?? decision.keyEnvironment] ?? ""
    }

    func composedDecisions(
        _ spec: ComposeSpec, local: (any DecisionEngine)?, gate: GenerationGate, besideDecode: Bool, cloudGate: CloudGate
    ) throws -> (any DecisionEngine)? {
        let clouds = spec.cascade.dropLast().filter { spec.models[$0]?.isCloud == true }
        guard let local, !clouds.isEmpty else { return GatedDecisionEngine.serving(local, gate: gate, besideDecode: besideDecode) }
        do {
            let remotes: [any DecisionEngine] = try clouds.map { name in
                guard case .cloud(_, let id)? = spec.models[name]?.source else { throw BasaltError.invalidRequestStructure(details: "\(name) is not a cloud model") }
                return GuardedDecisionEngine(try RemoteDecisionEngine(model: id, apiKey: key(spec, name)), gate: cloudGate)
            }
            log("Cloud decisions → \(clouds.joined(separator: " → ")) → \(local.modelID); confidence below \(spec.escalateBelow ?? decision.escalateBelow) goes on to the next stage")
            let gated = GatedDecisionEngine.serving(local, gate: gate, besideDecode: besideDecode) ?? local
            return CascadeDecisionEngine(stages: remotes + [gated], escalateBelow: spec.escalateBelow ?? decision.escalateBelow)
        } catch {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    func answerRouter(_ spec: ComposeSpec, chat: ModelOrchestratorActor, cloudGate: CloudGate) async throws -> AnswerRouter? {
        guard !spec.routes.isEmpty else { return nil }
        var targets: [String: any AnswerTarget] = [spec.answer: LocalAnswer(chat)]
        for name in spec.answerModels where name != spec.answer {
            targets[name] = try await answerTarget(spec, name, cloudGate: cloudGate)
        }
        log("Answer routes: " + spec.routes.enumerated().map { "\($0.offset): \($0.element.model)\($0.element.fallback.map { " → \($0)" } ?? "")" }.joined(separator: ", "))
        return AnswerRouter(routes: spec.routes, targets: targets, fallback: LocalAnswer(chat))
    }

    func answerTarget(_ spec: ComposeSpec, _ name: String, cloudGate: CloudGate) async throws -> any AnswerTarget {
        guard let model = spec.models[name] else { throw ExitCode.failure }
        switch model.source {
        case .cloud(_, let id):
            let client = try OpenRouterClient(config: .init(apiKey: key(spec, name), baseURL: Self.openRouterAnswers, timeout: 120, maxRetries: 1))
            return CloudAnswer(client: client, model: id, gate: cloudGate)
        case .local(let source):
            let path = try await resolvedPath(source, quant: nil)
            guard model.keep == .onDemand else {
                guard let loaded = try await loadPerceiver(source, path: path, role: "answer") else { throw ExitCode.failure }
                return LocalAnswer(loaded)
            }
            guard let orchestrator = try await loadPerceiver(source, path: path, role: "answer", loads: false, onDemand: true) else { throw ExitCode.failure }
            let idle = model.idleUnload ?? ComposeSpec.defaultIdleUnload
            log("Serving \(source) on demand for answers: unloaded after \(idle) s idle")
            return OnDemandAnswer(OnDemandModel(model: orchestrator, path: path, idle: .seconds(idle)))
        }
    }
}
