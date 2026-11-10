import Foundation
import ArgumentParser
import BasaltCore
import BasaltCloud
import BasaltInference
import BasaltServer

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

    func composeHost(_ spec: ComposeSpec, start: ComposeStart) async throws -> ComposeHost {
        let host = ComposeHost(serve: self, spec: spec, start: start)
        do {
            try await host.start(defaultDecisions: start.decisions)
        } catch {
            log("\(red("[error]")) compose: \(error.localizedDescription)")
            throw ExitCode.failure
        }
        log("Compose \(bold(spec.name)): pipelines \(spec.pipelineNames.map { $0 == ComposeSpec.defaultPipeline ? "/v1/pipeline" : "/v1/pipelines/\($0)" }.joined(separator: ", ")); GET /v1/compose for status")
        return host
    }
}

extension Optional {
    func asyncMap<T>(_ transform: (Wrapped) async throws -> T) async rethrows -> T? {
        guard let value = self else { return nil }
        return try await transform(value)
    }
}
