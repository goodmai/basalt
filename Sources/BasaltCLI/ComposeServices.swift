import Foundation
import BasaltCore
import BasaltCloud
import BasaltInference

struct ComposeServices: Sendable {
    var resolve: @Sendable (_ source: String) async throws -> String
    var load: @Sendable (_ source: String, _ path: String, _ role: String, _ onDemand: Bool) async throws -> ModelOrchestratorActor?
    var decider: @Sendable (_ model: String) async throws -> any DecisionEngine
    var decisions: @Sendable (_ view: ComposeSpec, _ local: (any DecisionEngine)?, _ gate: GenerationGate, _ besideDecode: Bool, _ cloudGate: CloudGate) throws -> (any DecisionEngine)?
    var key: @Sendable (_ model: ComposeSpec.Model?) -> String
    var cloudLimits: @Sendable (_ spec: ComposeSpec) -> [String: Double]
    var probe: @Sendable (_ model: ComposeHeld) async -> String? = { _ in nil }
}

extension ComposeServices {
    init(serve: ServeCommand, chat: MLXInferenceEngine) {
        self.init(
            resolve: { try await serve.resolvedPath($0, quant: nil) },
            load: { source, path, role, onDemand in try await serve.loadPerceiver(source, path: path, role: role, loads: !onDemand, onDemand: onDemand) },
            decider: { try await serve.loadDecisions($0, chat: chat) },
            decisions: { view, local, gate, besideDecode, cloudGate in
                try serve.composedDecisions(view, local: local, gate: gate, besideDecode: besideDecode, cloudGate: cloudGate)
            },
            key: { ProcessInfo.processInfo.environment[$0?.apiKeyEnv ?? serve.decision.keyEnvironment] ?? "" },
            cloudLimits: { serve.cloudLimits($0) },
            probe: ComposeProbe.live)
    }
}
