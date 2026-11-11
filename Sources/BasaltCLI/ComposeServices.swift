import Foundation
import BasaltCore
import BasaltCloud

struct ComposeServices: Sendable {
    var resolve: @Sendable (_ source: String) async throws -> String
    var load: @Sendable (_ source: String, _ path: String, _ role: String, _ onDemand: Bool) async throws -> ModelOrchestratorActor?
    var decisions: @Sendable (_ view: ComposeSpec, _ local: (any DecisionEngine)?, _ gate: GenerationGate, _ besideDecode: Bool, _ cloudGate: CloudGate) throws -> (any DecisionEngine)?
    var key: @Sendable (_ model: ComposeSpec.Model?) -> String
    var cloudLimits: @Sendable (_ spec: ComposeSpec) -> [String: Double]
}

extension ComposeServices {
    init(serve: ServeCommand) {
        self.init(
            resolve: { try await serve.resolvedPath($0, quant: nil) },
            load: { source, path, role, onDemand in try await serve.loadPerceiver(source, path: path, role: role, loads: !onDemand, onDemand: onDemand) },
            decisions: { view, local, gate, besideDecode, cloudGate in
                try serve.composedDecisions(view, local: local, gate: gate, besideDecode: besideDecode, cloudGate: cloudGate)
            },
            key: { ProcessInfo.processInfo.environment[$0?.apiKeyEnv ?? serve.decision.keyEnvironment] ?? "" },
            cloudLimits: { serve.cloudLimits($0) })
    }
}
