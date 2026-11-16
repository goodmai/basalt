import Foundation
import BasaltCore
import BasaltCloud
import BasaltInference

struct ComposeServices: Sendable {
    var resolve: @Sendable (_ source: String) async throws -> String
    var load: @Sendable (_ source: String, _ path: String, _ role: String, _ onDemand: Bool) async throws -> (any ModelOrchestrating)?
    var decider: @Sendable (_ model: String) async throws -> any DecisionEngine
    var decisions: @Sendable (_ view: ComposeSpec, _ locals: [String: any DecisionEngine], _ gate: GenerationGate, _ besideDecode: Bool, _ cloudGate: CloudGate) throws -> (any DecisionEngine)?
    var key: @Sendable (_ model: ComposeSpec.Model?) -> String
    var cloudLimits: @Sendable (_ spec: ComposeSpec) -> [String: Double]
    var probe: @Sendable (_ model: ComposeHeld) async -> String? = { _ in nil }
    var speaker: @Sendable (_ source: String, _ path: String) async throws -> SpeechSynthesizerLoader.LoadedSynthesizer = { source, path in
        try await SpeechSynthesizerLoader.load(identifier: source, directory: URL(fileURLWithPath: path))
    }
    var imageGenerator: @Sendable (
        _ source: String,
        _ path: String,
        _ sharedTextEmbedder: (@Sendable (String) async throws -> [Float])?
    ) async throws -> any ImageSynthesizer = { source, path, sharedTextEmbedder in
        try await ImageSynthesizerLoader.load(
            identifier: source,
            directory: URL(fileURLWithPath: path),
            sharedTextEmbedder: sharedTextEmbedder
        )
    }
    var sharedTextEmbedder: (@Sendable (String) async throws -> [Float])? = nil
    var memoryPressure: @Sendable () async -> Bool = { SystemMemoryPressure().sample().pressure != .normal }
}

extension ComposeServices {
    init(serve: ServeCommand, chat: MLXInferenceEngine, chatPath: String? = nil, memory: MemoryCoordinator? = nil) {
        self.init(
            resolve: { try await serve.resolvedPath($0, quant: nil) },
            load: { source, path, role, onDemand in try await serve.loadPerceiver(source, path: path, role: role, loads: !onDemand, onDemand: onDemand) },
            decider: { try await serve.loadDecisions($0, chat: chat, chatPath: chatPath) },
            decisions: { view, locals, gate, besideDecode, cloudGate in
                try serve.composedDecisions(view, locals: locals, gate: gate, besideDecode: besideDecode, cloudGate: cloudGate)
            },
            key: { ProcessInfo.processInfo.environment[$0?.apiKeyEnv ?? serve.decision.keyEnvironment] ?? "" },
            cloudLimits: { serve.cloudLimits($0) },
            probe: ComposeProbe.live,
            speaker: { source, path in
                try await SpeechSynthesizerLoader.load(identifier: source, directory: URL(fileURLWithPath: path))
            },
            imageGenerator: { source, path, sharedTextEmbedder in
                try await ImageSynthesizerLoader.load(
                    identifier: source,
                    directory: URL(fileURLWithPath: path),
                    sharedTextEmbedder: sharedTextEmbedder
                )
            },
            sharedTextEmbedder: { prompt in
                try await chat.encodePromptForImage(prompt)
            },
            memoryPressure: {
                if let memory {
                    let state = await memory.sample()
                    return state.directive.clearsCache || state.sample.pressure != .normal
                }
                return SystemMemoryPressure().sample().pressure != .normal
            })
    }
}
