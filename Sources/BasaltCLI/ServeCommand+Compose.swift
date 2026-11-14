import Foundation
import ArgumentParser
import BasaltCore
import BasaltCloud
import BasaltInference
import BasaltServer

extension ServeCommand {
    static let openRouterAnswers = URL(string: "https://openrouter.ai/api/v1")!

    mutating func prepareCompose() throws -> ComposeSpec? {
        guard let spec = try composeSpec() else { return nil }
        try applyComposeSpec(spec)
        return spec
    }

    mutating func applyComposeSpec(_ spec: ComposeSpec) throws {
        if model == nil, modelPath == nil {
            let source = spec.local(spec.answer) ?? spec.answer
            if source.hasPrefix("/") || source.hasPrefix(".") || source.hasPrefix("~") {
                modelPath = (source as NSString).expandingTildeInPath
            } else {
                model = source
            }
        }
        applyComposeExperimental(spec.experimental, usesSpeech: spec.usesSpeechAnalyzer)
        applyComposeMCP(spec.mcp)
        try experimental.validate()
        try expertStreaming.validate()
        try kvCache.validateCombination()
        try mcpOptions.validate()
    }

    private mutating func applyComposeExperimental(_ exp: ComposeSpec.Experimental, usesSpeech: Bool) {
        if exp.zeroCopyBody { experimental.zeroCopyBody = true }
        if exp.smallMKernel { experimental.smallMKernel = true }
        if usesSpeech { experimental.speechAnalyzer = true }
        if experimental.imageFallback == nil, exp.imageFallback == "ocr" { experimental.imageFallback = .ocr }
        if experimental.aneVisionTower == nil, let tower = exp.aneVisionTower { experimental.aneVisionTower = tower }
        if experimental.telemetryConfidenceEvery == 0, let every = exp.telemetryConfidenceEvery { experimental.telemetryConfidenceEvery = every }
        if exp.adaptiveThinkingBudget { thinking.experimentalAdaptiveThinkingBudget = true }
        if exp.samplers { sampling.experimentalSamplers = true }
        if exp.mtpDraftHead { speculative.mtpDraftHead = true }
        if exp.expertStreaming { expertStreaming.enabled = true }
        if expertStreaming.cacheGB == nil, let cacheGB = exp.expertCacheGB { expertStreaming.cacheGB = cacheGB }
        if kvCache.headSplit == nil, let split = exp.kvHeadSplit { kvCache.headSplit = split }
        if kvCache.questTopK == nil, let topK = exp.questTopK { kvCache.questTopK = topK }
    }

    private mutating func applyComposeMCP(_ specMCP: ComposeSpec.MCP) {
        if specMCP.enabled { mcpOptions.mcp = true }
        if specMCP.rest { mcpOptions.rest = true }
        if specMCP.researchTools { mcpOptions.researchTools = true }
        if mcpOptions.enable.isEmpty { mcpOptions.enable = specMCP.enable }
        if mcpOptions.disable.isEmpty { mcpOptions.disable = specMCP.disable }
        if mcpOptions.imageDirs.isEmpty { mcpOptions.imageDirs = specMCP.imageDirs }
    }

    func composeSpec() throws -> ComposeSpec? {
        guard let reference = pipeline.compose else { return nil }
        do {
            return try ComposeFile.load(reference, context: .machine.overriding(assignments: pipeline.composeSet, profiles: pipeline.composeProfile))
        } catch {
            log("\(red("[error]")) \(error.description)")
            throw ExitCode.failure
        }
    }

    func cloudLimits(_ spec: ComposeSpec?) -> [String: Double] {
        var limits: [String: Double] = [:]
        for model in spec?.models.values ?? [:].values {
            if case .cloud(_, let id) = model.source, let budget = model.budget { limits[id] = budget }
        }
        if let total = spec?.budget { limits[CloudGateCore.total] = total }
        if let cloud = decision.cloudModel, let budget = decision.budgetPerDay { limits[cloud] = budget }
        return limits
    }

    func cloudGate(_ spec: ComposeSpec?) -> CloudGate {
        CloudGate(limits: cloudLimits(spec), storage: DecisionOptions.spendFile)
    }

    func key(_ spec: ComposeSpec, _ name: String) -> String {
        ProcessInfo.processInfo.environment[spec.models[name]?.apiKeyEnv ?? decision.keyEnvironment] ?? ""
    }

    func composedDecisions(
        _ spec: ComposeSpec, local: (any DecisionEngine)?, gate: GenerationGate, besideDecode: Bool, cloudGate: CloudGate
    ) throws -> (any DecisionEngine)? {
        try composedDecisions(spec, locals: spec.cascade.last.flatMap { last in local.map { [last: $0] } } ?? [:], gate: gate, besideDecode: besideDecode, cloudGate: cloudGate)
    }

    func composedDecisions(
        _ spec: ComposeSpec, locals: [String: any DecisionEngine], gate: GenerationGate, besideDecode: Bool, cloudGate: CloudGate
    ) throws -> (any DecisionEngine)? {
        guard let last = spec.cascade.last, locals[last] != nil else { return nil }
        do {
            let stages: [any DecisionEngine] = try spec.cascade.compactMap { name in
                guard case .cloud(let provider, let id)? = spec.models[name]?.source else {
                    return locals[name].map { GatedDecisionEngine.serving($0, gate: gate, besideDecode: besideDecode) ?? $0 }
                }
                return GuardedDecisionEngine(try RemoteDecisionEngine(model: id, apiKey: key(spec, name), provider: CloudDecisionProvider(rawValue: provider) ?? .openRouter), gate: cloudGate)
            }
            guard stages.count > 1 else { return stages.first }
            log("Decisions → \(spec.cascade.joined(separator: " → ")); confidence below \(spec.escalateBelow ?? decision.escalateBelow) goes on to the next stage")
            return CascadeDecisionEngine(stages: stages, escalateBelow: spec.escalateBelow ?? decision.escalateBelow)
        } catch {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    func composeHost(_ spec: ComposeSpec, start: ComposeStart, engine: MLXInferenceEngine) async throws -> ComposeHost {
        let host = ComposeHost(services: ComposeServices(serve: self, chat: engine), spec: spec, start: start)
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
