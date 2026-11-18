import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltServer
import BasaltSpeech

struct DirectServeBoot<Orchestrator: ModelOrchestrating>: Sendable {
    let config: ServerConfig
    let engine: MLXInferenceEngine
    let orchestrator: Orchestrator
    let resolvedPath: String
    let port: Int
}

extension ServeCommand {
    func loadSpeechSynthesizer(
        gate: GenerationGate,
        memory: MemoryCoordinator?
    ) async throws -> (any SpeechSynthesizer)? {
        guard let ttsModel else { return nil }
        if ttsModel == ComposeSpec.appleSpeech {
            log("Serving \(ComposeSpec.appleSpeech) (/v1/audio/speech, Apple System Voice)")
            return AppleSpeechSynthesizer()
        }
        let path = try await resolvedPath(ttsModel, quant: nil)
        do {
            let loaded = try await SpeechSynthesizerLoader.load(
                identifier: ttsModel,
                directory: URL(fileURLWithPath: path)
            )
            if modelProbe {
                _ = try await SpeechSynthesizerLoader.probeSpeechSynthesizer(loaded.synthesizer)
            }
            log("Serving \(loaded.candidate.model) (/v1/audio/speech, \(loaded.candidate.compute.rawValue))")
            let gpuBusyProbe: @Sendable () async -> Bool = {
                if await gate.inFlight > 0 { return true }
                return await gate.queued > 0
            }
            let memoryPressureProbe: @Sendable () async -> Bool = {
                if let memory {
                    let state = await memory.sample()
                    return state.directive.clearsCache || state.sample.pressure != .normal
                }
                return SystemMemoryPressure().sample().pressure != .normal
            }
            return RoutedSpeechSynthesizer(
                tiers: [
                    RoutedSpeechSynthesizer.Tier(candidate: loaded.candidate, synthesizer: loaded.synthesizer),
                    RoutedSpeechSynthesizer.Tier(candidate: .appleSpeech, synthesizer: AppleSpeechSynthesizer()),
                ],
                onGPUBusy: loaded.candidate.compute == .mlx ? .fallbackAppleSpeech : .keep,
                gpuBusyProbe: gpuBusyProbe,
                memoryPressureProbe: memoryPressureProbe
            )
        } catch {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }
}

extension ServeCommand {
    func runDirect<Orchestrator: ModelOrchestrating>(_ spec: ComposeSpec?, _ boot: DirectServeBoot<Orchestrator>) async throws {
        let memory = noMemoryCoordinator ? nil : MemoryCoordinator()
        let loadedEmbeddings = try await loadEmbeddings()
        let loadedDecisions = try await loadDecisions(chat: boot.engine, chatPath: boot.resolvedPath)
        let perceivePaths = try await pipelinePerceivePaths()
        let plan = try await admitComputePlan(
            chatPath: boot.resolvedPath,
            embeddings: loadedEmbeddings,
            decisions: loadedDecisions,
            perceive: perceivePaths
        )
        let embeddings = BesideDecodeEmbeddingEngine.serving(loadedEmbeddings, besideDecode: plan.runsBesideDecode("embed"))
        let cloudGate = cloudGate(spec)
        var decisions = try spec.map {
            try composedDecisions($0, local: loadedDecisions, gate: boot.orchestrator.gate, besideDecode: plan.runsBesideDecode("decide"), cloudGate: cloudGate)
        } ?? servedDecisions(loadedDecisions, gate: boot.orchestrator.gate, besideDecode: plan.runsBesideDecode("decide"))

        try await load(boot.orchestrator, from: boot.resolvedPath)
        var stages = try await loadPipelineStages(perceivePaths, plan: plan)
        let start = ComposeStart(
            chat: boot.orchestrator,
            decider: loadedDecisions,
            decisions: decisions,
            cloudGate: cloudGate,
            startup: stages,
            besideDecode: plan.runsBesideDecode("decide")
        )
        let host = try await spec.asyncMap {
            try await composeHost($0, start: start, engine: boot.engine, chatPath: boot.resolvedPath, memory: memory)
        }
        if let spec, let host {
            await host.probe(spec.models.keys.sorted(), in: spec)
            for (name, failure) in await host.failures.sorted(by: { $0.key < $1.key }) { log("\(red("[error]")) \(name): \(failure)") }
        }
        var activeSpeech = try await loadSpeechSynthesizer(gate: boot.orchestrator.gate, memory: memory)
        var activeImage = try await loadImageSynthesizer(gate: boot.orchestrator.gate)
        if let host, let system = await host.registry.system(ComposeSpec.defaultPipeline) {
            (stages, decisions) = (system.stages, system.decisions)
            if activeSpeech == nil { activeSpeech = system.speech }
            if activeImage == nil { activeImage = system.image }
        }
        let (startREST, startMCP) = (!mcp || rest, mcp)
        let record = try spec.flatMap { (!startREST || host == nil) ? nil : try ComposeRecord.write(name: $0.name, port: boot.port, file: pipeline.compose ?? "") }
        defer { record.map(ComposeRecord.remove) }
        try await prepare(boot.orchestrator, virtualKeys: boot.config.virtualKeys, memory: memory)

        if startREST { log("REST → http://\(boot.config.host):\(boot.config.restPort)") }
        if startMCP  { log("MCP → JSON-RPC 2.0 on stdin/stdout") }
        let mcpServer = startMCP ? try makeMCPServer(
            orchestrator: boot.orchestrator, resolvedPath: boot.resolvedPath, decisions: decisions, stages: stages,
            pipelines: host?.registry, imageSynthesizer: activeImage) : nil

        try await Self.runServers(
            ServerLaunch(
                config: boot.config, startREST: startREST, mcpServer: mcpServer, embeddings: embeddings, decisions: decisions,
                pipeline: stages, pipelines: host?.registry, compose: host.map(\.control),
                statsInterval: stats ? min(max(1, statsInterval), 86_400) : nil, canaryInterval: canaryMinutes.map { .seconds($0 * 60) },
                announceReady: network.announceReady, speechSynthesizer: activeSpeech, imageSynthesizer: activeImage
            ),
            orchestrator: boot.orchestrator
        )
    }
}
