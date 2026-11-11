import Foundation
import BasaltCore
import BasaltInference
import BasaltServer

struct ComposeBoot: Sendable {
    let config: ServerConfig
    let engine: MLXInferenceEngine
    let orchestrator: ModelOrchestratorActor
    let chatPath: String
}

extension ServeCommand {
    func runComposed(_ spec: ComposeSpec, _ boot: ComposeBoot) async throws {
        let launch = try await composedLaunch(spec, boot)
        try await prepare(boot.orchestrator, virtualKeys: boot.config.virtualKeys)
        log("REST → http://\(boot.config.host):\(boot.config.restPort); the models load after the bind, GET /v1/health/ready says when they are ready")
        defer { ComposeRecord.remove(spec.name) }
        try await Self.runServers(launch, orchestrator: boot.orchestrator)
    }

    func composedLaunch(_ spec: ComposeSpec, _ boot: ComposeBoot) async throws -> ServerLaunch {
        let embeddings = try await loadEmbeddings()
        let slot = ComposeSlot(spec: spec, cloudGate: cloudGate(spec)) { [self] model in ProcessInfo.processInfo.environment[model?.apiKeyEnv ?? decision.keyEnvironment] ?? "" }
        let file = pipeline.compose ?? ""
        return ServerLaunch(
            config: boot.config, startREST: true, mcpServer: nil, embeddings: BesideDecodeEmbeddingEngine.serving(embeddings, besideDecode: false),
            pipelines: slot.registry, compose: slot.control,
            statsInterval: stats ? min(max(1, statsInterval), 86_400) : nil, canaryInterval: canaryMinutes.map { .seconds($0 * 60) },
            announceReady: network.announceReady,
            onBind: { port in Self.recordBind(spec.name, port: port, file: file) },
            startup: { [self] in try await startComposed(spec, boot, embeddings: embeddings, slot: slot) })
    }

    func startComposed(_ spec: ComposeSpec, _ boot: ComposeBoot, embeddings: (any EmbeddingEngine)?, slot: ComposeSlot) async throws {
        let loadedDecisions = try await loadDecisions(chat: boot.engine)
        let perceivePaths = try await pipelinePerceivePaths()
        let plan = try await admitComputePlan(chatPath: boot.chatPath, embeddings: embeddings, decisions: loadedDecisions, perceive: perceivePaths)
        let besideDecode = plan.runsBesideDecode("decide")
        let decisions = try composedDecisions(spec, local: loadedDecisions, gate: boot.orchestrator.gate, besideDecode: besideDecode, cloudGate: slot.cloudGate)
        try await load(boot.orchestrator, from: boot.chatPath)
        let stages = try await loadPipelineStages(perceivePaths, plan: plan)
        var start = ComposeStart(chat: boot.orchestrator, decider: loadedDecisions, decisions: decisions, cloudGate: slot.cloudGate, startup: stages, besideDecode: besideDecode)
        start.registry = slot.registry
        await slot.started(try await composeHost(spec, start: start, engine: boot.engine))
        log("Compose \(bold(spec.name)) is ready")
    }

    static func recordBind(_ name: String, port: Int, file: String) {
        do {
            _ = try ComposeRecord.write(name: name, port: port, file: file)
        } catch {
            fputs("[compose] could not write the run record for \(name): \(error.localizedDescription)\n", stderr)
        }
    }

    func prepare(_ orchestrator: ModelOrchestratorActor, virtualKeys: [VirtualKey]) async throws {
        if !noMemoryCoordinator { await orchestrator.coordinateMemory(with: MemoryCoordinator()) }
        if !noThermalPolicy { await orchestrator.coordinateThermal() }
        await experimental.attachMediaFallbacks(to: orchestrator)
        try await attachCloudFallback(to: orchestrator, virtualKeys: virtualKeys)
    }
}
