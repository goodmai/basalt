import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltServer

struct ServerLaunch: Sendable {
    let config: ServerConfig
    let startREST: Bool
    let mcpServer: MCPServer?
    var embeddings: (any EmbeddingEngine)? = nil
    var decisions: (any DecisionEngine)? = nil
    var pipeline = PipelineStages()
    var statsInterval: Int? = nil
    var canaryInterval: Duration? = nil
    var shutdownDeadline: Duration = .seconds(5)
    var handleSignals = true
}

extension ServeCommand {
    func validate() throws {
        try sessionLog.validateRetention()
        _ = try Self.adapterDirectories(adapters)
        guard maxConcurrent >= 1 else {
            throw ValidationError("--max-concurrent must be at least 1, got \(maxConcurrent)")
        }
        guard batchSlots >= 1 else {
            throw ValidationError("--batch-slots must be at least 1, got \(batchSlots)")
        }
        guard maxQueue >= 0 else {
            throw ValidationError("--max-queue must be 0 or more, got \(maxQueue)")
        }
        if let canaryMinutes, !(1...10_080).contains(canaryMinutes) {
            throw ValidationError("--canary-minutes must be between 1 and 10080, got \(canaryMinutes)")
        }
        try network.validateRates()
        guard (0...1).contains(promptCache.cacheBlendRatio) else {
            throw ValidationError("--cache-blend-ratio must be between 0 and 1, got \(promptCache.cacheBlendRatio)")
        }
        try kvCache.validateCombination()
        try promptCacheDisk.validate(sessionLog: sessionLog.mode)
        guard promptCacheDisk.megabytes == 0 || promptCache.promptCacheMB != 0 else {
            throw ValidationError("--prompt-cache-disk-mb persists the in-memory prompt cache, which --prompt-cache-mb 0 turns off")
        }
    }

    var targetModel: String { model ?? modelPath ?? Self.defaultModel }

    func assessMemory() async {
        log("🔍 Performing dry-run memory assessment for \(bold(targetModel))…")
        let assessment = await MemoryEvaluator(expertStreaming: expertStreaming.setting).evaluate(
            modelId: targetModel, kvPrecision: kvCache.bits?.budgetPrecision, kvZones: kvCache.zones)
        let lines = assessment.reportLines(
            kvZones: kvCache.zoned, kvEvict: kvCache.evict,
            fits: "✓ Dry-run passed: Model fits in memory budget.", exceeds: "❌ Dry-run warning: Model exceeds available RAM!")
        fputs("\n" + (lines + ["", ""]).joined(separator: "\n"), stderr)
    }

    func makeEngine() throws -> MLXInferenceEngine {
        var configuration = EngineConfiguration.standard
        sampling.configure(&configuration)
        promptCache.configure(&configuration)
        speculative.configure(&configuration)
        (configuration.kvPrecision, configuration.kvZones) = (kvCache.bits?.fixedPrecision, kvCache.zones)
        (configuration.kvHeadSplit, configuration.questTopK) = (kvCache.headSplit, kvCache.questTopK)
        configuration.batchSlots = maxConcurrent >= 2 ? batchSlots : 1
        configuration.wiresWorkingSet = wiredLimit
        configuration.expertStreaming = expertStreaming.setting
        configuration.probesModel = modelProbe
        configuration.visionTowerBundle = experimental.aneVisionTower
        return MLXInferenceEngine(configuration, promptCacheDisk: try promptCacheDisk.store())
    }

    static func runServers(_ launch: ServerLaunch, orchestrator: ModelOrchestratorActor) async throws {
        defer {
            await withTaskCancellationShield {
                await orchestrator.shutdown(deadline: launch.shutdownDeadline)
            }
        }
        try await withThrowingTaskGroup(of: Void.self) { group in
            if launch.startREST {
                let restServer = RESTServer(
                    orchestrator: orchestrator,
                    config: launch.config,
                    embeddings: launch.embeddings,
                    shutdownDeadline: launch.shutdownDeadline,
                    handleSignals: launch.handleSignals
                ).withDecisions(launch.decisions).withPipeline(launch.pipeline)
                group.addTask { try await restServer.run() }
            }
            if let mcpServer = launch.mcpServer {
                if launch.startREST {
                    group.addTask { await mcpServer.run() }
                } else {
                    group.addTask {
                        try await mcpServer.runServiceGroup(shutdownDeadline: launch.shutdownDeadline, handleSignals: launch.handleSignals)
                    }
                }
            }
            if let interval = launch.canaryInterval {
                group.addTask { await Canary.run(every: interval, on: orchestrator) }
            }
            if let interval = launch.statsInterval {
                group.addTask {
                    while (try? await Task.sleep(for: .seconds(interval))) != nil {
                        let snapshot = await orchestrator.telemetrySnapshot()
                        fputs("\(snapshot.formattedStatsLine)\n", stderr)
                    }
                }
            }
            try await group.next()
            group.cancelAll()
        }
    }
}
