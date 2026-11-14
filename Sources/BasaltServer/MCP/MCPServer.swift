import Foundation
import Logging
import ServiceLifecycle
import Synchronization
import BasaltCore
import BasaltHarness

public final class MCPServer: Service, Sendable {
    let orchestrator: ModelOrchestratorActor
    let modelId: String?
    let library: HarnessExperienceLibrary?
    let harnessRoots: [URL]
    let harnessLimits: HarnessLoadLimits
    let researchTools: Bool
    let toolFilter: MCPToolFilter
    let imageDirectories: [URL]
    let decisions: (any DecisionEngine)?
    let pipeline: PipelineStages
    private let makeInput: @Sendable () -> AsyncStream<String>
    let writer: @Sendable (String) -> Void

    public init(
        orchestrator: ModelOrchestratorActor,
        modelId: String? = nil,
        library: HarnessExperienceLibrary? = nil,
        harnessRoots: [URL] = HarnessLoader.allowedRoots(),
        harnessLimits: HarnessLoadLimits = .standard,
        writer: @Sendable @escaping (String) -> Void = { print($0); fflush(stdout) }
    ) {
        self.orchestrator = orchestrator
        self.modelId = modelId
        self.library = library
        self.harnessRoots = harnessRoots
        self.harnessLimits = harnessLimits
        self.researchTools = false
        self.toolFilter = .standard
        self.imageDirectories = []
        self.decisions = nil
        self.pipeline = PipelineStages()
        self.makeInput = Self.stdinStream
        self.writer = writer
    }

    package init(
        orchestrator: ModelOrchestratorActor,
        modelId: String? = nil,
        input: AsyncStream<String>,
        writer: @Sendable @escaping (String) -> Void
    ) {
        self.orchestrator = orchestrator
        self.modelId = modelId
        self.library = nil
        self.harnessRoots = HarnessLoader.allowedRoots()
        self.harnessLimits = .standard
        self.researchTools = false
        self.toolFilter = .standard
        self.imageDirectories = []
        self.decisions = nil
        self.pipeline = PipelineStages()
        self.makeInput = { input }
        self.writer = writer
    }

    private init(
        _ server: MCPServer,
        researchTools: Bool,
        toolFilter: MCPToolFilter,
        imageDirectories: [URL],
        decisions: (any DecisionEngine)?,
        pipeline: PipelineStages
    ) {
        orchestrator = server.orchestrator
        modelId = server.modelId
        library = server.library
        harnessRoots = server.harnessRoots
        harnessLimits = server.harnessLimits
        self.researchTools = researchTools
        self.toolFilter = toolFilter
        self.imageDirectories = imageDirectories
        self.decisions = decisions
        self.pipeline = pipeline
        makeInput = server.makeInput
        writer = server.writer
    }

    public func withResearchTools(_ enabled: Bool) -> MCPServer {
        MCPServer(self, researchTools: enabled, toolFilter: toolFilter, imageDirectories: imageDirectories, decisions: decisions, pipeline: pipeline)
    }

    public func withToolFilter(_ filter: MCPToolFilter) -> MCPServer {
        MCPServer(self, researchTools: researchTools, toolFilter: filter, imageDirectories: imageDirectories, decisions: decisions, pipeline: pipeline)
    }

    public func withImageDirectories(_ directories: [URL]) -> MCPServer {
        MCPServer(self, researchTools: researchTools, toolFilter: toolFilter, imageDirectories: directories.map(MCPImages.resolved), decisions: decisions, pipeline: pipeline)
    }

    package func withPipeline(_ stages: PipelineStages) -> MCPServer {
        MCPServer(self, researchTools: researchTools, toolFilter: toolFilter, imageDirectories: imageDirectories, decisions: decisions, pipeline: stages)
    }

    public func withDecisions(_ engine: (any DecisionEngine)?) -> MCPServer {
        MCPServer(self, researchTools: researchTools, toolFilter: toolFilter, imageDirectories: imageDirectories, decisions: GatedDecisionEngine.serving(engine, gate: orchestrator.gate), pipeline: pipeline)
    }
}

extension MCPServer {
    private static func stdinStream() -> AsyncStream<String> {
        AsyncStream { continuation in
            let stopped = Atomic<Bool>(false)
            continuation.onTermination = { _ in
                stopped.store(true, ordering: .relaxed)
            }
            Thread.detachNewThread {
                while !stopped.load(ordering: .relaxed), let line = readLine(strippingNewline: true) {
                    if case .terminated = continuation.yield(line) { break }
                }
                continuation.finish()
            }
        }
    }

    public func run() async {
        for await line in makeInput().cancelOnGracefulShutdown() {
            guard !Task.isCancelled && !Task.isShuttingDownGracefully else { break }
            guard !line.isEmpty else { continue }
            await dispatch(line)
        }
    }

    package func runServiceGroup(shutdownDeadline: Duration = .seconds(5), handleSignals: Bool = true) async throws {
        var logger = Logger(label: "Basalt")
        logger.logLevel = .warning
        let drain = GenerationDrainService(orchestrator: orchestrator, deadline: shutdownDeadline)
        let group = ServiceGroup(configuration: .init(
            services: [
                .init(service: self, successTerminationBehavior: .gracefullyShutdownGroup),
                .init(service: drain, successTerminationBehavior: .gracefullyShutdownGroup),
            ],
            gracefulShutdownSignals: handleSignals ? [.sigterm, .sigint] : [],
            logger: logger
        ))
        try await group.run()
    }

    internal func dispatch(_ line: String) async {
        switch MCPRoute(line: line) {
        case .failed(let id, let failure): writer(MCPWire.error(id: id, failure))
        case .ignored: break
        case .initialize(let id): writer(MCPWire.result(id: id, MCPInitializeResult.current))
        case .toolsList(let id): writer(MCPWire.result(id: id, MCPToolCatalog.catalog(researchTools: researchTools, filter: toolFilter)))
        case .toolsCall(let id, let params): await handleToolsCall(id: id, params: params)
        }
    }
}
