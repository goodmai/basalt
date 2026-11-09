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
    let imageDirectories: [URL]
    let decisions: (any DecisionEngine)?
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
        self.imageDirectories = []
        self.decisions = nil
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
        self.imageDirectories = []
        self.decisions = nil
        self.makeInput = { input }
        self.writer = writer
    }

    private init(_ server: MCPServer, researchTools: Bool, imageDirectories: [URL], decisions: (any DecisionEngine)?) {
        orchestrator = server.orchestrator
        modelId = server.modelId
        library = server.library
        harnessRoots = server.harnessRoots
        harnessLimits = server.harnessLimits
        self.researchTools = researchTools
        self.imageDirectories = imageDirectories
        self.decisions = decisions
        makeInput = server.makeInput
        writer = server.writer
    }

    public func withResearchTools(_ enabled: Bool) -> MCPServer {
        MCPServer(self, researchTools: enabled, imageDirectories: imageDirectories, decisions: decisions)
    }

    public func withImageDirectories(_ directories: [URL]) -> MCPServer {
        MCPServer(self, researchTools: researchTools, imageDirectories: directories.map(MCPImages.resolved), decisions: decisions)
    }

    public func withDecisions(_ engine: (any DecisionEngine)?) -> MCPServer {
        MCPServer(self, researchTools: researchTools, imageDirectories: imageDirectories, decisions: engine)
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
        case .toolsList(let id): writer(MCPWire.result(id: id, MCPToolCatalog.catalog(researchTools: researchTools)))
        case .toolsCall(let id, let params): await handleToolsCall(id: id, params: params)
        }
    }
}
