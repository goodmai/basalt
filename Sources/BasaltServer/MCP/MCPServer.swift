import Foundation
import Logging
import ServiceLifecycle
import BasaltCore
import BasaltHarness

public final class MCPServer: Service, Sendable {
    struct RuntimeOptions: Sendable {
        var researchTools = false
        var toolFilter: MCPToolFilter = .standard
        var imageDirectories: [URL] = []
        var decisions: (any DecisionEngine)?
        var pipeline = PipelineStages()
        var pipelines: PipelineRegistry?
        var imageSynthesizer: (any ImageSynthesizer)?
        var imageStoreDirectory: URL?
        var videoSynthesizer: (any VideoSynthesizer)?
        var videoStoreDirectory: URL?
        var hitlApprovals: HITLApprovalActor?
        var selfEditSessions = SelfEditSessionStore()
        var selfEditDirectory: URL = SelfEditSessionRecord.defaultDirectory()
    }

    let orchestrator: any ModelOrchestrating
    let modelId: String?
    let library: HarnessExperienceLibrary?
    let harnessRoots: [URL]
    let harnessLimits: HarnessLoadLimits
    let options: RuntimeOptions
    private let makeInput: @Sendable () -> AsyncStream<String>
    let writer: @Sendable (String) -> Void

    var researchTools: Bool { options.researchTools }
    var toolFilter: MCPToolFilter { options.toolFilter }
    var imageDirectories: [URL] { options.imageDirectories }
    var decisions: (any DecisionEngine)? { options.decisions }
    var pipeline: PipelineStages { options.pipeline }
    var pipelines: PipelineRegistry? { options.pipelines }
    var imageSynthesizer: (any ImageSynthesizer)? { options.imageSynthesizer }
    var imageStoreDirectory: URL? { options.imageStoreDirectory }
    var videoSynthesizer: (any VideoSynthesizer)? { options.videoSynthesizer }
    var videoStoreDirectory: URL? { options.videoStoreDirectory }
    var hitlApprovals: HITLApprovalActor? { options.hitlApprovals }
    var selfEditSessions: SelfEditSessionStore { options.selfEditSessions }
    var selfEditDirectory: URL { options.selfEditDirectory }

    public init(
        orchestrator: any ModelOrchestrating,
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
        self.options = RuntimeOptions()
        self.makeInput = Self.stdinStream
        self.writer = writer
    }

    package init(
        orchestrator: any ModelOrchestrating,
        modelId: String? = nil,
        input: AsyncStream<String>,
        writer: @Sendable @escaping (String) -> Void
    ) {
        self.orchestrator = orchestrator
        self.modelId = modelId
        self.library = nil
        self.harnessRoots = HarnessLoader.allowedRoots()
        self.harnessLimits = .standard
        self.options = RuntimeOptions()
        self.makeInput = { input }
        self.writer = writer
    }

    init(_ server: MCPServer, options: RuntimeOptions) {
        orchestrator = server.orchestrator
        modelId = server.modelId
        library = server.library
        harnessRoots = server.harnessRoots
        harnessLimits = server.harnessLimits
        self.options = options
        makeInput = server.makeInput
        writer = server.writer
    }

    public func withResearchTools(_ enabled: Bool) -> MCPServer {
        var next = options
        next.researchTools = enabled
        return MCPServer(self, options: next)
    }

    public func withToolFilter(_ filter: MCPToolFilter) -> MCPServer {
        var next = options
        next.toolFilter = filter
        return MCPServer(self, options: next)
    }

    public func withImageDirectories(_ directories: [URL]) -> MCPServer {
        var next = options
        next.imageDirectories = directories.map(MCPImages.resolved)
        return MCPServer(self, options: next)
    }

    package func withPipeline(_ stages: PipelineStages) -> MCPServer {
        var next = options
        next.pipeline = stages
        return MCPServer(self, options: next)
    }

    package func withSelfEditDirectory(_ directory: URL) -> MCPServer {
        var next = options
        next.selfEditDirectory = directory
        return MCPServer(self, options: next)
    }

    package func withPipelines(_ registry: PipelineRegistry?) -> MCPServer {
        var next = options
        next.pipelines = registry
        return MCPServer(self, options: next)
    }

    public func withDecisions(_ engine: (any DecisionEngine)?) -> MCPServer {
        var next = options
        next.decisions = GatedDecisionEngine.serving(engine, gate: orchestrator.gate)
        return MCPServer(self, options: next)
    }

    public func withHITLApprovals(_ approvals: HITLApprovalActor?) -> MCPServer {
        var next = options
        next.hitlApprovals = approvals
        return MCPServer(self, options: next)
    }

    package func withImageSynthesizer(_ synthesizer: (any ImageSynthesizer)?, storeDirectory: URL? = nil) -> MCPServer {
        var next = options
        next.imageSynthesizer = synthesizer
        if let storeDirectory { next.imageStoreDirectory = storeDirectory }
        return MCPServer(self, options: next)
    }

    package func withVideoSynthesizer(_ synthesizer: (any VideoSynthesizer)?, storeDirectory: URL? = nil) -> MCPServer {
        var next = options
        next.videoSynthesizer = synthesizer
        if let storeDirectory { next.videoStoreDirectory = storeDirectory }
        return MCPServer(self, options: next)
    }
}

extension MCPServer {
    static func lineStream(from handle: FileHandle) -> AsyncStream<String> {
        AsyncStream { continuation in
            let reader = Task {
                do {
                    for try await line in handle.bytes.lines {
                        guard !Task.isCancelled else { break }
                        if case .terminated = continuation.yield(line) { break }
                    }
                } catch {}
                continuation.finish()
            }
            continuation.onTermination = { _ in
                reader.cancel()
            }
        }
    }

    private static func stdinStream() -> AsyncStream<String> {
        lineStream(from: .standardInput)
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
        case .ping(let id): writer(MCPWire.result(id: id, [String: String]()))
        case .initialize(let id): writer(MCPWire.result(id: id, MCPInitializeResult.current))
        case .toolsList(let id):
            let catalog = MCPToolCatalog.catalog(
                researchTools: researchTools,
                filter: toolFilter,
                includeImageTools: imageSynthesizer != nil,
                includeVideoTools: videoSynthesizer != nil
            )
            writer(MCPWire.result(id: id, catalog))
        case .toolsCall(let id, let params): await handleToolsCall(id: id, params: params)
        }
    }
}
