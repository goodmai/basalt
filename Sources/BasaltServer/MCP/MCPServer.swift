import Foundation
import Logging
import ServiceLifecycle
import Synchronization
import BasaltCore
import BasaltHarness

public final class MCPServer: Service, Sendable {
    private let orchestrator: ModelOrchestratorActor
    private let modelId: String?
    private let library: HarnessExperienceLibrary?
    private let harnessRoots: [URL]
    private let harnessLimits: HarnessLoadLimits
    private let researchTools: Bool
    private let makeInput: @Sendable () -> AsyncStream<String>
    private let writer: @Sendable (String) -> Void

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
        self.makeInput = { input }
        self.writer = writer
    }

    private init(_ server: MCPServer, researchTools: Bool) {
        orchestrator = server.orchestrator
        modelId = server.modelId
        library = server.library
        harnessRoots = server.harnessRoots
        harnessLimits = server.harnessLimits
        self.researchTools = researchTools
        makeInput = server.makeInput
        writer = server.writer
    }

    public func withResearchTools(_ enabled: Bool) -> MCPServer {
        MCPServer(self, researchTools: enabled)
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

    private func handleToolsCall(id: JSONRPCId?, params: AnyCodable?) async {
        do throws(MCPFailure) {
            let call = try MCPArguments.toolCall(params)
            guard let tool = MCPTool(name: call.name) else { throw MCPFailure.methodNotFound("Unknown tool: \(call.name)") }
            let output = try await result(of: tool, args: call.arguments)
            writer(MCPWire.result(id: id, output))
        } catch {
            writer(MCPWire.error(id: id, error))
        }
    }

    private func result(of tool: MCPTool, args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        switch tool {
        case .status: await status(args)
        case .checkHardware: .json(await SystemProfiler().generateReport())
        case .inspectModel: try await inspectModel(args)
        case .switchModel: try await switchModel(args)
        case .generate: try await generate(args)
        case .health: await health()
        case .addKnowledge: try await addKnowledge(args)
        case .harnessValidate: try validateHarness(args)
        case .harnessLibraryQuery: try await queryLibrary(args)
        case .harnessOptimize: try await optimizeHarness(args)
        case .research(let tool): try await research(tool, args)
        }
    }

    private func research(_ tool: ResearchTool, _ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        guard researchTools || !tool.isExperiment else {
            throw .methodNotFound("\(tool.rawValue) is an experiment tool: start the server with serve --research-tools to use it")
        }
        let call = try ResearchCall(tool, args)
        do throws(BasaltError) {
            return try await orchestrator.research { model throws(BasaltError) in try await call.run(on: model) }
        } catch {
            throw MCPFailure(error)
        }
    }
}

extension ResearchCall {
    func run(on model: any ModelResearch) async throws(BasaltError) -> ToolResult {
        switch self {
        case .describe: return .json(try await model.anatomy())
        case .tensors(let filter): return .json(try await model.tensors(matching: filter))
        case .tensor(let request): return .json(try await model.tensor(request))
        case .promptCache(let namespace): return .json(await model.promptCacheReport(namespace: namespace))
        case .trace(let request): return .json(try await model.trace(request))
        case .intervene(let intervention): return .json(try await model.intervene(intervention))
        case .delta(let delta): return .json(try await model.applyDelta(delta))
        case .overlays: return .json(await model.overlays())
        case .undo(let overlay):
            try await model.undo(overlay)
            return .text("Overlay \(overlay) undone")
        case .compare(let request): return .json(try await model.compare(request))
        case .train(let request):
            guard let lab = model as? any ModelLab else { throw .invalidRequestStructure(details: "this engine cannot train adapters") }
            return .json(try await lab.train(request))
        }
    }
}

extension MCPServer {
    private func status(_ args: [String: Any]?) async -> ToolResult {
        let loaded = await orchestrator.currentModelId
        let health = await orchestrator.healthSnapshot(modelId: loaded.isEmpty ? modelId : loaded)
        let telemetry = await orchestrator.telemetrySnapshot()
        return .json(StatusOutput(health: health, loaded: loaded, requested: args?["modelId"] as? String, telemetry: telemetry))
    }

    private func health() async -> ToolResult {
        let health = await orchestrator.healthSnapshot(modelId: modelId)
        return .text("status=\(health.status) ready=\(health.isReady) version=\(health.version)")
    }

    private func inspectModel(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        let target = try MCPArguments.modelId(args)
        let report = await ModelInspector().inspect(modelIdOrPath: target)
        return .json(report, isError: !report.intact)
    }

    private func switchModel(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        let target = try MCPArguments.modelId(args)
        do throws(BasaltError) {
            try await orchestrator.switchModel(to: target)
        } catch {
            throw MCPFailure(error)
        }
        return .text("Successfully switched model to '\(target)'")
    }

    private func generate(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        let request = try MCPArguments.generation(args)
        do throws(BasaltError) {
            return .generated(try await orchestrator.generate(request: request))
        } catch {
            throw MCPFailure(error)
        }
    }

    private func addKnowledge(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        let knowledge = try KnowledgeArguments(args, libraryConfigured: library != nil)
        guard let library else { throw .libraryMissing }
        do {
            let entry = try await library.addKnowledge(
                topic: knowledge.topic, content: knowledge.content, domain: knowledge.domain,
                cue: knowledge.cue, confidence: knowledge.confidence
            )
            return .text("Stored '\(entry.topic)' in the Experience Library as \(entry.domain)/harnesses/\(entry.slug).")
        } catch {
            throw .knowledge(error)
        }
    }

    private func queryLibrary(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        guard let library else { throw .libraryMissing }
        do {
            return .json(try await library.query(args?["query"] as? String, domain: args?["domain"] as? String))
        } catch {
            throw .internalError("Cannot read the Experience Library: \(error)")
        }
    }

    private func loadHarness(_ args: [String: Any]?) throws(MCPFailure) -> (spec: HarnessSurfaceSpec, harness: LoadedHarness) {
        let input = try HarnessInputArguments(args)
        do {
            switch input.source {
            case .directory(let root):
                return (input.spec, try HarnessLoader.load(root: root, spec: input.spec, allowedRoots: harnessRoots, limits: harnessLimits))
            case .files(let files):
                return (input.spec, try HarnessLoader.inline(files, limits: harnessLimits))
            }
        } catch {
            throw .invalidParams("Cannot load harness: \(error)")
        }
    }

    private func validateHarness(_ args: [String: Any]?) throws(MCPFailure) -> ToolResult {
        let input = try loadHarness(args)
        var report = HarnessValidator.validate(input.harness.tree, spec: input.spec)
        report.warnings += input.harness.skipped.map { "skipped \($0)" }
        return .json(report, isError: !report.isValid)
    }

    private func optimizeHarness(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        let arguments = try HarnessOptimizeArguments(args)
        let input = try loadHarness(args)
        var tokens = HarnessOptimizer.TokenBudgets()
        tokens.proposer = arguments.proposerMaxTokens
        do {
            let optimizer = HarnessOptimizer(orchestrator: orchestrator, spec: input.spec, library: library, tokens: tokens)
            return .json(try await optimizer.optimize(
                task: arguments.task, base: input.harness.tree, k: arguments.k, skipped: input.harness.skipped
            ))
        } catch {
            throw .optimization(error)
        }
    }
}
