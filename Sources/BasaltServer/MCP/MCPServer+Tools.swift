import Foundation
import BasaltCore
import BasaltHarness

extension MCPServer {
    func handleToolsCall(id: JSONRPCId?, params: AnyCodable?) async {
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
        case .introspect: try await introspect(args)
        case .media(let tool): try await media(tool, args)
        }
    }

    private func research(_ tool: ResearchTool, _ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        guard researchTools || !tool.isExperiment else {
            throw .methodNotFound("\(tool.rawValue) is an experiment tool: start the server with serve --research-tools to use it")
        }
        let call = try ResearchCall(tool, args)
        if case .tensor(let request) = call, request.spectrum, !researchTools {
            throw .methodNotFound("model_tensor spectrum holds the model for seconds: start the server with serve --research-tools to use it")
        }
        do throws(BasaltError) {
            return try await orchestrator.research { model throws(BasaltError) in try await call.run(on: model) }
        } catch {
            throw MCPFailure(error)
        }
    }
}

extension ResearchCall {
    static func lab(_ model: any ModelResearch) throws(BasaltError) -> any ModelLab {
        guard let lab = model as? any ModelLab else {
            throw .invalidRequestStructure(details: "the inference engine of this server has no lab tools")
        }
        return lab
    }

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
        case .adapter, .gate, .pack, .train, .trainJepa, .distill, .contrast, .commitSteering: return try await run(on: Self.lab(model))
        }
    }

    func run(on lab: any ModelLab) async throws(BasaltError) -> ToolResult {
        switch self {
        case .adapter(let source, let gate): .json(try await lab.applyAdapter(source, gate: gate))
        case .gate(let request): .json(try await lab.gate(request))
        case .pack(let request): .json(try await lab.pack(request))
        case .train(let request): .json(try await lab.train(request))
        case .trainJepa(let request): .json(try await lab.trainJEPA(request))
        case .distill(let request): .json(try await lab.distill(request))
        case .contrast(let request): .json(try await lab.contrast(request))
        case .commitSteering(let commit): .json(try await lab.commitSteering(commit))
        case .describe, .tensors, .tensor, .promptCache, .trace, .intervene, .delta, .overlays, .undo, .compare: throw .invalidRequestStructure(details: "not a lab call")
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
        let verify = try MCPArguments.optionalBoolean("verify", in: args) ?? false
        let report = await ModelInspector().inspect(modelIdOrPath: target, verifyingWith: verify ? HuggingFaceHub() : nil)
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
        let request = try MCPArguments.generation(args, images: try MCPImages.parse(args?["images"], allowedIn: imageDirectories))
        do throws(BasaltError) {
            return .generated(try await orchestrator.generate(request: try await orchestrator.transcribingImages(in: request).request))
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

extension MCPServer {
    private func introspect(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        guard researchTools else {
            throw .methodNotFound("introspect is an experiment tool: start the server with serve --research-tools to use it")
        }
        let request = try IntrospectRequest(mcp: args)
        do throws(BasaltError) {
            return .json(try await Introspector(orchestrator: orchestrator, interventions: request.allowInterventions).introspect(request))
        } catch {
            throw MCPFailure(error)
        }
    }
}

extension MCPServer {
    private func media(_ tool: MediaTool, _ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        switch tool {
        case .transcribe: try await transcribe(args)
        case .ocr: try await ocr(args)
        case .decide: try await decide(args)
        }
    }

    private func transcribe(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        guard let audio = try MCPArguments.string("audio", in: args), let format = try MCPArguments.string("format", in: args) else {
            throw .invalidParams("Missing required arguments: audio and format")
        }
        let language = try MCPArguments.string("language", in: args)
        let temperature = try MCPArguments.number("temperature", in: args).map(Double.init)
        do throws(BasaltError) {
            guard audio.utf8.count <= (AudioClip.maxBytes + 2) / 3 * 4 else {
                throw .invalidRequestStructure(details: "audio is over the limit of \(AudioClip.maxBytes >> 20) MB")
            }
            guard let data = Data(base64Encoded: audio) else { throw .invalidRequestStructure(details: "audio is not valid base64") }
            let request = try TranscriptionRequest(audio: AudioClip(encoded: data, format: format), language: language, temperature: temperature)
            return .text(try await orchestrator.transcribe(request).text)
        } catch {
            throw MCPFailure(error)
        }
    }

    private func decide(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        guard let decisions else { throw .methodNotFound("basalt_systemone_decide needs a decision model: start serve with --decision-model <OpenJev repo|dir>") }
        guard let args, JSONSerialization.isValidJSONObject(args), let data = try? JSONSerialization.data(withJSONObject: args),
              let request = SystemOneController.request(from: data)
        else { throw .invalidParams("basalt_systemone_decide needs state (text, object or array) and questions {id: {type, instructions, criteria?}}") }
        do throws(BasaltError) {
            return .json(try await SystemOneController.decide(
                decisions, request, log: orchestrator.sessionLog, origin: RequestOrigin(source: "mcp", endpoint: "basalt_systemone_decide", stream: false), sessionId: nil))
        } catch {
            throw MCPFailure(error)
        }
    }
}
