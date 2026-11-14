import ArgumentParser
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import Testing
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltCore
import BasaltInference
@testable import BasaltServer

@Suite("compose experimental and mcp sections: YAML parsing, relative paths, line-numbered refusals, and ServeCommand lowering (#485)")
struct ComposeExperimentalParsingTests {
    static let base = """
        version: 1
        name: exp-desk
        models:
          chat:
            source: mlx-community/Qwen3-4B-4bit
        pipeline:
          answer: chat
        """

    static func makeTowerBundle(in folder: URL, name: String = "tower") throws -> URL {
        let tower = folder.appending(path: name)
        try FileManager.default.createDirectory(at: tower, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: tower.appending(path: "metadata.json"))
        try Data("aimodel".utf8).write(to: tower.appending(path: "tower.aimodel"))
        return tower
    }

    @Test("All experimental and mcp fields parse, resolve relative paths against the compose folder, and round-trip through ServeCommand")
    func fullSectionRoundTrip() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "compose-exp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let tower = try Self.makeTowerBundle(in: folder)
        let images = folder.appending(path: "imgs")
        try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
        let file = folder.appending(path: "desk.yaml")
        let yaml = Self.base + """

            experimental:
              zero_copy_body: true
              smallm_kernel: true
              speech_analyzer: true
              image_fallback: ocr
              ane_vision_tower: ./tower
              telemetry_confidence_every: 4
              adaptive_thinking_budget: true
              samplers: true
              mtp_draft_head: true
              expert_streaming: true
              expert_cache_gb: 2.5
              kv_head_split: 0.5
            mcp:
              enabled: true
              rest: true
              research_tools: false
              enable: [surgery, intervene]
              disable: [switch_model]
              image_dirs: [./imgs, ~/pics]
            """
        try Data(yaml.utf8).write(to: file)

        let spec = try ComposeFile.load(file.path, context: ComposeContext())
        #expect(spec.experimental.aneVisionTower == tower.standardizedFileURL.path)
        #expect(spec.mcp.imageDirs == [images.standardizedFileURL.path, (NSHomeDirectory() as NSString).appendingPathComponent("pics")])

        let fromUp = try ServeCommand.parse(spec.serveArguments)
        var fromDirect = try ServeCommand.parse(["--compose", file.path])
        let prepared = try #require(try fromDirect.prepareCompose())
        #expect(prepared == spec)

        for cmd in [fromUp, fromDirect] {
            #expect(cmd.model == "mlx-community/Qwen3-4B-4bit")
            #expect(cmd.experimental.zeroCopyBody && cmd.experimental.smallMKernel && cmd.experimental.speechAnalyzer)
            #expect(cmd.experimental.imageFallback == .ocr)
            #expect(cmd.experimental.aneVisionTower == tower.standardizedFileURL.path)
            #expect(cmd.experimental.telemetryConfidenceEvery == 4)
            #expect(cmd.thinking.experimentalAdaptiveThinkingBudget && cmd.sampling.experimentalSamplers)
            #expect(cmd.speculative.mtpDraftHead && cmd.expertStreaming.enabled && cmd.expertStreaming.cacheGB == 2.5)
            #expect(cmd.kvCache.headSplit == 0.5)
            #expect(cmd.mcp && cmd.rest && !cmd.researchTools)
            #expect(try cmd.mcpOptions.toolFilter().allows("steer_search", researchTools: false))
            #expect(try cmd.mcpOptions.toolFilter().allows("intervene", researchTools: false))
            #expect(try !cmd.mcpOptions.toolFilter().allows("switch_model", researchTools: false))
            #expect(cmd.imageDirs == spec.mcp.imageDirs)
        }
    }

    @Test("quest_top_k parses and lowers cleanly when kv_head_split is unset")
    func questTopKRoundTrip() throws {
        let spec = try ComposeFile.parse(Self.base + "\nexperimental:\n  quest_top_k: 64\n", file: "q.yaml")
        let cmd = try ServeCommand.parse(spec.serveArguments)
        #expect(cmd.kvCache.questTopK == 64 && cmd.kvCache.headSplit == nil)
    }

    @Test("Invalid experimental and mcp fields fail with exact file, line number, path, and reason", arguments: [
        ("experimental:\n  unknown_flag: true\n", "experimental.unknown_flag", "unknown key", 9),
        ("experimental:\n  telemetry_confidence_every: -1\n", "experimental.telemetry_confidence_every", "0 or a positive integer", 9),
        ("experimental:\n  telemetry_confidence_every: fast\n", "experimental.telemetry_confidence_every", "must be an integer", 9),
        ("experimental:\n  image_fallback: caption\n", "experimental.image_fallback", "must be ocr", 9),
        ("experimental:\n  ane_vision_tower: /nonexistent/tower\n", "experimental.ane_vision_tower", "not a Core AI vision tower bundle", 9),
        ("experimental:\n  expert_cache_gb: 2.0\n", "experimental.expert_cache_gb", "needs experimental.expert_streaming: true", 9),
        ("experimental:\n  expert_streaming: true\n  expert_cache_gb: 0\n", "experimental.expert_cache_gb", "must be positive", 10),
        ("experimental:\n  kv_head_split: 0.01\n", "experimental.kv_head_split", "between 0.05 and 1", 9),
        ("experimental:\n  quest_top_k: 0\n", "experimental.quest_top_k", "between 1 and 4096", 9),
        ("experimental:\n  kv_head_split: 0.5\n  quest_top_k: 32\n", "experimental.quest_top_k", "does not combine", 10),
        ("mcp:\n  enable: [not_a_tool]\n", "mcp.enable", "unknown MCP tool or group: not_a_tool", 9),
        ("mcp:\n  disable: [bogus_group]\n", "mcp.disable", "unknown MCP tool or group: bogus_group", 9),
        ("mcp:\n  extra: true\n", "mcp.extra", "unknown key", 9),
    ])
    func validationRefusals(_ tail: String, _ path: String, _ reason: String, _ line: Int) {
        do {
            _ = try ComposeFile.parse(Self.base + "\n" + tail, file: "bad.yaml")
            Issue.record("expected refusal for \(path)")
        } catch {
            let text = error.description
            #expect(text.contains("bad.yaml:\(line):"), "\(text)")
            #expect(text.contains(path), "\(text)")
            #expect(text.contains(reason), "\(text)")
        }
    }
}

@Suite("compose live /v1/compose/apply hot-reloads runtime experimental settings and refuses startup-only changes (#485)", .timeLimit(.minutes(1)))
struct ComposeExperimentalLiveReloadTests {
    static func makeLiveHost() async throws -> (MockInferenceEngine, ModelOrchestratorActor, ComposeHost, RESTServer) {
        let spec = try ComposeFile.parse(ComposeExperimentalParsingTests.base, file: "live.yaml")
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "mlx-community/Qwen3-4B-4bit")
        let chat = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        await chat.setModelId("Qwen3-4B-4bit")
        let services = ComposeServices(
            resolve: { $0 }, load: { _, _, _, _ in nil }, decider: { _ in StubDecisionEngine() },
            decisions: { _, _, _, _, _ in nil }, key: { _ in "" }, cloudLimits: { _ in [:] }
        )
        var start = ComposeStart(chat: chat, decider: nil, decisions: nil, cloudGate: CloudGate(limits: [:], storage: nil), startup: PipelineStages(), besideDecode: true)
        start.admission = ComposeAdmission(sizes: ComposeFileTests.sizes(["mlx-community/Qwen3-4B-4bit": ComposeFileTests.gb])) { (16 * ComposeFileTests.gb, .desktop) }
        let host = ComposeHost(services: services, spec: spec, start: start)
        try await host.start(defaultDecisions: nil)
        let registry = await host.registry
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "mlx-community/Qwen3-4B-4bit"))
            .withPipelines(registry, compose: host.control)
        return (engine, chat, host, server)
    }

    @Test("POST /v1/compose/apply hot-reloads speech_analyzer, image_fallback, telemetry_confidence_every, and adaptive_thinking_budget on the live orchestrator")
    func hotReloadOverHTTP() async throws {
        let (engine, chat, host, server) = try await Self.makeLiveHost()
        _ = try await chat.generate(request: GenerationRequest(prompt: "hi", messages: [ChatMessageInput(role: .user, text: "What is 2 + 2?")]))
        let before = try #require(await engine.lastRequest)
        #expect(!before.recordConfidence && before.thinkingBudget == nil)
        let mediaBefore = await chat.media
        #expect(!mediaBefore.transcribesImagesForTextModels && mediaBefore.speech == nil)

        let updatedYAML = ComposeExperimentalParsingTests.base + """

            experimental:
              speech_analyzer: true
              image_fallback: ocr
              telemetry_confidence_every: 1
              adaptive_thinking_budget: true
            """
        let payload = ByteBuffer(bytes: try JSONEncoder().encode(ComposeApply(yaml: updatedYAML)))
        let diff: ComposeDiff = try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: "/v1/compose/apply", method: .post, headers: [.contentType: "application/json"], body: payload) { response in
                #expect(response.status == .ok)
                return try JSONDecoder().decode(ComposeDiff.self, from: Data(buffer: response.body))
            }
        }
        #expect(diff.restart == nil)
        #expect(diff.experimental == ["speech_analyzer", "image_fallback", "telemetry_confidence_every", "adaptive_thinking_budget"])
        #expect(diff.lines.contains("  experimental: speech_analyzer, image_fallback, telemetry_confidence_every, adaptive_thinking_budget"))
        let mediaAfter = await chat.media
        #expect(mediaAfter.transcribesImagesForTextModels && mediaAfter.speech != nil)
        #expect(await host.registry.system(ComposeSpec.defaultPipeline)?.stages.speech != nil)

        _ = try await chat.generate(request: GenerationRequest(prompt: "hi", messages: [ChatMessageInput(role: .user, text: "What is 2 + 2?")]))
        let after = try #require(await engine.lastRequest)
        #expect(after.recordConfidence)
        #expect(after.thinkingBudget == 512)
    }

    @Test("POST /v1/compose/diff reports restart reasons and POST /v1/compose/apply refuses startup-only experimental and mcp changes", arguments: [
        ("experimental:\n  zero_copy_body: true\n", "experimental.zero_copy_body"),
        ("experimental:\n  smallm_kernel: true\n", "experimental.smallm_kernel"),
        ("experimental:\n  samplers: true\n", "experimental.samplers"),
        ("experimental:\n  mtp_draft_head: true\n", "experimental.mtp_draft_head"),
        ("experimental:\n  expert_streaming: true\n", "experimental.expert_streaming"),
        ("experimental:\n  kv_head_split: 0.5\n", "experimental.kv_head_split"),
        ("experimental:\n  quest_top_k: 64\n", "experimental.quest_top_k"),
        ("mcp:\n  enabled: true\n", "mcp options"),
    ])
    func restartOnlyRefusedOverHTTP(_ tail: String, _ expectedFragment: String) async throws {
        let (_, _, host, server) = try await Self.makeLiveHost()
        let payload = ByteBuffer(bytes: try JSONEncoder().encode(ComposeApply(yaml: ComposeExperimentalParsingTests.base + "\n" + tail)))
        try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: "/v1/compose/diff", method: .post, headers: [.contentType: "application/json"], body: payload) { response in
                #expect(response.status == .ok)
                let diff = try JSONDecoder().decode(ComposeDiff.self, from: Data(buffer: response.body))
                #expect(diff.restart?.contains(expectedFragment) == true, "\(String(describing: diff.restart))")
            }
            try await client.execute(uri: "/v1/compose/apply", method: .post, headers: [.contentType: "application/json"], body: payload) { response in
                #expect(response.status == .badRequest)
                #expect(String(buffer: response.body).contains(expectedFragment))
            }
        }
        let currentSpec = await host.spec
        #expect(currentSpec.experimental == ComposeSpec.Experimental() && currentSpec.mcp == ComposeSpec.MCP())
    }
}
