import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltServer

@Suite("basalt compose phase 3: named pipelines, compose up as a diff on a running system, compose ps and down (#412)", .timeLimit(.minutes(1)))
struct ComposeOperationsTests {
    static let base = """
        version: 1
        name: ops
        models:
          chat:
            source: mlx-community/gemma-4-12B-it-qat-4bit
          jev:
            source: cloud:openrouter/~typesafe/jev-latest
            api_key_env: HOME
        pipeline:
          decide:
            cascade: [jev, chat]
          answer: chat
        """

    static func with(_ extra: String, pipelines: String = "") -> String { base.replacingOccurrences(of: "pipeline:\n", with: extra + "pipeline:\n") + "\n" + pipelines }

    static let haiku = "  haiku:\n    source: cloud:openrouter/anthropic/claude-haiku-5.5\n    api_key_env: HOME\n"
    static let support = "pipelines:\n  support:\n    decide:\n      cascade: [chat]\n    answer:\n      routes:\n        - model: haiku\n          fallback: chat\n        - model: chat\n"

    @Test("The diff loads added models, reloads changed ones, keeps the rest; a cloud budget change needs no reload")
    func diff() throws {
        let old = try ComposeFile.parse(Self.base, file: "a.yaml")
        let new = try ComposeFile.parse(Self.with(Self.haiku, pipelines: Self.support), file: "b.yaml")
        let diff = ComposeDiff.between(old, new)
        #expect(diff.added == ["haiku"] && diff.kept == ["chat", "jev"] && diff.removed.isEmpty && diff.restart == nil)
        #expect(diff.pipelines == ["default", "support"] && diff.lines.first == "  load: haiku")
        var budgeted = old
        budgeted.models["jev"]?.budget = 1
        #expect(ComposeDiff.between(old, budgeted).kept == ["chat", "jev"])
        #expect(ComposeDiff.between(new, old).removed == ["haiku"])
    }

    @Test("A restart is needed for another system, another chat model, another memory mode, a sequential perceiver or another local decider")
    func restarts() throws {
        let old = try ComposeFile.parse(Self.base, file: "a.yaml")
        var renamed = old
        renamed.name = "other"
        #expect(ComposeDiff.between(old, renamed).restart?.contains("another system") == true)
        var chat = old
        chat.models["chat"] = .init(source: .local("mlx-community/other"))
        #expect(ComposeDiff.between(old, chat).restart?.contains("chat model") == true)
        var memory = old
        memory.memory = .sequential
        #expect(ComposeDiff.between(old, memory).restart?.contains("memory mode") == true)
        var swapped = old
        swapped.models["asr"] = .init(source: .local("divergentlabs/asr"), keep: .sequential)
        swapped.audio = ["asr"]
        #expect(ComposeDiff.between(old, swapped).restart != nil)
        var decider = old
        decider.models["laya"] = .init(source: .local("./laya"))
        decider.cascade = ["jev", "laya"]
        #expect(ComposeDiff.between(old, decider).restart?.contains("local decision model") == true)
    }

    @Test("Named pipelines parse; each is checked like the default one, shares its local decider, and keeps sequential perceivers to the default pipeline")
    func namedPipelines() throws {
        let spec = try ComposeFile.parse(Self.with(Self.haiku, pipelines: Self.support), file: "c.yaml")
        #expect(spec.pipelineNames == ["default", "support"] && spec.view("support")?.answerModels == ["chat", "haiku"])
        #expect(spec.view("missing") == nil)
        for (yaml, reason) in [
            (Self.with(Self.haiku, pipelines: Self.support.replacingOccurrences(of: "  support:", with: "  Support:")), "lowercase"),
            (Self.with(Self.haiku + "  laya:\n    source: ./laya\n", pipelines: Self.support.replacingOccurrences(of: "cascade: [chat]", with: "cascade: [laya]")), "shares the default pipeline's local decision model"),
            (Self.with(Self.haiku + "  asr:\n    source: divergentlabs/asr\n    keep: sequential\n", pipelines: Self.support.replacingOccurrences(of: "    decide:", with: "    perceive:\n      audio: [asr]\n    decide:")), "sequential perceivers"),
            (Self.with(Self.haiku), "declared but not used by any pipeline"),
        ] {
            do {
                _ = try ComposeFile.parse(yaml, file: "c.yaml")
                Issue.record("parsed a file that should be refused for \(reason)")
            } catch {
                #expect(error.description.contains(reason), "\(error.description)")
            }
        }
    }

    @Test("/v1/pipelines/<name> serves the named system, an unknown name is 404, and a replaced registry is used by the next request")
    func registryRoutes() async throws {
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        let registry = PipelineRegistry([ComposeSpec.defaultPipeline: PipelineSystem(decisions: nil, stages: PipelineStages())])
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock-model")).withPipelines(registry, compose: nil)
        let body = ByteBuffer(bytes: try JSONSerialization.data(withJSONObject: ["text": "Hi", "task": "Say hi."]))
        func post(_ path: String) async throws -> HTTPResponse.Status {
            try await Application(router: server.makeRouter()).test(.router) { client in
                try await client.execute(uri: path, method: .post, headers: [.contentType: "application/json"], body: body) { $0.status }
            }
        }
        #expect(try await post("/v1/pipeline") == .ok)
        #expect(try await post("/v1/pipelines/support") == .notFound)
        await registry.replace([ComposeSpec.defaultPipeline: PipelineSystem(decisions: nil, stages: PipelineStages()), "support": PipelineSystem(decisions: nil, stages: PipelineStages())])
        #expect(try await post("/v1/pipelines/support") == .ok)
    }

    @Test("A running host applies a diff that adds a cloud answer model and a named pipeline, refuses one that changes the chat model, and reports status")
    func hostApply() async throws {
        let spec = try ComposeFile.parse(Self.base, file: "a.yaml")
        let serve = try ServeCommand.parse(spec.serveArguments)
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        let gate = serve.cloudGate(spec)
        let start = ComposeStart(chat: chat, decider: StubDecisionEngine(), decisions: nil, cloudGate: gate, startup: PipelineStages(), besideDecode: true)
        let host = ComposeHost(serve: serve, spec: spec, start: start)
        try await host.start(defaultDecisions: nil)
        #expect(await host.registry.names == ["default"])
        let diff = try await host.apply(ComposeApply(yaml: Self.with(Self.haiku, pipelines: Self.support)))
        #expect(diff.added == ["haiku"] && diff.restart == nil)
        #expect(await host.registry.names == ["default", "support"])
        let support = try #require(await host.registry.system("support"))
        #expect(support.stages.answers?.targets["haiku"]?.isCloud == true)
        await #expect(throws: BasaltError.self) { _ = try await host.apply(ComposeApply(yaml: Self.base.replacingOccurrences(of: "gemma-4-12B-it-qat-4bit", with: "other"))) }
        let status = await host.status()
        #expect(status.pipelines == ["default", "support"] && status.models.map(\.name) == ["chat", "haiku", "jev"])
        #expect(status.models.first { $0.name == "haiku" }?.state == "cloud" && status.models.first { $0.name == "chat" }?.state == "loaded")
    }

    @Test("GET /v1/compose returns the status and POST /v1/compose/apply the diff, through the server's auth")
    func endpoints() async throws {
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        let control = ComposeControl(
            status: { ComposeStatus(name: "ops", pipelines: ["default"], models: [ComposeModelStatus(name: "chat", source: "org/chat", keep: "resident", state: "loaded")], spentToday: 0.01) },
            apply: { request throws(BasaltError) in
                guard request.yaml.contains("version"), request.base == "/srv/compose" else { throw .invalidRequestStructure(details: "compose apply: not a compose file") }
                var diff = ComposeDiff()
                diff.added = ["haiku"]
                return diff
            })
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock-model")).withPipelines(nil, compose: control)
        try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: "/v1/compose", method: .get) { response in
                let json = try JSONSerialization.jsonObject(with: Data(buffer: response.body)) as? [String: Any]
                #expect(response.status == .ok && json?["name"] as? String == "ops" && json?["spent_today"] as? Double == 0.01)
            }
            try await client.execute(uri: "/v1/compose/apply", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: #"{"yaml": "version: 1", "base": "/srv/compose"}"#)) { response in
                #expect(response.status == .ok && (try? JSONDecoder().decode(ComposeDiff.self, from: Data(buffer: response.body)))?.added == ["haiku"])
            }
            try await client.execute(uri: "/v1/compose/apply", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: #"{"yaml": "nope"}"#)) { #expect($0.status == .badRequest) }
            try await client.execute(uri: "/v1/compose/apply", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: "[]")) { #expect($0.status == .badRequest) }
        }
    }

    @Test("The record names the running system's pid and port, and is dropped once its process is gone")
    func record() throws {
        let name = "test-\(UUID().uuidString.prefix(8).lowercased())"
        defer { try? FileManager.default.removeItem(at: ComposeRecord.url(name)) }
        _ = try ComposeRecord.write(name: name, port: 18_123, file: "x.yaml")
        #expect(ComposeRecord.running(name)?.port == 18_123 && ComposeRecord.all().contains { $0.name == name })
        _ = try ComposeRecord.write(name: name, port: 18_123, file: "x.yaml", pid: 999_999)
        #expect(ComposeRecord.running(name) == nil && ComposeRecord.read(name) == nil)
    }
}
