import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltServer

@Suite("compose up changes the chat model and the local decider on a running system, without a restart (#423 A2)", .timeLimit(.minutes(1)))
struct ComposeRoleSwapTests {
    static let base = ComposeOperationsTests.base
    static let laya = base.replacingOccurrences(of: "pipeline:\n", with: "  laya:\n    source: ./laya\npipeline:\n").replacingOccurrences(of: "cascade: [jev, chat]", with: "cascade: [jev, laya]")
    static let chatB = base.replacingOccurrences(of: "mlx-community/gemma-4-12B-it-qat-4bit", with: "org/chat-b")

    @Test("The chat slot and the decider leave the load and unload lists; the old chat model counts as replaced in admission")
    func diff() throws {
        let gb = ComposeFileTests.gb
        let old = try ComposeFile.parse(Self.base, file: "a.yaml")
        let switched = ComposeDiff.between(old, try ComposeFile.parse(Self.chatB, file: "b.yaml"))
        #expect(switched.chat == "chat" && switched.reloaded.isEmpty && switched.kept == ["jev"] && switched.restart == nil)
        #expect(switched.lines.contains("  chat: chat") && switched.decider == nil)
        var renamed = old
        (renamed.models["big"], renamed.answer, renamed.cascade) = (.init(source: .local("org/big")), "big", ["jev", "big"])
        let toBig = ComposeDiff.between(old, renamed)
        #expect(toBig.chat == "big" && toBig.added == ["chat"] && toBig.removed.isEmpty && toBig.decider == nil)
        #expect(ComposeDiff.between(renamed, old).removed == ["chat"] && ComposeDiff.between(renamed, old).chat == "chat")
        let decider = ComposeDiff.between(old, try ComposeFile.parse(Self.laya, file: "c.yaml"))
        #expect(decider.decider == "laya" && decider.added == ["laya"] && decider.kept == ["chat", "jev"] && decider.restart == nil)
        #expect(decider.lines.contains("  decide: laya"))
        #expect(ComposeDiff.between(try ComposeFile.parse(Self.laya, file: "c.yaml"), old).decider == "self")
        let next = try ComposeFile.parse(Self.chatB, file: "b.yaml")
        let sizes = ComposeFileTests.sizes(["mlx-community/gemma-4-12B-it-qat-4bit": 6 * gb, "org/chat-b": 9 * gb])
        #expect(switched.admission(previous: old, next: next, sizes: sizes, budgetBytes: 20 * gb) == .success(.loadFirst))
        #expect(switched.admission(previous: old, next: next, sizes: sizes, budgetBytes: 12 * gb) == .success(.unloadFirst))
    }

    @Test("apply loads the new chat model into the serving orchestrator and keeps the cloud model")
    func switchesChat() async throws {
        let (host, chat, _) = try await Self.host()
        let diff = try await host.apply(ComposeApply(yaml: Self.chatB))
        #expect(diff.chat == "chat" && diff.restart == nil)
        #expect(await chat.loaded.path == "/models/chat-b")
        #expect(await chat.isLoaded)
        #expect(await host.spec.models["chat"]?.source == .local("org/chat-b"))
        #expect(await host.status().models.first { $0.name == "chat" }?.state == "loaded")
    }

    @Test("apply builds the new decider, the pipelines and /v1/systemone use it, and the old one is let go")
    func switchesDecider() async throws {
        let (host, _, asked) = try await Self.host()
        let diff = try await host.apply(ComposeApply(yaml: Self.laya))
        #expect(diff.decider == "laya" && asked.last?.hasSuffix("laya") == true)
        #expect(await host.registry.system(ComposeSpec.defaultPipeline)?.decisions?.modelID.hasSuffix("laya") == true)
        #expect(await host.status().models.first { $0.name == "laya" }?.state == "decision")
        _ = try await host.apply(ComposeApply(yaml: Self.base))
        #expect(await host.registry.system(ComposeSpec.defaultPipeline)?.decisions?.modelID == "self")
        #expect(await host.status().models.allSatisfy { $0.name != "laya" })
    }

    @Test("A failure after the chat switch rolls the chat model back and leaves the running system as it was")
    func rollsBack() async throws {
        let (host, chat, _) = try await Self.host()
        let broken = Self.chatB.replacingOccurrences(of: "pipeline:\n", with: "  laya:\n    source: ./broken\npipeline:\n").replacingOccurrences(of: "cascade: [jev, chat]", with: "cascade: [jev, laya]")
        await #expect(throws: BasaltError.self) { _ = try await host.apply(ComposeApply(yaml: broken)) }
        #expect(await chat.loaded.path == "/models/chat")
        #expect(await chat.isLoaded)
        let spec = await host.spec
        #expect(spec.models["chat"]?.source == .local("mlx-community/gemma-4-12B-it-qat-4bit") && spec.models["laya"] == nil)
        #expect(await host.registry.system(ComposeSpec.defaultPipeline)?.decisions?.modelID == "self")
    }

    @Test("/v1/systemone asks the default pipeline's current decisions, so a replaced registry is used by the next request")
    func systemOneFollowsRegistry() async throws {
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        let registry = PipelineRegistry([ComposeSpec.defaultPipeline: PipelineSystem(decisions: StubDecisionEngine(), stages: PipelineStages())])
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock-model")).withPipelines(registry, compose: nil)
        func post() async throws -> HTTPResponse.Status {
            try await Application(router: server.makeRouter()).test(.router) { client in
                try await client.execute(uri: "/v1/systemone", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: SystemOneAPITests.body)) { $0.status }
            }
        }
        #expect(try await post() == .ok)
        await registry.replace([ComposeSpec.defaultPipeline: PipelineSystem(decisions: nil, stages: PipelineStages())])
        #expect(try await post() == .notFound)
    }

    static func host(_ yaml: String = base) async throws -> (ComposeHost, ModelOrchestratorActor, KeyLog) {
        let asked = KeyLog()
        let services = ComposeServices(
            resolve: { "/models/" + ($0.split(separator: "/").last.map(String.init) ?? $0) },
            load: { _, _, _, _ in nil },
            decider: { model in
                asked.record(model)
                if model.hasSuffix("missing") { throw BasaltError.invalidRequestStructure(details: "not a decision bundle") }
                return NamedDecider(modelID: model)
            },
            decisions: { _, local, _, _, _ in
                if local?.modelID.hasSuffix("broken") == true { throw BasaltError.invalidRequestStructure(details: "no cascade for a broken decider") }
                return local
            },
            key: { _ in "test-key" },
            cloudLimits: { _ in [:] })
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        let spec = try ComposeFile.parse(yaml, file: "a.yaml")
        var start = ComposeStart(chat: chat, decider: NamedDecider(modelID: spec.deciderModel.flatMap(spec.local) ?? "self"), decisions: nil, cloudGate: CloudGate(limits: [:], storage: nil), startup: PipelineStages(), besideDecode: true)
        start.admission = ComposeAdmission(sizes: ComposeFileTests.sizes([:])) { (64 * ComposeFileTests.gb, .desktop) }
        let host = ComposeHost(services: services, spec: spec, start: start)
        try await host.start(defaultDecisions: nil)
        return (host, chat, asked)
    }
}

struct NamedDecider: DecisionEngine {
    let modelID: String

    func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try await StubDecisionEngine().decide(request)
    }
}
