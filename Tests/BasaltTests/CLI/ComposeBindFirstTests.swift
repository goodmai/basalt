import Testing
import Foundation
import ArgumentParser
import BasaltInference
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltServer

@Suite("compose binds first and loads after: ps says loading, ready is 503 and a second up waits until the system takes the change (#423 4c-2)", .timeLimit(.minutes(1)))
struct ComposeBindFirstTests {
    @Test("Before the host starts, local models are loading, cloud models show their key state, and apply is busy")
    func slotBeforeStart() async throws {
        let yaml = ComposeOperationsTests.base.replacingOccurrences(of: "api_key_env: HOME", with: "api_key_env: \(ComposeHealthTests.unsetKey)")
        let slot = ComposeSlot(spec: try ComposeFile.parse(yaml, file: "s.yaml"), cloudGate: CloudGate(limits: [:], storage: nil)) { _ in "" }
        let status = await slot.status()
        #expect(status.models.map(\.state) == [.loading, .noKey] && status.waiting == ["chat (loading)", "jev (no-key)"])
        let busy = await #expect(throws: BasaltError.self) { _ = try await slot.apply(ComposeApply(yaml: yaml)) }
        guard case .serverBusy? = busy else {
            Issue.record("apply before start was not busy: \(String(describing: busy))")
            return
        }
    }

    @Test("A live server answers during the load; once the host starts it is ready and takes a change from compose up")
    func liveLoad() async throws {
        let spec = try ComposeFile.parse(ComposeOperationsTests.base, file: "b.yaml")
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        let slot = ComposeSlot(spec: spec, cloudGate: CloudGate(limits: [:], storage: nil)) { _ in "test-key" }
        let (ports, bound) = AsyncStream.makeStream(of: Int.self)
        let (releases, release) = AsyncStream.makeStream(of: Void.self)
        let (starts, started) = AsyncStream.makeStream(of: Void.self)
        var launch = ServerLaunch(config: try ServerConfig(modelPath: "org/mock-model", restPort: 0, host: "127.0.0.1"), startREST: true, mcpServer: nil, pipelines: slot.registry, compose: slot.control)
        (launch.handleSignals, launch.onBind) = (false, { bound.yield($0) })
        launch.startup = {
            for await _ in releases { break }
            await slot.started(try await Self.host(spec, chat: chat, registry: slot.registry))
            started.yield()
        }
        let server = Task { try await ServeCommand.runServers(launch, orchestrator: chat) }
        defer { server.cancel() }
        let port = try #require(await ports.first { _ in true })
        let record = ComposeRecord(name: spec.name, pid: ProcessInfo.processInfo.processIdentifier, port: port, file: "b.yaml")
        let (loading, body) = try await ComposeCommand.call(record, "/v1/health/ready")
        #expect(loading == 503 && String(decoding: body, as: UTF8.self).contains("chat (loading)"))
        let change = ComposeApply(yaml: ComposeOperationsTests.with(ComposeOperationsTests.haiku, pipelines: ComposeOperationsTests.support))
        await #expect(throws: ExitCode.self) { _ = try await ComposeCommand.apply(change, to: record, until: .now) }
        await #expect(throws: ExitCode.self) { try await ComposeCommand.waitReady(record, until: .now + .milliseconds(100)) }
        release.yield()
        _ = await starts.first { _ in true }
        #expect(try await ComposeCommand.call(record, "/v1/health/ready").0 == 200)
        #expect(try await ComposeCommand.apply(change, to: record, until: .now + .seconds(30)).added == ["haiku"])
        try await ComposeCommand.waitReady(record, until: .now + .seconds(30))
    }

    @Test("serve's compose launch writes the record at bind, loads and probes in its startup task, then serves the host")
    func composedLaunch() async throws {
        let name = "test-\(UUID().uuidString.prefix(8).lowercased())"
        defer { try? FileManager.default.removeItem(at: ComposeRecord.url(name)) }
        let spec = try ComposeFile.parse(ComposeOperationsTests.base.replacingOccurrences(of: "name: ops", with: "name: \(name)"), file: "l.yaml")
        let serve = try ServeCommand.parse(spec.serveArguments + ["--compose", "l.yaml"])
        let (_, chat) = try await PipelineSwapEdgeTests.chat(loaded: false)
        let boot = ComposeBoot(config: try ServerConfig(modelPath: "/models/chat"), engine: MLXInferenceEngine(), orchestrator: chat, chatPath: "/models/chat")
        let launch = try await serve.composedLaunch(spec, boot)
        #expect(await launch.compose?.status().waiting == ["chat (loading)"])
        launch.onBind?(18_125)
        #expect(ComposeRecord.read(name)?.port == 18_125)
        let startup = try #require(launch.startup)
        try await startup()
        #expect(await chat.isLoaded)
        let status = try #require(await launch.compose?.status())
        #expect(status.waiting.isEmpty && status.models.first { $0.name == "chat" }?.state == .ready)
        #expect(await launch.pipelines?.names == ["default"])
    }

    @Test("compose up on a system that is still loading waits, applies the file once the system is up, and with --wait returns when it is ready")
    func upWaits() async throws {
        let name = "test-\(UUID().uuidString.prefix(8).lowercased())"
        let file = FileManager.default.temporaryDirectory.appending(path: "\(name).yaml")
        defer { try? FileManager.default.removeItem(at: ComposeRecord.url(name)); try? FileManager.default.removeItem(at: file) }
        let yaml = ComposeOperationsTests.with(ComposeOperationsTests.haiku, pipelines: ComposeOperationsTests.support).replacingOccurrences(of: "name: ops", with: "name: \(name)")
        try Data(yaml.utf8).write(to: file)
        let spec = try ComposeFile.parse(ComposeOperationsTests.base.replacingOccurrences(of: "name: ops", with: "name: \(name)"), file: "u.yaml")
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        let slot = ComposeSlot(spec: spec, cloudGate: CloudGate(limits: [:], storage: nil)) { _ in "test-key" }
        let (attempts, attempted) = AsyncStream.makeStream(of: Void.self)
        let (releases, release) = AsyncStream.makeStream(of: Void.self)
        let (ports, bound) = AsyncStream.makeStream(of: Int.self)
        let control = ComposeControl(status: { await slot.status() }, apply: { request throws(BasaltError) in
            attempted.yield()
            return try await slot.apply(request)
        })
        var launch = ServerLaunch(config: try ServerConfig(modelPath: "org/mock-model", restPort: 0, host: "127.0.0.1"), startREST: true, mcpServer: nil, pipelines: slot.registry, compose: control)
        (launch.handleSignals, launch.onBind) = (false, { bound.yield($0) })
        launch.startup = {
            for await _ in releases { break }
            await slot.started(try await Self.host(spec, chat: chat, registry: slot.registry))
        }
        let server = Task { try await ServeCommand.runServers(launch, orchestrator: chat) }
        defer { server.cancel() }
        _ = try ComposeRecord.write(name: name, port: try #require(await ports.first { _ in true }), file: file.path)
        let up = Task { try await ComposeCommand.Up.parse(["-f", file.path, "--wait", "--wait-timeout", "30"]).run() }
        _ = await attempts.first { _ in true }
        release.yield()
        try await up.value
        #expect(await slot.host?.spec.models["haiku"] != nil)
    }

    static func host(_ spec: ComposeSpec, chat: ModelOrchestratorActor, registry: PipelineRegistry) async throws -> ComposeHost {
        let services = ComposeServices(
            resolve: { $0 }, load: { _, _, _, _ in nil }, decider: { NamedDecider(modelID: $0) },
            decisions: { _, local, _, _, _ in local }, key: { _ in "test-key" }, cloudLimits: { _ in [:] })
        var start = ComposeStart(chat: chat, decider: NamedDecider(modelID: "self"), decisions: nil, cloudGate: CloudGate(limits: [:], storage: nil), startup: PipelineStages(), besideDecode: true)
        (start.admission, start.registry) = (ComposeAdmission(sizes: ComposeFileTests.sizes([:])) { (64 * ComposeFileTests.gb, .desktop) }, registry)
        let host = ComposeHost(services: services, spec: spec, start: start)
        try await host.start(defaultDecisions: nil)
        return host
    }
}
