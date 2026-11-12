import Testing
import Foundation
import ArgumentParser
import Hummingbird
import HummingbirdTesting
import NIOCore
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltServer

@Suite("compose health: closed model states, /v1/health/live and /ready, missing cloud keys, pid reuse, and down that fails on timeout (#423 4c-1)", .timeLimit(.minutes(1)))
struct ComposeHealthTests {
    static let unsetKey = "BASALT_TEST_KEY_THAT_IS_NEVER_SET"

    @Test("Model states are a closed set; only states that can serve a request count as ready")
    func states() {
        #expect(ComposeModelState(OnDemandCore.Phase.unloaded) == .idleUnloaded && ComposeModelState(OnDemandCore.Phase.loaded) == .ready)
        #expect(ComposeModelState(OnDemandCore.Phase.loading) == .loading && !ComposeModelState.loading.serves)
        #expect(ComposeHost.cloudState(nil, key: "") == .noKey && ComposeHost.cloudState(nil, key: "k") == .cloud)
        #expect(ComposeHost.cloudState(.budget(spent: 1, limit: 1), key: "k") == .overBudget && ComposeHost.cloudState(.paused(until: .now), key: "k") == .rateLimited)
        #expect(ComposeHost.cloudState(.disabled("401"), key: "k") == .refused && !ComposeModelState.refused.serves && !ComposeModelState.noKey.serves)
        #expect(Set(ComposeModelState.allCases.filter(\.serves)) == [.ready, .decision, .cloud, .idleUnloaded, .overBudget, .rateLimited])
    }

    @Test("A cloud model whose key variable is unset fails compose up before anything loads, and a running host refuses it on apply")
    func missingKeys() async throws {
        let yaml = ComposeOperationsTests.base.replacingOccurrences(of: "api_key_env: HOME", with: "api_key_env: \(Self.unsetKey)")
        let spec = try ComposeFile.parse(yaml, file: "k.yaml")
        #expect(spec.missingKeys([:]) == ["jev reads its key from $\(Self.unsetKey), which is not set"])
        #expect(spec.missingKeys([Self.unsetKey: "sk-test"]).isEmpty && spec.missingKeys([Self.unsetKey: ""]).count == 1)
        #expect(throws: ExitCode.self) { try ComposeCommand.announce(spec, environment: [:]) }
        let (host, _, _) = try await ComposeRoleSwapTests.host()
        let refused = await #expect(throws: BasaltError.self) { _ = try await host.apply(ComposeApply(yaml: yaml)) }
        #expect(refused.map { "\($0)" }?.contains("not set") == true)
    }

    @Test("/v1/health/live answers 200; /v1/health/ready is 503 with the waiting models until every one can serve")
    func readiness() async throws {
        let (_, loaded) = try await PipelineSwapEdgeTests.chat()
        let (_, empty) = try await PipelineSwapEdgeTests.chat(loaded: false)
        let loading = ComposeModelStatus(name: "big", source: "org/big", keep: "resident", state: .loading)
        let ready = ComposeModelStatus(name: "chat", source: "org/chat", keep: "resident", state: .ready)
        #expect(try await Self.get("/v1/health/live", empty, compose: nil) == (.ok, nil))
        #expect(try await Self.get("/v1/health/ready", loaded, compose: nil) == (.ok, []))
        #expect(try await Self.get("/v1/health/ready", empty, compose: nil) == (.serviceUnavailable, ["chat (unloaded)"]))
        #expect(try await Self.get("/v1/health/ready", loaded, compose: Self.control([ready, loading])) == (.serviceUnavailable, ["big (loading)"]))
        #expect(try await Self.get("/v1/health/ready", loaded, compose: Self.control([ready])) == (.ok, []))
    }

    @Test("A record whose pid now belongs to another process is not running; down fails when the process outlives the grace period")
    func lifecycle() async throws {
        let name = "test-\(UUID().uuidString.prefix(8).lowercased())"
        defer { try? FileManager.default.removeItem(at: ComposeRecord.url(name)) }
        _ = try ComposeRecord.write(name: name, port: 18_124, file: "x.yaml")
        #expect(ComposeRecord.read(name)?.started == ComposeRecord.startTime(ProcessInfo.processInfo.processIdentifier))
        #expect(ComposeRecord.running(name) != nil)
        var reused = try #require(ComposeRecord.read(name))
        reused.started = 1
        try JSONEncoder().encode(reused).write(to: ComposeRecord.url(name))
        #expect(ComposeRecord.running(name) == nil)
        let stubborn = try await Self.spawn("trap '' INT; echo ready; exec sleep 20")
        defer { stubborn.terminate() }
        await #expect(throws: ExitCode.self) { try await ComposeCommand.stop(Self.record(stubborn), grace: .milliseconds(300)) }
        let polite = try await Self.spawn("echo ready; exec sleep 20")
        try await ComposeCommand.stop(Self.record(polite), grace: .seconds(10))
        polite.waitUntilExit()
        #expect(!polite.isRunning)
    }

    static func spawn(_ script: String) async throws -> Process {
        let (process, output) = (Process(), Pipe())
        (process.executableURL, process.arguments, process.standardOutput) = (URL(fileURLWithPath: "/bin/sh"), ["-c", script], output)
        try process.run()
        await withCheckedContinuation { (ready: CheckedContinuation<Void, Never>) in
            output.fileHandleForReading.readabilityHandler = { handle in
                handle.readabilityHandler = nil
                ready.resume()
            }
        }
        return process
    }

    static func record(_ process: Process) -> ComposeRecord {
        let pid = process.processIdentifier
        return ComposeRecord(name: "stub", pid: pid, port: 0, file: "", started: ComposeRecord.startTime(pid))
    }

    static func control(_ models: [ComposeModelStatus]) -> ComposeControl {
        ComposeControl(status: { ComposeStatus(name: "ops", pipelines: ["default"], models: models, spentToday: 0) }, apply: { _ throws(BasaltError) in ComposeDiff() })
    }

    static func get(_ path: String, _ chat: ModelOrchestratorActor, compose: ComposeControl?) async throws -> (HTTPResponse.Status, [String]?) {
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock-model")).withPipelines(nil, compose: compose)
        return try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: path, method: .get) { response in
                let json = try JSONSerialization.jsonObject(with: Data(buffer: response.body)) as? [String: Any]
                return (response.status, json?["waiting"] as? [String])
            }
        }
    }
}
