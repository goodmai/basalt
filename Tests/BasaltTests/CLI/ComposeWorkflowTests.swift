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

@Suite("compose staged workflow: pull, plan with sizes from the Hub or refused, diff as a dry run, progress events (#423 4d)", .timeLimit(.minutes(1)))
struct ComposeWorkflowTests {
    static let gb: Int64 = 1_000_000_000

    @Test("The Hub's file list gives the weight size; a repo without weights gives none")
    func hubSizes() throws {
        let json = #"{"id":"org/m","siblings":[{"rfilename":"model-00001.safetensors","size":4000000000},{"rfilename":"model-00002.safetensors","size":1000000000},{"rfilename":"config.json","size":900}]}"#
        #expect(HuggingFaceHub.weightBytes(try JSONDecoder().decode(HFModelDetails.self, from: Data(json.utf8))) == 5 * Self.gb)
        #expect(HuggingFaceHub.weightBytes(try JSONDecoder().decode(HFModelDetails.self, from: Data(#"{"id":"org/m","siblings":[{"rfilename":"README.md"}]}"#.utf8))) == nil)
    }

    @Test("Sizes come from the cache, then the Hub, then --assume-size; local paths and cloud models are not pulled or asked")
    func layeredSizes() async throws {
        let yaml = ComposeOperationsTests.with("  laya:\n    source: ./laya\n  big:\n    source: org/not-cached-big\n    keep: on-demand\n",
                                               pipelines: "pipelines:\n  long:\n    decide:\n      cascade: [laya]\n    answer:\n      routes:\n        - model: big\n        - model: chat\n")
        let spec = try ComposeFile.parse(yaml, file: "w.yaml")
        #expect(ComposeCommand.repos(spec).contains("org/not-cached-big") && !ComposeCommand.repos(spec).contains { $0.contains("laya") || $0.contains("haiku") })
        let asked = KeyLog()
        let remote = await ComposeCommand.remoteSizes(spec) { repo in
            asked.record(repo)
            return repo == "org/not-cached-big" ? 9 * Self.gb : nil
        }
        #expect(remote["org/not-cached-big"] == 9 * Self.gb && asked.all.contains("org/not-cached-big"))
        let sizes = ComposeCommand.sizes(remote: remote, assume: 2 * Self.gb)
        #expect(sizes.bytes("org/not-cached-big") == 9 * Self.gb && sizes.bytes("org/unknown-elsewhere") == 2 * Self.gb)
        #expect(ComposeCommand.sizes(remote: [:], assume: nil).bytes("org/unknown-elsewhere") == nil)
    }

    @Test("Pull downloads each repo once with progress every tenth, and reports the ones that failed")
    func pull() async throws {
        let spec = try ComposeFile.parse(ComposeOperationsTests.base, file: "p.yaml")
        let pulled = KeyLog()
        let failed = await ComposeCommand.pull(spec) { repo, progress in
            pulled.record(repo)
            for done in stride(from: Int64(0), through: 100, by: 5) { progress(done, 100) }
        }
        #expect(failed.isEmpty && pulled.all == ["mlx-community/gemma-4-12B-it-qat-4bit"])
        #expect(await ComposeCommand.pull(spec) { _, _ in throw HFError.modelNotFound("x") } == ["mlx-community/gemma-4-12B-it-qat-4bit"])
        let progress = PullProgress()
        let lines = stride(from: Int64(0), through: 100, by: 5).compactMap { progress.step($0, 100) }
        #expect(lines.count == 10 && lines.last?.hasPrefix("100%") == true && progress.step(1, 0) == nil)
    }

    @Test("Preview shows the change without making it, and says when a key is missing or a restart is needed; apply and probes emit events")
    func previewAndEvents() async throws {
        let failing: @Sendable (ComposeHeld) async -> String? = { model in if case .decision = model { "startup probe failed: no decision" } else { nil } }
        let (host, _, _) = try await ComposeRoleSwapTests.host(probe: failing)
        let change = ComposeOperationsTests.with(ComposeOperationsTests.haiku, pipelines: ComposeOperationsTests.support)
        let preview = try await host.preview(ComposeApply(yaml: change))
        let unchanged = await host.spec.models["haiku"] == nil
        #expect(preview.added == ["haiku"] && preview.refusal == nil && unchanged)
        let keyless = change.replacingOccurrences(of: "api_key_env: HOME\n", with: "api_key_env: \(ComposeHealthTests.unsetKey)\n")
        #expect(try await host.preview(ComposeApply(yaml: keyless)).refusal?.contains("not set") == true)
        #expect(try await host.preview(ComposeApply(yaml: change.replacingOccurrences(of: "name: ops", with: "name: other"))).restart != nil)
        _ = try await host.apply(ComposeApply(yaml: change))
        _ = try? await host.apply(ComposeApply(yaml: change.replacingOccurrences(of: "name: ops", with: "name: other")))
        let laya = ComposePipelineDeciderTests.yaml(ComposePipelineDeciderTests.laya)
        _ = try await host.apply(ComposeApply(yaml: laya))
        let kinds = await host.events.history.map(\.kind)
        #expect(kinds.contains("applied") && kinds.contains("refused") && kinds.contains("failed"))
        #expect(ComposeCommand.event(#"data: {"kind":"loading","model":"chat","message":"org/chat","at":"2026-11-12T00:00:00Z"}"#) == "loading chat org/chat" && ComposeCommand.event("event: x") == nil)
    }

    @Test("/v1/compose/diff answers the preview and /v1/compose/events streams events, the history first")
    func endpoints() async throws {
        let events = ComposeEvents()
        await events.emit("loading", model: "chat", "org/chat")
        var control = ComposeControl(status: { ComposeStatus(name: "ops", pipelines: [], models: [], spentToday: 0) }, apply: { _ throws(BasaltError) in ComposeDiff() })
        control.preview = { _ throws(BasaltError) in
            var diff = ComposeDiff()
            diff.refusal = "does not fit"
            return diff
        }
        (control.events, control.heartbeat) = (events, .milliseconds(50))
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock-model")).withPipelines(nil, compose: control)
        try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: "/v1/compose/diff", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: #"{"yaml": "version: 1"}"#)) { response in
                #expect(response.status == .ok && (try? JSONDecoder().decode(ComposeDiff.self, from: Data(buffer: response.body)))?.refusal == "does not fit")
            }
        }
        try await Application(router: server.makeRouter()).test(.live) { client in
            let port = try #require(client.port)
            let (lines, _) = try await URLSession.shared.bytes(from: try #require(URL(string: "http://localhost:\(port)/v1/compose/events")))
            var first: String?
            for try await line in lines.lines where line.hasPrefix("data: ") {
                first = ComposeCommand.event(line)
                break
            }
            await events.emit("ready", model: "chat", "org/chat")
            #expect(first == "loading chat org/chat")
        }
    }

    @Test("compose diff against the running system prints the change, fails on a needed restart, and says when the system is not running")
    func diffCommand() async throws {
        let name = "test-\(UUID().uuidString.prefix(8).lowercased())"
        let file = FileManager.default.temporaryDirectory.appending(path: "\(name).yaml")
        defer { try? FileManager.default.removeItem(at: ComposeRecord.url(name)); try? FileManager.default.removeItem(at: file) }
        let base = ComposeOperationsTests.base.replacingOccurrences(of: "name: ops", with: "name: \(name)")
        let spec = try ComposeFile.parse(base, file: "d.yaml")
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        let registry = PipelineRegistry([:])
        let host = try await ComposeBindFirstTests.host(spec, chat: chat, registry: registry)
        let (ports, bound) = AsyncStream.makeStream(of: Int.self)
        var launch = ServerLaunch(config: try ServerConfig(modelPath: "org/mock-model", restPort: 0, host: "127.0.0.1"), startREST: true, mcpServer: nil, pipelines: registry, compose: host.control)
        (launch.handleSignals, launch.onBind) = (false, { bound.yield($0) })
        let server = Task { try await ServeCommand.runServers(launch, orchestrator: chat) }
        defer { server.cancel() }
        _ = try ComposeRecord.write(name: name, port: try #require(await ports.first { _ in true }), file: file.path)
        try Data(ComposeOperationsTests.with(ComposeOperationsTests.haiku, pipelines: ComposeOperationsTests.support).replacingOccurrences(of: "name: ops", with: "name: \(name)").utf8).write(to: file)
        try await ComposeCommand.Diff.parse(["-f", file.path]).run()
        #expect(await host.spec.models["haiku"] == nil)
        try Data(base.replacingOccurrences(of: "version: 1\n", with: "version: 1\nmemory: sequential\n").utf8).write(to: file)
        await #expect(throws: ExitCode.self) { try await ComposeCommand.Diff.parse(["-f", file.path]).run() }
        try FileManager.default.removeItem(at: ComposeRecord.url(name))
        try await ComposeCommand.Diff.parse(["-f", file.path]).run()
    }
}
