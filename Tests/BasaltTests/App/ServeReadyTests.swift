import Foundation
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltServer

@Suite("serve --port 0 --announce-ready: the server binds a free port and reports it once, for the Mac app's supervisor (#434 phase 1)", .timeLimit(.minutes(1)))
struct ServeReadyTests {
    @Test("The ready handler gets the bound port, health answers on it, and cancelling the run stops the server")
    func reportsPort() async throws {
        let engine = MockInferenceEngine()
        let chat = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await chat.loadModel(path: "/models/chat")
        let (ports, report) = AsyncStream<Int>.makeStream()
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock-model", restPort: 0), handleSignals: false)
            .withReadyHandler { report.yield($0) }
        let running = Task { try await server.run() }
        var iterator = ports.makeAsyncIterator()
        let port = try #require(await iterator.next())
        #expect(port > 0)
        let health = try await EngineClientProbe.status(port)
        #expect(health == 200)
        running.cancel()
        _ = await running.result
    }

    @Test("The ready line is one JSON object after a fixed prefix, with port, pid and version")
    func line() throws {
        let line = RESTServer.readyLine(port: 54_321, pid: 77)
        #expect(line.hasPrefix("basalt-ready {"))
        let json = try #require(try JSONSerialization.jsonObject(with: Data(line.dropFirst("basalt-ready ".count).utf8)) as? [String: Any])
        #expect(json["port"] as? Int == 54_321 && json["pid"] as? Int == 77 && json["version"] as? String == HealthResponse.version)
    }

    @Test("--announce-ready parses, is refused beside --mcp, and --port 0 is accepted")
    func flags() throws {
        let serve = try ServeCommand.parse(["--model", "org/m", "--port", "0", "--announce-ready"])
        #expect(serve.network.announceReady && serve.network.port == 0)
        #expect(throws: (any Error).self) { try ServeCommand.parse(["--model", "org/m", "--announce-ready", "--mcp"]) }
    }
}

enum EngineClientProbe {
    static func status(_ port: Int) async throws -> Int {
        let (_, response) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/api/v1/health")!)
        return (response as? HTTPURLResponse)?.statusCode ?? 0
    }
}
