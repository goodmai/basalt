import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import HTTPTypes
import Logging
import NIOCore
import ServiceLifecycle
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltServer

@Suite("Admission over HTTP and drain on graceful shutdown (#74)", .serialized)
struct ServerDrainTests {
    @Test("A full queue answers a non-streaming request with 503 and Retry-After")
    func busyIsServiceUnavailableWithRetryAfter() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(
            engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 1, maxQueue: 0))
        try await orchestrator.loadModel(path: "mock")
        let holder = try await orchestrator.gate.admit()
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "mock"))
        try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(
                uri: "/api/v1/generate", method: .post, headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"prompt": "hi"}"#)
            ) { res in
                #expect(res.status == .serviceUnavailable)
                #expect(res.headers[.retryAfter] == "1")
            }
        }
        await orchestrator.gate.release(holder)
    }

    @Test("On graceful shutdown the drain service ends an open stream within the deadline")
    func drainOnGracefulShutdown() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")
        await engine.setArtificialDelay(30)
        let stream = try await orchestrator.generateStream(request: GenerationRequest(prompt: "long"))
        let consumer = Task { for await _ in stream {} }

        let group = ServiceGroup(configuration: .init(
            services: [.init(service: GenerationDrainService(orchestrator: orchestrator, deadline: .milliseconds(200)))],
            logger: Logger(label: "test")))
        let running = Task { try await group.run() }
        try await Task.sleep(for: .milliseconds(50))
        let started = ContinuousClock.now
        await group.triggerGracefulShutdown()
        await consumer.value
        try await running.value
        #expect(ContinuousClock.now - started < .seconds(3))
        #expect(await orchestrator.gate.inFlight == 0)
    }

    static let streamEndpoints: [(uri: String, body: String)] = [
        ("/api/v1/generate/stream", #"{"prompt": "hi"}"#),
        ("/v1/chat/completions", #"{"stream": true, "messages": [{"role": "user", "content": "hi"}]}"#),
        ("/v1/responses", #"{"stream": true, "input": "hi"}"#),
        ("/v1/messages", #"{"stream": true, "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}"#),
    ]

    @Test("A streaming request is refused with an HTTP 503 and Retry-After before any SSE byte when the queue is full")
    func streamRefusedBeforeSSE() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(
            engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 1, maxQueue: 0))
        try await orchestrator.loadModel(path: "mock")
        let holder = try await orchestrator.gate.admit()
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "mock"))
        try await Application(router: server.makeRouter()).test(.router) { client in
            for endpoint in Self.streamEndpoints {
                try await client.execute(
                    uri: endpoint.uri, method: .post, headers: [.contentType: "application/json"],
                    body: ByteBuffer(string: endpoint.body)
                ) { res in
                    #expect(res.status == .serviceUnavailable, "\(endpoint.uri)")
                    #expect(res.headers[.retryAfter] == "1", "\(endpoint.uri)")
                    #expect(res.headers[.contentType] != "text/event-stream", "\(endpoint.uri)")
                }
            }
        }
        await orchestrator.gate.release(holder)
    }

    @Test("A streaming prompt that cannot fit in memory is 413 before the SSE starts; a normal one still streams")
    func streamTooLargeBeforeSSE() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "mock"))
        try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(
                uri: "/v1/chat/completions", method: .post, headers: [.contentType: "application/json"],
                body: ByteBuffer(string: Self.streamEndpoints[1].body)
            ) { res in
                #expect(res.status == .ok)
                #expect(res.headers[.contentType] == "text/event-stream")
            }
            await engine.setMemory(kvBudget: 0, promptTokens: 10)
            for endpoint in Self.streamEndpoints {
                try await client.execute(
                    uri: endpoint.uri, method: .post, headers: [.contentType: "application/json"],
                    body: ByteBuffer(string: endpoint.body)
                ) { res in
                    #expect(res.status == .contentTooLarge, "\(endpoint.uri)")
                }
            }
        }
    }

    @Test("Shutting down inside an already-cancelled task waits for in-flight work without busy-spinning")
    func cancelledShutdownDoesNotBusySpin() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")
        let holder = try await orchestrator.gate.admit()
        let releaser = Task {
            try? await Task.sleep(for: .milliseconds(300))
            await orchestrator.gate.release(holder)
        }
        let started = ContinuousClock.now
        let (ready, markReady) = AsyncStream<Void>.makeStream()
        let cancelled = Task {
            markReady.finish()
            while !Task.isCancelled { await Task.yield() }
            #expect(Task.isCancelled)
            await orchestrator.shutdown(deadline: .seconds(1))
        }
        for await _ in ready {}
        cancelled.cancel()
        await cancelled.value
        await releaser.value
        let elapsed = ContinuousClock.now - started
        let polls = await orchestrator.gate.drainPolls
        #expect(elapsed >= .milliseconds(250))
        #expect(elapsed < .milliseconds(900))
        #expect(polls >= 1 && polls <= 45)
        #expect(await orchestrator.gate.inFlight == 0)
        #expect(await engine.persistCount == 1)
    }

    @Test("MCP-only serve persists the prompt cache and flushes the session log on stdin EOF")
    func mcpOnlyServePersistsCacheAndClosesSessionLogOnEOF() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-mcp-eof-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let sessionLog = SessionLogWriter(mode: .full, directory: tempDir, bufferLimit: 65_536, defaultSession: "mcp-eof")
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker(), sessionLog: sessionLog)
        try await orchestrator.loadModel(path: "mock")

        let input = AsyncStream<String> { continuation in
            continuation.yield(#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#)
            continuation.yield(#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"gemma_generate","arguments":{"prompt":"hello"}}}"#)
            continuation.finish()
        }
        let mcp = MCPServer(orchestrator: orchestrator, modelId: "mock", input: input) { _ in }
        try await ServeCommand.runServers(
            orchestrator: orchestrator,
            config: try ServerConfig(modelPath: "mock", restPort: 0),
            startREST: false,
            mcpServer: mcp,
            shutdownDeadline: .seconds(1),
            handleSignals: false
        )

        #expect(await engine.persistCount == 1)
        let logFile = try #require(SessionLogReader.file(for: "mcp-eof", in: tempDir))
        let contents = try String(contentsOf: logFile, encoding: .utf8)
        #expect(contents.contains("\"type\":\"request\""))
        #expect(contents.contains("\"type\":\"response\""))
    }

    @Test("MCP graceful shutdown exits an idle input stream without hanging and persists once")
    func mcpGracefulShutdownExitsIdleInputAndPersistsOnce() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")

        let (input, continuation) = AsyncStream<String>.makeStream()
        defer { continuation.finish() }
        let mcp = MCPServer(orchestrator: orchestrator, modelId: "mock", input: input) { _ in }
        let drain = GenerationDrainService(orchestrator: orchestrator, deadline: .milliseconds(200))
        let group = ServiceGroup(configuration: .init(
            services: [
                .init(service: mcp, successTerminationBehavior: .gracefullyShutdownGroup),
                .init(service: drain, successTerminationBehavior: .gracefullyShutdownGroup),
            ],
            logger: Logger(label: "test")
        ))
        let running = Task { try await group.run() }
        try await Task.sleep(for: .milliseconds(50))
        let started = ContinuousClock.now
        await group.triggerGracefulShutdown()
        try await running.value
        #expect(ContinuousClock.now - started < .seconds(1))
        #expect(await engine.persistCount == 1)
    }

    @Test("Combined REST and MCP serve persists the prompt cache once, not twice, when MCP reaches EOF")
    func combinedServePersistsOnceOnMCPEOF() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")

        let (input, continuation) = AsyncStream<String>.makeStream()
        let feeder = Task {
            try? await Task.sleep(for: .milliseconds(100))
            continuation.yield(#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#)
            continuation.finish()
        }
        let mcp = MCPServer(orchestrator: orchestrator, modelId: "mock", input: input) { _ in }
        try await ServeCommand.runServers(
            orchestrator: orchestrator,
            config: try ServerConfig(modelPath: "mock", restPort: 0),
            startREST: true,
            mcpServer: mcp,
            shutdownDeadline: .milliseconds(500),
            handleSignals: false
        )
        await feeder.value
        #expect(await engine.persistCount == 1)
    }

    @Test("Throwing from engine.generate releases the gate ticket via async defer")
    func generateErrorReleasesGateTicketViaAsyncDefer() async throws {
        let engine = MockInferenceEngine()
        let orchestrator = ModelOrchestratorActor(
            engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 1, maxQueue: 0))
        try await orchestrator.loadModel(path: "mock")
        await engine.setShouldFailOnGenerate(true)

        await #expect(throws: BasaltError.self) {
            _ = try await orchestrator.generate(request: GenerationRequest(prompt: "boom"))
        }
        #expect(await orchestrator.gate.inFlight == 0)

        await engine.setShouldFailOnGenerate(false)
        let recovery = try await orchestrator.generate(request: GenerationRequest(prompt: "ok"))
        #expect(recovery.generatedText == "Mock response")
        #expect(await orchestrator.gate.inFlight == 0)
    }
}
