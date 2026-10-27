import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import HTTPTypes
import Logging
import NIOCore
import ServiceLifecycle
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
        await engine.setArtificialDelay(2)
        let holding = try await orchestrator.generateStream(request: GenerationRequest(prompt: "hold"))
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
        _ = holding
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
        await engine.setArtificialDelay(2)
        let holding = try await orchestrator.generateStream(request: GenerationRequest(prompt: "hold"))
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
        _ = holding
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
}
