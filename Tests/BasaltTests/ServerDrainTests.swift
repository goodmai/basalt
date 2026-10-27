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
}
