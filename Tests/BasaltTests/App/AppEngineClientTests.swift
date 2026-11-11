import Foundation
import Hummingbird
import HummingbirdTesting
import Testing
@testable import BasaltAppCore
@testable import BasaltCore
@testable import BasaltServer

@Suite("Mac app core: the HTTP engine client against the real REST server on a loopback port — streaming, the bearer token, refusals, cancellation, stats (#434 phase 1)", .timeLimit(.minutes(1)))
struct AppEngineClientTests {
    static let token = "per-launch-token"

    static func serve<T: Sendable>(_ engine: MockInferenceEngine, token: String? = token, _ body: @Sendable (EngineEndpoint) async throws -> T) async throws -> T {
        let chat = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await chat.loadModel(path: "/models/chat")
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock-model", apiKey: token))
        return try await Application(router: server.makeRouter()).test(.live) { client in
            let port = try #require(client.port)
            return try await body(EngineEndpoint(base: try #require(URL(string: "http://localhost:\(port)")), token: token))
        }
    }

    static func request(_ text: String) -> ChatRequest {
        var conversation = ConversationSnapshot(id: ConversationID())
        conversation.messages = [MessageSnapshot(id: MessageID(), role: .user, parts: [.text(text)])]
        return ChatRequest(conversation)
    }

    static func collect(_ stream: AsyncThrowingStream<ChatDelta, any Error>) async throws -> [ChatDelta] {
        var deltas: [ChatDelta] = []
        for try await delta in stream { deltas.append(delta) }
        return deltas
    }

    @Test("A chat streams reasoning, the answer, the finish reason and usage; health and stats answer with the token")
    func streams() async throws {
        let engine = MockInferenceEngine()
        await engine.setMockReasoning("weighing it")
        await engine.setResponder { _ in "Basalt is volcanic." }
        try await Self.serve(engine) { endpoint in
            let client = EngineClient.http(endpoint)
            let deltas = try await Self.collect(client.chat(Self.request("What is basalt?")))
            #expect(deltas.first == .reasoning("weighing it"))
            #expect(deltas.compactMap { if case .text(let text) = $0 { text } else { nil } }.joined() == "Basalt is volcanic.")
            #expect(deltas.contains(.finish("stop")) && deltas.contains { if case .usage(_, 10) = $0 { true } else { false } })
            #expect(await client.health())
            let stats = try await client.stats()
            #expect(stats.totalRequests >= 1)
        }
    }

    @Test("A wrong token is refused with 401 and the server's message, on chat and on stats")
    func refused() async throws {
        try await Self.serve(MockInferenceEngine()) { endpoint in
            var wrong = endpoint
            wrong.token = "nope"
            let client = EngineClient.http(wrong)
            await #expect { _ = try await Self.collect(client.chat(Self.request("hi"))) } throws: { error in
                if case EngineError.refused(401, _) = error { true } else { false }
            }
            await #expect { _ = try await client.stats() } throws: { error in
                if case EngineError.refused(401, _) = error { true } else { false }
            }
        }
    }

    @Test("Cancelling the consumer ends its iteration without an error, the request is dropped, and the server takes the next chat")
    func cancels() async throws {
        let engine = MockInferenceEngine()
        await engine.holdStreams()
        try await Self.serve(engine) { endpoint in
            let client = EngineClient.http(endpoint)
            let consumer = Task {
                var deltas: [ChatDelta] = []
                for try await delta in client.chat(Self.request("long")) {
                    deltas.append(delta)
                }
                return deltas
            }
            await engine.waitForRequest()
            consumer.cancel()
            #expect(try await consumer.value.isEmpty)
            await engine.releaseStreams()
            let next = try await Self.collect(client.chat(Self.request("again")))
            #expect(next.contains(.finish("stop")))
        }
    }

    @Test("A server that is not there is a transport error, and health is false")
    func unreachable() async throws {
        let client = EngineClient.http(.loopback(port: 1, token: nil))
        await #expect { _ = try await Self.collect(client.chat(Self.request("hi"))) } throws: { error in
            if case EngineError.transport = error { true } else { false }
        }
        #expect(await client.health() == false)
    }
}
