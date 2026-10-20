import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("response_format over the wire (#42)")
struct ResponseFormatAPITests {

    private func app() async throws -> (Application<RouterResponder<BasicRequestContext>>, MockInferenceEngine) {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orch.loadModel(path: "mock")
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock"))
        return (Application(router: server.makeRouter()), engine)
    }

    private func post(_ app: Application<RouterResponder<BasicRequestContext>>, _ uri: String, _ json: String,
                      _ check: @escaping @Sendable (TestResponse) throws -> Void) async throws {
        try await app.test(.router) { client in
            try await client.execute(uri: uri, method: .post, headers: [.contentType: "application/json"],
                                     body: ByteBuffer(string: json)) { try check($0) }
        }
    }

    private static func errorMessage(_ response: TestResponse) -> String? {
        let body = try? JSONSerialization.jsonObject(with: Data(String(buffer: response.body).utf8)) as? [String: Any]
        return (body?["error"] as? [String: Any])?["message"] as? String
    }

    static let schema = #"{"type": "object", "properties": {"tags": {"type": "array", "items": {"enum": ["a", "b"]}}}, "required": ["tags"]}"#

    @Test("Chat completions: json_schema reaches the engine as a compiled-checked schema, json_object as any object, text as nothing")
    func chatCompletionsFormats() async throws {
        let (app, engine) = try await app()
        try await post(app, "/v1/chat/completions", """
            {"messages": [{"role": "user", "content": "hi"}],
             "response_format": {"type": "json_schema", "json_schema": {"name": "t", "strict": true, "schema": \(Self.schema)}}}
            """) { #expect($0.status == .ok) }
        guard case .jsonSchema(let sent)? = await engine.lastRequest?.responseFormat else {
            Issue.record("json_schema did not reach the engine"); return
        }
        let grammar = try JSONGrammar.compile(schemaJSON: sent)
        var matcher = JSONGrammarMatcher(grammar: grammar)
        #expect(matcher.accept(Array(#"{"tags":["a"]}"#.utf8)) && matcher.isAccepting)

        try await post(app, "/v1/chat/completions", #"""
            {"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}}
            """#) { #expect($0.status == .ok) }
        #expect(await engine.lastRequest?.responseFormat == .jsonObject)

        try await post(app, "/v1/chat/completions", #"""
            {"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "text"}}
            """#) { #expect($0.status == .ok) }
        #expect(await engine.lastRequest?.responseFormat == nil)
    }

    @Test("Responses: text.format json_schema carries the schema inline")
    func responsesTextFormat() async throws {
        let (app, engine) = try await app()
        try await post(app, "/v1/responses", """
            {"input": "hi", "text": {"format": {"type": "json_schema", "name": "t", "schema": \(Self.schema)}}}
            """) { #expect($0.status == .ok) }
        guard case .jsonSchema? = await engine.lastRequest?.responseFormat else {
            Issue.record("text.format did not reach the engine"); return
        }
    }

    @Test("An unsupported schema feature, an unknown type or a missing schema is a 400 that says what is wrong")
    func rejectsWhatItCannotEnforce() async throws {
        let (app, engine) = try await app()
        try await post(app, "/v1/chat/completions", #"""
            {"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_schema",
             "json_schema": {"name": "t", "schema": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}}}}}
            """#) { response in
            #expect(response.status == .badRequest)
            #expect(Self.errorMessage(response) == "response_format: unsupported JSON Schema keyword 'minimum' at #/properties/n")
        }
        try await post(app, "/v1/responses", #"""
            {"input": "hi", "text": {"format": {"type": "xml"}}}
            """#) { response in
            #expect(response.status == .badRequest)
            #expect(Self.errorMessage(response)?.contains("'xml' is not supported") == true)
        }
        try await post(app, "/v1/chat/completions", #"""
            {"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_schema", "json_schema": {"name": "t"}}}
            """#) { response in
            #expect(response.status == .badRequest)
            #expect(Self.errorMessage(response) == "response_format json_schema needs a schema")
        }
        #expect(await engine.lastRequest == nil)
    }
}
