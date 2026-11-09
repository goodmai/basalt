import Testing
import Foundation
import Synchronization
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltServer

final class DecisionCalls: Sendable {
    let requests = Mutex<[SystemOneRequest]>([])
}

struct StubDecisionEngine: DecisionEngine {
    let modelID = "org/stub-jev"
    let calls = DecisionCalls()

    func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try SystemOneCore.validate(request)
        calls.requests.withLock { $0.append(request) }
        let answers = request.questions.mapValues { SystemOneCore.answer($0, scores: SystemOneCore.options($0).keys.indices.map(Double.init), temperature: 1) }
        return SystemOneResponse(
            model: modelID, answers: answers,
            usage: SystemOneUsage(prefixTokens: 10, candidateBranches: 2, prefixEvaluations: 1, branchForwardCalls: 1, suffixTokens: 8), calibrated: true)
    }
}

@Suite("/v1/systemone and basalt_systemone_decide on a stub decision engine (#234)")
struct SystemOneAPITests {
    static let body = #"{"state": "The color is red.", "questions": {"color": {"type": "choice", "instructions": "Which color?", "criteria": {"red": "Red.", "blue": "Blue."}}, "rain": {"type": "noul", "instructions": "It rains."}}}"#

    private func app(_ engine: StubDecisionEngine?) throws -> Application<RouterResponder<BasicRequestContext>> {
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), maxTokens: 65536, telemetry: TelemetryTracker())
        return Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock-model")).withDecisions(engine).makeRouter())
    }

    private func post(_ client: some TestClientProtocol, _ body: String) async throws -> (HTTPResponse.Status, [String: Any]) {
        try await client.execute(uri: "/v1/systemone", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: body)) { res in
            (res.status, (try? JSONSerialization.jsonObject(with: Data(String(buffer: res.body).utf8)) as? [String: Any]) ?? [:])
        }
    }

    @Test("A valid request returns answers per question ID, usage in snake case and the model ID")
    func answers() async throws {
        let engine = StubDecisionEngine()
        try await app(engine).test(.router) { client in
            let (status, json) = try await post(client, Self.body)
            #expect(status == .ok)
            let answers = try #require(json["answers"] as? [String: [String: Any]])
            #expect(answers["color"]?["choice"] as? String == "red")
            #expect((answers["rain"]?["noul"] as? Double).map { $0 > 0.5 } == true)
            #expect((json["usage"] as? [String: Any])?["prefix_evaluations"] as? Int == 1)
            #expect(json["model"] as? String == "org/stub-jev")
            #expect(json["escalate"] as? Bool == false)
        }
        #expect(engine.calls.requests.withLock { $0.count } == 1)
    }

    @Test("Malformed bodies get 400 and invalid questions the engine's validation message; no decision model gets 404")
    func refusals() async throws {
        try await app(StubDecisionEngine()).test(.router) { client in
            let malformed = try await post(client, #"{"state": 7, "questions": {}}"#).0
            #expect(malformed == .badRequest)
            let (status, _) = try await post(client, #"{"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "A"}}}}"#)
            #expect(status == .badRequest)
        }
        try await app(nil).test(.router) { client in
            let missing = try await post(client, Self.body).0
            #expect(missing == .notFound)
        }
    }

    @Test("basalt_systemone_decide answers through MCP with the same parser, and says which flag to pass without a decision model")
    func mcpTool() async throws {
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine())
        let lines = MCPLines()
        let arguments = try #require(JSONSerialization.jsonObject(with: Data(Self.body.utf8)) as? [String: Any])
        let server = MCPServer(orchestrator: orchestrator, writer: { lines.append($0) }).withDecisions(StubDecisionEngine())
        await server.dispatch(toolCall(1, "basalt_systemone_decide", arguments))
        let result = try lines.toolText(0)
        #expect((result["answers"] as? [String: [String: Any]])?["color"]?["choice"] as? String == "red")
        let bare = MCPLines()
        await MCPServer(orchestrator: orchestrator, writer: { bare.append($0) }).dispatch(toolCall(2, "basalt_systemone_decide", arguments))
        #expect(try bare.json(0)["error"] != nil)
        #expect(MCPToolCatalog.list.tools.contains { $0.name == "basalt_systemone_decide" })
    }
}
