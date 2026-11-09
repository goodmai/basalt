import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltCloud
@testable import BasaltInference
@testable import BasaltServer

@Suite("Cascade System One on /v1/systemone, basalt_systemone_decide and the /v1/pipeline decide stage: stub remote, tiny local readout (#407, #228)", .serialized, .timeLimit(.minutes(1)))
struct CascadeContractTests {
    static let body = #"{"state": "The car is red.", "questions": {"color": {"type": "choice", "instructions": "Which color?", "criteria": {"red": "Red.", "blue": "Blue."}}, "unsure_rain": {"type": "noul", "instructions": "It rains."}}}"#

    static func cascade(_ base: URL, gate: GenerationGate) async throws -> CascadeDecisionEngine {
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-readout"), model: try TinyModel.make(), processor: ByteInputProcessor(), tokenizer: ASCIITokenizer())))
        let local = ReadoutDecisionEngine(engine: engine, modelID: "tiny-readout", readout: .uncalibrated)
        return CascadeDecisionEngine(remote: try RemoteStub.engine(base, "ok"), local: local, escalateBelow: 0.6, gate: gate)
    }

    static func expectSplit(_ json: [String: Any]) {
        let answers = json["answers"] as? [String: [String: Any]]
        #expect(answers?["color"]?["choice"] as? String == "blue")
        #expect(answers?["unsure_rain"]?["noul"] is Double)
        let stages = (json["usage"] as? [String: Any])?["stages"] as? [[String: Any]]
        #expect(stages?.map { $0["model"] as? String } == ["typesafe/jev-stub", "tiny-readout"])
        #expect(stages?.map { $0["questions"] as? [String] } == [["color"], ["unsure_rain"]])
        #expect((json["usage"] as? [String: Any])?["cost"] as? Double == 0.00001)
    }

    @Test("/v1/systemone: the confident question is answered remotely, the unsure one by the tiny local readout, in one response")
    func rest() async throws {
        try await RemoteStub.serve { base, log in
            let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), maxTokens: 65536, telemetry: TelemetryTracker())
            let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock-model")).withDecisions(try await Self.cascade(base, gate: orchestrator.gate))
            #expect(server.decisions is CascadeDecisionEngine)
            let json = try await Application(router: server.makeRouter()).test(.router) { client in
                try await client.execute(uri: "/v1/systemone", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: Self.body)) { res in
                    #expect(res.status == .ok)
                    return (try? JSONSerialization.jsonObject(with: Data(String(buffer: res.body).utf8)) as? [String: Any]) ?? [:]
                }
            }
            Self.expectSplit(json)
            #expect(log.calls.withLock { $0.count } == 1)
        }
    }

    @Test("basalt_systemone_decide answers through the same cascade")
    func mcp() async throws {
        try await RemoteStub.serve { base, log in
            let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine())
            let lines = MCPLines()
            let server = MCPServer(orchestrator: orchestrator, writer: { lines.append($0) }).withDecisions(try await Self.cascade(base, gate: orchestrator.gate))
            await server.dispatch(toolCall(1, "basalt_systemone_decide", try #require(JSONSerialization.jsonObject(with: Data(Self.body.utf8)) as? [String: Any])))
            Self.expectSplit(try lines.toolText(0))
            #expect(log.calls.withLock { $0.count } == 1)
        }
    }

    @Test("/v1/pipeline sends a typed text state to the cascade, but a state read from a document stays on the machine and is decided locally")
    func pipeline() async throws {
        try await RemoteStub.serve { base, log in
            let (_, chat) = try await PipelineAPITests.chat()
            let cascade = try await Self.cascade(base, gate: chat.gate)
            let typed = try await PipelineAPITests.post(chat, decisions: cascade, ["text": "The car is red.", "questions": PipelineAPITests.questions])
            #expect(typed.status == .ok)
            #expect(log.calls.withLock { $0.count } == 1)
            #expect(PipelineAPITests.stages(typed.json).first?["model"] as? String == "typesafe/jev-stub")
            await chat.setDocumentTranscriber(ScriptedDocumentTranscriber(text: "Invoice total: 42"))
            let image = ["kind": "image", "data": try OCRFixture.png(width: 4, height: 4).base64EncodedString()]
            let read = try await PipelineAPITests.post(chat, decisions: cascade, ["attachments": [image], "questions": PipelineAPITests.questions])
            #expect(read.status == .ok)
            #expect(log.calls.withLock { $0.count } == 1)
            #expect(PipelineAPITests.stages(read.json).map { $0["model"] as? String }.contains("tiny-readout"))
        }
    }
}
