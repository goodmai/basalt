import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import HTTPTypes
import NIOCore
@testable import BasaltCore
@testable import BasaltServer

@Suite("Rejections are unchanged by the shared pipeline (#106)", .serialized)
struct RejectionGoldenTests {
    static let cases: [(name: String, uri: String, body: String)] = {
        let endpoints = [
            ("messages", "/v1/messages", #""max_tokens": 64, "messages""#),
            ("chat", "/v1/chat/completions", #""messages""#),
            ("responses", "/v1/responses", #""input""#),
        ]
        var cases: [(String, String, String)] = []
        for (name, uri, key) in endpoints {
            let turn = name == "responses" ? #""hi""# : #"[{"role": "user", "content": "hi"}]"#
            let empty = name == "responses" ? #""""# : "[]"
            cases += [
                ("\(name)-not-json", uri, "not json"),
                ("\(name)-empty", uri, "{\(key): \(empty)}"),
                ("\(name)-wrong-type", uri, "{\(key): 3}"),
                ("\(name)-undeclared-tool", uri, #"{"#+key+": "+turn+#", "tools": [{"type": "function", "function": {"name": "a", "parameters": {}}}], "tool_choice": {"type": "function", "function": {"name": "b"}}}"#),
                ("\(name)-bad-temperature", uri, "{\(key): \(turn), \"temperature\": 5}"),
                ("\(name)-path-model", uri, "{\(key): \(turn), \"model\": \"/etc/passwd\"}"),
            ]
            if name != "messages" {
                cases.append(("\(name)-bad-format", uri, "{\(key): \(turn), \"response_format\": {\"type\": \"json_schema\"}}"))
            }
        }
        cases.append(("messages-bad-thinking", "/v1/messages", #"{"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "enabled", "budget_tokens": 64}}"#))
        return cases
    }()

    static func observed() async throws -> [String: [String: Any]] {
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "mock"))
        let raw = try await Application(router: server.makeRouter()).test(.router) { client in
            var raw: [(name: String, status: Int, contentType: String, body: String)] = []
            for c in cases {
                raw.append(try await client.execute(uri: c.uri, method: .post, headers: [.contentType: "application/json"],
                                                    body: ByteBuffer(string: c.body)) { res in
                    (c.name, Int(res.status.code), res.headers[.contentType] ?? "", String(buffer: res.body))
                })
            }
            return raw
        }
        return Dictionary(uniqueKeysWithValues: raw.map { entry in
            (entry.name, ["status": entry.status, "contentType": entry.contentType,
                          "body": (try? JSONSerialization.jsonObject(with: Data(entry.body.utf8))) ?? entry.body] as [String: Any])
        })
    }

    @Test("Every rejection keeps its status, content type and body")
    func rejectionsMatchTheGolden() async throws {
        let observed = try await Self.observed()
        if let path = ProcessInfo.processInfo.environment["BASALT_WRITE_REJECTION_GOLDEN"] {
            try JSONSerialization.data(withJSONObject: observed, options: [.sortedKeys, .prettyPrinted]).write(to: URL(fileURLWithPath: path))
            return
        }
        let golden = try #require(try JSONSerialization.jsonObject(with: Data(RejectionGolden.json.utf8)) as? [String: Any])
        #expect(Set(golden.keys) == Set(observed.keys))
        for (name, expected) in golden {
            #expect(NSDictionary(dictionary: observed[name] ?? [:]).isEqual(to: expected as? [String: Any] ?? [:]), "\(name)")
        }
    }
}
