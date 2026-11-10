import Testing
import Foundation
import Synchronization
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltCloud
@testable import BasaltServer

final class RemoteStubLog: Sendable {
    struct Call: Sendable {
        let path: String
        let authorization: String?
        let data: Data

        var body: [String: Any]? { (try? JSONSerialization.jsonObject(with: data)).flatMap { $0 as? [String: Any] } }
    }

    let calls = Mutex<[Call]>([])
}

enum RemoteStub {
    static let key = "sk-or-v1-0123456789abcdef0123456789abcdef"

    static func reply(to body: [String: Any]?) -> String {
        let questions = body?["questions"] as? [String: [String: Any]] ?? [:]
        let answers = questions.map { id, question -> String in
            let unsure = id.hasPrefix("unsure")
            guard question["type"] as? String == "choice", let keys = (question["criteria"] as? [String: String])?.keys.sorted(), let top = keys.first else {
                return "\"\(id)\": {\"type\": \"noul\", \"noul\": \(unsure ? 0.6 : 0.97)}"
            }
            let probabilities = keys.map { "\"\($0)\": \($0 == top ? 0.9 : 0.1 / Double(keys.count - 1))" }.joined(separator: ", ")
            return "\"\(id)\": {\"type\": \"choice\", \"choice\": \"\(top)\", \"probabilities\": {\(probabilities)}, \"confidence\": \(unsure ? 0.3 : 0.95)}"
        }
        return "{\"model\": \"typesafe/jev-stub\", \"answers\": {\(answers.joined(separator: ", "))}, \"usage\": {\"input_tokens\": 9, \"cost\": 0.00001}, \"provider\": \"TypeSafe\"}"
    }

    static func router(_ log: RemoteStubLog, hang: AsyncStream<Void>) -> Router<BasicRequestContext> {
        let router = Router()
        @Sendable func record(_ request: Request) async throws -> RemoteStubLog.Call {
            let body = Data(try await request.body.collect(upTo: 1 << 20).readableBytesView)
            let call = RemoteStubLog.Call(path: request.uri.path, authorization: request.headers[.authorization], data: body)
            log.calls.withLock { $0.append(call) }
            return call
        }
        router.post("ok/decisions") { request, _ in
            Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: ByteBuffer(string: reply(to: try await record(request).body))))
        }
        router.post("fail/decisions") { request, _ in
            let call = try await record(request)
            return Response(status: .serviceUnavailable, body: ResponseBody(byteBuffer: ByteBuffer(string: "upstream died; you sent \(call.authorization ?? "")")))
        }
        router.post("garbage/decisions") { request, _ in
            _ = try await record(request)
            return Response(status: .ok, body: ResponseBody(byteBuffer: ByteBuffer(string: "<html>\(key)</html>")))
        }
        router.post("hang/decisions") { request, _ in
            _ = try await record(request)
            for await _ in hang {}
            return Response(status: .ok)
        }
        router.post("redirect/decisions") { request, _ in
            _ = try await record(request)
            return Response(status: .temporaryRedirect, headers: [.location: "/ok/decisions"])
        }
        return router
    }

    static func serve<T: Sendable>(_ body: @Sendable (URL, RemoteStubLog) async throws -> T) async throws -> T {
        let log = RemoteStubLog()
        let (hang, release) = AsyncStream<Void>.makeStream()
        let app = Application(router: router(log, hang: hang))
        return try await app.test(.live) { client in
            defer { release.finish() }
            let port = try #require(client.port)
            return try await body(try #require(URL(string: "http://localhost:\(port)")), log)
        }
    }

    static func engine(_ base: URL, _ route: String, timeout: TimeInterval = 30) throws -> RemoteDecisionEngine {
        try RemoteDecisionEngine(model: "~typesafe/jev-latest", apiKey: key, baseURL: base.appending(path: route), timeout: timeout)
    }
}

@Suite("Remote System One on OpenRouter's decisions endpoint: decoder, redaction, the key, fallback against a stub server (#407)", .timeLimit(.minutes(1)))
struct RemoteDecisionEngineTests {
    static let recorded = #"{"model":"typesafe/jev-1.14-20261014","answers":{"department":{"type":"choice","choice":"sales","probabilities":{"account":0,"billing":0.05,"returns":0,"sales":0.94,"shipping":0.01,"technical":0},"confidence":0.93},"late":{"type":"noul","noul":0.02},"severity":{"type":"score","score":0.02,"legend":{"0":"No impact.","1":"Minor.","2":"Major.","3":"Critical."},"probabilities":{"0":0.99,"1":0.01,"2":0,"3":0},"confidence":0.99}},"usage":{"input_tokens":581,"output_tokens":89,"cost":0.000024402,"is_byok":false},"id":"gen-dec-1","provider":"TypeSafe"}"#

    static let request = SystemOneRequest(state: "Ticket: discount for order #1213?", questions: [
        "department": SystemOneQuestion(type: .choice, instructions: "Which department?", criteria: .named(Dictionary(uniqueKeysWithValues: ["account", "billing", "returns", "sales", "shipping", "technical"].map { ($0, "The \($0) team.") }))),
        "late": SystemOneQuestion(type: .noul, instructions: "The order was late."),
        "severity": SystemOneQuestion(type: .score, instructions: "How severe?", criteria: .levels(["No impact.", "Minor.", "Major.", "Critical."])),
    ])

    @Test("A recorded Jev response decodes: type fields and the legend object are ignored, noul confidence is |2p − 1|, the cost is kept")
    func decodesRecorded() throws {
        let response = try #require(RemoteDecisionCore.response(Data(Self.recorded.utf8), for: Self.request))
        #expect(response.model == "typesafe/jev-1.14-20261014")
        #expect(response.answers["department"]?.choice == "sales")
        #expect(response.answers["department"]?.confidence == 0.93)
        #expect(abs((response.answers["late"]?.confidence ?? 0) - 0.96) < 1e-9)
        #expect(response.answers["severity"]?.score == 0.02)
        #expect(response.answers["severity"]?.legend == ["No impact.", "Minor.", "Major.", "Critical."])
        #expect(response.usage.cost == 0.000024402)
        #expect(response.answers.values.allSatisfy { $0.abstain == false })
    }

    @Test("Answers that do not fit their question are dropped so the cascade escalates them; bodies without answers are malformed")
    func dropsInvalid() throws {
        let bad = #"{"model":"m","answers":{"department":{"choice":"marketing","confidence":0.9},"late":{"noul":1.5},"severity":{"score":7,"confidence":0.9}},"usage":{"cost":-1}}"#
        let response = try #require(RemoteDecisionCore.response(Data(bad.utf8), for: Self.request))
        #expect(response.answers.isEmpty)
        #expect(response.usage.cost == nil)
        let unknownKey = #"{"model":"m","answers":{"department":{"choice":"sales","probabilities":{"sales":0.5,"spam":0.5}}}}"#
        #expect(RemoteDecisionCore.response(Data(unknownKey.utf8), for: Self.request)?.answers.isEmpty == true)
        #expect(RemoteDecisionCore.response(Data(#"{"error":{"message":"No endpoints"}}"#.utf8), for: Self.request) == nil)
        #expect(RemoteDecisionCore.response(Data("not json".utf8), for: Self.request) == nil)
    }

    @Test("The body is our request plus the model, with secrets in the state redacted before it leaves the machine")
    func bodyRedacts() throws {
        var request = Self.request
        request.state = "Use key sk-ant-api03-abcdefghijklmnopqrstuvwxyz to log in."
        let body = try #require(try JSONSerialization.jsonObject(with: try RemoteDecisionCore.body(request, model: "~typesafe/jev-latest")) as? [String: Any])
        #expect(body["model"] as? String == "~typesafe/jev-latest")
        #expect((body["state"] as? String)?.contains("sk-ant") == false)
        #expect((body["state"] as? String)?.contains("[REDACTED:anthropic_key]") == true)
        #expect(((body["questions"] as? [String: Any])?["severity"] as? [String: Any])?["type"] as? String == "score")
    }

    @Test("The key is required and never shows in the engine's description, debug description or dump")
    func keyIsHidden() throws {
        #expect(throws: BasaltError.self) { try RemoteDecisionEngine(model: "m", apiKey: "") }
        let engine = try RemoteStub.engine(URL(string: "http://127.0.0.1:1")!, "ok")
        var dumped = ""
        dump(engine, to: &dumped)
        for text in [String(describing: engine), String(reflecting: engine), dumped] {
            #expect(!text.contains(RemoteStub.key))
        }
    }

    @Test("A good answer comes back, and the key is sent only to the configured base URL")
    func answers() async throws {
        try await RemoteStub.serve { base, log in
            let response = try await RemoteStub.engine(base, "ok").decide(Self.request)
            #expect(response.answers["department"]?.choice == "account")
            #expect(response.usage.cost == 0.00001)
            let calls = log.calls.withLock { $0 }
            #expect(calls.map(\.path) == ["/ok/decisions"])
            #expect(calls.first?.authorization == "Bearer \(RemoteStub.key)")
            #expect(calls.first?.body?["model"] as? String == "~typesafe/jev-latest")
        }
    }

    @Test("A 5xx, malformed JSON, a timeout and a redirect fail with a short reason that never contains the key; a redirect is not followed", arguments: [
        ("fail", "server error"), ("garbage", "malformed"), ("hang", "timeout"), ("redirect", "refused"),
    ])
    func failures(route: String, reason: String) async throws {
        try await RemoteStub.serve { base, log in
            let engine = try RemoteStub.engine(base, route, timeout: route == "hang" ? 0.5 : 30)
            let error = await #expect(throws: BasaltError.self) { try await engine.decide(Self.request) }
            let message = error?.localizedDescription ?? ""
            #expect(message.contains(reason))
            #expect(!message.contains(RemoteStub.key))
            let paths = log.calls.withLock { $0.map(\.path) }
            #expect(route == "hang" ? paths.allSatisfy { $0 == "/hang/decisions" } : paths == ["/\(route)/decisions"])
        }
    }

    @Test("On each failure the cascade answers the whole request locally, and the session record names the fallback without the key", arguments: ["fail", "garbage", "hang", "redirect"])
    func cascadeFallsBack(route: String) async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "cascade-\(UUID().uuidString)")
        try await RemoteStub.serve { base, _ in
            let local = StubDecisionEngine()
            let cascade = CascadeDecisionEngine(remote: try RemoteStub.engine(base, route, timeout: 0.5), local: local, escalateBelow: 0.6, gate: GenerationGate())
            let writer = SessionLogWriter(mode: .full, directory: directory, defaultSession: "t")
            let response = try await SystemOneController.decide(cascade, Self.request, log: writer, origin: RequestOrigin(source: "systemone", endpoint: "/v1/systemone", stream: false), sessionId: nil)
            await writer.close()
            #expect(response.answers.count == 3)
            #expect(local.calls.requests.withLock { $0.first?.questions.count } == 3)
            #expect(response.usage.stages?.first?.fallback?.isEmpty == false)
        }
        let record = try #require(try SessionLogTests.lines(in: directory).first { $0["type"] as? String == "decision" })
        #expect(((record["stages"] as? [[String: Any]])?.first?["fallback"] as? String)?.contains("cloud decisions") == true)
        #expect(!SessionLogTests.rawText(in: directory).contains(RemoteStub.key))
    }
}
