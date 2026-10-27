import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import HTTPTypes
import NIOCore
@testable import BasaltCore
@testable import BasaltServer

@Suite("JSONL session log (#106, part 2)", .serialized)
struct SessionLogTests {
    static func tempDirectory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("sessions-\(UUID().uuidString)")
    }

    static func lines(in directory: URL) throws -> [[String: Any]] {
        let files = (FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL } ?? []).filter { $0.pathExtension == "jsonl" }.sorted { $0.path < $1.path }
        return try files.flatMap { file in
            try String(contentsOf: file, encoding: .utf8).split(separator: "\n").map {
                try #require(try JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any])
            }
        }
    }

    static func rawText(in directory: URL) -> String {
        (FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)?.compactMap { $0 as? URL } ?? [])
            .filter { $0.pathExtension == "jsonl" }
            .compactMap { try? String(contentsOf: $0, encoding: .utf8) }.joined()
    }

    static func server(_ mode: SessionLogMode, directory: URL, engine: MockInferenceEngine = MockInferenceEngine()) async throws
        -> (Application<RouterResponder<BasicRequestContext>>, ModelOrchestratorActor, SessionLogWriter)
    {
        let writer = SessionLogWriter(mode: mode, directory: directory, defaultSession: "test")
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker(), sessionLog: writer)
        try await orchestrator.loadModel(path: "mock")
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "mock"))
        return (Application(router: server.makeRouter()), orchestrator, writer)
    }

    static func post(_ client: some TestClientProtocol, _ uri: String, _ body: String, headers: HTTPFields = [:]) async throws -> HTTPResponse.Status {
        var all = headers
        all[.contentType] = "application/json"
        return try await client.execute(uri: uri, method: .post, headers: all, body: ByteBuffer(string: body)) { $0.status }
    }

    @Test("One record structure for all three protocols; only source and endpoint differ")
    func sameStructureAcrossProtocols() async throws {
        let dir = Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (app, _, writer) = try await Self.server(.full, directory: dir)
        try await app.test(.router) { client in
            let status1 = try await Self.post(client, "/v1/messages", #"{"max_tokens": 64, "messages": [{"role": "user", "content": "same prompt"}]}"#)
            #expect(status1 == .ok)
            let status2 = try await Self.post(client, "/v1/chat/completions", #"{"messages": [{"role": "user", "content": "same prompt"}]}"#)
            #expect(status2 == .ok)
            let status3 = try await Self.post(client, "/v1/responses", #"{"input": [{"role": "user", "content": "same prompt"}]}"#)
            #expect(status3 == .ok)
        }
        await writer.flush()
        let lines = try Self.lines(in: dir)
        let requests = lines.filter { $0["type"] as? String == "request" }
        let responses = lines.filter { $0["type"] as? String == "response" }
        #expect(requests.count == 3)
        #expect(responses.count == 3)
        #expect(requests.map { $0["source"] as? String } == ["anthropic", "openai-chat", "openai-responses"])
        #expect(Set(requests.map { Set($0.keys) }).count == 1)
        #expect(Set(responses.map { Set($0.keys) }).count == 1)
        let digests = Set(requests.compactMap { ($0["prompt"] as? [String: Any])?["sha256"] as? String })
        #expect(digests.count == 1)
        #expect(Set(responses.map { $0["request_id"] as? String }) == Set(requests.map { $0["id"] as? String }))
        #expect(responses.allSatisfy { $0["status"] as? String == "ok" && $0["tokens"] != nil && $0["timing"] != nil })
        #expect(lines.first?["type"] as? String == "session_start")
    }

    @Test("Only the orchestrator and the shared pipeline write the log, never a controller")
    func controllersStayThin() throws {
        let server = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BasaltServer")
        for file in ["REST/AnthropicController.swift", "REST/OpenAIController.swift", "REST/GenerateController.swift", "MCP/MCPServer.swift"] {
            let source = try String(contentsOf: server.appendingPathComponent(file), encoding: .utf8)
            #expect(!source.contains("sessionLog") && !source.contains("SessionLog"), "\(file)")
        }
    }

    @Test("meta carries metrics, never the prompt or the reply")
    func metaCarriesNoContent() async throws {
        let dir = Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let engine = MockInferenceEngine()
        await engine.setResponder { _ in "reply with secret-456" }
        let (app, _, writer) = try await Self.server(.meta, directory: dir, engine: engine)
        try await app.test(.router) { client in
            _ = try await Self.post(client, "/v1/chat/completions", #"{"messages": [{"role": "user", "content": "my secret-123"}]}"#)
        }
        await writer.flush()
        let text = Self.rawText(in: dir)
        #expect(!text.contains("secret-123"))
        #expect(!text.contains("secret-456"))
        let response = try #require(try Self.lines(in: dir).first { $0["type"] as? String == "response" })
        #expect(response["finish_reason"] as? String == "stop")
        #expect(response["tokens"] != nil && response["timing"] != nil)
    }

    @Test("full stores the conversation and the reply, in a 0700 directory and a 0600 file")
    func fullModeContentAndPermissions() async throws {
        let dir = Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (app, _, writer) = try await Self.server(.full, directory: dir)
        try await app.test(.router) { client in
            _ = try await Self.post(client, "/v1/chat/completions", #"{"messages": [{"role": "user", "content": "keep me"}]}"#)
        }
        await writer.flush()
        let lines = try Self.lines(in: dir)
        let request = try #require(lines.first { $0["type"] as? String == "request" })
        let messages = try #require((request["content"] as? [String: Any])?["messages"] as? [[String: Any]])
        #expect(messages.first?["text"] as? String == "keep me")
        let response = try #require(lines.first { $0["type"] as? String == "response" })
        #expect((response["output"] as? [String: Any])?["text"] as? String == "Mock response")
        let fm = FileManager.default
        #expect((try fm.attributesOfItem(atPath: dir.path)[.posixPermissions] as? Int) == 0o700)
        let file = try #require(fm.enumerator(at: dir, includingPropertiesForKeys: nil)?.compactMap { $0 as? URL }.first { $0.pathExtension == "jsonl" })
        #expect((try fm.attributesOfItem(atPath: file.path)[.posixPermissions] as? Int) == 0o600)
        #expect((try fm.attributesOfItem(atPath: file.deletingLastPathComponent().path)[.posixPermissions] as? Int) == 0o700)
    }

    @Test("A rejected request leaves one error line with body_bytes and neither header value nor body text")
    func rejectionsHideHeadersAndBody() async throws {
        let dir = Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (app, _, writer) = try await Self.server(.full, directory: dir)
        var secretHeaders = HTTPFields()
        secretHeaders[.authorization] = "Bearer header-secret-1"
        secretHeaders[HTTPField.Name("x-api-key")!] = "header-secret-2"
        let headers = secretHeaders
        let body = "{not json body-secret-3"
        try await app.test(.router) { client in
            let status = try await Self.post(client, "/v1/messages", body, headers: headers)
            #expect(status == .badRequest)
        }
        await writer.flush()
        let lines = try Self.lines(in: dir).filter { $0["type"] as? String != "session_start" }
        #expect(lines.count == 1)
        #expect(lines.first?["type"] as? String == "error")
        #expect(lines.first?["body_bytes"] as? Int == body.utf8.count)
        #expect(lines.first?["code"] as? Int == 400)
        let text = Self.rawText(in: dir)
        #expect(!text.contains("header-secret") && !text.contains("body-secret"))
    }

    @Test("A streaming client that drops mid-generation leaves a cancelled response after its request")
    func cancelledStream() async throws {
        let dir = Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let engine = MockInferenceEngine()
        await engine.setArtificialDelay(5)
        let writer = SessionLogWriter(mode: .full, directory: dir, defaultSession: "test")
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker(), sessionLog: writer)
        try await orchestrator.loadModel(path: "mock")
        var stream: AsyncStream<StreamChunk>? = try await orchestrator.generateStream(
            request: GenerationRequest(prompt: "long").withOrigin(RequestOrigin(source: "openai-chat", endpoint: "/v1/chat/completions", stream: true)))
        _ = stream
        stream = nil
        let deadline = ContinuousClock.now + .seconds(3)
        var lines: [[String: Any]] = []
        while ContinuousClock.now < deadline {
            await writer.flush()
            lines = (try? Self.lines(in: dir)) ?? []
            if lines.contains(where: { $0["type"] as? String == "response" }) { break }
            try await Task.sleep(for: .milliseconds(50))
        }
        let request = try #require(lines.first { $0["type"] as? String == "request" })
        let response = try #require(lines.first { $0["type"] as? String == "response" })
        #expect(response["status"] as? String == "cancelled")
        #expect(response["request_id"] as? String == request["id"] as? String)
        #expect(request["stream"] as? Bool == true)
    }

    @Test("An unwritable log directory disables the log once and never breaks a request")
    func failOpen() async throws {
        let parent = Self.tempDirectory()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: parent.path)
            try? FileManager.default.removeItem(at: parent)
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o500], ofItemAtPath: parent.path)
        let (app, _, writer) = try await Self.server(.full, directory: parent.appendingPathComponent("sessions"))
        try await app.test(.router) { client in
            let status4 = try await Self.post(client, "/v1/chat/completions", #"{"messages": [{"role": "user", "content": "hi"}]}"#)
            #expect(status4 == .ok)
            let status5 = try await Self.post(client, "/v1/chat/completions", #"{"messages": [{"role": "user", "content": "hi"}]}"#)
            #expect(status5 == .ok)
        }
        #expect(await !writer.isEnabled)
    }

    @Test("Overflow drops records and says how many")
    func overflowIsCounted() async throws {
        let dir = Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let writer = SessionLogWriter(mode: .meta, directory: dir, bufferLimit: 2, defaultSession: "test")
        for i in 0..<5 {
            await writer.logRejection(origin: .unspecified, sessionId: nil, code: 400, message: "m\(i)", bodyBytes: 1)
        }
        await writer.flush()
        let lines = try Self.lines(in: dir)
        #expect(lines.filter { $0["type"] as? String == "error" }.count == 2)
        #expect(lines.first { $0["type"] as? String == "dropped" }?["count"] as? Int == 3)
    }

    @Test("Model switches are recorded with their duration")
    func modelSwitch() async throws {
        let dir = Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let writer = SessionLogWriter(mode: .meta, directory: dir, defaultSession: "test")
        await writer.logModelSwitch(from: "a", to: "b", duration: .milliseconds(1_250))
        await writer.flush()
        let event = try #require(try Self.lines(in: dir).first { $0["type"] as? String == "model_switch" })
        #expect(event["from"] as? String == "a" && event["to"] as? String == "b" && event["duration_ms"] as? Int == 1_250)
    }

    @Test("Retention removes files older than the window, then the oldest beyond the size cap")
    func retention() throws {
        let dir = Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let day = dir.appendingPathComponent("2026-10-01")
        try FileManager.default.createDirectory(at: day, withIntermediateDirectories: true)
        let now = Date()
        func file(_ name: String, ageDays: Double, bytes: Int) throws -> URL {
            let url = day.appendingPathComponent(name)
            try Data(count: bytes).write(to: url)
            try FileManager.default.setAttributes([.modificationDate: now.addingTimeInterval(-ageDays * 86_400)], ofItemAtPath: url.path)
            return url
        }
        let old = try file("old.jsonl", ageDays: 2, bytes: 10)
        let olderBig = try file("older-big.jsonl", ageDays: 0.5, bytes: 600)
        let newest = try file("new.jsonl", ageDays: 0.1, bytes: 600)
        SessionLogWriter.applyRetention(directory: dir, days: 1, maxTotalBytes: 1_000, now: now)
        let fm = FileManager.default
        #expect(!fm.fileExists(atPath: old.path))
        #expect(!fm.fileExists(atPath: olderBig.path))
        #expect(fm.fileExists(atPath: newest.path))
    }

    @Test("The writer never touches stdout, which MCP stdio uses for JSON-RPC")
    func noStdout() throws {
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BasaltCore/SessionLog.swift"), encoding: .utf8)
        for marker in ["print(", "stdout", "STDOUT_FILENO", "FileHandle.standardOutput"] {
            #expect(!source.contains(marker), "\(marker)")
        }
    }
}
