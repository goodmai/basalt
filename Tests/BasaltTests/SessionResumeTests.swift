import Testing
import Foundation
import HTTPTypes
import ArgumentParser
@testable import BasaltCore
@testable import BasaltServer
@testable import BasaltCLI

final class ManualClock: @unchecked Sendable {
    private let lock = NSLock()
    private var current = Date(timeIntervalSince1970: 1_800_000_000)
    var now: Date { lock.withLock { current } }
    func advance(_ seconds: TimeInterval) { lock.withLock { current = current.addingTimeInterval(seconds) } }
}

@Suite("Session identity and resume (#106, part 4)", .serialized)
struct SessionResumeTests {
    static func turns(_ texts: [String]) -> [ChatMessageInput] {
        texts.enumerated().map { ChatMessageInput(role: $0.offset.isMultiple(of: 2) ? .user : .assistant, text: $0.element) }
    }

    static func exchange(_ writer: SessionLogWriter, _ texts: [String], id: String, session: String? = nil, reply: String = "ok") async {
        let request = GenerationRequest(prompt: "", messages: turns(texts)).withRouting(sessionId: session, budgetKey: nil)
        await writer.logRequest(request, id: id, origin: .unspecified, modelRequested: nil)
        var result = SessionLogResult()
        result.absorb(.text(reply))
        await writer.logResponse(request, id: id, status: "ok", result: result, queueWait: .zero)
    }

    static func files(_ dir: URL) -> [String] {
        SessionLogReader.sessions(in: dir).map(\.id).sorted()
    }

    @Test("X-Basalt-Session wins over x-session-id; either one names the session")
    func headerOrder() {
        var both = HTTPFields()
        both[GenerationRequest.basaltSessionHeader] = "primary"
        both[GenerationRequest.sessionHeader] = "alias"
        #expect(GenerationRequest(prompt: "p").routed(by: both).sessionId == "primary")
        var alias = HTTPFields()
        alias[GenerationRequest.sessionHeader] = "alias"
        #expect(GenerationRequest(prompt: "p").routed(by: alias).sessionId == "alias")
    }

    @Test("Without an id, a request that extends an earlier conversation joins its session; an unrelated one starts a new session")
    func inferredSessions() async throws {
        let dir = SessionLogTests.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let writer = SessionLogWriter(mode: .meta, directory: dir, defaultSession: "server")
        await Self.exchange(writer, ["hello"], id: "a1")
        await Self.exchange(writer, ["hello", "hi", "more"], id: "a2")
        await Self.exchange(writer, ["something else"], id: "b1")
        await writer.flush()
        let ids = Self.files(dir)
        #expect(ids.count == 2)
        #expect(ids.allSatisfy { $0.hasPrefix("s-") })
        let lines = ids.map { id in SessionLogReader.lines(of: SessionLogReader.file(for: id, in: dir)!) }
        #expect(Set(lines.map { $0.filter { $0.type == "request" }.count }) == [1, 2])
    }

    @Test("After a restart a known id appends to its file after session_resume and continues the parent chain")
    func resumeAfterRestart() async throws {
        let dir = SessionLogTests.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let first = SessionLogWriter(mode: .full, directory: dir, defaultSession: "server-1")
        await Self.exchange(first, ["q1"], id: "r1", session: "S")
        await Self.exchange(first, ["q1", "a1", "q2"], id: "r2", session: "S")
        await Self.exchange(first, ["q1", "a1", "q2", "a2", "q3"], id: "r3", session: "S")
        await first.close()

        let second = SessionLogWriter(mode: .full, directory: dir, defaultSession: "server-2")
        await Self.exchange(second, ["q1", "a1", "q2", "a2", "q3", "a3", "q4"], id: "r4", session: "S")
        await second.close()

        #expect(Self.files(dir) == ["S"])
        let lines = SessionLogReader.lines(of: try #require(SessionLogReader.file(for: "S", in: dir)))
        #expect(lines.filter { $0.type == "session_start" }.count == 1)
        let resumeIndex = try #require(lines.firstIndex { $0.type == "session_resume" })
        let r4Index = try #require(lines.firstIndex { $0.id == "r4" })
        #expect(resumeIndex < r4Index)
        let r4 = lines[r4Index]
        #expect(r4.content?.parent == "r3")
        #expect(r4.content?.prefixCount == 5)
        #expect(r4.content?.messages?.count == 2)
        #expect(SessionLogReader.conversation(of: "r4", in: lines)?.map(\.text) == ["q1", "a1", "q2", "a2", "q3", "a3", "q4"])
    }

    @Test("An idle session is suspended and the next request resumes it with the gap")
    func idleSuspendAndResume() async throws {
        let dir = SessionLogTests.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let clock = ManualClock()
        let writer = SessionLogWriter(mode: .meta, directory: dir, defaultSession: "server", idleTimeout: 60, clock: { clock.now })
        await Self.exchange(writer, ["hello"], id: "r1", session: "S")
        await writer.flush()
        clock.advance(30)
        await writer.flush()
        clock.advance(45)
        await writer.flush()
        await Self.exchange(writer, ["hello", "hi", "again"], id: "r2", session: "S")
        await writer.flush()
        let lines = SessionLogReader.lines(of: try #require(SessionLogReader.file(for: "S", in: dir)))
        let types = lines.map(\.type)
        #expect(types == ["session_start", "request", "response", "session_suspend", "session_resume", "request", "response"])
        let raw = try String(contentsOf: try #require(SessionLogReader.file(for: "S", in: dir)), encoding: .utf8)
        #expect(raw.contains(#""idle_s":75"#))
        #expect(raw.contains(#""gap_s":0"#))
    }

    @Test("basalt chat --resume rebuilds the history from a full session and refuses a meta one")
    func chatResume() async throws {
        let dir = SessionLogTests.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let full = SessionLogWriter(mode: .full, directory: dir, defaultSession: "x")
        await Self.exchange(full, ["first question"], id: "c1", session: "chat-full", reply: "first answer")
        await Self.exchange(full, ["first question", "first answer", "second question"], id: "c2", session: "chat-full", reply: "second answer")
        await full.close()
        let resumed = try ChatCommand.resumed("chat-full", in: dir)
        #expect(resumed.session == "chat-full")
        #expect(resumed.history.map(\.text) == ["first question", "first answer", "second question", "second answer"])
        #expect(ChatCommand.request("third", history: resumed.history, maxTokens: 64, sessionId: resumed.session).messages?.count == 5)

        let meta = SessionLogWriter(mode: .meta, directory: dir, defaultSession: "y")
        await Self.exchange(meta, ["private"], id: "m1", session: "chat-meta")
        await meta.close()
        #expect(throws: ValidationError.self) { _ = try ChatCommand.resumed("chat-meta", in: dir) }
        #expect(throws: ValidationError.self) { _ = try ChatCommand.resumed("missing", in: dir) }
        #expect(try SessionsCommand.resolve("latest", in: dir).id == "chat-meta")
    }
}
