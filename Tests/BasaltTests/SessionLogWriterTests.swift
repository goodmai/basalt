import Testing
import Foundation
@testable import BasaltCore

@Suite("SessionLogWriter: state changes are synchronous reduces (#300)")
struct SessionLogWriterTests {
    struct Row: Sendable, CustomTestStringConvertible {
        let testDescription: String
        let events: [SessionLogState.Event]
        var pending: [String] = []
        var dropped = 0
        var disabled = false
        var handles: [String: Int32] = [:]
        var requests: [String: String] = [:]
    }

    static let line = Data("{}\n".utf8)
    static let t0 = Date(timeIntervalSince1970: 1_800_000_000)

    static let rows: [Row] = [
        Row(testDescription: "A full buffer counts a drop instead of appending",
            events: [.appended(session: "a", line: line, limit: 1), .appended(session: "b", line: line, limit: 1)],
            pending: ["a"], dropped: 1),
        Row(testDescription: "A line that failed to encode is neither buffered nor dropped",
            events: [.appended(session: "a", line: nil, limit: 4)]),
        Row(testDescription: "Reporting drops resets the count and buffers past the limit",
            events: [.appended(session: "a", line: line, limit: 0), .dropsReported(session: "server", line: line)],
            pending: ["server"]),
        Row(testDescription: "Draining empties the buffer and keeps the drop count",
            events: [.appended(session: "a", line: line, limit: 0), .appended(session: "b", line: line, limit: 4), .drained],
            dropped: 1),
        Row(testDescription: "A failed log drops its buffer and buffers nothing more",
            events: [.appended(session: "a", line: line, limit: 4), .failed, .appended(session: "b", line: line, limit: 4)],
            disabled: true),
        Row(testDescription: "A suspended session gives back its handle",
            events: [.opened(session: "a", fd: 3), .opened(session: "b", fd: 4), .suspended(session: "a", at: t0)],
            handles: ["b": 4]),
        Row(testDescription: "Closing forgets every handle",
            events: [.opened(session: "a", fd: 3), .closed]),
        Row(testDescription: "An answer forgets only its own request's session",
            events: [.started(id: "r1", session: "a", at: t0), .started(id: "r2", session: "b", at: t0),
                     .answered(id: "r1", session: "a", at: t0)],
            requests: ["r2": "b"]),
    ]

    static func turns(_ texts: [String]) -> [ChatMessageInput] {
        texts.enumerated().map { ChatMessageInput(role: $0.offset.isMultiple(of: 2) ? .user : .assistant, text: $0.element) }
    }

    @Test("reduce", arguments: rows)
    func reduce(_ row: Row) {
        var state = SessionLogState()
        for event in row.events {
            let next = SessionLogState.reduce(state, event)
            #expect(next.checkInvariants(after: state).isEmpty)
            state = next
        }
        #expect(state.pending.map(\.session) == row.pending)
        #expect(state.dropped == row.dropped)
        #expect(state.disabled == row.disabled)
        #expect(state.handles == row.handles)
        #expect(state.requestSessions == row.requests)
    }

    @Test("Started, resumed and remembered events record the session, its folder, its seeded turns and its new turn")
    func conversationEvents() {
        let url = URL(fileURLWithPath: "/logs/day/s.jsonl")
        let events: [SessionLogState.Event] = [
            .started(id: "r1", session: "s", at: Self.t0),
            .resumed(session: "s", from: url, seed: [("old", SessionConversations.digests(Self.turns(["a"])))]),
            .remembered(id: "r1", session: "s", digests: nil),
            .remembered(id: "r2", session: "s", digests: SessionConversations.digests(Self.turns(["a", "b", "c"]))),
        ]
        let state = events.reduce(SessionLogState(), SessionLogState.reduce)
        #expect(state.requestSessions == ["r1": "s"])
        #expect(state.resumedPaths == ["s": url])
        #expect(state.activity.lastActivity == ["s": Self.t0])
        #expect(state.activity.start(of: "s", at: Self.t0) == .seen)
        #expect(state.conversations.parent(of: SessionConversations.digests(Self.turns(["a", "b", "c", "d"])), in: "s")?.id == "r2")
        #expect(state.conversations.parent(of: SessionConversations.digests(Self.turns(["a", "x"])), in: "s")?.id == "old")
    }

    @Test("Two requests interleaved at the actor's await points: each response lands in its request's session, even after an idle flush suspended it")
    func interleavedRequests() async throws {
        let dir = SessionLogTests.tempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let clock = ManualClock()
        let writer = SessionLogWriter(mode: .full, directory: dir, defaultSession: "server", idle: .init(timeout: 60, clock: { clock.now }))
        let a = GenerationRequest(prompt: "", messages: Self.turns(["question a"])).withRouting(sessionId: "A", budgetKey: nil)
        let b = GenerationRequest(prompt: "", messages: Self.turns(["question b"]))
        var reply = SessionLogResult()
        reply.absorb(.text("ok"))
        var previous = await writer.state
        func checked() async -> SessionLogState {
            let state = await writer.state
            #expect(state.checkInvariants(after: previous).isEmpty)
            previous = state
            return state
        }

        await writer.logRequest(a, id: "ra", origin: .unspecified, modelRequested: nil)
        await writer.logRequest(b, id: "rb", origin: .unspecified, modelRequested: nil)
        let sessionB = try #require(await checked().requestSessions["rb"])
        await writer.flush()
        clock.advance(61)
        await writer.logResponse(b, id: "rb", status: "ok", result: reply, queueWait: .zero)
        await writer.flush()
        #expect(await checked().handles.keys.sorted() == [sessionB])
        await writer.logResponse(a, id: "ra", status: "ok", result: reply, queueWait: .zero)
        await writer.flush()
        #expect(await checked().requestSessions.isEmpty)
        await writer.close()

        let linesA = SessionLogReader.lines(of: try #require(SessionLogReader.file(for: "A", in: dir)))
        let linesB = SessionLogReader.lines(of: try #require(SessionLogReader.file(for: sessionB, in: dir)))
        #expect(linesA.map(\.type) == ["session_start", "request", "session_suspend", "response"])
        #expect(linesB.map(\.type) == ["session_start", "request", "response"])
    }
}
