import Testing
import Foundation
@testable import BasaltCore

@Suite("Session log value core: formatting, redaction, conversations, idle rotation, retention")
struct SessionLogCoreTests {
    static let secret = "sk-" + String(repeating: "a1B2", count: 10)
    static let stamp = SessionLogRecord.Stamp(ts: "2026-11-04T00:00:00.000Z", session: "s")

    static func turns(_ texts: [String]) -> [ChatMessageInput] {
        texts.enumerated().map { ChatMessageInput(role: $0.offset.isMultiple(of: 2) ? .user : .assistant, text: $0.element) }
    }

    @Test("Meta mode keeps the prompt digest and drops the content")
    func metaRequest() {
        let request = GenerationRequest(prompt: "", messages: Self.turns(["hi \(Self.secret)"]))
        let record = SessionLogFormat(mode: .meta, redacts: true)
            .request(request, id: "r1", origin: .unspecified, modelRequested: "m", parent: nil, at: Self.stamp)
        #expect(record.content == nil)
        #expect(record.prompt == SessionLogRecord.Digest("hi \(Self.secret)"))
        #expect(record.messagesCount == 1)
        #expect(record.ts == Self.stamp.ts && record.session == "s" && record.type == "request")
    }

    @Test("Full mode stores only the turns after the parent prefix, redacted unless redaction is off", arguments: [true, false])
    func fullRequest(redacts: Bool) throws {
        let request = GenerationRequest(prompt: "", messages: Self.turns(["a", "b", "key \(Self.secret)"]))
        let format = SessionLogFormat(mode: .full, redacts: redacts)
        let content = try #require(
            format.request(request, id: "r2", origin: .unspecified, modelRequested: nil, parent: ("r1", 2), at: Self.stamp).content)
        #expect(content.parent == "r1")
        #expect(content.prefixCount == 2)
        let text = try #require(content.messages?.map(\.text))
        #expect(text.count == 1)
        #expect(text[0].contains(Self.secret) == !redacts)
    }

    @Test("A prompt-only request keeps its scrubbed prompt and no messages")
    func promptContent() {
        let content = SessionLogFormat(mode: .full, redacts: true)
            .content(of: GenerationRequest(prompt: "p \(Self.secret)"), parent: nil)
        #expect(content.messages == nil)
        #expect(content.prompt?.contains("[REDACTED:openai_key]") == true)
    }

    @Test("Output text appears only in full mode, scrubbed; empty reasoning and tool calls stay absent")
    func output() {
        var result = SessionLogResult()
        result.absorb(.text("out \(Self.secret)"))
        let meta = SessionLogFormat(mode: .meta, redacts: true).output(of: result)
        #expect(meta.text == nil)
        #expect(meta.chars == result.text.count)
        let full = SessionLogFormat(mode: .full, redacts: true).output(of: result)
        #expect(full.text?.contains(Self.secret) == false)
        #expect(full.reasoning == nil)
        #expect(full.toolCalls == nil)
        #expect(full.sha256 == SessionLogRecord.Digest(result.text).sha256)
    }

    @Test("A response without metadata has no tokens or timing and carries the error")
    func responseWithoutMetadata() {
        let record = SessionLogFormat(mode: .meta, redacts: true).response(
            id: "r", status: "error", result: SessionLogResult(), queueWait: .zero, error: "boom", at: Self.stamp)
        #expect(record.tokens == nil && record.timing == nil && record.memory == nil)
        #expect(record.error == "boom")
        #expect(record.requestId == "r")
    }

    @Test("Durations round down to whole milliseconds")
    func milliseconds() {
        #expect(SessionLogFormat.milliseconds(.milliseconds(1_250)) == 1_250)
        #expect(SessionLogFormat.milliseconds(.microseconds(1_999)) == 1)
        #expect(SessionLogFormat.milliseconds(.zero) == 0)
    }

    @Test("Records encode as one newline-terminated line; sorted lines keep slashes, plain lines escape them")
    func lineEncoding() throws {
        let event = SessionLogRecord.Event("session_start", at: .init(ts: "t", session: "a/b"))
        let sorted = try #require(SessionLogFormat.line(event))
        let plain = try #require(SessionLogFormat.line(event, sorted: false))
        #expect(sorted.last == 0x0A && plain.last == 0x0A)
        #expect(String(decoding: sorted, as: UTF8.self) == #"{"session":"a/b","ts":"t","type":"session_start","v":1}"# + "\n")
        #expect(String(decoding: plain, as: UTF8.self).contains(#""a\/b""#))
    }

    @Test("A conversation that extends an earlier one finds its session and parent; the window keeps the last eight")
    func conversations() {
        var index = SessionConversations()
        let none = index.remember(nil, id: "x", session: "s1")
        let first = index.remember(Self.turns(["a"]), id: "r1", session: "s1")
        let parent = index.remember(Self.turns(["a", "b", "c"]), id: "r2", session: "s1")
        #expect(none == nil && first == nil)
        #expect(parent?.id == "r1" && parent?.count == 1)
        #expect(index.session(continuing: SessionConversations.digests(Self.turns(["a", "b", "c", "d"]))) == "s1")
        #expect(index.session(continuing: SessionConversations.digests(Self.turns(["z"]))) == nil)
        for i in 0..<SessionConversations.window { _ = index.remember(Self.turns(["w\(i)"]), id: "w\(i)", session: "s1") }
        let evicted = index.remember(Self.turns(["a", "b", "c", "d"]), id: "r3", session: "s1")
        #expect(evicted == nil)
        #expect(index.knows("s1") && !index.knows("s2"))
    }

    @Test("A session starts once, resumes with its gap after a suspend, and is idle only at the timeout")
    func activity() {
        var activity = SessionActivity()
        let t0 = Date(timeIntervalSince1970: 1_000)
        let starts = [activity.start("s", at: t0), activity.start("s", at: t0)]
        #expect(starts == [.firstSeen, .seen])
        #expect(SessionActivity.idleSeconds(since: t0, now: t0.addingTimeInterval(59.9), timeout: 60) == nil)
        #expect(SessionActivity.idleSeconds(since: t0, now: t0.addingTimeInterval(60), timeout: 60) == 60)
        activity.suspend("s", at: t0.addingTimeInterval(61))
        #expect(activity.lastActivity["s"] == nil)
        let resumed = activity.start("s", at: t0.addingTimeInterval(161))
        #expect(resumed == .resumed(gapS: 100))
        #expect(activity.lastActivity["s"] == t0.addingTimeInterval(161))
        activity.touch("s", at: t0)
        #expect(activity.lastActivity["s"] == t0)
    }

    @Test("Retention deletes expired files first, then the oldest until the total fits the cap")
    func retention() {
        let now = Date(timeIntervalSince1970: 10_000_000)
        func file(_ name: String, ageDays: Double, bytes: Int64) -> SessionLogRetention.File {
            .init(url: URL(fileURLWithPath: "/logs/\(name).jsonl"), modified: now.addingTimeInterval(-ageDays * 86_400), size: bytes)
        }
        let old = file("old", ageDays: 2, bytes: 10)
        let olderBig = file("older-big", ageDays: 0.5, bytes: 600)
        let newest = file("new", ageDays: 0.1, bytes: 600)
        let files = [newest, old, olderBig]
        #expect(SessionLogRetention(days: 1, maxTotalBytes: 1_000).deletions(among: files, now: now) == [old.url, olderBig.url])
        #expect(SessionLogRetention(days: 1, maxTotalBytes: 1_200).deletions(among: files, now: now) == [old.url])
        #expect(SessionLogRetention(days: 3, maxTotalBytes: 0).deletions(among: files, now: now) == [old.url, olderBig.url, newest.url])
        #expect(SessionLogRetention(days: -5, maxTotalBytes: .max).deletions(among: files, now: now) == [newest.url, old.url, olderBig.url])
    }
}
