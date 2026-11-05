import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCLI

@Suite("A long chat folds its oldest turns into a digest instead of failing past the KV room (#23)")
struct ChatHistoryDigestTests {
    static func exchanges(_ count: Int, size: Int = 800) -> ChatHistoryDigest {
        var digest = ChatHistoryDigest()
        for i in 0..<count {
            digest.record("question \(i) " + String(repeating: "q", count: size), "answer \(i) " + String(repeating: "a", count: size))
        }
        return digest
    }

    static func roomByBytes(fitting: Int) -> @Sendable (GenerationRequest) async -> Int {
        { request in fitting - ChatMessageInput.plainText(request.messages ?? []).utf8.count / 4 }
    }

    @Test("A history that fits is sent unchanged, with no digest")
    func fittingHistoryIsUntouched() async {
        var digest = Self.exchanges(3)
        let request = await digest.fitted("next", maxTokens: 4096, sessionId: "s", room: Self.roomByBytes(fitting: 100_000))
        #expect(digest.lines.isEmpty)
        #expect(request.messages == Self.exchanges(3).turns + [ChatMessageInput(role: .user, text: "next")])
    }

    @Test("An overflowing history folds the oldest half, keeps the newest turns verbatim and starts on a user turn")
    func overflowFoldsOldestHalf() async throws {
        var digest = Self.exchanges(20)
        let newest = digest.turns.suffix(4)
        let request = await digest.fitted("next", maxTokens: 4096, sessionId: "s", room: Self.roomByBytes(fitting: 5_000))
        #expect(await Self.roomByBytes(fitting: 5_000)(request) >= ChatHistoryDigest.replyFloor)
        #expect(digest.turns.count < 40)
        #expect(digest.turns.suffix(4) == newest)
        #expect(digest.turns.first?.role == .user)
        let messages = try #require(request.messages)
        #expect(messages.first?.text.hasPrefix(ChatHistoryDigest.header) == true)
        #expect(messages.first?.text.contains("user: question 0 ") == true)
        #expect(messages.last == ChatMessageInput(role: .user, text: "next"))
    }

    @Test("When nothing is left to fold the prompt carries the digest and the loop ends")
    func foldingStopsWhenHistoryIsEmpty() async throws {
        var digest = Self.exchanges(5)
        let request = await digest.fitted("next", maxTokens: 4096, sessionId: "s") { _ in 0 }
        #expect(digest.turns.isEmpty)
        let messages = try #require(request.messages)
        #expect(messages.count == 1)
        #expect(messages[0].text.hasPrefix(ChatHistoryDigest.header))
        #expect(messages[0].text.hasSuffix("\n\nnext"))
    }

    @Test("A resumed history that starts with a system message still leaves a user turn first")
    func foldRealignsOnUserTurn() {
        var digest = ChatHistoryDigest(turns: [ChatMessageInput(role: .system, text: "be brief")] + Self.exchanges(4).turns)
        #expect(digest.foldOlderHalf())
        #expect(digest.turns.first?.role == .user)
        #expect(digest.lines.first == "system: be brief")
    }

    @Test("Digest lines collapse whitespace, are cut at the line length, and keep only the newest lines")
    func digestLinesAreBounded() {
        let long = ChatMessageInput(role: .assistant, text: "a\n\n  b " + String(repeating: "x", count: 500))
        let line = ChatHistoryDigest.line(long)
        #expect(line.hasPrefix("assistant: a b x"))
        #expect(line.count == "assistant: ".count + ChatHistoryDigest.lineLength + 1)
        var digest = Self.exchanges(100, size: 10)
        while digest.foldOlderHalf() {}
        #expect(digest.lines.count == ChatHistoryDigest.maxLines)
        #expect(digest.lines.last?.hasPrefix("assistant: answer 99") == true)
    }

    @Test("Against the orchestrator's KV room: a chat past the budget folds until a reply fits")
    func foldsAgainstOrchestratorRoom() async throws {
        let fitting = 4_000
        let (orchestrator, _, dir) = try await MemoryCapTests.orchestrator(kvBudget: Int64(fitting) * MemoryCapTests.bytesPerToken)
        defer { try? FileManager.default.removeItem(at: dir) }
        var digest = Self.exchanges(20)
        let before = ChatCommand.request("next", history: digest.turns, maxTokens: 65_536, sessionId: "s")
        #expect(await orchestrator.effectiveMaxTokens(for: before) == 0)
        let request = await digest.fitted("next", maxTokens: 65_536, sessionId: "s") {
            await orchestrator.effectiveMaxTokens(for: $0)
        }
        #expect(!digest.lines.isEmpty)
        #expect(await orchestrator.effectiveMaxTokens(for: request) >= ChatHistoryDigest.replyFloor)
    }
}
