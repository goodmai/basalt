import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCLI

@Suite("basalt chat reuses the prompt cache between turns (#174)", .serialized)
struct ChatPromptCacheTests {

    @Test("A chat request carries a per-session cache key, so the engine keeps its KV")
    func chatRequestCarriesSessionCacheKey() {
        let request = ChatCommand.request("hi", history: [], maxTokens: 8, sessionId: "chat-1")
        #expect(request.cacheKey == "cli:chat-1")
        #expect(request.sessionId == "chat-1")
        #expect(ChatCommand.request("hi", history: [], maxTokens: 8, sessionId: "chat-2").cacheKey != request.cacheKey)
    }

    @Test("Turn 2 of a basalt chat reuses turn 1's KV instead of prefilling the whole history again")
    func secondTurnReusesFirstTurnKV() async throws {
        let engine = try await TinyModel.engine(promptCacheBytes: 1 << 26)
        let first = ChatCommand.request(Conversation.firstTurn, history: [], maxTokens: 6, sessionId: "chat-t")
        let reply = try await engine.generate(request: first)
        #expect(reply.cachedPromptTokens == 0)

        let history = [ChatMessageInput(role: .user, text: Conversation.firstTurn), ChatMessageInput(role: .assistant, text: reply.generatedText)]
        let second = try await engine.generate(
            request: ChatCommand.request("and the tests", history: history, maxTokens: 6, sessionId: "chat-t"))
        #expect(second.cachedPromptTokens >= TinyModel.tokens(Conversation.firstTurn).count - 1)
    }
}
