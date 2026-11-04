import Foundation
import Testing
import MLXLMCommon
import Tokenizers
@testable import BasaltCore
@testable import BasaltInference

@Suite("Real chat templates render a fixed conversation to the committed golden prompt, which round-trips through the tokenizer (#327)", .serialized)
struct TemplateGoldenTests {
    static let conversation = [
        ChatMessageInput(role: .system, text: "You are a coding agent."),
        ChatMessageInput(role: .user, text: "Is it warm in Paris?"),
        ChatMessageInput(role: .assistant, text: "<think>\nThe user asks about Paris; I should check the weather.\n</think>\n\nLet me check."),
        ChatMessageInput(role: .user, text: "Please do."),
        ChatMessageInput(role: .assistant, text: "", toolCalls: [
            ToolCallOutput(id: "call_1", name: "get_weather", arguments: #"{"city": "Paris", "unit": "celsius"}"#),
        ]),
        ChatMessageInput(role: .tool, text: #"{"temp": 21}"#, toolCallId: "call_1", toolName: "get_weather"),
    ]

    static let cachedRepos = TemplateGoldens.byRepo.keys.sorted().filter { ControlTokenGuardTests.cachedModel($0) != nil }

    @Test("The prompt the template renders through the engine's tokenizer is the golden string: detokenizing it gives the golden back, and tokenizing the golden gives the same tokens",
          .enabled(if: !cachedRepos.isEmpty, "needs a cached model template"), arguments: cachedRepos)
    func rendersGolden(repo: String) async throws {
        let directory = try #require(ControlTokenGuardTests.cachedModel(repo))
        let golden = try #require(TemplateGoldens.byRepo[repo])
        let tokenizer = try await MLXInferenceEngine.upstreamTokenizer(at: directory)
        let tools = MLXInferenceEngine.toolSpecs([ChatTemplateFidelityTests.weatherTool])
        let chat = MLXInferenceEngine.chat(Self.conversation, lastText: Self.conversation[Self.conversation.count - 1].text)
        let messages = DefaultMessageGenerator().generate(from: UserInput(chat: chat, tools: tools))
        let tokens = try tokenizer.applyChatTemplate(messages: messages, tools: tools)
        #expect(tokenizer.decode(tokens: tokens) == golden, "\(repo)")
        #expect(tokens == tokenizer.encode(text: golden, addSpecialTokens: false), "\(repo)")
    }
}
