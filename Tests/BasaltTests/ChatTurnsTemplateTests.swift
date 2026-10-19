import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
@testable import BasaltCore
@testable import BasaltInference

struct RoleTaggedInputProcessor: UserInputProcessor {
    static let generationPrompt = "<|im_start|>assistant\n"
    static let afterLastContent = "<|im_end|>\n" + generationPrompt

    static func render(_ messages: [[String: any Sendable]]) -> String {
        messages.map { message in
            let calls = (message["tool_calls"] as? [[String: any Sendable]] ?? [])
                .compactMap { ($0["function"] as? [String: any Sendable])?["name"] as? String }
                .map { "<tool_call>\($0)</tool_call>" }
            return "<|im_start|>\(message["role"] as? String ?? "")\n\(message["content"] as? String ?? "")\(calls.joined())<|im_end|>\n"
        }.joined() + generationPrompt
    }

    func prepare(input: UserInput) throws -> LMInput {
        let text = Self.render(DefaultMessageGenerator().generate(from: input))
        return LMInput(tokens: MLXArray(ByteTokenizer().encode(text: text, addSpecialTokens: false).map(Int32.init)))
    }
}

@Suite("Structured chat turns through the native chat template (#86)", .serialized)
struct ChatTurnsTemplateTests {
    static let firstTurn = [
        ChatMessageInput(role: .system, text: "You are a coding agent."),
        ChatMessageInput(role: .user, text: "list the files"),
    ]

    static func secondTurn(answer: String) -> [ChatMessageInput] {
        firstTurn + [
            ChatMessageInput(role: .assistant, text: answer),
            ChatMessageInput(role: .user, text: "and the hidden ones"),
        ]
    }

    static func request(_ messages: [ChatMessageInput], key: String?) -> GenerationRequest {
        GenerationRequest(prompt: ChatMessageInput.plainText(messages), maxTokens: 12, temperature: 0, cacheKey: key, messages: messages)
    }

    static func tokenCount(_ messages: [ChatMessageInput]) -> Int {
        RoleTaggedInputProcessor.render(DefaultMessageGenerator().generate(messages: MLXInferenceEngine.chat(messages, lastText: messages.last!.text))).unicodeScalars.count
    }

    static func engine(_ model: any LanguageModel) async -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(promptCacheBytes: 1 << 30)
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-role-tagged"),
            model: model,
            processor: RoleTaggedInputProcessor(),
            tokenizer: ByteTokenizer())))
        return engine
    }

    @Test("Each role maps to its Chat.Message; tool calls keep id and decoded arguments, free-text arguments survive under input")
    func messagesMapToChatRoles() throws {
        let chat = MLXInferenceEngine.chat([
            ChatMessageInput(role: .system, text: "s"),
            ChatMessageInput(role: .user, text: "u"),
            ChatMessageInput(role: .assistant, text: "a", toolCalls: [
                ToolCallOutput(id: "call_1", name: "Bash", arguments: #"{"command":"ls"}"#),
                ToolCallOutput(id: "", name: "apply_patch", arguments: "*** Begin Patch"),
            ]),
            ChatMessageInput(role: .tool, text: "README.md", toolCallId: "call_1", toolName: "Bash"),
        ], lastText: "replaced")
        let raw = DefaultMessageGenerator().generate(messages: chat)
        #expect(raw.map { $0["role"] as? String } == ["system", "user", "assistant", "tool"])
        #expect(raw.map { $0["content"] as? String } == ["s", "u", "a", "replaced"])
        let calls = try #require(raw[2]["tool_calls"] as? [[String: any Sendable]])
        #expect(calls.map { $0["id"] as? String } == ["call_1", nil])
        let functions = calls.compactMap { $0["function"] as? [String: any Sendable] }
        #expect((functions[0]["arguments"] as? [String: any Sendable])?["command"] as? String == "ls")
        #expect((functions[1]["arguments"] as? [String: any Sendable])?["input"] as? String == "*** Begin Patch")
        #expect(raw[3]["tool_call_id"] as? String == "call_1")
        #expect(raw[3]["name"] as? String == "Bash")
    }

    @Test("Turn 2 (history plus a new assistant and user turn) reuses the whole turn-1 prompt on an attention model")
    func secondTurnReusesFirstOnAttentionModel() async throws {
        let engine = await Self.engine(try TinyModel.make())
        let turn1 = try await engine.generate(request: Self.request(Self.firstTurn, key: "ns"))
        #expect(turn1.cachedPromptTokens == 0)
        #expect(turn1.promptTokens == Self.tokenCount(Self.firstTurn))
        #expect(!turn1.generatedText.isEmpty)

        let second = Self.secondTurn(answer: turn1.generatedText)
        let turn2 = try await engine.generate(request: Self.request(second, key: "ns"))
        let full2 = try await engine.generate(request: Self.request(second, key: nil))
        #expect(turn2.cachedPromptTokens == turn1.promptTokens)
        #expect(turn2.promptTokens == Self.tokenCount(second))
        #expect(turn2.generatedText == full2.generatedText)
    }

    @Test("On a hybrid model the snapshot sits at the end of the last user turn's text, so turn 2 reuses all but the generation-prompt suffix")
    func secondTurnReusesSnapshotOnHybridModel() async throws {
        let engine = await Self.engine(try TinyHybrid.make())
        let turn1 = try await engine.generate(request: Self.request(Self.firstTurn, key: "ns"))
        #expect(turn1.cachedPromptTokens == 0)

        let second = Self.secondTurn(answer: turn1.generatedText)
        let turn2 = try await engine.generate(request: Self.request(second, key: "ns"))
        let full2 = try await engine.generate(request: Self.request(second, key: nil))
        #expect(turn2.cachedPromptTokens == turn1.promptTokens - RoleTaggedInputProcessor.afterLastContent.count)
        #expect(turn2.generatedText == full2.generatedText)

        let third = second + [ChatMessageInput(role: .assistant, text: turn2.generatedText), ChatMessageInput(role: .user, text: "thanks")]
        let turn3 = try await engine.generate(request: Self.request(third, key: "ns"))
        #expect(turn3.cachedPromptTokens == turn2.promptTokens - RoleTaggedInputProcessor.afterLastContent.count)
        #expect(pinCount(await engine.promptCaches) == 0)
    }
}
