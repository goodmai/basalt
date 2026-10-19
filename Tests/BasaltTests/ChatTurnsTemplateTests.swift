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

struct StrictAlternationError: Error {}

struct StrictAlternationInputProcessor: UserInputProcessor {
    var refusesEverything = false

    func prepare(input: UserInput) throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let roles = messages.compactMap { $0["role"] as? String }
        let dialogue = roles.drop { $0 == "system" }
        let alternates = dialogue.enumerated().allSatisfy { $1 == ($0 % 2 == 0 ? "user" : "assistant") }
        let hasToolCalls = messages.contains { $0["tool_calls"] != nil }
        guard !refusesEverything, alternates, !dialogue.isEmpty, !dialogue.contains("system"), !hasToolCalls else {
            throw StrictAlternationError()
        }
        let text = RoleTaggedInputProcessor.render(messages)
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

    static func engine(_ model: any LanguageModel, processor: any UserInputProcessor = RoleTaggedInputProcessor()) async -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(promptCacheBytes: 1 << 30)
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-role-tagged"),
            model: model,
            processor: processor,
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

    static let agentLoop = [
        ChatMessageInput(role: .system, text: "instructions"),
        ChatMessageInput(role: .system, text: "developer note"),
        ChatMessageInput(role: .user, text: "list the files"),
        ChatMessageInput(role: .assistant, text: "", toolCalls: [ToolCallOutput(id: "call_1", name: "Bash", arguments: #"{"command":"ls"}"#)]),
        ChatMessageInput(role: .tool, text: "README.md", toolCallId: "call_1", toolName: "Bash"),
        ChatMessageInput(role: .system, text: "be brief"),
    ]

    @Test("A template that demands alternation, a leading system and no tool role still answers, through the alternating turns")
    func strictTemplateFallsBackToAlternatingTurns() async throws {
        let engine = await Self.engine(try TinyModel.make(), processor: StrictAlternationInputProcessor())
        let response = try await engine.generate(request: Self.request(Self.agentLoop, key: "ns"))
        #expect(!response.generatedText.isEmpty)
        #expect(response.promptTokens == Self.tokenCount(ChatMessageInput.alternating(Self.agentLoop)))
    }

    @Test("A template that rejects even the alternating turns is a 400, not a hardware failure")
    func templateRejectingAlternatingTurnsIsBadRequest() async throws {
        let engine = await Self.engine(try TinyModel.make(), processor: StrictAlternationInputProcessor(refusesEverything: true))
        do {
            _ = try await engine.generate(request: Self.request(Self.agentLoop, key: "ns"))
            Issue.record("expected the request to be refused")
        } catch {
            #expect(error.httpStatus == 400)
        }
    }
}

@Suite("Alternating chat turns for strict templates (#86)")
struct AlternatingChatTurnsTests {
    typealias M = ChatMessageInput

    @Test("Every system message merges into one leading system turn")
    func systemsMergeToFront() {
        let turns = M.alternating([M(role: .system, text: "a"), M(role: .user, text: "u"), M(role: .system, text: "b")])
        #expect(turns == [M(role: .system, text: "a\n\nb"), M(role: .user, text: "u")])
    }

    @Test("A tool turn becomes user text naming the tool")
    func toolBecomesUserText() {
        let turns = M.alternating([
            M(role: .user, text: "u"), M(role: .assistant, text: "a"),
            M(role: .tool, text: "README.md", toolCallId: "call_1", toolName: "Bash"),
        ])
        #expect(turns.last == M(role: .user, text: "Tool result (Bash): README.md"))
    }

    @Test("Assistant tool calls become name(arguments) text on the assistant turn")
    func toolCallsBecomeText() {
        let turns = M.alternating([
            M(role: .user, text: "u"),
            M(role: .assistant, text: "checking", toolCalls: [ToolCallOutput(id: "c", name: "Bash", arguments: #"{"command":"ls"}"#)]),
        ])
        #expect(turns.last == M(role: .assistant, text: "checking\nBash({\"command\":\"ls\"})"))
    }

    @Test("Adjacent turns of one role merge with a blank line")
    func adjacentSameRoleMerge() {
        let turns = M.alternating([
            M(role: .user, text: "u"),
            M(role: .assistant, text: "", toolCalls: [ToolCallOutput(id: "c", name: "Bash", arguments: "{}")]),
            M(role: .tool, text: "x", toolName: "Bash"), M(role: .user, text: "next"),
        ])
        #expect(turns.map(\.role) == [.user, .assistant, .user])
        #expect(turns.last?.text == "Tool result (Bash): x\n\nnext")
    }

    @Test("Without any user turn the whole conversation goes as one user message")
    func noUserTurnIsOneUserMessage() {
        let messages = [M(role: .system, text: "only instructions")]
        #expect(M.alternating(messages) == [M(role: .user, text: "only instructions")])
    }
}
