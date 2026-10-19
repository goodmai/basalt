import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
import Jinja
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

struct NotATemplateError: Error {}

func templateRefusal(_ message: String) -> any Error {
    do {
        _ = try Template("{{ raise_exception('\(message)') }}").render([:])
        return NotATemplateError()
    } catch {
        return error
    }
}

struct NonLeadingSystemRefusingInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        guard !messages.dropFirst().contains(where: { $0["role"] as? String == "system" }) else {
            throw templateRefusal("System message must be at the beginning.")
        }
        let text = RoleTaggedInputProcessor.render(messages)
        return LMInput(tokens: MLXArray(ByteTokenizer().encode(text: text, addSpecialTokens: false).map(Int32.init)))
    }
}

struct FailingInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput { throw NotATemplateError() }
}

struct StrictAlternationInputProcessor: UserInputProcessor {
    var refusesEverything = false
    var refusesAsRuntimeError = false

    func prepare(input: UserInput) throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let roles = messages.compactMap { $0["role"] as? String }
        let dialogue = roles.drop { $0 == "system" }
        let alternates = dialogue.enumerated().allSatisfy { $1 == ($0 % 2 == 0 ? "user" : "assistant") }
        let hasToolCalls = messages.contains { $0["tool_calls"] != nil }
        guard !refusesEverything, alternates, !dialogue.isEmpty, !dialogue.contains("system"), !hasToolCalls else {
            if refusesAsRuntimeError { throw JinjaError.runtime("Conversation roles must alternate") }
            throw templateRefusal("Conversation roles must alternate user/assistant/user/assistant/...")
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

    @Test("A template that refuses with a Jinja runtime error also falls back to the alternating turns")
    func runtimeTemplateErrorFallsBack() async throws {
        let engine = await Self.engine(try TinyModel.make(), processor: StrictAlternationInputProcessor(refusesAsRuntimeError: true))
        let response = try await engine.generate(request: Self.request(Self.agentLoop, key: "ns"))
        #expect(response.promptTokens == Self.tokenCount(ChatMessageInput.alternating(Self.agentLoop)))
    }

    @Test("A template that only rejects a non-leading system answers after system placement, with tool calls still structured")
    func systemPlacementKeepsToolCallsStructured() async throws {
        let engine = await Self.engine(try TinyModel.make(), processor: NonLeadingSystemRefusingInputProcessor())
        let response = try await engine.generate(request: Self.request(Self.agentLoop, key: "ns"))
        let placed = ChatMessageInput.leadingSystem(Self.agentLoop)
        #expect(!response.generatedText.isEmpty)
        #expect(response.promptTokens == Self.tokenCount(placed))
        #expect(response.promptTokens != Self.tokenCount(ChatMessageInput.alternating(Self.agentLoop)))
        let rendered = RoleTaggedInputProcessor.render(DefaultMessageGenerator().generate(messages: MLXInferenceEngine.chat(placed, lastText: placed.last!.text)))
        #expect(rendered.contains("<tool_call>Bash</tool_call>"))
        #expect(rendered.contains("<|im_start|>tool\nREADME.md"))
    }

    @Test("A prepare error that is not a template refusal stays a 503")
    func nonTemplatePrepareErrorIsServiceUnavailable() async throws {
        let engine = await Self.engine(try TinyModel.make(), processor: FailingInputProcessor())
        do {
            _ = try await engine.generate(request: Self.request(Self.agentLoop, key: "ns"))
            Issue.record("expected the request to fail")
        } catch {
            #expect(error.httpStatus == 503)
        }
    }

    @Test("A template that rejects even the alternating turns is a 400, not a hardware failure")
    func templateRejectingAlternatingTurnsIsBadRequest() async throws {
        let engine = await Self.engine(try TinyModel.make(), processor: StrictAlternationInputProcessor(refusesEverything: true))
        do {
            _ = try await engine.generate(request: Self.request(Self.agentLoop, key: "ns"))
            Issue.record("expected the request to be refused")
        } catch {
            #expect(error.httpStatus == 400)
            #expect(String(describing: error).contains("after normalising"))
        }
    }
}

@Suite("Alternating chat turns for strict templates (#86)")
struct AlternatingChatTurnsTests {
    typealias M = ChatMessageInput

    @Test("Systems before the first user merge into the leading system; later ones become user text in place")
    func systemsBeforeFirstUserLeadLaterOnesStayInPlace() {
        let turns = M.alternating([
            M(role: .system, text: "a"), M(role: .system, text: "b"), M(role: .user, text: "u"),
            M(role: .assistant, text: "x"), M(role: .system, text: "now summarize"),
        ])
        #expect(turns == [
            M(role: .system, text: "a\n\nb"), M(role: .user, text: "u"),
            M(role: .assistant, text: "x"), M(role: .user, text: "System: now summarize"),
        ])
    }

    @Test("A system right after a user turn merges into that user turn")
    func midSystemMergesIntoAdjacentUser() {
        let turns = M.leadingSystem([M(role: .user, text: "u"), M(role: .system, text: "b"), M(role: .user, text: "v")])
        #expect(turns == [M(role: .user, text: "u\n\nSystem: b\n\nv")])
    }

    @Test("System placement keeps tool messages and tool calls as they are")
    func leadingSystemPreservesToolTurns() {
        let call = ToolCallOutput(id: "call_1", name: "Bash", arguments: #"{"command":"ls"}"#)
        let tool = M(role: .tool, text: "README.md", toolCallId: "call_1", toolName: "Bash")
        let turns = M.leadingSystem([
            M(role: .system, text: "s"), M(role: .user, text: "u"),
            M(role: .assistant, text: "", toolCalls: [call]), tool, M(role: .system, text: "be brief"),
        ])
        #expect(turns == [
            M(role: .system, text: "s"), M(role: .user, text: "u"),
            M(role: .assistant, text: "", toolCalls: [call]), tool, M(role: .user, text: "System: be brief"),
        ])
    }

    @Test("A leading assistant turn folds into the first user turn so the dialogue opens with user")
    func leadingAssistantFoldsIntoFirstUser() {
        #expect(M.alternating([M(role: .system, text: "s"), M(role: .assistant, text: "hi"), M(role: .user, text: "u")])
            == [M(role: .system, text: "s"), M(role: .user, text: "Assistant: hi\n\nu")])
        #expect(M.alternating([M(role: .assistant, text: "hi"), M(role: .user, text: "u"), M(role: .assistant, text: "a")])
            == [M(role: .user, text: "Assistant: hi\n\nu"), M(role: .assistant, text: "a")])
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
