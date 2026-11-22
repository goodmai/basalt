import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

private struct InklingStubTokenizer: MLXLMCommon.Tokenizer, Sendable {
    static let messageModel = 200001
    static let text = 200004
    static let endSampling = 200006
    static let thinking = 200008
    static let endMessage = 200010
    static let toolJSON = 200049

    private let vocab: [Int: String] = [
        200001: "<|message_model|>", 200004: "<|content_text|>", 200006: "<|content_model_end_sampling|>",
        200008: "<|content_thinking|>", 200010: "<|end_message|>", 200049: "<|content_invoke_tool_json|>",
        11: "Check the forecast first. ", 12: "Then answer.", 13: "<|end_message|> is only text here.",
        14: "Sunny for three days.", 15: "weather.get", 16: #"{"name":"weather.get","args":{"days":3}}"#,
    ]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if let exact = vocab.first(where: { $0.value == text })?.key { return [exact] }
        return text.utf8.map { Int($0 % 200) + 1 }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { id in
            if skipSpecialTokens && id >= 200000 { return nil }
            return vocab[id] ?? UnicodeScalar(max(32, min(126, id))).map(String.init) ?? " "
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { vocab.first { $0.value == token && $0.key >= 200000 }?.key }
    func convertIdToToken(_ id: Int) -> String? { vocab[id] }
    var bosToken: String? { nil }
    var eosToken: String? { "<|content_model_end_sampling|>" }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [1, 2, 3]
    }
}

private struct ScriptedTokens: TokenIteratorProtocol {
    private var tokens: IndexingIterator<[Int]>
    let maxTokens: Int? = nil
    private(set) var tokenCount = 0
    let promptPrefillTime: TimeInterval = 0.001

    init(_ tokens: [Int]) { self.tokens = tokens.makeIterator() }

    mutating func next() -> Int? {
        guard let token = tokens.next() else { return nil }
        tokenCount += 1
        return token
    }
}

@Suite("Inkling 2 P3: message-frame protocol, reasoning tap and JSON tool calls (#650)", .serialized)
struct InklingProtocolTests {
    fileprivate typealias T = InklingStubTokenizer

    static let weatherTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": "weather.get", "description": "Weather",
            "parameters": [
                "type": "object", "properties": ["days": ["type": "integer"] as [String: any Sendable]] as [String: any Sendable],
                "required": ["days"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    static let turn = [
        T.messageModel, T.thinking, 11, 12, T.endMessage,
        T.messageModel, T.text, 13, 14, T.endMessage,
        T.messageModel, 15, T.toolJSON, 16, T.endMessage,
        T.endSampling,
    ]

    @Test("The reasoning parser reports only thinking frames, flags the first non-thinking frame, and treats marker-shaped text as text")
    func reasoningParser() throws {
        var parser = try #require(InklingReasoningStateMachine(tokenizer: T()))
        let steps = Self.turn.map { parser.push($0) }
        let reasoning = steps.compactMap { if case .reasoningChunk(let text) = $0 { text } else { nil } }.joined()
        #expect(reasoning == "Check the forecast first. Then answer.")
        #expect(steps.filter { $0 == .enteredNonReasoning }.count == 2)
        #expect(steps[6] == .enteredNonReasoning)
    }

    @Test("Inkling2Model selects the .inkling protocol, and the template's tool frame infers it too")
    func formatSelection() throws {
        let model = try #require(try TinyModel.make(.inkling2Loop) as? Inkling2Model)
        #expect(model.toolCallFormat == .inkling)
        let template = try String(contentsOf: Inkling2P0Tests.fixtures.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        #expect(ToolCallFormat.inferred(fromChatTemplate: template) == .inkling)
    }

    @Test("Through the relay a turn yields reasoning, text and a schema-checked tool call, with no protocol tokens in the content")
    func relayEndToEnd() async throws {
        let tokenizer = T()
        let context = ModelContext(
            configuration: ModelConfiguration(id: "tiny-inkling2"), model: try TinyModel.make(.inkling2Loop),
            processor: ByteInputProcessor(), tokenizer: tokenizer)
        #expect(MLXInferenceEngine.stopTokenIds(context).contains(T.endSampling))
        let (stream, task, prefill, box) = MLXInferenceEngine.handOffWithOnyx(
            promptTokenCount: 3, context: context, iterator: ScriptedTokens(Self.turn), tools: [Self.weatherTool], toolCallPolicy: .init())
        let container = ModelContainer(context: context)
        let engine = MLXInferenceEngine()
        await engine.attach(container, profile: InklingProfile())
        let request = GenerationRequest(
            prompt: "Weather?", maxTokens: 32,
            tools: [ToolDefinition(name: "weather.get", description: "Weather", parametersJSON: #"{"type":"object","properties":{"days":{"type":"integer"}},"required":["days"]}"#)])
        let relay = GenerationRelay(GenerationRelay.Setup(request: request, promptTokens: 3, onyxReasoning: box))
        let inputs = MLXInferenceEngine.RelayInputs(
            stream: stream, task: task, prefillSeconds: prefill, promptTokens: [1, 2, 3], namespace: nil, leaseID: 0,
            epoch: ModelEpoch(rawValue: 0), constraint: nil, container: container, faults: FaultPlan())
        let chunks = AsyncStream<StreamChunk> { continuation in
            Task { await engine.relay(relay, inputs, into: continuation) }
        }
        var (reasoning, answer, calls) = ("", "", [ToolCallOutput]())
        for await chunk in chunks {
            switch chunk {
            case .reasoning(let text): reasoning += text
            case .text(let text): answer += text
            case .toolCall(let call): calls.append(call)
            case .metadata: break
            }
        }
        #expect(reasoning == "Check the forecast first. Then answer.")
        #expect(answer == "<|end_message|> is only text here.Sunny for three days.")
        #expect(calls.map(\.name) == ["weather.get"])
        let arguments = try JSONSerialization.jsonObject(with: Data(try #require(calls.first).arguments.utf8)) as? [String: Int]
        #expect(arguments == ["days": 3])
    }
}
