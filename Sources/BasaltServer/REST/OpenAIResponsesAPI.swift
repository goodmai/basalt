import Foundation
import Hummingbird
import BasaltCore

extension OpenAIController {
    @Sendable
    func responses(request: Request, context: BasicRequestContext) async throws -> Response {
        let pipeline = self.pipeline(source: "openai-responses", endpoint: "/v1/responses")
        return try await pipeline.handle(request) { data in
            let body = try pipeline.decode(ChatPipeline.Body<ResponsesRequest>.self, from: data, failure: "Cannot decode responses request")
            let dto = body.dto
            let conversation = try pipeline.parsed(dto.conversation)
            try pipeline.require(!conversation.isEmpty, "input or messages must not be empty")
            let json = ChatPipeline.object(in: data)
            let tools = try pipeline.tools(in: json)
            let format = try pipeline.responseFormat(in: json, data: data)
            let stream = dto.stream == true
            let prepared = try pipeline.prepare(ChatPipeline.Input(
                chat: dto.instructionMessages + conversation,
                maxTokens: dto.maxOutputTokens ?? dto.maxTokens, sampling: dto.sampling(try pipeline.antiRepetition(body.antiRepetition, bodyBytes: data.count)),
                reasoningEffort: dto.effectiveReasoningEffort, tools: tools, thinkingBudget: dto.thinkingBudget,
                responseFormat: format, model: dto.model, stream: stream), headers: request.headers)
            if stream { try await pipeline.admitStream(prepared.request) }
            let model = dto.model ?? modelId
            return await pipeline.respond(to: prepared, stream: stream) { chunks in
                OpenAIResponsesStream(model: model).response(streaming: chunks)
            } completed: { result in
                try JSONEncoder().encode(ResponsesResponse(result: result, model: model))
            }
        }
    }
}

extension OpenAIController {
    struct ResponsesRequest: Codable, Sendable {
        let model: String?
        let instructions: String?
        let input: ResponseInput?
        let messages: [ChatMessage]?
        let maxOutputTokens: Int?
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let minP: Double?
        let dynamicTemperature: Bool?
        let stream: Bool?
        let reasoning: ReasoningConfig?
        let reasoningEffort: String?
        let tools: [ResponseTool]?
        let additionalTools: [ResponseTool]?
        let topK: Int?
        let presencePenalty: Double?
        let repetitionPenalty: Double?
        let thinkingBudget: Int?
        let speculativeLookup: Bool?
        let grammarJumpForward: Bool?

        struct ReasoningConfig: Codable, Sendable {
            let effort: String?
        }

        var effectiveReasoningEffort: String? { reasoning?.effort ?? reasoningEffort }

        enum CodingKeys: String, CodingKey {
            case model, instructions, input, messages, temperature, stream, tools, reasoning
            case maxOutputTokens = "max_output_tokens"
            case maxTokens = "max_tokens"
            case topP = "top_p"
            case minP = "min_p"
            case dynamicTemperature = "dynamic_temperature"
            case reasoningEffort = "reasoning_effort"
            case additionalTools = "additional_tools"
            case topK = "top_k"
            case presencePenalty = "presence_penalty"
            case repetitionPenalty = "repetition_penalty"
            case thinkingBudget = "thinking_budget"
            case speculativeLookup = "speculative_lookup"
            case grammarJumpForward = "grammar_jump_forward"
        }

        func conversation() throws(BasaltError) -> [ChatMessageInput] {
            switch input {
            case .text(let t):        t.isEmpty ? [] : [ChatMessageInput(role: .user, text: t)]
            case .messages(let msgs): try OpenAIController.chatMessages(msgs)
            case nil:                 try OpenAIController.chatMessages(messages ?? [])
            }
        }

        var instructionMessages: [ChatMessageInput] {
            instructions.flatMap { $0.isEmpty ? nil : [ChatMessageInput(role: .system, text: $0)] } ?? []
        }

        func sampling(_ antiRepetition: ChatPipeline.AntiRepetition) -> ChatPipeline.Sampling {
            ChatPipeline.Sampling(
                temperature: temperature, topP: topP, minP: minP, dynamicTemperature: dynamicTemperature, topK: topK,
                presencePenalty: presencePenalty, repetitionPenalty: repetitionPenalty,
                speculativeLookup: speculativeLookup, grammarJumpForward: grammarJumpForward, antiRepetition: antiRepetition)
        }
    }

    enum ResponseInput: Codable, Sendable {
        case text(String)
        case messages([ChatMessage])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let str = try? container.decode(String.self) {
                self = .text(str)
                return
            }
            let msgs = try container.decode([ChatMessage].self)
            self = .messages(msgs)
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .text(let s): try container.encode(s)
            case .messages(let m): try container.encode(m)
            }
        }
    }

    struct ResponseTool: Codable, Sendable {
        let type: String
        let name: String?
        let description: String?
    }
}

extension OpenAIController {
    struct ResponsesResponse: Codable, Sendable {
        let id: String
        let object: String
        let createdAt: Int
        let status: String
        let model: String
        let output: [OutputItem]
        let usage: Usage

        struct OutputItem: Codable, Sendable {
            let id: String
            let type: String
            let status: String
            var role: String? = nil
            var content: [OutputContent]? = nil
            var callId: String? = nil
            var name: String? = nil
            var arguments: String? = nil
            var summary: [OutputContent]? = nil

            enum CodingKeys: String, CodingKey {
                case id, type, status, role, content, name, arguments, summary
                case callId = "call_id"
            }
        }

        struct OutputContent: Codable, Sendable {
            let type: String
            let text: String?
        }

        struct Usage: Codable, Sendable {
            let inputTokens: Int
            let inputTokensDetails: CachedTokens
            let outputTokens: Int
            let totalTokens: Int
            var outputTokensDetails: ReasoningTokens? = nil

            enum CodingKeys: String, CodingKey {
                case inputTokens = "input_tokens"
                case inputTokensDetails = "input_tokens_details"
                case outputTokens = "output_tokens"
                case totalTokens = "total_tokens"
                case outputTokensDetails = "output_tokens_details"
            }
        }

        enum CodingKeys: String, CodingKey {
            case id, object, status, model, output, usage
            case createdAt = "created_at"
        }
    }
}

extension OpenAIController.ResponsesResponse {
    init(result: GenerationResponse, model: String) {
        self = Self(
            id: "resp_\(UUID().uuidString)", object: "response", createdAt: Int(Date().timeIntervalSince1970),
            status: "completed", model: model, output: Self.output(for: result),
            usage: Usage(
                inputTokens: result.promptTokens,
                inputTokensDetails: OpenAIController.CachedTokens(cachedTokens: result.cachedPromptTokens),
                outputTokens: result.completionTokens,
                totalTokens: result.promptTokens + result.completionTokens,
                outputTokensDetails: OpenAIController.ReasoningTokens(reasoningTokens: result.reasoningTokens ?? 0)))
    }

    static func output(for result: GenerationResponse) -> [OutputItem] {
        let reasoning = result.reasoningText.map { text in
            [OutputItem(id: "rs_\(UUID().uuidString)", type: "reasoning", status: "completed",
                        content: [OutputContent(type: "reasoning_text", text: text)], summary: [])]
        } ?? []
        let message = result.generatedText.isEmpty && result.toolCalls != nil ? [] : [
            OutputItem(id: "msg_\(UUID().uuidString)", type: "message", status: "completed", role: "assistant",
                       content: [OutputContent(type: "output_text", text: result.generatedText)])
        ]
        let calls = (result.toolCalls ?? []).map { call in
            OutputItem(id: "fc_\(UUID().uuidString)", type: "function_call", status: "completed",
                       callId: call.id, name: call.name, arguments: call.arguments)
        }
        return reasoning + message + calls
    }
}

struct OpenAIResponsesStream: EventStreamEncoder {
    enum Item: Sendable {
        case reasoning(id: String, text: String)
        case message(id: String, text: String)
        case functionCall(id: String, call: ToolCallOutput)

        var json: [String: Any] {
            switch self {
            case .reasoning(let id, let text):
                ["id": id, "type": "reasoning", "summary": [] as [Any], "content": [["type": "reasoning_text", "text": text]]]
            case .message(let id, let text):
                OpenAIResponsesStream.message(id: id, status: "completed", text: text)
            case .functionCall(let id, let call):
                OpenAIResponsesStream.functionCall(id: id, call: call, arguments: call.arguments, status: "completed")
            }
        }

        var isFunctionCall: Bool {
            if case .functionCall = self { true } else { false }
        }
    }

    let id = "resp_\(UUID().uuidString)"
    let reasoningId = "rs_\(UUID().uuidString)"
    let createdAt = Int(Date().timeIntervalSince1970)
    let model: String
    var messageId = "msg_\(UUID().uuidString)"
    var sequence = 0
    var completed: [Item] = []
    var reasoning: String?
    var messageIndex: Int?
    var text = ""
    var final: GenerationResponse?
    var out: [String] = []

    var calledTools: Bool { completed.contains(where: \.isFunctionCall) }

    mutating func opening() -> [String] {
        out = []
        send("response.created", ["response": responseObject(status: "in_progress", output: [])])
        return out
    }

    mutating func frames(for chunk: StreamChunk) -> [String] {
        out = []
        switch chunk {
        case .text(let delta): appendText(delta)
        case .reasoning(let delta): appendReasoning(delta)
        case .toolCall(let call): emitToolCall(call)
        case .metadata(let meta): final = meta
        }
        return out
    }

    mutating func closing() -> [String] {
        out = []
        if final?.finishReason == .error {
            send("response.failed", ["response": responseObject(status: "failed", output: [], extra: [
                "error": ["code": "server_error", "message": ModelsController.generationFailed]
            ])])
            return out
        }
        closeReasoning()
        if messageIndex != nil || !calledTools { _ = openMessage() }
        closeMessage()
        send("response.completed", ["response": responseObject(status: "completed", output: completed.map(\.json), extra: [
            "usage": usage
        ])])
        return out
    }

    var usage: [String: Any] {
        let inTokens = final?.promptTokens ?? 0
        let outTokens = final?.completionTokens ?? 0
        return [
            "input_tokens": inTokens, "output_tokens": outTokens, "total_tokens": inTokens + outTokens,
            "input_tokens_details": ["cached_tokens": final?.cachedPromptTokens ?? 0],
            "output_tokens_details": ["reasoning_tokens": final?.reasoningTokens ?? 0]]
    }

    static func message(id: String, status: String, text: String?) -> [String: Any] {
        let content: [Any] = text.map { [["type": "output_text", "text": $0, "annotations": [] as [Any]]] } ?? []
        return ["id": id, "type": "message", "status": status, "role": "assistant", "content": content]
    }

    static func functionCall(id: String, call: ToolCallOutput, arguments: String, status: String) -> [String: Any] {
        ["id": id, "type": "function_call", "call_id": call.id, "name": call.name, "arguments": arguments, "status": status]
    }

    private mutating func send(_ event: String, _ fields: [String: Any]) {
        var payload = fields
        payload["type"] = event
        payload["sequence_number"] = sequence
        sequence += 1
        if let json = Self.json(payload) { out.append("event: \(event)\ndata: \(json)\n\n") }
    }

    private func responseObject(status: String, output: [Any], extra: [String: Any] = [:]) -> [String: Any] {
        var response: [String: Any] = ["id": id, "object": "response", "created_at": createdAt,
                                       "status": status, "model": model, "output": output]
        response.merge(extra) { _, new in new }
        return response
    }

    private mutating func appendText(_ delta: String) {
        if messageIndex == nil, calledTools, delta.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return }
        let index = openMessage()
        text += delta
        send("response.output_text.delta", ["item_id": messageId, "output_index": index, "content_index": 0, "delta": delta])
    }

    private mutating func appendReasoning(_ delta: String) {
        guard messageIndex == nil, completed.isEmpty else { return }
        if reasoning == nil {
            send("response.output_item.added", ["output_index": 0, "item": [
                "id": reasoningId, "type": "reasoning", "summary": [] as [Any], "content": [] as [Any]
            ] as [String: Any]])
        }
        reasoning = (reasoning ?? "") + delta
        send("response.reasoning_text.delta", ["item_id": reasoningId, "output_index": 0, "content_index": 0, "delta": delta])
    }

    private mutating func closeReasoning() {
        guard let reasoning else { return }
        send("response.reasoning_text.done", [
            "item_id": reasoningId, "output_index": completed.count, "content_index": 0, "text": reasoning
        ])
        let done = Item.reasoning(id: reasoningId, text: reasoning)
        send("response.output_item.done", ["output_index": completed.count, "item": done.json])
        completed.append(done)
        self.reasoning = nil
    }

    private mutating func closeMessage() {
        guard let index = messageIndex else { return }
        send("response.output_text.done", ["item_id": messageId, "output_index": index, "content_index": 0, "text": text])
        send("response.content_part.done", [
            "item_id": messageId, "output_index": index, "content_index": 0,
            "part": ["type": "output_text", "text": text, "annotations": [] as [Any]]
        ])
        let done = Item.message(id: messageId, text: text)
        send("response.output_item.done", ["output_index": index, "item": done.json])
        completed.append(done)
        messageIndex = nil
        messageId = "msg_\(UUID().uuidString)"
        text = ""
    }

    private mutating func openMessage() -> Int {
        if let index = messageIndex { return index }
        closeReasoning()
        let index = completed.count
        send("response.output_item.added", ["output_index": index, "item": Self.message(id: messageId, status: "in_progress", text: nil)])
        send("response.content_part.added", [
            "item_id": messageId, "output_index": index, "content_index": 0,
            "part": ["type": "output_text", "text": "", "annotations": [] as [Any]]
        ])
        messageIndex = index
        return index
    }

    private mutating func emitToolCall(_ call: ToolCallOutput) {
        closeReasoning()
        closeMessage()
        let index = completed.count
        let itemId = "fc_\(UUID().uuidString)"
        send("response.output_item.added", [
            "output_index": index, "item": Self.functionCall(id: itemId, call: call, arguments: "", status: "in_progress")
        ])
        send("response.function_call_arguments.delta", ["item_id": itemId, "output_index": index, "delta": call.arguments])
        send("response.function_call_arguments.done", ["item_id": itemId, "output_index": index, "arguments": call.arguments])
        let done = Item.functionCall(id: itemId, call: call)
        send("response.output_item.done", ["output_index": index, "item": done.json])
        completed.append(done)
    }
}
