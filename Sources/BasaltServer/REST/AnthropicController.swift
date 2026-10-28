import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

// MARK: — Anthropic Messages API (no auth required)
//
// Implements POST /v1/messages so Claude Code can use Basalt as a drop-in backend:
//
//   export ANTHROPIC_API_KEY=local        # any non-empty string
//   export ANTHROPIC_BASE_URL=http://localhost:8080
//   claude                                # routes all traffic to local Basalt server
//
// The server accepts any ANTHROPIC_API_KEY value and never validates it.

struct AnthropicController: Sendable {

    let orchestrator: ModelOrchestratorActor
    let modelId: String

    // MARK: — Anthropic DTOs

    struct ContentBlock: Codable, Sendable {
        let type: String
        var text: String? = nil
        var id: String? = nil
        var name: String? = nil
        var input: AnyCodable? = nil
        var toolUseId: String? = nil
        var content: MessageContent? = nil
        var isError: Bool? = nil
        var thinking: String? = nil
        var signature: String? = nil

        init(type: String = "text", text: String) {
            self.type = type
            self.text = text
        }

        init(thinking: String) {
            type = "thinking"
            self.thinking = thinking
            signature = ""
        }

        init(toolUse call: ToolCallOutput, input: AnyCodable?) {
            type = "tool_use"
            id = AnthropicController.toolUseID(call.id)
            name = call.name
            self.input = input ?? AnyCodable([String: Any]())
        }

        var promptText: String? {
            type == "image" ? ToolTranscript.imageOmitted : text
        }

        var inputJSON: String {
            input.flatMap { try? Self.sortedEncoder.encode($0) }.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        }

        private static let sortedEncoder: JSONEncoder = {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            return encoder
        }()

        enum CodingKeys: String, CodingKey {
            case type, text, id, name, input, content, thinking, signature
            case toolUseId = "tool_use_id"
            case isError = "is_error"
        }
    }

    static func objectInput(_ arguments: String) -> AnyCodable? {
        let decoded = try? JSONDecoder().decode(AnyCodable.self, from: Data(arguments.utf8))
        return decoded?.rawValue is [String: Any] ? decoded : nil
    }

    static func nonObjectNote(_ call: ToolCallOutput) -> String {
        "\n[tool call to \(call.name): arguments are not an object, sent as {}: \(call.arguments)]"
    }

    static func toolUseID(_ id: String) -> String {
        id.hasPrefix("toolu_") ? id : "toolu_\(id.hasPrefix("call_") ? id.dropFirst(5) : Substring(id))"
    }

    struct Message: Codable, Sendable {
        let role: String
        let content: MessageContent
    }

    enum MessageContent: Codable, Sendable {
        case text(String)
        case blocks([ContentBlock])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let str = try? container.decode(String.self) {
                self = .text(str)
            } else {
                let blocks = try container.decode([ContentBlock].self)
                self = .blocks(blocks)
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .text(let str): try container.encode(str)
            case .blocks(let b): try container.encode(b)
            }
        }

        var textValue: String {
            switch self {
            case .text(let s): return s
            case .blocks(let b): return b.compactMap(\.promptText).joined(separator: "\n")
            }
        }
    }

    /// system can be a plain string OR an array of content blocks (beta API format)
    enum SystemPrompt: Codable, Sendable {
        case text(String)
        case blocks([ContentBlock])

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let str = try? c.decode(String.self) {
                self = .text(str)
            } else {
                self = .blocks((try? c.decode([ContentBlock].self)) ?? [])
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .text(let s):   try c.encode(s)
            case .blocks(let b): try c.encode(b)
            }
        }

        var textValue: String {
            switch self {
            case .text(let s):   return s
            case .blocks(let b): return b.compactMap(\.promptText).joined(separator: "\n")
            }
        }
    }

    struct MessagesRequest: Codable, Sendable {
        let model: String?
        let messages: [Message]
        let system: SystemPrompt?
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let minP: Double?
        let dynamicTemperature: Bool?
        let stream: Bool?
        let thinking: ThinkingConfig?
        let reasoningEffort: String?
        let topK: Int?
        let speculativeLookup: Bool?

        enum CodingKeys: String, CodingKey {
            case model, messages, system, temperature, stream, thinking
            case maxTokens = "max_tokens"
            case topP = "top_p"
            case minP = "min_p"
            case dynamicTemperature = "dynamic_temperature"
            case reasoningEffort = "reasoning_effort"
            case topK = "top_k"
            case speculativeLookup = "speculative_lookup"
        }
    }

    struct ThinkingConfig: Codable, Sendable {
        let type: String
        let budgetTokens: Int?

        enum CodingKeys: String, CodingKey {
            case type
            case budgetTokens = "budget_tokens"
        }
    }

    struct MessagesResponse: Codable, Sendable {
        let id: String
        let type: String
        let role: String
        let content: [ContentBlock]
        let model: String
        let stopReason: String
        let stopSequence: String?
        let usage: Usage

        struct Usage: Codable, Sendable {
            let inputTokens: Int
            let cacheReadInputTokens: Int
            let outputTokens: Int

            init(_ result: GenerationResponse) {
                inputTokens = result.promptTokens - result.cachedPromptTokens
                cacheReadInputTokens = result.cachedPromptTokens
                outputTokens = result.completionTokens
            }

            enum CodingKeys: String, CodingKey {
                case inputTokens = "input_tokens"
                case cacheReadInputTokens = "cache_read_input_tokens"
                case outputTokens = "output_tokens"
            }
        }

        enum CodingKeys: String, CodingKey {
            case id, type, role, content, model, usage
            case stopReason = "stop_reason"
            case stopSequence = "stop_sequence"
        }
    }

    // MARK: — POST /v1/messages

    @Sendable
    func messages(request: Request, context: BasicRequestContext) async throws -> Response {
        let pipeline = ChatPipeline(orchestrator: orchestrator, label: "Anthropic", source: "anthropic", endpoint: "/v1/messages") { status, message in
            Self.errorResponse(status: status, message: message,
                               type: status == .serviceUnavailable ? "overloaded_error" : "invalid_request_error")
        }
        return try await pipeline.handle(request) { data in
            let dto = try pipeline.decode(MessagesRequest.self, from: data, failure: "Decode error")
            try pipeline.require(!dto.messages.isEmpty, "messages must not be empty")
            let maxTokens = dto.maxTokens ?? 8192
            if dto.thinking?.type == "enabled", let budget = dto.thinking?.budgetTokens, budget < 1 || budget >= maxTokens {
                throw pipeline.rejection(.badRequest, "thinking.budget_tokens must be >= 1 and less than max_tokens (\(maxTokens))")
            }
            let tools = try pipeline.tools(in: data)
            let prepared = try pipeline.prepare(
                chat: Self.chatMessages(system: dto.system, messages: dto.messages), headers: request.headers,
                maxTokens: maxTokens,
                sampling: .init(temperature: dto.temperature, topP: dto.topP, minP: dto.minP,
                                dynamicTemperature: dto.dynamicTemperature, topK: dto.topK,
                                speculativeLookup: dto.speculativeLookup),
                reasoningEffort: Self.reasoningEffort(dto), tools: tools,
                thinkingBudget: dto.thinking?.type == "enabled" ? dto.thinking?.budgetTokens : nil,
                responseFormat: nil, model: dto.model, stream: dto.stream == true)
            if dto.stream == true { try await pipeline.admitStream(prepared.request) }
            let genRequest = prepared.request
            let requestedModel = prepared.requestedModel

            if dto.stream == true {
                return streamMessagesResponse(genRequest: genRequest, model: dto.model ?? modelId, requestedModel: requestedModel)
            }

            do {
                let result = try await orchestrator.generate(request: genRequest, modelId: requestedModel)
                let response = MessagesResponse(
                    id: "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))",
                    type: "message",
                    role: "assistant",
                    content: Self.responseContent(reasoning: result.reasoningText, text: result.generatedText, toolCalls: result.toolCalls ?? []),
                    model: dto.model ?? modelId,
                    stopReason: Self.stopReason(result.finishReason, calledTools: result.toolCalls?.isEmpty == false),
                    stopSequence: nil,
                    usage: .init(result)
                )
                return try jsonResponse(status: .ok, body: response)
            } catch let err as BasaltError {
                return errorResponse(
                    status: HTTPResponse.Status(code: err.httpStatus),
                    message: ModelsController.publicMessage(for: err),
                    type: "api_error"
                )
            }
        }
    }

    // MARK: — Streaming (Anthropic SSE format)

    private func streamMessagesResponse(genRequest: GenerationRequest, model: String, requestedModel: String? = nil) -> Response {
        let headers: HTTPFields = [
            .contentType: "text/event-stream",
            .cacheControl: "no-cache",
            .connection: "keep-alive"
        ]
        let body = ResponseBody(asyncSequence: anthropicSSEStream(genRequest: genRequest, model: model, requestedModel: requestedModel))
        return Response(status: .ok, headers: headers, body: body)
    }

    private func anthropicSSEStream(genRequest: GenerationRequest, model: String, requestedModel: String? = nil) -> AsyncStream<ByteBuffer> {
        AsyncStream { continuation in
            let task = Task {
                let msgId = "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"

                func send(_ event: String, _ data: [String: Any]) {
                    guard let json = try? JSONSerialization.data(withJSONObject: data),
                          let jsonStr = String(data: json, encoding: .utf8) else { return }
                    var buf = ByteBuffer()
                    buf.writeString("event: \(event)\ndata: \(jsonStr)\n\n")
                    continuation.yield(buf)
                }

                // message_start
                send("message_start", [
                    "type": "message_start",
                    "message": [
                        "id": msgId,
                        "type": "message",
                        "role": "assistant",
                        "content": [] as [[String: Any]],
                        "model": model,
                        "stop_reason": NSNull(),
                        "stop_sequence": NSNull(),
                        "usage": ["input_tokens": 0, "output_tokens": 0]
                    ] as [String: Any]
                ])

                // ping
                send("ping", ["type": "ping"])

                // Content blocks per the Messages streaming spec: reasoning goes into its own
                // `thinking` block, the answer into a `text` block after it — a thinking_delta
                // inside a text block is something SDKs drop or reject.
                var openBlock: (index: Int, type: String)?
                var nextIndex = 0
                func open(_ type: String) {
                    if let current = openBlock {
                        if current.type == type { return }
                        send("content_block_stop", ["type": "content_block_stop", "index": current.index])
                    }
                    let block: [String: Any] = type == "thinking"
                        ? ["type": "thinking", "thinking": "", "signature": ""]
                        : ["type": "text", "text": ""]
                    send("content_block_start", ["type": "content_block_start", "index": nextIndex, "content_block": block])
                    openBlock = (nextIndex, type)
                    nextIndex += 1
                }

                var calledTools = false
                func emitToolUse(_ call: ToolCallOutput) {
                    let note = Self.objectInput(call.arguments) == nil ? Self.nonObjectNote(call) : nil
                    if let note {
                        open("text")
                        send("content_block_delta", [
                            "type": "content_block_delta",
                            "index": openBlock!.index,
                            "delta": ["type": "text_delta", "text": note]
                        ])
                    }
                    if let current = openBlock {
                        send("content_block_stop", ["type": "content_block_stop", "index": current.index])
                        openBlock = nil
                    }
                    let block: [String: Any] = ["type": "tool_use", "id": Self.toolUseID(call.id), "name": call.name, "input": [String: Any]()]
                    send("content_block_start", ["type": "content_block_start", "index": nextIndex, "content_block": block])
                    send("content_block_delta", [
                        "type": "content_block_delta",
                        "index": nextIndex,
                        "delta": ["type": "input_json_delta", "partial_json": note == nil ? call.arguments : "{}"]
                    ])
                    send("content_block_stop", ["type": "content_block_stop", "index": nextIndex])
                    nextIndex += 1
                    calledTools = true
                }

                var outputTokens = 0
                var final: GenerationResponse?

                do {
                    let stream = try await orchestrator.generateStream(request: genRequest, modelId: requestedModel)
                    for await chunk in stream {
                        switch chunk {
                        case .toolCall(let call):
                            outputTokens += 1
                            emitToolUse(call)
                        case .text(let text):
                            outputTokens += 1
                            if calledTools, openBlock == nil, text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
                            open("text")
                            send("content_block_delta", [
                                "type": "content_block_delta",
                                "index": openBlock!.index,
                                "delta": ["type": "text_delta", "text": text]
                            ])
                        case .reasoning(let reasoning):
                            outputTokens += 1
                            guard openBlock?.type != "text", !calledTools else { continue }
                            open("thinking")
                            send("content_block_delta", [
                                "type": "content_block_delta",
                                "index": openBlock!.index,
                                "delta": ["type": "thinking_delta", "thinking": reasoning]
                            ])
                        case .metadata(let meta):
                            final = meta
                        }
                    }
                } catch {
                    let errPayload: [String: Any] = [
                        "type": "error",
                        "error": [
                            "type": "api_error",
                            "message": ModelsController.publicMessage(for: error)
                        ]
                    ]
                    if let json = try? JSONSerialization.data(withJSONObject: errPayload),
                       let str = String(data: json, encoding: .utf8) {
                        var buf = ByteBuffer()
                        buf.writeString("event: error\ndata: \(str)\n\n")
                        continuation.yield(buf)
                    }
                    continuation.finish()
                    return
                }

                if openBlock != nil || !calledTools {
                    open("text")
                    send("content_block_stop", ["type": "content_block_stop", "index": openBlock!.index])
                }

                // message_delta
                send("message_delta", [
                    "type": "message_delta",
                    // The engine's own counts when it reported them; chunk count otherwise.
                    "delta": ["stop_reason": Self.stopReason(final?.finishReason ?? .stop, calledTools: calledTools),
                              "stop_sequence": NSNull()],
                    "usage": [
                        "input_tokens": (final?.promptTokens ?? 0) - (final?.cachedPromptTokens ?? 0),
                        "cache_read_input_tokens": final?.cachedPromptTokens ?? 0,
                        "output_tokens": final?.completionTokens ?? outputTokens
                    ]
                ])

                // message_stop
                send("message_stop", ["type": "message_stop"])

                continuation.finish()
            }
            // A client that disconnects cancels the generation instead of leaving it running.
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    static func stopReason(_ finish: GenerationResponse.FinishReason, calledTools: Bool) -> String {
        calledTools ? "tool_use" : (finish.isTruncated ? "max_tokens" : "end_turn")
    }

    static func responseContent(reasoning: String? = nil, text answer: String, toolCalls: [ToolCallOutput]) -> [ContentBlock] {
        let inputs = toolCalls.map { objectInput($0.arguments) }
        let text = answer + zip(toolCalls, inputs).filter { $0.1 == nil }.map { nonObjectNote($0.0) }.joined()
        let keepText = toolCalls.isEmpty || !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return (reasoning.map { [ContentBlock(thinking: $0)] } ?? []) + (keepText ? [ContentBlock(text: text)] : [])
            + zip(toolCalls, inputs).map { ContentBlock(toolUse: $0.0, input: $0.1) }
    }

    static func chatMessages(system: SystemPrompt?, messages: [Message]) -> [ChatMessageInput] {
        var chat: [ChatMessageInput] = []
        if let text = system?.textValue, !text.isEmpty {
            chat.append(ChatMessageInput(role: .system, text: text))
        }

        var toolNames: [String: String] = [:]
        for case .blocks(let blocks) in messages.map(\.content) {
            for block in blocks where block.type == "tool_use" {
                if let id = block.id, let name = block.name { toolNames[id] = name }
            }
        }

        for msg in messages {
            let role: ChatMessageInput.Role = msg.role == "assistant" ? .assistant : .user
            let blocks: [ContentBlock]
            switch msg.content {
            case .text(let s):   blocks = [ContentBlock(text: s)]
            case .blocks(let b): blocks = b
            }
            var text: [String] = []
            var calls: [ToolCallOutput] = []
            var emitted = false
            func flush() {
                guard !text.isEmpty || !calls.isEmpty else { return }
                chat.append(ChatMessageInput(role: role, text: text.joined(separator: "\n"), toolCalls: calls.isEmpty ? nil : calls))
                text = []
                calls = []
                emitted = true
            }
            for block in blocks {
                switch block.type {
                case "tool_use":
                    calls.append(ToolCallOutput(id: block.id ?? "", name: block.name ?? "tool", arguments: block.inputJSON))
                case "tool_result":
                    flush()
                    let output = block.content?.textValue ?? ""
                    chat.append(ChatMessageInput(
                        role: .tool, text: block.isError == true ? "Error: \(output)" : output,
                        toolCallId: block.toolUseId, toolName: block.toolUseId.flatMap { toolNames[$0] }))
                    emitted = true
                default:
                    if let t = block.promptText { text.append(t) }
                }
            }
            flush()
            if !emitted {
                chat.append(ChatMessageInput(role: role, text: ""))
            }
        }
        return chat
    }

    // MARK: — Helpers

    private func jsonResponse<T: Encodable>(status: HTTPResponse.Status, body: T) throws -> Response {
        let data = try JSONEncoder().encode(body)
        var buf = ByteBuffer()
        buf.writeBytes(data)
        return Response(status: status, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buf))
    }

    /// Anthropic error envelope format: {"type":"error","error":{"type":"...","message":"..."}}
    static func reasoningEffort(_ dto: MessagesRequest) -> String? {
        if let effort = dto.reasoningEffort { return effort }
        guard let thinking = dto.thinking else { return nil }
        if thinking.type == "disabled" { return "off" }
        return thinking.budgetTokens.map { $0 > 8000 ? "high" : ($0 > 2000 ? "medium" : "low") } ?? "medium"
    }

    private func errorResponse(status: HTTPResponse.Status, message: String, type errorType: String) -> Response {
        Self.errorResponse(status: status, message: message, type: errorType)
    }

    static func errorResponse(status: HTTPResponse.Status, message: String, type errorType: String) -> Response {
        let body: [String: Any] = [
            "type": "error",
            "error": ["type": errorType, "message": message]
        ]
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        var buf = ByteBuffer()
        buf.writeBytes(data)
        return Response(status: status, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buf))
    }
}

private extension HTTPResponse.Status {
    init(code: Int) {
        self = HTTPResponse.Status(code: code, reasonPhrase: "")
    }
}

