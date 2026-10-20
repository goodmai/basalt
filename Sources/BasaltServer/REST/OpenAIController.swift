import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

// MARK: — OpenAI-compatible endpoints (no auth required)
//
// Usage with Claude Code:
//   export OPENAI_API_KEY=local
//   export OPENAI_BASE_URL=http://localhost:8080
//   claude --model openai/gemma-4-31b-it-4bit
//
// Routes (open on loopback; behind the API key and rate limit otherwise):
//   GET  /v1/models
//   POST /v1/chat/completions   — OpenAI chat format, streaming supported
//   POST /v1/generate           — raw prompt

struct OpenAIController: Sendable {

    let orchestrator: ModelOrchestratorActor
    let modelId: String

    // MARK: — DTOs

    /// content can be a plain string OR an array of typed blocks (Claude Code sends arrays)
    enum MessageContent: Codable, Sendable {
        case text(String)
        case blocks([ContentBlock])

        struct ContentBlock: Codable, Sendable {
            let type: String?
            let text: String?
        }

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
            case .blocks(let b):
                return b.compactMap { $0.type?.contains("image") == true ? ToolTranscript.imageOmitted : $0.text }
                    .joined(separator: "\n")
            }
        }
    }

    struct ChatMessage: Codable, Sendable {
        let type: String?
        let role: String?
        let content: MessageContent?   // null when tool_calls are present
        // Responses API input items besides messages: a tool call the model made earlier
        // (`function_call`: name + arguments, `custom_tool_call`: name + input) and its result
        // (`*_output`: output — a string, or content blocks when a tool returned images).
        let name: String?
        let arguments: String?
        let input: String?
        let output: MessageContent?
        let toolCalls: [ChatToolCall]?
        var callId: String? = nil
        var toolCallId: String? = nil
        var reasoning: String? = nil

        init(type: String? = nil, role: String? = nil, content: MessageContent?,
             name: String? = nil, arguments: String? = nil, input: String? = nil, output: MessageContent? = nil,
             toolCalls: [ChatToolCall]? = nil) {
            self.type = type
            self.role = role
            self.content = content
            self.name = name
            self.arguments = arguments
            self.input = input
            self.output = output
            self.toolCalls = toolCalls
        }

        enum CodingKeys: String, CodingKey {
            case type, role, content, name, arguments, input, output
            case toolCalls = "tool_calls"
            case callId = "call_id"
            case toolCallId = "tool_call_id"
            case reasoning
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encodeIfPresent(type, forKey: .type)
            try c.encodeIfPresent(role, forKey: .role)
            try c.encode(content, forKey: .content)
            try c.encodeIfPresent(reasoning, forKey: .reasoning)
            try c.encodeIfPresent(name, forKey: .name)
            try c.encodeIfPresent(arguments, forKey: .arguments)
            try c.encodeIfPresent(input, forKey: .input)
            try c.encodeIfPresent(output, forKey: .output)
            try c.encodeIfPresent(toolCalls, forKey: .toolCalls)
            try c.encodeIfPresent(callId, forKey: .callId)
            try c.encodeIfPresent(toolCallId, forKey: .toolCallId)
        }
    }

    struct ChatToolCall: Codable, Sendable {
        var id: String?
        var type: String? = "function"
        let function: Function

        struct Function: Codable, Sendable {
            let name: String
            let arguments: String?
        }

        init(_ call: ToolCallOutput) {
            id = call.id
            function = Function(name: call.name, arguments: call.arguments)
        }
    }

    static func finishReason(_ finish: GenerationResponse.FinishReason?, calledTools: Bool) -> String {
        calledTools ? "tool_calls" : (finish == .length ? "length" : "stop")
    }

    struct ChatCompletionRequest: Codable, Sendable {
        let model: String?
        let messages: [ChatMessage]
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let minP: Double?
        let dynamicTemperature: Bool?
        let stream: Bool?
        let reasoningEffort: String?
        let topK: Int?
        let presencePenalty: Double?
        let repetitionPenalty: Double?
        let thinkingBudget: Int?
        let streamOptions: StreamOptions?

        struct StreamOptions: Codable, Sendable {
            let includeUsage: Bool?

            enum CodingKeys: String, CodingKey {
                case includeUsage = "include_usage"
            }
        }

        enum CodingKeys: String, CodingKey {
            case model, messages, temperature, stream
            case streamOptions = "stream_options"
            case maxTokens = "max_tokens"
            case topP = "top_p"
            case minP = "min_p"
            case dynamicTemperature = "dynamic_temperature"
            case reasoningEffort = "reasoning_effort"
            case topK = "top_k"
            case presencePenalty = "presence_penalty"
            case repetitionPenalty = "repetition_penalty"
            case thinkingBudget = "thinking_budget"
        }
    }

    struct ResponsesRequest: Codable, Sendable {
        let model: String?
        /// The Responses API's system prompt — Codex sends its whole agent prompt here.
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
        /// Responses API form: `"reasoning": {"effort": "low"}`.
        let reasoning: ReasoningConfig?
        /// Chat Completions form, accepted here too for clients that reuse it.
        let reasoningEffort: String?
        let tools: [ResponseTool]?
        let additionalTools: [ResponseTool]?
        let topK: Int?
        let presencePenalty: Double?
        let repetitionPenalty: Double?
        let thinkingBudget: Int?

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

    struct ResponsesResponse: Codable, Sendable {
        let id: String
        let object: String
        let createdAt: Int
        let status: String
        let model: String
        let output: [OutputItem]
        let usage: Usage

        /// A `message` (role + content) or a `function_call` (call_id + name + arguments).
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

    struct ChatCompletionResponse: Codable, Sendable {
        let id: String
        let object: String
        let created: Int
        let model: String
        let choices: [Choice]
        let usage: Usage

        struct Choice: Codable, Sendable {
            let index: Int
            let message: ChatMessage
            let finishReason: String
            enum CodingKeys: String, CodingKey {
                case index, message
                case finishReason = "finish_reason"
            }
        }

        struct Usage: Codable, Sendable {
            let promptTokens: Int
            let promptTokensDetails: CachedTokens
            let completionTokens: Int
            let totalTokens: Int
            var completionTokensDetails: ReasoningTokens? = nil
            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case promptTokensDetails = "prompt_tokens_details"
                case completionTokens = "completion_tokens"
                case totalTokens = "total_tokens"
                case completionTokensDetails = "completion_tokens_details"
            }
        }
    }

    struct ReasoningTokens: Codable, Sendable {
        let reasoningTokens: Int

        enum CodingKeys: String, CodingKey {
            case reasoningTokens = "reasoning_tokens"
        }
    }

    static func chatUsage(_ result: GenerationResponse) -> [String: Any] {
        [
            "prompt_tokens": result.promptTokens, "completion_tokens": result.completionTokens,
            "total_tokens": result.promptTokens + result.completionTokens,
            "prompt_tokens_details": ["cached_tokens": result.cachedPromptTokens],
            "completion_tokens_details": ["reasoning_tokens": result.reasoningTokens ?? 0],
        ]
    }

    struct CachedTokens: Codable, Sendable {
        let cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }
    }

    // MARK: — GET /v1/models

    @Sendable
    func listModels(request: Request, context: BasicRequestContext) async throws -> Response {
        let snapshot = await orchestrator.healthSnapshot(modelId: modelId)
        let body: [String: Any] = [
            "object": "list",
            "data": [[
                "id": snapshot.modelId ?? modelId,
                "object": "model",
                "created": Int(Date().timeIntervalSince1970),
                "owned_by": "basalt"
            ] as [String: Any]]
        ]
        return makeJSONResponse(body)
    }

    // MARK: — POST /v1/chat/completions

    @Sendable
    func chatCompletions(request: Request, context: BasicRequestContext) async throws -> Response {
        let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
        let data = Data(buffer.readableBytesView)


        let dto: ChatCompletionRequest
        do {
            dto = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
        } catch {
            fputs("🔴 [OpenAI] decode error: \(error) (body: \(data.count) bytes)\n", stderr)
            return makeErrorResponse(status: .badRequest,
                                     message: "Cannot decode request: \(error.localizedDescription)",
                                     code: 400)
        }

        guard !dto.messages.isEmpty else {
            return makeErrorResponse(status: .badRequest, message: "messages must not be empty", code: 400)
        }

        let requested: RequestedTools
        do { requested = try Self.requestedTools(fromRequestBody: data) } catch {
            return makeErrorResponse(status: .badRequest, message: error.message, code: 400)
        }
        let responseFormat: ResponseFormat?
        do { responseFormat = try Self.responseFormat(fromRequestBody: data) } catch {
            return makeErrorResponse(status: .badRequest, message: error.message, code: 400)
        }
        let chat = Self.chatMessages(dto.messages)
        let prompt = ChatMessageInput.plainText(chat)
        let service = ServiceRequestDetector.prepare(prompt: prompt, maxTokens: dto.maxTokens, headers: request.headers)
        let genRequest = GenerationRequest(
            prompt: service.prompt,
            maxTokens: dto.maxTokens,
            temperature: dto.temperature,
            topP: dto.topP,
            minP: dto.minP,
            dynamicTemperature: dto.dynamicTemperature,
            reasoningEffort: dto.reasoningEffort,
            isServiceRequest: service.isService,
            tools: requested.tools,
            toolChoice: requested.choice,
            cacheKey: PromptCacheKey.resolve(isService: service.isService),
            messages: service.prompt == prompt ? chat : nil,
            topK: dto.topK,
            presencePenalty: dto.presencePenalty,
            repetitionPenalty: dto.repetitionPenalty,
            thinkingBudget: dto.thinkingBudget,
            responseFormat: responseFormat
        )
        if let rejection = invalidParameters(genRequest) { return rejection }
        let model = dto.model.map(ModelsController.restModel(from:))
        let requestedModel = model?.repoId
        if model == .path {
            return makeErrorResponse(status: .badRequest, message: ModelsController.pathRejected, code: 400)
        }

        if dto.stream == true {
            return streamResponse(genRequest: genRequest, model: dto.model ?? modelId, requestedModel: requestedModel,
                                  includeUsage: dto.streamOptions?.includeUsage == true)
        }

        do {
            let result = try await orchestrator.generate(request: genRequest, modelId: requestedModel)
            let calls = result.toolCalls ?? []
            let onlyCalls = !calls.isEmpty && result.generatedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            var message = ChatMessage(role: "assistant", content: onlyCalls ? nil : .text(result.generatedText),
                                      toolCalls: calls.isEmpty ? nil : calls.map(ChatToolCall.init))
            message.reasoning = result.reasoningText
            let response = ChatCompletionResponse(
                id: "chatcmpl-\(UUID().uuidString)",
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: dto.model ?? modelId,
                choices: [.init(
                    index: 0,
                    message: message,
                    finishReason: Self.finishReason(result.finishReason, calledTools: !calls.isEmpty)
                )],
                usage: .init(
                    promptTokens: result.promptTokens,
                    promptTokensDetails: .init(cachedTokens: result.cachedPromptTokens),
                    completionTokens: result.completionTokens,
                    totalTokens: result.promptTokens + result.completionTokens,
                    completionTokensDetails: ReasoningTokens(reasoningTokens: result.reasoningTokens ?? 0)
                )
            )
            let encoded = try JSONEncoder().encode(response)
            return makeDataResponse(encoded)
        } catch {
            fputs("🔴 [OpenAI] generate failed: \(error)\n", stderr)
            let code = (error as? BasaltError)?.httpStatus ?? 500
            return makeErrorResponse(status: HTTPResponse.Status(code: code),
                                     message: ModelsController.publicMessage(for: error), code: code)
        }
    }

    // MARK: — POST /v1/responses (Codex CLI / Cursor / OpenAI Responses API)

    @Sendable
    func responses(request: Request, context: BasicRequestContext) async throws -> Response {
        let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
        let data = Data(buffer.readableBytesView)

        let dto: ResponsesRequest
        do {
            dto = try JSONDecoder().decode(ResponsesRequest.self, from: data)
        } catch {
            return makeErrorResponse(status: .badRequest,
                                     message: "Cannot decode responses request: \(error.localizedDescription)",
                                     code: 400)
        }

        let conversation: [ChatMessageInput]
        switch dto.input {
        case .text(let t):        conversation = t.isEmpty ? [] : [ChatMessageInput(role: .user, text: t)]
        case .messages(let msgs): conversation = Self.chatMessages(msgs)
        case nil:                 conversation = Self.chatMessages(dto.messages ?? [])
        }

        guard !conversation.isEmpty else {
            return makeErrorResponse(status: .badRequest, message: "input or messages must not be empty", code: 400)
        }
        // `instructions` is the system prompt of the Responses API; dropping it loses the agent's rules.
        let instructions = dto.instructions.flatMap { $0.isEmpty ? nil : ChatMessageInput(role: .system, text: $0) }
        let chat = (instructions.map { [$0] } ?? []) + conversation
        let prompt = ChatMessageInput.plainText(chat)

        let requested: RequestedTools
        do { requested = try Self.requestedTools(fromRequestBody: data) } catch {
            return makeErrorResponse(status: .badRequest, message: error.message, code: 400)
        }
        let responseFormat: ResponseFormat?
        do { responseFormat = try Self.responseFormat(fromRequestBody: data) } catch {
            return makeErrorResponse(status: .badRequest, message: error.message, code: 400)
        }
        let maxT = dto.maxOutputTokens ?? dto.maxTokens
        let service = ServiceRequestDetector.prepare(prompt: prompt, maxTokens: maxT, headers: request.headers)

        let genRequest = GenerationRequest(
            prompt: service.prompt,
            maxTokens: maxT,
            temperature: dto.temperature,
            topP: dto.topP,
            minP: dto.minP,
            dynamicTemperature: dto.dynamicTemperature,
            reasoningEffort: dto.effectiveReasoningEffort,
            isServiceRequest: service.isService,
            tools: requested.tools,
            toolChoice: requested.choice,
            cacheKey: PromptCacheKey.resolve(isService: service.isService),
            messages: service.prompt == prompt ? chat : nil,
            topK: dto.topK,
            presencePenalty: dto.presencePenalty,
            repetitionPenalty: dto.repetitionPenalty,
            thinkingBudget: dto.thinkingBudget,
            responseFormat: responseFormat
        )
        if let rejection = invalidParameters(genRequest) { return rejection }
        let model = dto.model.map(ModelsController.restModel(from:))
        let requestedModel = model?.repoId
        if model == .path {
            return makeErrorResponse(status: .badRequest, message: ModelsController.pathRejected, code: 400)
        }

        if dto.stream == true {
            return streamResponses(genRequest: genRequest, model: dto.model ?? modelId, requestedModel: requestedModel)
        }

        do {
            let result = try await orchestrator.generate(request: genRequest, modelId: requestedModel)
            let response = ResponsesResponse(
                id: "resp_\(UUID().uuidString)",
                object: "response",
                createdAt: Int(Date().timeIntervalSince1970),
                status: "completed",
                model: dto.model ?? modelId,
                // The message (when the model said anything or called nothing), then its tool calls.
                output: (result.reasoningText.map { [.init(
                    id: "rs_\(UUID().uuidString)", type: "reasoning", status: "completed",
                    content: [.init(type: "reasoning_text", text: $0)], summary: [])] } ?? [])
                    + (result.generatedText.isEmpty && result.toolCalls != nil ? [] : [
                    .init(
                        id: "msg_\(UUID().uuidString)",
                        type: "message",
                        status: "completed",
                        role: "assistant",
                        content: [.init(type: "output_text", text: result.generatedText)]
                    )
                ]) + (result.toolCalls ?? []).map { call in
                    .init(id: "fc_\(UUID().uuidString)", type: "function_call", status: "completed",
                          callId: call.id, name: call.name, arguments: call.arguments)
                },
                usage: .init(
                    inputTokens: result.promptTokens,
                    inputTokensDetails: .init(cachedTokens: result.cachedPromptTokens),
                    outputTokens: result.completionTokens,
                    totalTokens: result.promptTokens + result.completionTokens,
                    outputTokensDetails: ReasoningTokens(reasoningTokens: result.reasoningTokens ?? 0)
                )
            )
            let encoded = try JSONEncoder().encode(response)
            return makeDataResponse(encoded)
        } catch {
            let code = (error as? BasaltError)?.httpStatus ?? 500
            return makeErrorResponse(status: HTTPResponse.Status(code: code),
                                     message: ModelsController.publicMessage(for: error), code: code)
        }
    }

    // MARK: — POST /v1/generate (raw prompt)

    @Sendable
    func generateRaw(request: Request, context: BasicRequestContext) async throws -> Response {
        let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
        let data = Data(buffer.readableBytesView)
        do {
            let dto = PromptCacheKey.scoped(try JSONDecoder().decode(GenerationRequest.self, from: data))
            let result = try await orchestrator.generate(request: dto)
            let encoded = try JSONEncoder().encode(result)
            return makeDataResponse(encoded)
        } catch {
            return makeErrorResponse(status: .badRequest,
                                     message: ModelsController.publicMessage(for: error), code: 400)
        }
    }

    // MARK: — Streaming

    private func streamResponse(
        genRequest: GenerationRequest, model: String, requestedModel: String? = nil, includeUsage: Bool
    ) -> Response {
        Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
                .connection: "keep-alive"
            ],
            body: ResponseBody(asyncSequence: sseStream(
                genRequest: genRequest, model: model, requestedModel: requestedModel, includeUsage: includeUsage))
        )
    }

    private func sseStream(
        genRequest: GenerationRequest, model: String, requestedModel: String? = nil, includeUsage: Bool
    ) -> AsyncStream<ByteBuffer> {
        AsyncStream { continuation in
            let task = Task {
                let id = "chatcmpl-\(UUID().uuidString)"
                let created = Int(Date().timeIntervalSince1970)
                func send(delta: [String: Any], finish: String? = nil) {
                    send(["choices": [["index": 0, "delta": delta, "finish_reason": finish ?? NSNull()] as [String: Any]]])
                }
                func send(_ fields: [String: Any]) {
                    var chunk: [String: Any] = [
                        "id": id, "object": "chat.completion.chunk",
                        "created": created, "model": model
                    ]
                    chunk.merge(fields) { _, new in new }
                    if let json = try? JSONSerialization.data(withJSONObject: chunk),
                       let str = String(data: json, encoding: .utf8) {
                        var buf = ByteBuffer(); buf.writeString("data: \(str)\n\n")
                        continuation.yield(buf)
                    }
                }
                var toolCallCount = 0
                var final: GenerationResponse?
                do {
                    let stream = try await orchestrator.generateStream(request: genRequest, modelId: requestedModel)
                    for await chunk in stream {
                        switch chunk {
                        case .toolCall(let call):
                            send(delta: ["role": "assistant", "tool_calls": [[
                                "index": toolCallCount, "id": call.id, "type": "function",
                                "function": ["name": call.name, "arguments": call.arguments]
                            ] as [String: Any]]])
                            toolCallCount += 1
                        case .text(let text):
                            if toolCallCount > 0, text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
                            send(delta: ["role": "assistant", "content": text])
                        case .reasoning(let reasoning):
                            send(delta: ["role": "assistant", "reasoning": reasoning])
                        case .metadata(let meta):
                            final = meta
                        }
                    }
                    send(delta: [:], finish: Self.finishReason(final?.finishReason, calledTools: toolCallCount > 0))
                    if includeUsage, let final {
                        send(["choices": [] as [Any], "usage": Self.chatUsage(final)])
                    }
                    var done2 = ByteBuffer(); done2.writeString("data: [DONE]\n\n")
                    continuation.yield(done2)
                } catch {
                    let errPayload: [String: Any] = [
                        "error": [
                            "message": ModelsController.publicMessage(for: error),
                            "type": "server_error"
                        ]
                    ]
                    if let json = try? JSONSerialization.data(withJSONObject: errPayload),
                       let str = String(data: json, encoding: .utf8) {
                        var buf = ByteBuffer()
                        buf.writeString("event: error\ndata: \(str)\n\n")
                        continuation.yield(buf)
                    }
                }
                continuation.finish()
            }
            // A client that disconnects cancels the generation instead of leaving it running.
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func streamResponses(genRequest: GenerationRequest, model: String, requestedModel: String? = nil) -> Response {
        Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
                .connection: "keep-alive"
            ],
            body: ResponseBody(asyncSequence: sseResponsesStream(genRequest: genRequest, model: model, requestedModel: requestedModel))
        )
    }

    /// OpenAI Responses API streaming. Output items get indices in the order they appear:
    /// a `reasoning` item when the model thinks, the answer `message`, and one `function_call`
    /// item per tool call (a turn that only calls tools has no message). Each item is opened
    /// with output_item.added and closed with output_item.done; text streams as
    /// output_text.delta inside a content part, arguments as function_call_arguments.delta.
    /// The turn ends with response.completed (response.failed on error) — Codex ends a turn
    /// there; there is no `[DONE]` sentinel in this protocol. Streaming the reasoning also keeps
    /// the connection alive while a model thinks for minutes — Codex drops a stream that stays
    /// silent past its idle timeout.
    private func sseResponsesStream(genRequest: GenerationRequest, model: String, requestedModel: String? = nil) -> AsyncStream<ByteBuffer> {
        AsyncStream { continuation in
            let task = Task {
                let id = "resp_\(UUID().uuidString)"
                var messageId = "msg_\(UUID().uuidString)"
                let reasoningId = "rs_\(UUID().uuidString)"
                let createdAt = Int(Date().timeIntervalSince1970)
                var seq = 0

                func send(_ event: String, _ fields: [String: Any]) {
                    var payload = fields
                    payload["type"] = event
                    payload["sequence_number"] = seq
                    seq += 1
                    if let json = try? JSONSerialization.data(withJSONObject: payload),
                       let str = String(data: json, encoding: .utf8) {
                        var buf = ByteBuffer()
                        buf.writeString("event: \(event)\ndata: \(str)\n\n")
                        continuation.yield(buf)
                    }
                }
                func responseObject(status: String, output: [Any], extra: [String: Any] = [:]) -> [String: Any] {
                    var r: [String: Any] = ["id": id, "object": "response", "created_at": createdAt,
                                            "status": status, "model": model, "output": output]
                    r.merge(extra) { _, new in new }
                    return r
                }

                var completed: [[String: Any]] = []   // finished output items, in index order
                var reasoning: String?                // open reasoning item's text
                var messageIndex: Int?                // open message item's index
                var text = ""
                var final: GenerationResponse?

                func messageItem(status: String, text: String?) -> [String: Any] {
                    let content: [Any] = text.map { [["type": "output_text", "text": $0, "annotations": [] as [Any]]] } ?? []
                    return ["id": messageId, "type": "message", "status": status, "role": "assistant", "content": content]
                }
                func closeReasoning() {
                    guard let r = reasoning else { return }
                    send("response.reasoning_text.done", ["item_id": reasoningId, "output_index": completed.count, "content_index": 0, "text": r])
                    let done: [String: Any] = ["id": reasoningId, "type": "reasoning", "summary": [] as [Any],
                                               "content": [["type": "reasoning_text", "text": r]]]
                    send("response.output_item.done", ["output_index": completed.count, "item": done])
                    completed.append(done)
                    reasoning = nil
                }
                func closeMessage() {
                    guard let index = messageIndex else { return }
                    send("response.output_text.done", ["item_id": messageId, "output_index": index, "content_index": 0, "text": text])
                    send("response.content_part.done", [
                        "item_id": messageId, "output_index": index, "content_index": 0,
                        "part": ["type": "output_text", "text": text, "annotations": [] as [Any]]
                    ])
                    let done = messageItem(status: "completed", text: text)
                    send("response.output_item.done", ["output_index": index, "item": done])
                    completed.append(done)
                    messageIndex = nil
                    // Text after a tool call is a new message: its own id, its own text.
                    messageId = "msg_\(UUID().uuidString)"
                    text = ""
                }
                func openMessage() -> Int {
                    if let index = messageIndex { return index }
                    closeReasoning()
                    let index = completed.count
                    send("response.output_item.added", ["output_index": index, "item": messageItem(status: "in_progress", text: nil)])
                    send("response.content_part.added", [
                        "item_id": messageId, "output_index": index, "content_index": 0,
                        "part": ["type": "output_text", "text": "", "annotations": [] as [Any]]
                    ])
                    messageIndex = index
                    return index
                }
                func emitToolCall(_ call: ToolCallOutput) {
                    closeReasoning()
                    closeMessage()
                    let index = completed.count
                    let itemId = "fc_\(UUID().uuidString)"
                    var item: [String: Any] = ["id": itemId, "type": "function_call", "call_id": call.id,
                                               "name": call.name, "arguments": "", "status": "in_progress"]
                    send("response.output_item.added", ["output_index": index, "item": item])
                    send("response.function_call_arguments.delta", ["item_id": itemId, "output_index": index, "delta": call.arguments])
                    send("response.function_call_arguments.done", ["item_id": itemId, "output_index": index, "arguments": call.arguments])
                    item["arguments"] = call.arguments
                    item["status"] = "completed"
                    send("response.output_item.done", ["output_index": index, "item": item])
                    completed.append(item)
                }

                send("response.created", ["response": responseObject(status: "in_progress", output: [])])

                do {
                    let stream = try await orchestrator.generateStream(request: genRequest, modelId: requestedModel)
                    for await chunk in stream {
                        switch chunk {
                        case .text(let delta):
                            // Whitespace the template leaves after a tool call is not an answer.
                            if messageIndex == nil, completed.contains(where: { $0["type"] as? String == "function_call" }),
                               delta.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                continue
                            }
                            let index = openMessage()
                            text += delta
                            send("response.output_text.delta", [
                                "item_id": messageId, "output_index": index, "content_index": 0, "delta": delta
                            ])
                        case .reasoning(let delta):
                            // Reasoning after the answer started has no item to go to; drop it.
                            guard messageIndex == nil, completed.isEmpty else { continue }
                            if reasoning == nil {
                                send("response.output_item.added", ["output_index": 0, "item": [
                                    "id": reasoningId, "type": "reasoning", "summary": [] as [Any], "content": [] as [Any]
                                ] as [String: Any]])
                                reasoning = ""
                            }
                            reasoning! += delta
                            send("response.reasoning_text.delta", [
                                "item_id": reasoningId, "output_index": 0, "content_index": 0, "delta": delta
                            ])
                        case .toolCall(let call):
                            emitToolCall(call)
                        case .metadata(let meta):
                            final = meta
                        }
                    }
                    if Task.isCancelled { continuation.finish(); return }

                    closeReasoning()
                    // A turn that only called tools carries no message; any other turn has one,
                    // even an empty one.
                    if messageIndex != nil || !completed.contains(where: { $0["type"] as? String == "function_call" }) {
                        _ = openMessage()
                    }
                    closeMessage()

                    let inTokens = final?.promptTokens ?? 0
                    let outTokens = final?.completionTokens ?? 0
                    let usage: [String: Any] = [
                        "input_tokens": inTokens, "output_tokens": outTokens, "total_tokens": inTokens + outTokens,
                        "input_tokens_details": ["cached_tokens": final?.cachedPromptTokens ?? 0],
                        "output_tokens_details": ["reasoning_tokens": final?.reasoningTokens ?? 0]]
                    send("response.completed", ["response": responseObject(status: "completed", output: completed, extra: [
                        "usage": usage
                    ])])
                } catch {
                    send("response.failed", ["response": responseObject(status: "failed", output: [], extra: [
                        "error": ["code": "server_error", "message": ModelsController.publicMessage(for: error)]
                    ])])
                }
                continuation.finish()
            }
            // A client that disconnects cancels the generation instead of leaving it running.
            continuation.onTermination = { _ in task.cancel() }
        }
    }


    // MARK: — Tools

    struct RequestedTools {
        var tools: [ToolDefinition]?
        var choice: ToolChoice?
    }

    struct UndeclaredForcedTool: Error {
        let name: String
        var message: String { "tool_choice names \(name), which is not in tools" }
    }

    static func requestedTools(fromRequestBody data: Data) throws(UndeclaredForcedTool) -> RequestedTools {
        guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return RequestedTools() }
        let choice = body["tool_choice"] as? [String: Any]
        let forced = ((choice?["function"] as? [String: Any])?["name"] as? String) ?? (choice?["name"] as? String)
        let parallel = body["parallel_tool_calls"] as? Bool != false && choice?["disable_parallel_tool_use"] as? Bool != true
        let raw = ((body["tools"] as? [[String: Any]]) ?? []) + ((body["additional_tools"] as? [[String: Any]]) ?? [])
        let tools: [ToolDefinition] = raw.compactMap { entry in
            guard entry["type"] as? String == "function" || entry["input_schema"] != nil else { return nil }
            let spec = (entry["function"] as? [String: Any]) ?? entry
            guard let name = spec["name"] as? String, !name.isEmpty else { return nil }
            let parameters = (spec["parameters"] ?? spec["input_schema"])
                .flatMap { try? JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys, .withoutEscapingSlashes]) }
                .flatMap { String(data: $0, encoding: .utf8) }
            return ToolDefinition(name: name, description: spec["description"] as? String, parametersJSON: parameters)
        }
        let offered = tools.isEmpty ? nil : tools
        if ((body["tool_choice"] as? String) ?? (choice?["type"] as? String)) == "none" {
            return RequestedTools(tools: offered, choice: ToolChoice(callsDisabled: true))
        }
        if let forced, !tools.contains(where: { $0.name == forced }) { throw UndeclaredForcedTool(name: forced) }
        return RequestedTools(tools: offered,
                              choice: forced == nil && parallel ? nil : ToolChoice(forcedTool: forced, parallel: parallel))
    }

    struct InvalidResponseFormat: Error {
        let message: String
    }

    static func responseFormat(fromRequestBody data: Data) throws(InvalidResponseFormat) -> ResponseFormat? {
        guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let format = (body["response_format"] as? [String: Any])
                ?? ((body["text"] as? [String: Any])?["format"] as? [String: Any]) else { return nil }
        switch format["type"] as? String {
        case "text":
            return nil
        case "json_object":
            return .jsonObject
        case "json_schema":
            guard let schema = (format["json_schema"] as? [String: Any])?["schema"] ?? format["schema"],
                  let encoded = try? JSONSerialization.data(withJSONObject: schema, options: [.sortedKeys, .fragmentsAllowed]),
                  let text = String(data: encoded, encoding: .utf8) else {
                throw InvalidResponseFormat(message: "response_format json_schema needs a schema")
            }
            do {
                _ = try ResponseFormat.jsonSchema(text).grammar()
            } catch {
                throw InvalidResponseFormat(message: "response_format: \(error.message)")
            }
            return .jsonSchema(text)
        case let type:
            throw InvalidResponseFormat(message:
                "response_format type \(type.map { "'\($0)' is not supported" } ?? "is missing"); expected text, json_object or json_schema")
        }
    }

    // MARK: — Prompt

    static func chatMessages(_ messages: [ChatMessage]) -> [ChatMessageInput] {
        var toolNames: [String: String] = [:]
        for msg in messages {
            if let id = msg.callId, let name = msg.name { toolNames[id] = name }
            for call in msg.toolCalls ?? [] { if let id = call.id { toolNames[id] = call.function.name } }
        }
        var chat: [ChatMessageInput] = []
        func appendCall(_ call: ToolCallOutput) {
            if let last = chat.last, last.role == .assistant {
                chat[chat.count - 1] = ChatMessageInput(role: .assistant, text: last.text, toolCalls: (last.toolCalls ?? []) + [call])
            } else {
                chat.append(ChatMessageInput(role: .assistant, text: "", toolCalls: [call]))
            }
        }
        for msg in messages {
            switch msg.type {
            case "function_call", "custom_tool_call":
                appendCall(ToolCallOutput(id: msg.callId ?? "", name: msg.name ?? "tool", arguments: msg.arguments ?? msg.input ?? "{}"))
                continue
            case "function_call_output", "custom_tool_call_output":
                chat.append(ChatMessageInput(role: .tool, text: msg.output?.textValue ?? "",
                                             toolCallId: msg.callId, toolName: msg.callId.flatMap { toolNames[$0] }))
                continue
            case "reasoning":
                continue
            default:
                break
            }
            let text = msg.content?.textValue ?? ""
            let role: ChatMessageInput.Role
            switch msg.role {
            case "system", "developer": role = .system
            case "assistant":           role = .assistant
            case "tool":                role = .tool
            default:                    role = .user
            }
            let calls = (msg.toolCalls ?? []).map {
                ToolCallOutput(id: $0.id ?? "", name: $0.function.name, arguments: $0.function.arguments ?? "{}")
            }
            if role == .tool {
                chat.append(ChatMessageInput(role: .tool, text: text, toolCallId: msg.toolCallId,
                                             toolName: msg.toolCallId.flatMap { toolNames[$0] }))
            } else if !text.isEmpty || !calls.isEmpty {
                chat.append(ChatMessageInput(role: role, text: text, toolCalls: calls.isEmpty ? nil : calls))
            }
        }
        return chat
    }

    // MARK: — Helpers

    private func makeJSONResponse(_ body: [String: Any]) -> Response {
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        return makeDataResponse(data)
    }

    private func makeDataResponse(_ data: Data) -> Response {
        var buf = ByteBuffer(); buf.writeBytes(data)
        return Response(status: .ok, headers: [.contentType: "application/json"],
                        body: ResponseBody(byteBuffer: buf))
    }

    private func invalidParameters(_ request: GenerationRequest) -> Response? {
        do {
            _ = try request.validated()
            return nil
        } catch {
            return makeErrorResponse(status: HTTPResponse.Status(code: error.httpStatus),
                                     message: ModelsController.publicMessage(for: error), code: error.httpStatus)
        }
    }

    private func makeErrorResponse(status: HTTPResponse.Status, message: String, code: Int) -> Response {
        let body: [String: Any] = ["type": "error",
                                   "error": ["type": "invalid_request_error", "message": message] as [String: Any]]
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        var buf = ByteBuffer(); buf.writeBytes(data)
        return Response(status: status, headers: [.contentType: "application/json"],
                        body: ResponseBody(byteBuffer: buf))
    }
}

private extension HTTPResponse.Status {
    init(code: Int) { self = HTTPResponse.Status(code: code, reasonPhrase: "") }
}

