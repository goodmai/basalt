import Foundation
import Hummingbird
import HTTPTypes

// MARK: — OpenAI-compatible endpoints (no auth required)
//
// Usage with Claude Code:
//   export OPENAI_API_KEY=local
//   export OPENAI_BASE_URL=http://localhost:8080
//   claude --model openai/gemma-4-31b-it-4bit
//
// Routes (all public, no JWT):
//   GET  /v1/models
//   POST /v1/chat/completions   — OpenAI chat format, streaming supported
//   POST /v1/generate           — raw prompt, no auth

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
            case .blocks(let b): return b.compactMap(\.text).joined(separator: "\n")
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

        init(type: String? = nil, role: String? = nil, content: MessageContent?,
             name: String? = nil, arguments: String? = nil, input: String? = nil, output: MessageContent? = nil) {
            self.type = type
            self.role = role
            self.content = content
            self.name = name
            self.arguments = arguments
            self.input = input
            self.output = output
        }
    }

    struct ChatCompletionRequest: Codable, Sendable {
        let model: String?
        let messages: [ChatMessage]
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let stream: Bool?
        let reasoningEffort: String?
        let tools: [ResponseTool]?

        enum CodingKeys: String, CodingKey {
            case model, messages, temperature, stream, tools
            case maxTokens = "max_tokens"
            case topP = "top_p"
            case reasoningEffort = "reasoning_effort"
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
        let stream: Bool?
        /// Responses API form: `"reasoning": {"effort": "low"}`.
        let reasoning: ReasoningConfig?
        /// Chat Completions form, accepted here too for clients that reuse it.
        let reasoningEffort: String?
        let tools: [ResponseTool]?
        let additionalTools: [ResponseTool]?

        struct ReasoningConfig: Codable, Sendable {
            let effort: String?
        }

        var effectiveReasoningEffort: String? { reasoning?.effort ?? reasoningEffort }

        enum CodingKeys: String, CodingKey {
            case model, instructions, input, messages, temperature, stream, tools, reasoning
            case maxOutputTokens = "max_output_tokens"
            case maxTokens = "max_tokens"
            case topP = "top_p"
            case reasoningEffort = "reasoning_effort"
            case additionalTools = "additional_tools"
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

            enum CodingKeys: String, CodingKey {
                case id, type, status, role, content, name, arguments
                case callId = "call_id"
            }
        }

        struct OutputContent: Codable, Sendable {
            let type: String
            let text: String?
        }

        struct Usage: Codable, Sendable {
            let inputTokens: Int
            let outputTokens: Int
            let totalTokens: Int
            let inputTokensDetails: InputTokensDetails?

            struct InputTokensDetails: Codable, Sendable {
                let cachedTokens: Int
                enum CodingKeys: String, CodingKey {
                    case cachedTokens = "cached_tokens"
                }
            }

            enum CodingKeys: String, CodingKey {
                case inputTokens = "input_tokens"
                case outputTokens = "output_tokens"
                case totalTokens = "total_tokens"
                case inputTokensDetails = "input_tokens_details"
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
            let completionTokens: Int
            let totalTokens: Int
            let promptTokensDetails: PromptTokensDetails?

            struct PromptTokensDetails: Codable, Sendable {
                let cachedTokens: Int
                enum CodingKeys: String, CodingKey {
                    case cachedTokens = "cached_tokens"
                }
            }

            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case completionTokens = "completion_tokens"
                case totalTokens = "total_tokens"
                case promptTokensDetails = "prompt_tokens_details"
            }
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

        // Save raw body for diagnosis; visible at /tmp/basalt_openai_request.json
        let rawString = String(data: data, encoding: .utf8) ?? "<non-utf8>"
        try? rawString.write(toFile: "/tmp/basalt_openai_request.json", atomically: true, encoding: .utf8)

        let dto: ChatCompletionRequest
        do {
            dto = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
        } catch {
            fputs("🔴 [OpenAI] decode error: \(error)\nBody: \(rawString.prefix(300))\n", stderr)
            return makeErrorResponse(status: .badRequest,
                                     message: "Cannot decode request: \(error.localizedDescription)",
                                     code: 400)
        }

        guard !dto.messages.isEmpty else {
            return makeErrorResponse(status: .badRequest, message: "messages must not be empty", code: 400)
        }

        let prompt = messagesToPrompt(dto.messages)
        let service = ServiceRequestDetector.prepare(prompt: prompt, maxTokens: dto.maxTokens, headers: request.headers)

        let explicitSessionId = (HTTPField.Name("x-session-id").flatMap { request.headers[$0] })
            ?? (HTTPField.Name("x-conversation-id").flatMap { request.headers[$0] })
        let sessionId = explicitSessionId ?? "session_\(UUID().uuidString.prefix(8))"

        var prefillDelta: String? = nil
        var cachedTokensEstimate: Int? = nil
        if !service.isService {
            await orchestrator.parkInactiveSlots(except: sessionId)
            let delta = await orchestrator.computePrefillDelta(slotId: sessionId, prompt: service.prompt)
            prefillDelta = delta.newPromptSuffix
            cachedTokensEstimate = delta.cachedPromptTokens
        }

        let genRequest = GenerationRequest(
            prompt: service.prompt,
            maxTokens: dto.maxTokens,
            temperature: dto.temperature ?? 0.7,
            topP: dto.topP ?? 0.9,
            reasoningEffort: dto.reasoningEffort,
            isServiceRequest: service.isService,
            prefillDelta: prefillDelta,
            cachedTokensEstimate: cachedTokensEstimate
        )
        let requestedModel = dto.model.map { ModelsController.hfRepoId(from: $0) }  // strip claude-local/ prefix if present
        let hasTools = (dto.tools?.isEmpty == false)

        if dto.stream == true {
            return streamResponse(genRequest: genRequest, model: dto.model ?? modelId, requestedModel: requestedModel, sessionId: service.isService ? nil : sessionId, prompt: service.prompt)
        }

        do {
            let result = try await orchestrator.generate(request: genRequest, modelId: requestedModel)
            if !service.isService {
                await orchestrator.recordTurn(slotId: sessionId, prompt: service.prompt, completion: result.generatedText)
            }
            let outputText = hasTools ? (ToolCallRepairer.repairJSON(result.generatedText) ?? result.generatedText) : result.generatedText
            let response = ChatCompletionResponse(
                id: "chatcmpl-\(UUID().uuidString)",
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: dto.model ?? modelId,
                choices: [.init(
                    index: 0,
                    message: ChatMessage(role: "assistant", content: .text(outputText)),
                    finishReason: result.finishReason == GenerationResponse.FinishReason.stop ? "stop" : "length"
                )],
                usage: .init(
                    promptTokens: result.promptTokens,
                    completionTokens: result.completionTokens,
                    totalTokens: result.promptTokens + result.completionTokens,
                    promptTokensDetails: cachedTokensEstimate != nil ? .init(cachedTokens: cachedTokensEstimate!) : nil
                )
            )
            let encoded = (try? JSONEncoder().encode(response)) ?? Data()
            return makeDataResponse(encoded)
        } catch {
            fputs("🔴 [OpenAI] generate failed: \(error)\n", stderr)
            return makeErrorResponse(status: .internalServerError,
                                     message: error.localizedDescription, code: 500)
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

        let conversation: String
        if let input = dto.input {
            switch input {
            case .text(let t): conversation = t
            case .messages(let msgs): conversation = messagesToPrompt(msgs)
            }
        } else if let msgs = dto.messages, !msgs.isEmpty {
            conversation = messagesToPrompt(msgs)
        } else {
            conversation = ""
        }

        guard !conversation.isEmpty else {
            return makeErrorResponse(status: .badRequest, message: "input or messages must not be empty", code: 400)
        }
        // `instructions` is the system prompt of the Responses API; dropping it loses the agent's rules.
        let prompt = dto.instructions.map { $0.isEmpty ? conversation : "[System]: \($0)\n\n\(conversation)" } ?? conversation

        let maxT = dto.maxOutputTokens ?? dto.maxTokens
        let service = ServiceRequestDetector.prepare(prompt: prompt, maxTokens: maxT, headers: request.headers)

        let explicitSessionId = (HTTPField.Name("x-session-id").flatMap { request.headers[$0] })
            ?? (HTTPField.Name("x-conversation-id").flatMap { request.headers[$0] })
        let sessionId = explicitSessionId ?? "session_\(UUID().uuidString.prefix(8))"

        var prefillDelta: String? = nil
        var cachedTokensEstimate: Int? = nil
        if !service.isService {
            await orchestrator.parkInactiveSlots(except: sessionId)
            let delta = await orchestrator.computePrefillDelta(slotId: sessionId, prompt: service.prompt)
            prefillDelta = delta.newPromptSuffix
            cachedTokensEstimate = delta.cachedPromptTokens
        }

        let genRequest = GenerationRequest(
            prompt: service.prompt,
            maxTokens: maxT,
            temperature: dto.temperature ?? 0.7,
            topP: dto.topP ?? 0.9,
            reasoningEffort: dto.effectiveReasoningEffort,
            isServiceRequest: service.isService,
            prefillDelta: prefillDelta,
            cachedTokensEstimate: cachedTokensEstimate,
            tools: Self.functionTools(fromRequestBody: data)
        )
        let requestedModel = dto.model.map { ModelsController.hfRepoId(from: $0) }
        let hasTools = (dto.tools?.isEmpty == false) || (dto.additionalTools?.isEmpty == false)

        if dto.stream == true {
            return streamResponses(genRequest: genRequest, model: dto.model ?? modelId, requestedModel: requestedModel, sessionId: service.isService ? nil : sessionId, prompt: service.prompt)
        }

        do {
            let result = try await orchestrator.generate(request: genRequest, modelId: requestedModel)
            if !service.isService {
                await orchestrator.recordTurn(slotId: sessionId, prompt: service.prompt, completion: result.generatedText)
            }
            let outputText = hasTools ? (ToolCallRepairer.repairJSON(result.generatedText) ?? result.generatedText) : result.generatedText
            let response = ResponsesResponse(
                id: "resp_\(UUID().uuidString)",
                object: "response",
                createdAt: Int(Date().timeIntervalSince1970),
                status: "completed",
                model: dto.model ?? modelId,
                // The message (when the model said anything or called nothing), then its tool calls.
                output: (result.generatedText.isEmpty && result.toolCalls != nil ? [] : [
                    .init(
                        id: "msg_\(UUID().uuidString)",
                        type: "message",
                        status: "completed",
                        role: "assistant",
                        content: [.init(type: "output_text", text: outputText)]
                    )
                ]) + (result.toolCalls ?? []).map { call in
                    .init(id: "fc_\(UUID().uuidString)", type: "function_call", status: "completed",
                          callId: call.id, name: call.name, arguments: call.arguments)
                },
                usage: .init(
                    inputTokens: result.promptTokens,
                    outputTokens: result.completionTokens,
                    totalTokens: result.promptTokens + result.completionTokens,
                    inputTokensDetails: cachedTokensEstimate != nil ? .init(cachedTokens: cachedTokensEstimate!) : nil
                )
            )
            let encoded = (try? JSONEncoder().encode(response)) ?? Data()
            return makeDataResponse(encoded)
        } catch {
            return makeErrorResponse(status: .internalServerError,
                                     message: error.localizedDescription, code: 500)
        }
    }

    // MARK: — POST /v1/generate (no auth, raw prompt)

    @Sendable
    func generateNoAuth(request: Request, context: BasicRequestContext) async throws -> Response {
        let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
        let data = Data(buffer.readableBytesView)
        do {
            let dto = try JSONDecoder().decode(GenerationRequest.self, from: data)
            let result = try await orchestrator.generate(request: dto)
            let encoded = try JSONEncoder().encode(result)
            return makeDataResponse(encoded)
        } catch {
            return makeErrorResponse(status: .badRequest,
                                     message: error.localizedDescription, code: 400)
        }
    }

    // MARK: — Streaming

    private func streamResponse(
        genRequest: GenerationRequest,
        model: String,
        requestedModel: String? = nil,
        sessionId: String? = nil,
        prompt: String? = nil
    ) -> Response {
        Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
                .connection: "keep-alive"
            ],
            body: ResponseBody(asyncSequence: sseStream(
                genRequest: genRequest,
                model: model,
                requestedModel: requestedModel,
                sessionId: sessionId,
                prompt: prompt
            ))
        )
    }

    private func sseStream(
        genRequest: GenerationRequest,
        model: String,
        requestedModel: String? = nil,
        sessionId: String? = nil,
        prompt: String? = nil
    ) -> AsyncStream<ByteBuffer> {
        AsyncStream { continuation in
            let task = Task {
                let id = "chatcmpl-\(UUID().uuidString)"
                let created = Int(Date().timeIntervalSince1970)
                var accumulatedText = ""
                do {
                    let stream = try await orchestrator.generateStream(request: genRequest, modelId: requestedModel)
                    for await chunk in stream {
                        switch chunk {
                        case .toolCall:  // this path declares no tools, so a call has no client to go to
                            break
                        case .text(let text):
                            accumulatedText += text
                            let delta: [String: Any] = [
                                "id": id, "object": "chat.completion.chunk",
                                "created": created, "model": model,
                                "choices": [["index": 0,
                                             "delta": ["role": "assistant", "content": text] as [String: Any],
                                             "finish_reason": NSNull()] as [String: Any]]
                            ]
                            if let json = try? JSONSerialization.data(withJSONObject: delta),
                               let str = String(data: json, encoding: .utf8) {
                                var buf = ByteBuffer(); buf.writeString("data: \(str)\n\n")
                                continuation.yield(buf)
                            }
                        case .reasoning(let reasoning):
                            let delta: [String: Any] = [
                                "id": id, "object": "chat.completion.chunk",
                                "created": created, "model": model,
                                "choices": [["index": 0,
                                             "delta": ["role": "assistant", "reasoning": reasoning] as [String: Any],
                                             "finish_reason": NSNull()] as [String: Any]]
                            ]
                            if let json = try? JSONSerialization.data(withJSONObject: delta),
                               let str = String(data: json, encoding: .utf8) {
                                var buf = ByteBuffer(); buf.writeString("data: \(str)\n\n")
                                continuation.yield(buf)
                            }
                        case .metadata:
                            break
                        }
                    }
                    if let sessionId, let prompt, !accumulatedText.isEmpty {
                        await orchestrator.recordTurn(slotId: sessionId, prompt: prompt, completion: accumulatedText)
                    }
                    let done: [String: Any] = [
                        "id": id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [["index": 0, "delta": [:] as [String: Any], "finish_reason": "stop"] as [String: Any]]
                    ]
                    if let json = try? JSONSerialization.data(withJSONObject: done),
                       let str = String(data: json, encoding: .utf8) {
                        var buf = ByteBuffer(); buf.writeString("data: \(str)\n\n")
                        continuation.yield(buf)
                    }
                    var done2 = ByteBuffer(); done2.writeString("data: [DONE]\n\n")
                    continuation.yield(done2)
                } catch {
                    let errPayload: [String: Any] = [
                        "error": [
                            "message": error.localizedDescription,
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

    private func streamResponses(
        genRequest: GenerationRequest,
        model: String,
        requestedModel: String? = nil,
        sessionId: String? = nil,
        prompt: String? = nil
    ) -> Response {
        Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
                .connection: "keep-alive"
            ],
            body: ResponseBody(asyncSequence: sseResponsesStream(
                genRequest: genRequest,
                model: model,
                requestedModel: requestedModel,
                sessionId: sessionId,
                prompt: prompt
            ))
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
    private func sseResponsesStream(
        genRequest: GenerationRequest,
        model: String,
        requestedModel: String? = nil,
        sessionId: String? = nil,
        prompt: String? = nil
    ) -> AsyncStream<ByteBuffer> {
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

                    // #53: the finished turn becomes the conversation slot's cached prefix.
                    if let sessionId, let prompt, !text.isEmpty {
                        await orchestrator.recordTurn(slotId: sessionId, prompt: prompt, completion: text)
                    }

                    closeReasoning()
                    // A turn that only called tools carries no message; any other turn has one,
                    // even an empty one.
                    if messageIndex != nil || !completed.contains(where: { $0["type"] as? String == "function_call" }) {
                        _ = openMessage()
                    }
                    closeMessage()

                    let inTokens = final?.promptTokens ?? (genRequest.cachedTokensEstimate ?? 0)
                    let outTokens = final?.completionTokens ?? 0
                    send("response.completed", ["response": responseObject(status: "completed", output: completed, extra: [
                        "usage": ["input_tokens": inTokens, "output_tokens": outTokens, "total_tokens": inTokens + outTokens]
                    ])])
                } catch {
                    send("response.failed", ["response": responseObject(status: "failed", output: [], extra: [
                        "error": ["code": "server_error", "message": error.localizedDescription]
                    ])])
                }
                continuation.finish()
            }
            // A client that disconnects cancels the generation instead of leaving it running.
            continuation.onTermination = { _ in task.cancel() }
        }
    }


    // MARK: — Tools

    /// `function` tools from `tools` and `additional_tools`, in the Responses form
    /// (`{type, name, description, parameters}`) or the Chat Completions form
    /// (`{type, function: {name, description, parameters}}`). Read from the raw body because the
    /// parameter schema is arbitrary JSON. Built-in tools (local_shell, web_search, …) have no
    /// function schema for the template and are skipped.
    static func functionTools(fromRequestBody data: Data) -> [ToolDefinition]? {
        guard let body = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let raw = ((body["tools"] as? [[String: Any]]) ?? []) + ((body["additional_tools"] as? [[String: Any]]) ?? [])
        let tools: [ToolDefinition] = raw.compactMap { entry in
            guard entry["type"] as? String == "function" else { return nil }
            let spec = (entry["function"] as? [String: Any]) ?? entry
            guard let name = spec["name"] as? String, !name.isEmpty else { return nil }
            let parameters = spec["parameters"].flatMap { try? JSONSerialization.data(withJSONObject: $0) }
                .flatMap { String(data: $0, encoding: .utf8) }
            return ToolDefinition(name: name, description: spec["description"] as? String, parametersJSON: parameters)
        }
        return tools.isEmpty ? nil : tools
    }

    // MARK: — Prompt

    /// Flattens the conversation to plain text and lets the tokenizer's own chat
    /// template wrap it (see MLXInferenceEngine.generateStream).
    ///
    /// Do NOT hand-roll `<|turn>` / `<|channel>thought` markers here: a prompt ending
    /// in a thought marker puts the engine's stream parser into thinking mode for the
    /// whole generation, so every token lands in the reasoning channel and
    /// `generatedText` comes back empty. Mirrors AnthropicController.buildPrompt.
    private func messagesToPrompt(_ messages: [ChatMessage]) -> String {
        var parts: [String] = []
        for msg in messages {
            // Responses API history items: keep tool calls and their results in the transcript,
            // otherwise a coding agent's next turn loses what its tools returned.
            switch msg.type {
            case "function_call", "custom_tool_call":
                parts.append("[Assistant called \(msg.name ?? "tool")]: \(msg.arguments ?? msg.input ?? "{}")")
                continue
            case "function_call_output", "custom_tool_call_output":
                parts.append("[Tool result]: \(msg.output?.textValue ?? "")")
                continue
            case "reasoning":
                continue
            default:
                break
            }
            guard let text = msg.content?.textValue, !text.isEmpty else { continue }
            let role = msg.role ?? "user"
            switch role {
            case "system", "developer": parts.append("[System]: \(text)")
            case "user":      parts.append("[User]: \(text)")
            case "assistant": parts.append("[Assistant]: \(text)")
            default:          parts.append("[\(role.capitalized)]: \(text)")
            }
        }
        return parts.joined(separator: "\n\n")
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

