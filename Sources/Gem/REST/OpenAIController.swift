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

        init(type: String? = nil, role: String? = nil, content: MessageContent?) {
            self.type = type
            self.role = role
            self.content = content
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
        let input: ResponseInput?
        let messages: [ChatMessage]?
        let maxOutputTokens: Int?
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let stream: Bool?
        let reasoningEffort: String?
        let tools: [ResponseTool]?
        let additionalTools: [ResponseTool]?

        enum CodingKeys: String, CodingKey {
            case model, input, messages, temperature, stream, tools
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

        struct OutputItem: Codable, Sendable {
            let id: String
            let type: String
            let status: String
            let role: String
            let content: [OutputContent]
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
        let isService = ServiceRequestDetector.isTitleOrServiceRequest(prompt: prompt, maxTokens: dto.maxTokens)
        let optInCompact = request.headers.isFlagTrue("x-compact-prompt") || request.headers.isFlagTrue("x-service-request")
        let finalPrompt = (isService && optInCompact) ? ServiceRequestDetector.compactTitlePrompt(prompt) : prompt

        let explicitSessionId = (HTTPField.Name("x-session-id").flatMap { request.headers[$0] })
            ?? (HTTPField.Name("x-conversation-id").flatMap { request.headers[$0] })
        let sessionId = explicitSessionId ?? "session_\(UUID().uuidString.prefix(8))"

        var prefillDelta: String? = nil
        var cachedTokensEstimate: Int? = nil
        if !isService {
            await orchestrator.parkInactiveSlots(except: sessionId)
            let delta = await orchestrator.computePrefillDelta(slotId: sessionId, prompt: finalPrompt)
            prefillDelta = delta.newPromptSuffix
            cachedTokensEstimate = delta.cachedPromptTokens
        }

        let genRequest = GenerationRequest(
            prompt: finalPrompt,
            maxTokens: dto.maxTokens,
            temperature: dto.temperature ?? 0.7,
            topP: dto.topP ?? 0.9,
            reasoningEffort: dto.reasoningEffort,
            isServiceRequest: isService,
            prefillDelta: prefillDelta,
            cachedTokensEstimate: cachedTokensEstimate
        )
        let requestedModel = dto.model.map { ModelsController.hfRepoId(from: $0) }  // strip claude-local/ prefix if present
        let hasTools = (dto.tools?.isEmpty == false)

        if dto.stream == true {
            return streamResponse(genRequest: genRequest, model: dto.model ?? modelId, requestedModel: requestedModel, sessionId: isService ? nil : sessionId, prompt: finalPrompt)
        }

        do {
            let result = try await orchestrator.generate(request: genRequest, modelId: requestedModel)
            if !isService {
                await orchestrator.recordTurn(slotId: sessionId, prompt: finalPrompt, completion: result.generatedText)
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

        let prompt: String
        if let input = dto.input {
            switch input {
            case .text(let t): prompt = t
            case .messages(let msgs): prompt = messagesToPrompt(msgs)
            }
        } else if let msgs = dto.messages, !msgs.isEmpty {
            prompt = messagesToPrompt(msgs)
        } else {
            prompt = ""
        }

        guard !prompt.isEmpty else {
            return makeErrorResponse(status: .badRequest, message: "input or messages must not be empty", code: 400)
        }

        let maxT = dto.maxOutputTokens ?? dto.maxTokens
        let isService = ServiceRequestDetector.isTitleOrServiceRequest(prompt: prompt, maxTokens: maxT)
        let optInCompact = request.headers.isFlagTrue("x-compact-prompt") || request.headers.isFlagTrue("x-service-request")
        let finalPrompt = (isService && optInCompact) ? ServiceRequestDetector.compactTitlePrompt(prompt) : prompt

        let explicitSessionId = (HTTPField.Name("x-session-id").flatMap { request.headers[$0] })
            ?? (HTTPField.Name("x-conversation-id").flatMap { request.headers[$0] })
        let sessionId = explicitSessionId ?? "session_\(UUID().uuidString.prefix(8))"

        var prefillDelta: String? = nil
        var cachedTokensEstimate: Int? = nil
        if !isService {
            await orchestrator.parkInactiveSlots(except: sessionId)
            let delta = await orchestrator.computePrefillDelta(slotId: sessionId, prompt: finalPrompt)
            prefillDelta = delta.newPromptSuffix
            cachedTokensEstimate = delta.cachedPromptTokens
        }

        let genRequest = GenerationRequest(
            prompt: finalPrompt,
            maxTokens: maxT,
            temperature: dto.temperature ?? 0.7,
            topP: dto.topP ?? 0.9,
            reasoningEffort: dto.reasoningEffort,
            isServiceRequest: isService,
            prefillDelta: prefillDelta,
            cachedTokensEstimate: cachedTokensEstimate
        )
        let requestedModel = dto.model.map { ModelsController.hfRepoId(from: $0) }
        let hasTools = (dto.tools?.isEmpty == false) || (dto.additionalTools?.isEmpty == false)

        if dto.stream == true {
            return streamResponses(genRequest: genRequest, model: dto.model ?? modelId, requestedModel: requestedModel, sessionId: isService ? nil : sessionId, prompt: finalPrompt)
        }

        do {
            let result = try await orchestrator.generate(request: genRequest, modelId: requestedModel)
            if !isService {
                await orchestrator.recordTurn(slotId: sessionId, prompt: finalPrompt, completion: result.generatedText)
            }
            let outputText = hasTools ? (ToolCallRepairer.repairJSON(result.generatedText) ?? result.generatedText) : result.generatedText
            let response = ResponsesResponse(
                id: "resp_\(UUID().uuidString)",
                object: "response",
                createdAt: Int(Date().timeIntervalSince1970),
                status: "completed",
                model: dto.model ?? modelId,
                output: [
                    .init(
                        id: "item_\(UUID().uuidString)",
                        type: "message",
                        status: "completed",
                        role: "assistant",
                        content: [.init(type: "text", text: outputText)]
                    )
                ],
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
            Task {
                let id = "chatcmpl-\(UUID().uuidString)"
                let created = Int(Date().timeIntervalSince1970)
                var accumulatedText = ""
                do {
                    let stream = try await orchestrator.generateStream(request: genRequest, modelId: requestedModel)
                    for await chunk in stream {
                        switch chunk {
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

    private func sseResponsesStream(
        genRequest: GenerationRequest,
        model: String,
        requestedModel: String? = nil,
        sessionId: String? = nil,
        prompt: String? = nil
    ) -> AsyncStream<ByteBuffer> {
        AsyncStream { continuation in
            Task {
                let id = "resp_\(UUID().uuidString)"
                let itemId = "item_\(id)"
                var seq = 0

                func sendEvent(_ event: String, _ dataDict: [String: Any]) {
                    var payload = dataDict
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

                sendEvent("response.created", [
                    "id": id,
                    "response": [
                        "id": id,
                        "object": "response",
                        "status": "in_progress",
                        "model": model,
                        "output": [] as [Any]
                    ] as [String: Any]
                ])

                sendEvent("response.output_item.added", [
                    "id": id,
                    "output_index": 0,
                    "item": [
                        "id": itemId,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [] as [Any]
                    ] as [String: Any]
                ])

                var accumulatedText = ""
                var outputTokens = 0
                var finalResponse: GenerationResponse?

                do {
                    let stream = try await orchestrator.generateStream(request: genRequest, modelId: requestedModel)
                    for await chunk in stream {
                        switch chunk {
                        case .text(let text):
                            accumulatedText += text
                            outputTokens += 1
                            sendEvent("response.output_text.delta", [
                                "id": id,
                                "item_id": itemId,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": text
                            ])
                            sendEvent("response.text.delta", [
                                "id": id,
                                "item_id": itemId,
                                "output_index": 0,
                                "delta": text
                            ])
                        case .reasoning(let reasoning):
                            sendEvent("response.reasoning.delta", [
                                "item_id": itemId,
                                "output_index": 0,
                                "delta": reasoning
                            ])
                        case .metadata(let meta):
                            finalResponse = meta
                        }
                    }

                    if let sessionId, let prompt, !accumulatedText.isEmpty {
                        await orchestrator.recordTurn(slotId: sessionId, prompt: prompt, completion: accumulatedText)
                    }

                    sendEvent("response.output_item.done", [
                        "id": id,
                        "output_index": 0,
                        "status": "completed",
                        "item": [
                            "id": itemId,
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                [
                                    "type": "text",
                                    "text": accumulatedText
                                ]
                            ] as [Any]
                        ] as [String: Any]
                    ])

                    let inTokens = finalResponse?.promptTokens ?? (genRequest.cachedTokensEstimate ?? 0)
                    let outTokens = finalResponse?.completionTokens ?? outputTokens
                    let usage: [String: Any] = [
                        "input_tokens": inTokens,
                        "output_tokens": outTokens,
                        "total_tokens": inTokens + outTokens
                    ]

                    let completedResponse: [String: Any] = [
                        "id": id,
                        "object": "response",
                        "status": "completed",
                        "model": model,
                        "output": [
                            [
                                "id": itemId,
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    [
                                        "type": "text",
                                        "text": accumulatedText
                                    ]
                                ] as [Any]
                            ] as [String: Any]
                        ],
                        "usage": usage
                    ]

                    sendEvent("response.done", [
                        "id": id,
                        "object": "response",
                        "status": "completed",
                        "model": model,
                        "response": completedResponse,
                        "usage": usage
                    ])

                    var doneBuf = ByteBuffer()
                    doneBuf.writeString("data: [DONE]\n\n")
                    continuation.yield(doneBuf)
                } catch {
                    let errObj: [String: Any] = [
                        "error": [
                            "type": "server_error",
                            "message": error.localizedDescription
                        ] as [String: Any]
                    ]
                    sendEvent("error", errObj)
                }
                continuation.finish()
            }
        }
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

private extension HTTPFields {
    func isFlagTrue(_ name: String) -> Bool {
        guard let fieldName = HTTPField.Name(name) else { return false }
        return self[fieldName] == "true"
    }
}
