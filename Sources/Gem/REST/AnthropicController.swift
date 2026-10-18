import Foundation
import Hummingbird
import HTTPTypes

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
        let text: String?

        init(type: String = "text", text: String) {
            self.type = type
            self.text = text
        }
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
            case .blocks(let b): return b.compactMap(\.text).joined(separator: "\n")
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
            case .blocks(let b): return b.compactMap(\.text).joined(separator: "\n")
            }
        }
    }

    struct AnthropicTool: Codable, Sendable {
        let name: String
        let description: String?
    }

    struct MessagesRequest: Codable, Sendable {
        let model: String?
        let messages: [Message]
        let system: SystemPrompt?
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let stream: Bool?
        let thinking: ThinkingConfig?
        let reasoningEffort: String?
        let tools: [AnthropicTool]?

        enum CodingKeys: String, CodingKey {
            case model, messages, system, temperature, stream, thinking, tools
            case maxTokens = "max_tokens"
            case topP = "top_p"
            case reasoningEffort = "reasoning_effort"
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
            let outputTokens: Int
            let cacheReadInputTokens: Int?

            enum CodingKeys: String, CodingKey {
                case inputTokens = "input_tokens"
                case outputTokens = "output_tokens"
                case cacheReadInputTokens = "cache_read_input_tokens"
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
        // Accept any Authorization header (or none) — no validation in local mode
        let dto: MessagesRequest
        do {
            let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
            let data = Data(buffer.readableBytesView)
            let raw = String(data: data, encoding: .utf8) ?? "<non-utf8>"
            try? raw.write(toFile: "/tmp/basalt_anthropic_request.json", atomically: true, encoding: .utf8)
            do {
                dto = try JSONDecoder().decode(MessagesRequest.self, from: data)
            } catch {
                fputs("🔴 [Anthropic] decode error: \(error)\nBody: \(raw.prefix(400))\n", stderr)
                return errorResponse(status: .badRequest,
                                     message: "Decode error: \(error.localizedDescription)",
                                     type: "invalid_request_error")
            }
        }

        guard !dto.messages.isEmpty else {
            return errorResponse(status: .badRequest,
                                 message: "messages must not be empty",
                                 type: "invalid_request_error")
        }

        let prompt = buildPrompt(system: dto.system, messages: dto.messages)
        let service = ServiceRequestDetector.prepare(prompt: prompt, maxTokens: dto.maxTokens, headers: request.headers)

        let effort: String?
        if let re = dto.reasoningEffort {
            effort = re
        } else if let th = dto.thinking {
            effort = th.type == "disabled" ? "off" : (th.budgetTokens.map { $0 > 8000 ? "high" : ($0 > 2000 ? "medium" : "low") } ?? "medium")
        } else {
            effort = nil
        }

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
            maxTokens: dto.maxTokens ?? 8192,
            temperature: dto.temperature ?? 0.7,
            topP: dto.topP ?? 0.9,
            reasoningEffort: effort,
            isServiceRequest: service.isService,
            prefillDelta: prefillDelta,
            cachedTokensEstimate: cachedTokensEstimate
        )

        let requestedModel = dto.model.map { ModelsController.hfRepoId(from: $0) }  // strip claude-local/ prefix if present
        let hasTools = (dto.tools?.isEmpty == false)

        if dto.stream == true {
            return streamMessagesResponse(genRequest: genRequest, model: dto.model ?? modelId, requestedModel: requestedModel, sessionId: service.isService ? nil : sessionId, prompt: service.prompt)
        }

        do {
            let result = try await orchestrator.generate(request: genRequest, modelId: requestedModel)
            if !service.isService {
                await orchestrator.recordTurn(slotId: sessionId, prompt: service.prompt, completion: result.generatedText)
            }
            let outputText = hasTools ? (ToolCallRepairer.repairJSON(result.generatedText) ?? result.generatedText) : result.generatedText
            let response = MessagesResponse(
                id: "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))",
                type: "message",
                role: "assistant",
                content: [ContentBlock(text: outputText)],
                model: dto.model ?? modelId,
                stopReason: result.finishReason == .stop ? "end_turn" : "max_tokens",
                stopSequence: nil,
                usage: .init(inputTokens: result.promptTokens,
                             outputTokens: result.completionTokens,
                             cacheReadInputTokens: cachedTokensEstimate)
            )
            let data = (try? JSONEncoder().encode(response)) ?? Data()
            var buf = ByteBuffer()
            buf.writeBytes(data)
            return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buf))
        } catch let err {
            return errorResponse(
                status: HTTPResponse.Status(code: err.httpStatus),
                message: err.errorDescription ?? err.localizedDescription,
                type: "api_error"
            )
        }
    }

    // MARK: — Streaming (Anthropic SSE format)

    private func streamMessagesResponse(
        genRequest: GenerationRequest,
        model: String,
        requestedModel: String? = nil,
        sessionId: String? = nil,
        prompt: String? = nil
    ) -> Response {
        let headers: HTTPFields = [
            .contentType: "text/event-stream",
            .cacheControl: "no-cache",
            .connection: "keep-alive"
        ]
        let body = ResponseBody(asyncSequence: anthropicSSEStream(
            genRequest: genRequest,
            model: model,
            requestedModel: requestedModel,
            sessionId: sessionId,
            prompt: prompt
        ))
        return Response(status: .ok, headers: headers, body: body)
    }

    private func anthropicSSEStream(
        genRequest: GenerationRequest,
        model: String,
        requestedModel: String? = nil,
        sessionId: String? = nil,
        prompt: String? = nil
    ) -> AsyncStream<ByteBuffer> {
        AsyncStream { continuation in
            Task {
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

                var outputTokens = 0
                var accumulatedText = ""

                do {
                    let stream = try await orchestrator.generateStream(request: genRequest, modelId: requestedModel)
                    for await chunk in stream {
                        switch chunk {
                        case .text(let text):
                            outputTokens += 1
                            accumulatedText += text
                            open("text")
                            send("content_block_delta", [
                                "type": "content_block_delta",
                                "index": openBlock!.index,
                                "delta": ["type": "text_delta", "text": text]
                            ])
                        case .reasoning(let reasoning):
                            outputTokens += 1
                            // Reasoning after the answer started has no block to go to; drop it.
                            guard openBlock?.type != "text" else { continue }
                            open("thinking")
                            send("content_block_delta", [
                                "type": "content_block_delta",
                                "index": openBlock!.index,
                                "delta": ["type": "thinking_delta", "thinking": reasoning]
                            ])
                        case .metadata:
                            break
                        }
                    }

                    if let sessionId, let prompt, !accumulatedText.isEmpty {
                        await orchestrator.recordTurn(slotId: sessionId, prompt: prompt, completion: accumulatedText)
                    }
                } catch {
                    let errPayload: [String: Any] = [
                        "type": "error",
                        "error": [
                            "type": "api_error",
                            "message": error.localizedDescription
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

                // An empty answer still carries one (empty) text block.
                open("text")
                send("content_block_stop", ["type": "content_block_stop", "index": openBlock!.index])

                // message_delta
                send("message_delta", [
                    "type": "message_delta",
                    "delta": ["stop_reason": "end_turn", "stop_sequence": NSNull()],
                    "usage": [
                        "output_tokens": outputTokens,
                        "cache_read_input_tokens": genRequest.cachedTokensEstimate ?? 0
                    ]
                ])

                // message_stop
                send("message_stop", ["type": "message_stop"])

                continuation.finish()
            }
        }
    }

    // MARK: — Prompt builder

    /// Converts Anthropic messages + system prompt into a flat string.
    /// MLX engine wraps it in UserInput(chat: [.user(prompt)]) and applies
    /// the model's own chat template on top.
    private func buildPrompt(system: SystemPrompt?, messages: [Message]) -> String {
        var parts: [String] = []

        if let system {
            let text = system.textValue
            if !text.isEmpty { parts.append("[System]: \(text)") }
        }

        for msg in messages {
            let text = msg.content.textValue
            switch msg.role {
            case "user":      parts.append("[User]: \(text)")
            case "assistant": parts.append("[Assistant]: \(text)")
            default:          parts.append(text)
            }
        }

        return parts.joined(separator: "\n\n")
    }

    // MARK: — Helpers

    private func jsonResponse<T: Encodable>(status: HTTPResponse.Status, body: T) throws -> Response {
        let data = try JSONEncoder().encode(body)
        var buf = ByteBuffer()
        buf.writeBytes(data)
        return Response(status: status, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buf))
    }

    /// Anthropic error envelope format: {"type":"error","error":{"type":"...","message":"..."}}
    private func errorResponse(status: HTTPResponse.Status, message: String, type errorType: String) -> Response {
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

