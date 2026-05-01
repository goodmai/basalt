import Foundation
import Hummingbird

// MARK: — Anthropic Messages API (no auth required)
//
// Implements POST /v1/messages so Claude Code can use Gemm as a drop-in backend:
//
//   export ANTHROPIC_API_KEY=local        # any non-empty string
//   export ANTHROPIC_BASE_URL=http://localhost:8080
//   claude                                # routes all traffic to local Gemm server
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

    struct MessagesRequest: Codable, Sendable {
        let model: String?
        let messages: [Message]
        let system: String?
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let stream: Bool?

        enum CodingKeys: String, CodingKey {
            case model, messages, system, temperature, stream
            case maxTokens = "max_tokens"
            case topP = "top_p"
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

            enum CodingKeys: String, CodingKey {
                case inputTokens = "input_tokens"
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
    func messages(request: Request, context: GemmaRequestContext) async throws -> Response {
        // Accept any Authorization header (or none) — no validation in local mode
        let dto: MessagesRequest
        do {
            let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
            let data = Data(buffer.readableBytesView)
            dto = try JSONDecoder().decode(MessagesRequest.self, from: data)
        } catch {
            return errorResponse(status: .badRequest,
                                 message: "Invalid JSON: \(error.localizedDescription)",
                                 type: "invalid_request_error")
        }

        guard !dto.messages.isEmpty else {
            return errorResponse(status: .badRequest,
                                 message: "messages must not be empty",
                                 type: "invalid_request_error")
        }

        let prompt = buildPrompt(system: dto.system, messages: dto.messages)
        let genRequest = GenerationRequest(
            prompt: prompt,
            maxTokens: dto.maxTokens ?? 8192,
            temperature: dto.temperature ?? 0.7,
            topP: dto.topP ?? 0.9
        )

        if dto.stream == true {
            return streamMessagesResponse(genRequest: genRequest, model: dto.model ?? modelId)
        }

        do {
            let result = try await orchestrator.generate(request: genRequest)
            let response = MessagesResponse(
                id: "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))",
                type: "message",
                role: "assistant",
                content: [ContentBlock(text: result.generatedText)],
                model: dto.model ?? modelId,
                stopReason: result.finishReason == .stop ? "end_turn" : "max_tokens",
                stopSequence: nil,
                usage: .init(inputTokens: result.promptTokens,
                             outputTokens: result.completionTokens)
            )
            return try jsonResponse(status: .ok, body: response)
        } catch let err as GemError {
            return errorResponse(
                status: HTTPResponse.Status(code: err.httpStatus),
                message: err.errorDescription ?? err.localizedDescription,
                type: "api_error"
            )
        }
    }

    // MARK: — Streaming (Anthropic SSE format)

    private func streamMessagesResponse(genRequest: GenerationRequest, model: String) -> Response {
        let headers: HTTPFields = [
            .contentType: "text/event-stream",
            .cacheControl: "no-cache",
            .connection: "keep-alive"
        ]
        let body = ResponseBody(asyncSequence: anthropicSSEStream(genRequest: genRequest, model: model))
        return Response(status: .ok, headers: headers, body: body)
    }

    private func anthropicSSEStream(genRequest: GenerationRequest, model: String) -> AsyncStream<ByteBuffer> {
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

                // content_block_start
                send("content_block_start", [
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": ["type": "text", "text": ""]
                ])

                // ping
                send("ping", ["type": "ping"])

                var outputTokens = 0

                do {
                    let stream = try await orchestrator.generateStream(request: genRequest)
                    for await chunk in stream {
                        if case .text(let text) = chunk {
                            outputTokens += 1
                            send("content_block_delta", [
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": ["type": "text_delta", "text": text]
                            ])
                        }
                    }
                } catch {
                    var buf = ByteBuffer()
                    buf.writeString("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"\(error.localizedDescription)\"}}\n\n")
                    continuation.yield(buf)
                    continuation.finish()
                    return
                }

                // content_block_stop
                send("content_block_stop", ["type": "content_block_stop", "index": 0])

                // message_delta
                send("message_delta", [
                    "type": "message_delta",
                    "delta": ["stop_reason": "end_turn", "stop_sequence": NSNull()],
                    "usage": ["output_tokens": outputTokens]
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
    private func buildPrompt(system: String?, messages: [Message]) -> String {
        var parts: [String] = []

        if let system, !system.isEmpty {
            parts.append("[System]: \(system)")
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
