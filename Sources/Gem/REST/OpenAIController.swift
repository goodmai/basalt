import Foundation
import Hummingbird

// MARK: — OpenAI-compatible endpoints (no auth required)
//
// Enables integration with Claude Code and other OpenAI-compatible clients:
//
//   export OPENAI_API_KEY=local
//   export OPENAI_BASE_URL=http://localhost:8080
//   claude --model openai/gemma-4-31b-it-4bit
//
// Routes (all public, no JWT):
//   GET  /v1/models
//   POST /v1/chat/completions       — OpenAI chat format, streaming supported
//   POST /v1/generate               — simplified raw prompt, no auth

struct OpenAIController: Sendable {

    let orchestrator: ModelOrchestratorActor
    let modelId: String

    // MARK: — OpenAI DTOs

    struct ChatMessage: Codable, Sendable {
        let role: String
        let content: String
    }

    struct ChatCompletionRequest: Codable, Sendable {
        let model: String?
        let messages: [ChatMessage]
        let maxTokens: Int?
        let temperature: Double?
        let topP: Double?
        let stream: Bool?

        enum CodingKeys: String, CodingKey {
            case model, messages, temperature, stream
            case maxTokens = "max_tokens"
            case topP = "top_p"
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

            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case completionTokens = "completion_tokens"
                case totalTokens = "total_tokens"
            }
        }
    }

    // MARK: — GET /v1/models

    @Sendable
    func listModels(request: Request, context: GemmaRequestContext) async throws -> Response {
        let snapshot = await orchestrator.healthSnapshot(modelId: modelId)
        let data: [[String: Any]] = [
            [
                "id": snapshot.modelId ?? modelId,
                "object": "model",
                "created": Int(Date().timeIntervalSince1970),
                "owned_by": "gemm"
            ]
        ]
        let body: [String: Any] = ["object": "list", "data": data]
        let json = try JSONSerialization.data(withJSONObject: body)
        var buf = ByteBuffer()
        buf.writeBytes(json)
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buf))
    }

    // MARK: — POST /v1/chat/completions

    @Sendable
    func chatCompletions(request: Request, context: GemmaRequestContext) async throws -> Response {
        let dto: ChatCompletionRequest
        do {
            let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
            let data = Data(buffer.readableBytesView)
            dto = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
        } catch {
            return errorResponse(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", code: 400)
        }

        guard !dto.messages.isEmpty else {
            return errorResponse(status: .badRequest, message: "messages must not be empty", code: 400)
        }

        let prompt = messagesToPrompt(dto.messages)
        let genRequest = GenerationRequest(
            prompt: prompt,
            maxTokens: dto.maxTokens,
            temperature: dto.temperature ?? 0.7,
            topP: dto.topP ?? 0.9
        )

        if dto.stream == true {
            return streamChatResponse(genRequest: genRequest, modelId: dto.model ?? modelId)
        }

        do {
            let result = try await orchestrator.generate(request: genRequest)
            let response = ChatCompletionResponse(
                id: "chatcmpl-\(UUID().uuidString)",
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: dto.model ?? modelId,
                choices: [
                    .init(
                        index: 0,
                        message: ChatMessage(role: "assistant", content: result.generatedText),
                        finishReason: result.finishReason == .stop ? "stop" : "length"
                    )
                ],
                usage: .init(
                    promptTokens: result.promptTokens,
                    completionTokens: result.completionTokens,
                    totalTokens: result.promptTokens + result.completionTokens
                )
            )
            return try jsonResponse(status: .ok, body: response)
        } catch let err as GemError {
            return errorResponse(
                status: HTTPResponse.Status(code: err.httpStatus),
                message: err.errorDescription ?? err.localizedDescription,
                code: err.httpStatus
            )
        }
    }

    // MARK: — POST /v1/generate (no auth, raw prompt)

    @Sendable
    func generateNoAuth(request: Request, context: GemmaRequestContext) async throws -> Response {
        let dto: GenerationRequest
        do {
            let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
            let data = Data(buffer.readableBytesView)
            dto = try JSONDecoder().decode(GenerationRequest.self, from: data)
        } catch {
            return errorResponse(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", code: 400)
        }

        do {
            let result = try await orchestrator.generate(request: dto)
            return try jsonResponse(status: .ok, body: result)
        } catch let err as GemError {
            return errorResponse(
                status: HTTPResponse.Status(code: err.httpStatus),
                message: err.errorDescription ?? err.localizedDescription,
                code: err.httpStatus
            )
        }
    }

    // MARK: — Streaming

    private func streamChatResponse(genRequest: GenerationRequest, modelId: String) -> Response {
        let headers: HTTPFields = [
            .contentType: "text/event-stream",
            .cacheControl: "no-cache",
            .connection: "keep-alive"
        ]
        let body = ResponseBody(asyncSequence: sseStream(genRequest: genRequest, modelId: modelId))
        return Response(status: .ok, headers: headers, body: body)
    }

    private func sseStream(genRequest: GenerationRequest, modelId: String) -> AsyncStream<ByteBuffer> {
        AsyncStream { continuation in
            Task {
                do {
                    let stream = try await orchestrator.generateStream(request: genRequest)
                    let id = "chatcmpl-\(UUID().uuidString)"
                    let created = Int(Date().timeIntervalSince1970)

                    for await chunk in stream {
                        if case .text(let text) = chunk {
                            let delta: [String: Any] = [
                                "id": id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": modelId,
                                "choices": [
                                    [
                                        "index": 0,
                                        "delta": ["role": "assistant", "content": text],
                                        "finish_reason": NSNull()
                                    ]
                                ]
                            ]
                            if let data = try? JSONSerialization.data(withJSONObject: delta),
                               let json = String(data: data, encoding: .utf8) {
                                var buf = ByteBuffer()
                                buf.writeString("data: \(json)\n\n")
                                continuation.yield(buf)
                            }
                        }
                    }

                    // Final chunk
                    let done: [String: Any] = [
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": modelId,
                        "choices": [["index": 0, "delta": [:] as [String: Any], "finish_reason": "stop"]]
                    ]
                    if let data = try? JSONSerialization.data(withJSONObject: done),
                       let json = String(data: data, encoding: .utf8) {
                        var buf = ByteBuffer()
                        buf.writeString("data: \(json)\n\n")
                        continuation.yield(buf)
                    }

                    var doneBuf = ByteBuffer()
                    doneBuf.writeString("data: [DONE]\n\n")
                    continuation.yield(doneBuf)
                    continuation.finish()

                } catch {
                    var buf = ByteBuffer()
                    buf.writeString("event: error\ndata: {\"error\": \"\(error.localizedDescription)\"}\n\n")
                    continuation.yield(buf)
                    continuation.finish()
                }
            }
        }
    }

    // MARK: — Prompt formatting

    /// Converts OpenAI messages array to a flat prompt string.
    /// The MLX engine wraps it in UserInput(chat: [.user(prompt)]) and
    /// applies the model's own chat template on top, so we just concatenate
    /// the conversation turns in a readable format.
    private func messagesToPrompt(_ messages: [ChatMessage]) -> String {
        var parts: [String] = []
        for msg in messages {
            switch msg.role {
            case "system":
                parts.append("[System]: \(msg.content)")
            case "user":
                parts.append("[User]: \(msg.content)")
            case "assistant":
                parts.append("[Assistant]: \(msg.content)")
            default:
                parts.append(msg.content)
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

    private func errorResponse(status: HTTPResponse.Status, message: String, code: Int) -> Response {
        let envelope = ErrorResponse(error: message, code: code)
        let data = (try? JSONEncoder().encode(envelope)) ?? Data()
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
