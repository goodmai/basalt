import Foundation
import Hummingbird

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
            let type: String
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
        let role: String
        let content: MessageContent?   // null when tool_calls are present
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
    func listModels(request: Request, context: BasicRequestContext) async throws -> Response {
        let snapshot = await orchestrator.healthSnapshot(modelId: modelId)
        let body: [String: Any] = [
            "object": "list",
            "data": [[
                "id": snapshot.modelId ?? modelId,
                "object": "model",
                "created": Int(Date().timeIntervalSince1970),
                "owned_by": "gemm"
            ] as [String: Any]]
        ]
        return makeJSONResponse(body)
    }

    // MARK: — POST /v1/chat/completions

    @Sendable
    func chatCompletions(request: Request, context: BasicRequestContext) async throws -> Response {
        let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
        let data = Data(buffer.readableBytesView)

        // Save raw body for diagnosis; visible at /tmp/gemm_openai_request.json
        let rawString = String(data: data, encoding: .utf8) ?? "<non-utf8>"
        try? rawString.write(toFile: "/tmp/gemm_openai_request.json", atomically: true, encoding: .utf8)

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
        let genRequest = GenerationRequest(
            prompt: prompt,
            maxTokens: dto.maxTokens,
            temperature: dto.temperature ?? 0.7,
            topP: dto.topP ?? 0.9
        )
        let requestedModel = dto.model.map { ModelsController.hfRepoId(from: $0) }  // strip claude-local/ prefix if present

        if dto.stream == true {
            return streamResponse(genRequest: genRequest, model: dto.model ?? modelId, requestedModel: requestedModel)
        }

        do {
            let result = try await orchestrator.generate(request: genRequest, modelId: requestedModel)
            let response = ChatCompletionResponse(
                id: "chatcmpl-\(UUID().uuidString)",
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: dto.model ?? modelId,
                choices: [.init(
                    index: 0,
                    message: ChatMessage(role: "assistant", content: .text(result.generatedText)),
                    finishReason: result.finishReason == .stop ? "stop" : "length"
                )],
                usage: .init(
                    promptTokens: result.promptTokens,
                    completionTokens: result.completionTokens,
                    totalTokens: result.promptTokens + result.completionTokens
                )
            )
            let encoded = try JSONEncoder().encode(response)
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

    private func streamResponse(genRequest: GenerationRequest, model: String, requestedModel: String? = nil) -> Response {
        Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
                .connection: "keep-alive"
            ],
            body: ResponseBody(asyncSequence: sseStream(genRequest: genRequest, model: model, requestedModel: requestedModel))
        )
    }

    private func sseStream(genRequest: GenerationRequest, model: String, requestedModel: String? = nil) -> AsyncStream<ByteBuffer> {
        AsyncStream { continuation in
            Task {
                let id = "chatcmpl-\(UUID().uuidString)"
                let created = Int(Date().timeIntervalSince1970)
                do {
                    let stream = try await orchestrator.generateStream(request: genRequest, modelId: requestedModel)
                    for await chunk in stream {
                        switch chunk {
                        case .text(let text):
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
                    var buf = ByteBuffer()
                    buf.writeString("event: error\ndata: {\"error\":\"\(error.localizedDescription)\"}\n\n")
                    continuation.yield(buf)
                }
                continuation.finish()
            }
        }
    }

    // MARK: — Prompt

    private func messagesToPrompt(_ messages: [ChatMessage]) -> String {
        var result = "<bos>"
        for msg in messages {
            guard let text = msg.content?.textValue, !text.isEmpty else { continue }
            switch msg.role {
            case "system":
                result += "<|turn>system\n\(text)<turn|>\n"
            case "user":
                result += "<|turn>user\n\(text)<turn|>\n"
            case "assistant":
                result += "<|turn>model\n\(text)<turn|>\n"
            default:
                result += "<|turn>\(msg.role)\n\(text)<turn|>\n"
            }
        }
        result += "<|turn>model\n<|channel>thought\n"
        return result
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
