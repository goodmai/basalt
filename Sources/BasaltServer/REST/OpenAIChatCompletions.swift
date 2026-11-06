import Foundation
import Hummingbird
import BasaltCore

extension OpenAIController {
    @Sendable
    func chatCompletions(request: Request, context: BasicRequestContext) async throws -> Response {
        let pipeline = self.pipeline(source: "openai-chat", endpoint: "/v1/chat/completions")
        return try await pipeline.handle(request) { data in
            let body = try pipeline.decode(ChatPipeline.Body<ChatCompletionRequest>.self, from: data, failure: "Cannot decode request")
            let dto = body.dto
            try pipeline.require(!dto.messages.isEmpty, "messages must not be empty")
            let json = ChatPipeline.object(in: data)
            let tools = try pipeline.tools(in: json)
            let format = try pipeline.responseFormat(in: json, data: data)
            let stream = dto.stream == true
            let prepared = try pipeline.prepare(ChatPipeline.Input(
                chat: try pipeline.parsed { () throws(BasaltError) in try Self.chatMessages(dto.messages) }, maxTokens: dto.maxTokens,
                sampling: dto.sampling(try pipeline.antiRepetition(body.antiRepetition, bodyBytes: data.count)),
                reasoningEffort: dto.reasoningEffort, tools: tools, thinkingBudget: dto.thinkingBudget,
                responseFormat: format, model: dto.model, stream: stream), headers: request.headers)
            if stream { try await pipeline.admitStream(prepared.request) }
            let model = dto.model ?? modelId
            let includeUsage = dto.streamOptions?.includeUsage == true
            return await pipeline.respond(to: prepared, stream: stream) { chunks in
                OpenAIChatStream(model: model, includeUsage: includeUsage).response(streaming: chunks)
            } completed: { result in
                try JSONEncoder().encode(ChatCompletionResponse(result: result, model: model))
            }
        }
    }
}

extension OpenAIController {
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
        let speculativeLookup: Bool?
        let grammarJumpForward: Bool?
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
            case speculativeLookup = "speculative_lookup"
            case grammarJumpForward = "grammar_jump_forward"
        }

        func sampling(_ antiRepetition: ChatPipeline.AntiRepetition) -> ChatPipeline.Sampling {
            ChatPipeline.Sampling(
                temperature: temperature, topP: topP, minP: minP, dynamicTemperature: dynamicTemperature, topK: topK,
                presencePenalty: presencePenalty, repetitionPenalty: repetitionPenalty,
                speculativeLookup: speculativeLookup, grammarJumpForward: grammarJumpForward, antiRepetition: antiRepetition)
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
}

extension OpenAIController.ChatCompletionResponse {
    init(result: GenerationResponse, model: String) {
        let calls = result.toolCalls ?? []
        let onlyCalls = !calls.isEmpty && result.generatedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let message = OpenAIController.ChatMessage(
            role: "assistant", content: onlyCalls ? nil : .text(result.generatedText),
            toolCalls: calls.isEmpty ? nil : calls.map { OpenAIController.ChatToolCall($0) }, reasoning: result.reasoningText)
        let finishReason = OpenAIController.finishReason(result.finishReason, calledTools: !calls.isEmpty)
        self = Self(
            id: "chatcmpl-\(UUID().uuidString)", object: "chat.completion", created: Int(Date().timeIntervalSince1970),
            model: model, choices: [Choice(index: 0, message: message, finishReason: finishReason)],
            usage: Usage(
                promptTokens: result.promptTokens,
                promptTokensDetails: OpenAIController.CachedTokens(cachedTokens: result.cachedPromptTokens),
                completionTokens: result.completionTokens,
                totalTokens: result.promptTokens + result.completionTokens,
                completionTokensDetails: OpenAIController.ReasoningTokens(reasoningTokens: result.reasoningTokens ?? 0)))
    }
}

struct OpenAIChatStream: EventStreamEncoder {
    let id = "chatcmpl-\(UUID().uuidString)"
    let created = Int(Date().timeIntervalSince1970)
    let model: String
    let includeUsage: Bool
    var toolCallCount = 0
    var final: GenerationResponse?

    func opening() -> [String] { [] }

    mutating func frames(for chunk: StreamChunk) -> [String] {
        switch chunk {
        case .toolCall(let call):
            defer { toolCallCount += 1 }
            return frame(delta: ["role": "assistant", "tool_calls": [[
                "index": toolCallCount, "id": call.id, "type": "function",
                "function": ["name": call.name, "arguments": call.arguments]
            ] as [String: Any]]])
        case .text(let text):
            if toolCallCount > 0, text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return [] }
            return frame(delta: ["role": "assistant", "content": text])
        case .reasoning(let reasoning):
            return frame(delta: ["role": "assistant", "reasoning": reasoning, "reasoning_content": reasoning])
        case .metadata(let meta):
            final = meta
            return []
        }
    }

    func closing() -> [String] {
        if final?.finishReason == .error {
            let error: [String: Any] = ["error": ["message": ModelsController.generationFailed, "type": "server_error"]]
            return Self.json(error).map { ["event: error\ndata: \($0)\n\n"] } ?? []
        }
        var frames = frame(delta: [:], finish: OpenAIController.finishReason(final?.finishReason, calledTools: toolCallCount > 0))
        if includeUsage, let final {
            frames += frame(["choices": [] as [Any], "usage": OpenAIController.chatUsage(final)])
        }
        return frames + ["data: [DONE]\n\n"]
    }

    private func frame(delta: [String: Any], finish: String? = nil) -> [String] {
        frame(["choices": [["index": 0, "delta": delta, "finish_reason": finish ?? NSNull()] as [String: Any]]])
    }

    private func frame(_ fields: [String: Any]) -> [String] {
        var chunk: [String: Any] = ["id": id, "object": "chat.completion.chunk", "created": created, "model": model]
        chunk.merge(fields) { _, new in new }
        return Self.json(chunk).map { ["data: \($0)\n\n"] } ?? []
    }
}
