import Foundation
import Hummingbird
import BasaltCore

extension OpenAIController {
    @Sendable
    func chatCompletions(request: Request, context: BasicRequestContext) async throws -> Response {
        let pipeline = self.pipeline(source: "openai-chat", endpoint: "/v1/chat/completions")
        return try await pipeline.handleBuffer(request) { buffer in
            let parsed = try pipeline.fastRead(from: buffer, failure: "Cannot decode request")
            let dto = parsed.dto
            try pipeline.require(!dto.messages.isEmpty, "messages must not be empty")
            let tools = try pipeline.tools(parsed.tools)
            let format = try pipeline.responseFormat(parsed.responseFormat)
            let stream = dto.stream == true
            let anti = try pipeline.antiRepetition(parsed.antiRepetition, bodyBytes: buffer.readableBytes)
            let chat = try pipeline.parsed { () throws(BasaltError) in try Self.chatMessages(dto.messages) }
            let model = dto.model ?? modelId
            let prepared = try pipeline.prepare(ChatPipeline.Input(
                chat: chat, maxTokens: dto.maxTokens,
                sampling: dto.sampling(anti),
                reasoningEffort: dto.reasoningEffort, tools: tools, thinkingBudget: dto.thinkingBudget,
                responseFormat: format, model: dto.model, stream: stream), headers: request.headers)
            if anti.xrayTrace == true, !stream {
                try await pipeline.admitStream(prepared.request)
                let topLP = anti.topLogprobs ?? anti.nProbs ?? 5
                return try await XrayTokensController.chatXrayResponse(
                    chat: chat, maxTokens: dto.maxTokens, temperature: dto.temperature,
                    topLogprobs: topLP, model: model, orchestrator: orchestrator
                )
            }
            if stream { try await pipeline.admitStream(prepared.request) }
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

        struct ChoiceLogprobs: Codable, Sendable {
            let content: [OutputTokenTelemetry]?
            var summary: AttributedTokenTraceSummary? = nil
        }

        struct Choice: Codable, Sendable {
            let index: Int
            let message: ChatMessage
            let finishReason: String
            var logprobs: ChoiceLogprobs? = nil
            enum CodingKeys: String, CodingKey {
                case index, message, logprobs
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
        let choiceLogprobs = result.tokenTelemetry.map { ChoiceLogprobs(content: $0) }
        self = Self(
            id: "chatcmpl-\(UUID().uuidString)", object: "chat.completion", created: Int(Date().timeIntervalSince1970),
            model: model, choices: [Choice(index: 0, message: message, finishReason: finishReason, logprobs: choiceLogprobs)],
            usage: Usage(
                promptTokens: result.promptTokens,
                promptTokensDetails: OpenAIController.CachedTokens(cachedTokens: result.cachedPromptTokens),
                completionTokens: result.completionTokens,
                totalTokens: result.promptTokens + result.completionTokens,
                completionTokensDetails: OpenAIController.ReasoningTokens(reasoningTokens: result.reasoningTokens ?? 0)))
    }
}

struct OpenAIChatStream: EventStreamEncoder {
    let id: String
    let created: Int
    let model: String
    let includeUsage: Bool
    let writer: SSEFastWriter
    var toolCallCount = 0
    var final: GenerationResponse?

    init(
        model: String, includeUsage: Bool,
        id: String = "chatcmpl-\(UUID().uuidString)",
        created: Int = Int(Date().timeIntervalSince1970)
    ) {
        self.id = id
        self.created = created
        self.model = model
        self.includeUsage = includeUsage
        self.writer = SSEFastWriter(id: id, created: created, model: model)
    }

    func opening() -> [String] { [] }

    mutating func frames(for chunk: StreamChunk) -> [String] {
        var out: [String] = []
        emit(chunk) { out.append(String(decoding: $0.readableBytesView, as: UTF8.self)) }
        return out
    }

    func closing() -> [String] {
        var out: [String] = []
        emitClosing { out.append(String(decoding: $0.readableBytesView, as: UTF8.self)) }
        return out
    }

    mutating func emit(_ chunk: StreamChunk, yield: (ByteBuffer) -> Void) {
        switch chunk {
        case .toolCall(let call):
            let index = toolCallCount
            toolCallCount += 1
            yield(writer.toolCallDelta(index: index, call: call))
        case .text(let text):
            if toolCallCount > 0, text.unicodeScalars.allSatisfy({ CharacterSet.whitespacesAndNewlines.contains($0) }) {
                return
            }
            yield(writer.textDelta(text))
        case .reasoning(let reasoning):
            yield(writer.reasoningDelta(reasoning))
        case .metadata(let meta):
            final = meta
        }
    }

    func emitClosing(yield: (ByteBuffer) -> Void) {
        if final?.finishReason == .error {
            yield(SSEFastWriter.errorFrame(message: ModelsController.generationFailed))
            return
        }
        let finishReason = OpenAIController.finishReason(final?.finishReason, calledTools: toolCallCount > 0)
        if let telemetry = final?.tokenTelemetry {
            yield(XrayTokensController.streamingLogprobsFinishFrame(
                id: id, created: created, model: model, finishReason: finishReason, telemetry: telemetry))
        } else {
            yield(writer.finish(finishReason))
        }
        if includeUsage, let final {
            yield(writer.usage(final))
        }
        yield(SSEFastWriter.doneFrame())
    }

    func response(streaming chunks: AsyncStream<StreamChunk>) -> Response {
        let body = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                var encoder = self
                for await chunk in chunks {
                    encoder.emit(chunk) { _ = continuation.yield($0) }
                }
                if !Task.isCancelled {
                    encoder.emitClosing { _ = continuation.yield($0) }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive"],
            body: ResponseBody(asyncSequence: body)
        )
    }
}
