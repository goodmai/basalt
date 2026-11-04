import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct OpenAIController: Sendable {
    let orchestrator: ModelOrchestratorActor
    let modelId: String
}

extension OpenAIController {
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

    @Sendable
    func chatCompletions(request: Request, context: BasicRequestContext) async throws -> Response {
        let pipeline = self.pipeline(source: "openai-chat", endpoint: "/v1/chat/completions")
        return try await pipeline.handle(request) { data in
            let dto = try pipeline.decode(ChatCompletionRequest.self, from: data, failure: "Cannot decode request")
            try pipeline.require(!dto.messages.isEmpty, "messages must not be empty")
            let tools = try pipeline.tools(in: data)
            let format = try pipeline.responseFormat(in: data)
            let stream = dto.stream == true
            let prepared = try pipeline.prepare(
                chat: Self.chatMessages(dto.messages), headers: request.headers, maxTokens: dto.maxTokens,
                sampling: dto.sampling(try pipeline.antiRepetition(in: data)),
                reasoningEffort: dto.reasoningEffort, tools: tools, thinkingBudget: dto.thinkingBudget,
                responseFormat: format, model: dto.model, stream: stream)
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

    @Sendable
    func responses(request: Request, context: BasicRequestContext) async throws -> Response {
        let pipeline = self.pipeline(source: "openai-responses", endpoint: "/v1/responses")
        return try await pipeline.handle(request) { data in
            let dto = try pipeline.decode(ResponsesRequest.self, from: data, failure: "Cannot decode responses request")
            let conversation = dto.conversation
            try pipeline.require(!conversation.isEmpty, "input or messages must not be empty")
            let tools = try pipeline.tools(in: data)
            let format = try pipeline.responseFormat(in: data)
            let stream = dto.stream == true
            let prepared = try pipeline.prepare(
                chat: dto.instructionMessages + conversation, headers: request.headers,
                maxTokens: dto.maxOutputTokens ?? dto.maxTokens, sampling: dto.sampling(try pipeline.antiRepetition(in: data)),
                reasoningEffort: dto.effectiveReasoningEffort, tools: tools, thinkingBudget: dto.thinkingBudget,
                responseFormat: format, model: dto.model, stream: stream)
            if stream { try await pipeline.admitStream(prepared.request) }
            let model = dto.model ?? modelId
            return await pipeline.respond(to: prepared, stream: stream) { chunks in
                OpenAIResponsesStream(model: model).response(streaming: chunks)
            } completed: { result in
                try JSONEncoder().encode(ResponsesResponse(result: result, model: model))
            }
        }
    }

    @Sendable
    func generateRaw(request: Request, context: BasicRequestContext) async throws -> Response {
        let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
        let data = Data(buffer.readableBytesView)
        let dto: GenerationRequest
        do {
            dto = PromptCacheKey.scoped(try JSONDecoder().decode(GenerationRequest.self, from: data), headers: request.headers).routed(by: request.headers)
                .withOrigin(RequestOrigin(source: "generate", endpoint: "/v1/generate", stream: false))
        } catch {
            return ProviderErrorResponse.response(status: .badRequest, message: ModelsController.publicMessage(for: error), error: error, shape: .openAI)
        }
        do {
            let result = try await orchestrator.generate(request: dto)
            let encoded = try JSONEncoder().encode(result)
            return makeDataResponse(encoded)
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/v1/generate")
        }
    }
}

extension OpenAIController {
    private func makeJSONResponse(_ body: [String: Any]) -> Response {
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        return makeDataResponse(data)
    }

    private func makeDataResponse(_ data: Data) -> Response {
        var buf = ByteBuffer(); buf.writeBytes(data)
        return Response(status: .ok, headers: [.contentType: "application/json"],
                        body: ResponseBody(byteBuffer: buf))
    }

    func pipeline(source: String, endpoint: String) -> ChatPipeline {
        ChatPipeline(orchestrator: orchestrator, label: "OpenAI", source: source, endpoint: endpoint) { status, message in
            ProviderErrorResponse.response(status: status, message: message, shape: .openAI)
        }
    }
}
