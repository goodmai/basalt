import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct AnthropicController: Sendable {
    let orchestrator: ModelOrchestratorActor
    let modelId: String
}

extension AnthropicController {
    @Sendable
    func messages(request: Request, context: BasicRequestContext) async throws -> Response {
        let pipeline = ChatPipeline(orchestrator: orchestrator, label: "Anthropic", source: "anthropic", endpoint: "/v1/messages") { status, message in
            ProviderErrorResponse.response(status: status, message: message, shape: .anthropic)
        }
        return try await pipeline.handle(request) { data in
            let dto = try pipeline.decode(MessagesRequest.self, from: data, failure: "Decode error")
            try pipeline.require(!dto.messages.isEmpty, "messages must not be empty")
            if let problem = dto.thinkingBudgetProblem { throw pipeline.rejection(.badRequest, problem) }
            let tools = try pipeline.tools(in: data)
            let stream = dto.stream == true
            let prepared = try pipeline.prepare(
                chat: Self.chatMessages(system: dto.system, messages: dto.messages), headers: request.headers,
                maxTokens: dto.effectiveMaxTokens, sampling: dto.sampling(try pipeline.antiRepetition(in: data)),
                reasoningEffort: dto.effectiveReasoningEffort, tools: tools, thinkingBudget: dto.thinkingBudget,
                responseFormat: nil, model: dto.model, stream: stream)
            if stream { try await pipeline.admitStream(prepared.request) }
            let model = dto.model ?? modelId
            return await pipeline.respond(to: prepared, stream: stream) { chunks in
                AnthropicMessagesStream(model: model).response(streaming: chunks)
            } completed: { result in
                try JSONEncoder().encode(MessagesResponse(result: result, model: model))
            }
        }
    }
}
