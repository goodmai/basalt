import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct OpenAIController: Sendable {
    let orchestrator: ModelOrchestratorActor
    let modelId: String
    var zeroCopyBody = false
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
    @Sendable
    func score(request: Request, context: BasicRequestContext) async throws -> Response {
        let data = Data(try await request.body.collect(upTo: ChatPipeline.textBodyLimit).readableBytesView)
        guard let dto = try? JSONDecoder().decode(ScoreRequest.self, from: data) else {
            return ProviderErrorResponse.response(status: .badRequest, message: "A score request needs \"messages\", \"completion\" (token ids) and \"top_logprobs\".", shape: .openAI)
        }
        do {
            let report = try await orchestrator.research { model throws(BasaltError) in try await ResearchCall.lab(model).score(dto) }
            return makeDataResponse(try JSONEncoder().encode(report))
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/v1/score")
        }
    }

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
        ChatPipeline(orchestrator: orchestrator, label: "OpenAI", source: source, endpoint: endpoint, zeroCopyBody: zeroCopyBody) { status, message in
            ProviderErrorResponse.response(status: status, message: message, shape: .openAI)
        }
    }
}
