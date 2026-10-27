import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct ChatPipeline: Sendable {
    struct Rejection: Error {
        let response: Response
        let message: String
    }

    struct Sampling: Sendable {
        var temperature: Double?
        var topP: Double?
        var minP: Double?
        var dynamicTemperature: Bool?
        var topK: Int?
        var presencePenalty: Double?
        var repetitionPenalty: Double?
    }

    struct Prepared: Sendable {
        let request: GenerationRequest
        let requestedModel: String?
    }

    let orchestrator: ModelOrchestratorActor
    let label: String
    let source: String
    let endpoint: String
    let reject: @Sendable (HTTPResponse.Status, String) -> Response

    func rejection(_ status: HTTPResponse.Status, _ message: String) -> Rejection {
        Rejection(response: reject(status, message), message: message)
    }

    func handle(_ request: Request, _ work: (Data) async throws -> Response) async throws -> Response {
        let data = try await body(of: request)
        do {
            return try await work(data)
        } catch let rejection as Rejection {
            await orchestrator.sessionLog?.logRejection(
                origin: RequestOrigin(source: source, endpoint: endpoint, stream: false),
                sessionId: GenerationRequest.sessionId(in: request.headers), code: Int(rejection.response.status.code),
                message: rejection.message, bodyBytes: data.count)
            return rejection.response
        }
    }

    static let bodyLimit = 4 * 1024 * 1024

    func body(of request: Request) async throws -> Data {
        Data(try await request.body.collect(upTo: Self.bodyLimit).readableBytesView)
    }

    func decode<T: Decodable>(_ type: T.Type, from data: Data, failure: String) throws -> T {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            fputs("🔴 [\(label)] decode error: \(error) (body: \(data.count) bytes)\n", stderr)
            throw rejection(.badRequest, "\(failure): \(error.localizedDescription)")
        }
    }

    func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw rejection(.badRequest, message) }
    }

    func tools(in data: Data) throws -> OpenAIController.RequestedTools {
        do { return try OpenAIController.requestedTools(fromRequestBody: data) } catch {
            throw rejection(.badRequest, error.message)
        }
    }

    func responseFormat(in data: Data) throws -> ResponseFormat? {
        do { return try OpenAIController.responseFormat(fromRequestBody: data) } catch {
            throw rejection(.badRequest, error.message)
        }
    }

    func prepare(
        chat: [ChatMessageInput], headers: HTTPFields, maxTokens: Int?, sampling: Sampling, reasoningEffort: String?,
        tools: OpenAIController.RequestedTools, thinkingBudget: Int?, responseFormat: ResponseFormat?, model: String?,
        stream: Bool
    ) throws -> Prepared {
        let prompt = ChatMessageInput.plainText(chat)
        let service = ServiceRequestDetector.prepare(prompt: prompt, maxTokens: maxTokens, headers: headers)
        let request = GenerationRequest(
            prompt: service.prompt,
            maxTokens: maxTokens,
            temperature: sampling.temperature,
            topP: sampling.topP,
            minP: sampling.minP,
            dynamicTemperature: sampling.dynamicTemperature,
            reasoningEffort: reasoningEffort,
            isServiceRequest: service.isService,
            tools: tools.tools,
            toolChoice: tools.choice,
            cacheKey: PromptCacheKey.resolve(isService: service.isService),
            messages: service.prompt == prompt ? chat : nil,
            topK: sampling.topK,
            presencePenalty: sampling.presencePenalty,
            repetitionPenalty: sampling.repetitionPenalty,
            thinkingBudget: thinkingBudget,
            responseFormat: responseFormat
        ).routed(by: headers).withOrigin(RequestOrigin(source: source, endpoint: endpoint, stream: stream))
        do {
            _ = try request.validated()
        } catch {
            throw rejection(HTTPResponse.Status(code: error.httpStatus, reasonPhrase: ""), ModelsController.publicMessage(for: error))
        }
        let restModel = model.map(ModelsController.restModel(from:))
        if restModel == .path { throw rejection(.badRequest, ModelsController.pathRejected) }
        return Prepared(request: request, requestedModel: restModel?.repoId)
    }

    func admitStream(_ request: GenerationRequest) async throws {
        if let refusal = await orchestrator.admissionRefusal(for: request) {
            throw rejection(HTTPResponse.Status(code: refusal.httpStatus, reasonPhrase: ""), ModelsController.publicMessage(for: refusal))
        }
    }
}
