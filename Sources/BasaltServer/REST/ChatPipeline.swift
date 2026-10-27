import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct ChatPipeline: Sendable {
    struct Rejection: Error {
        let response: Response
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
    let reject: @Sendable (HTTPResponse.Status, String) -> Response

    static let bodyLimit = 4 * 1024 * 1024

    func body(of request: Request) async throws -> Data {
        Data(try await request.body.collect(upTo: Self.bodyLimit).readableBytesView)
    }

    func decode<T: Decodable>(_ type: T.Type, from data: Data, failure: String) throws -> T {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            fputs("🔴 [\(label)] decode error: \(error) (body: \(data.count) bytes)\n", stderr)
            throw Rejection(response: reject(.badRequest, "\(failure): \(error.localizedDescription)"))
        }
    }

    func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw Rejection(response: reject(.badRequest, message)) }
    }

    func tools(in data: Data) throws -> OpenAIController.RequestedTools {
        do { return try OpenAIController.requestedTools(fromRequestBody: data) } catch {
            throw Rejection(response: reject(.badRequest, error.message))
        }
    }

    func responseFormat(in data: Data) throws -> ResponseFormat? {
        do { return try OpenAIController.responseFormat(fromRequestBody: data) } catch {
            throw Rejection(response: reject(.badRequest, error.message))
        }
    }

    func prepare(
        chat: [ChatMessageInput], headers: HTTPFields, maxTokens: Int?, sampling: Sampling, reasoningEffort: String?,
        tools: OpenAIController.RequestedTools, thinkingBudget: Int?, responseFormat: ResponseFormat?, model: String?
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
        ).routed(by: headers)
        do {
            _ = try request.validated()
        } catch {
            throw Rejection(response: reject(HTTPResponse.Status(code: error.httpStatus, reasonPhrase: ""),
                                             ModelsController.publicMessage(for: error)))
        }
        let restModel = model.map(ModelsController.restModel(from:))
        if restModel == .path { throw Rejection(response: reject(.badRequest, ModelsController.pathRejected)) }
        return Prepared(request: request, requestedModel: restModel?.repoId)
    }

    func admitStream(_ request: GenerationRequest) async throws {
        if let refusal = await orchestrator.admissionRefusal(for: request) {
            throw Rejection(response: reject(HTTPResponse.Status(code: refusal.httpStatus, reasonPhrase: ""),
                                             ModelsController.publicMessage(for: refusal)))
        }
    }
}
