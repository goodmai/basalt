import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct ChatPipeline: Sendable {
    struct Rejection: Error {
        var response: Response
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
        var speculativeLookup: Bool?
        var grammarJumpForward: Bool?
        var antiRepetition = AntiRepetition()
    }

    struct AntiRepetition: Decodable, Sendable {
        var dryMultiplier: Double?
        var dryBase: Double?
        var dryAllowedLength: Int?
        var drySequenceBreakers: [String]?
        var xtcThreshold: Double?
        var xtcProbability: Double?

        enum CodingKeys: String, CodingKey {
            case dryMultiplier = "dry_multiplier"
            case dryBase = "dry_base"
            case dryAllowedLength = "dry_allowed_length"
            case drySequenceBreakers = "dry_sequence_breakers"
            case xtcThreshold = "xtc_threshold"
            case xtcProbability = "xtc_probability"
        }
    }

    struct Input: Sendable {
        let chat: [ChatMessageInput]
        let maxTokens: Int?
        let sampling: Sampling
        let reasoningEffort: String?
        let tools: OpenAIController.RequestedTools
        let thinkingBudget: Int?
        let responseFormat: ResponseFormat?
        let model: String?
        let stream: Bool
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

    func antiRepetition(in data: Data) throws -> AntiRepetition {
        try decode(AntiRepetition.self, from: data, failure: "Invalid DRY or XTC parameter")
    }

    func responseFormat(in data: Data) throws -> ResponseFormat? {
        do { return try OpenAIController.responseFormat(fromRequestBody: data) } catch {
            throw rejection(.badRequest, error.message)
        }
    }

    func prepare(_ input: Input, headers: HTTPFields) throws -> Prepared {
        let prompt = ChatMessageInput.plainText(input.chat)
        let service = ServiceRequestDetector.prepare(prompt: prompt, maxTokens: input.maxTokens, headers: headers)
        let request = GenerationRequest(
            prompt: service.prompt,
            maxTokens: input.maxTokens,
            temperature: input.sampling.temperature,
            topP: input.sampling.topP,
            minP: input.sampling.minP,
            dynamicTemperature: input.sampling.dynamicTemperature,
            reasoningEffort: input.reasoningEffort,
            isServiceRequest: service.isService,
            tools: input.tools.tools,
            toolChoice: input.tools.choice,
            cacheKey: PromptCacheKey.resolve(isService: service.isService, headers: headers),
            messages: service.prompt == prompt ? input.chat : nil,
            topK: input.sampling.topK,
            presencePenalty: input.sampling.presencePenalty,
            repetitionPenalty: input.sampling.repetitionPenalty,
            thinkingBudget: input.thinkingBudget,
            responseFormat: input.responseFormat,
            dryMultiplier: input.sampling.antiRepetition.dryMultiplier,
            dryBase: input.sampling.antiRepetition.dryBase,
            dryAllowedLength: input.sampling.antiRepetition.dryAllowedLength,
            drySequenceBreakers: input.sampling.antiRepetition.drySequenceBreakers,
            xtcThreshold: input.sampling.antiRepetition.xtcThreshold,
            xtcProbability: input.sampling.antiRepetition.xtcProbability
        ).withSpeculativeLookup(input.sampling.speculativeLookup)
            .withGrammarJumpForward(input.sampling.grammarJumpForward)
            .routed(by: headers).withOrigin(RequestOrigin(source: source, endpoint: endpoint, stream: input.stream))
        do {
            _ = try request.validated()
        } catch {
            throw rejection(HTTPResponse.Status(code: error.httpStatus, reasonPhrase: ""), ModelsController.publicMessage(for: error))
        }
        let restModel = input.model.map(ModelsController.restModel(from:))
        if restModel == .path { throw rejection(.badRequest, ModelsController.pathRejected) }
        return Prepared(request: request, requestedModel: restModel?.repoId)
    }

    func admitStream(_ request: GenerationRequest) async throws {
        if let refusal = await orchestrator.admissionRefusal(for: request) {
            var refused = rejection(HTTPResponse.Status(code: refusal.httpStatus, reasonPhrase: ""), ModelsController.publicMessage(for: refusal))
            if case .memoryPressure(let seconds) = refusal { refused.response.headers[.retryAfter] = String(seconds) }
            throw refused
        }
    }
}
