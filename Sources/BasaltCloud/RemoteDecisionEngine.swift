import Foundation
import BasaltCore

public enum CloudDecisionProvider: String, Sendable, CaseIterable {
    case openRouter = "openrouter"
    case gemini

    static let geminiInputPrice = 0.10 / 1_000_000

    public var baseURL: URL {
        switch self {
        case .openRouter: URL(string: "https://openrouter.ai/api/alpha")!
        case .gemini: URL(string: "https://generativelanguage.googleapis.com/v1beta")!
        }
    }

    public var keyEnvironment: String { self == .gemini ? "GEMINI_API_KEY" : "OPENROUTER_API_KEY" }

    public var timeout: TimeInterval { self == .gemini ? 20 : 3 }

    public static func parse(_ reference: String) -> (provider: CloudDecisionProvider, model: String) {
        reference.hasPrefix("gemini/") ? (.gemini, String(reference.dropFirst("gemini/".count))) : (.openRouter, reference)
    }
}

enum RemoteDecisionCore {
    struct Body: Encodable {
        let model: String
        let state: String
        let questions: [String: SystemOneQuestion]
        let calibrated: Bool?
    }

    struct GeminiBody: Encodable {
        struct Block: Encodable {
            let type: String
            var text: String?
            var data: String?
            var mimeType: String?

            enum CodingKeys: String, CodingKey {
                case type, text, data
                case mimeType = "mime_type"
            }
        }

        enum Input: Encodable {
            case text(String)
            case blocks([Block])

            func encode(to encoder: any Encoder) throws {
                var container = encoder.singleValueContainer()
                switch self {
                case .text(let text): try container.encode(text)
                case .blocks(let blocks): try container.encode(blocks)
                }
            }
        }

        let model: String
        let input: Input
        let questions: [String: SystemOneQuestion]
    }

    struct Wire: Decodable {
        struct Usage: Decodable {
            let cost: Double?
            let totalInputTokens: Int?

            enum CodingKeys: String, CodingKey {
                case cost
                case totalInputTokens = "total_input_tokens"
            }
        }

        struct Answer: Decodable {
            let noul: Double?
            let confidence: Double?
            let score: Double?
            let choice: String?
            let probabilities: [String: Double]?
        }

        let model: String?
        let answers: [String: Answer]
        let usage: Usage?
    }

    static func body(_ request: SystemOneRequest, model: String, provider: CloudDecisionProvider = .openRouter) throws(BasaltError) -> Data {
        let state = SecretRedactor.redact(request.state)
        do {
            guard provider == .gemini else { return try JSONEncoder().encode(Body(model: model, state: state, questions: request.questions, calibrated: request.calibrated)) }
            let media = (request.media ?? []).map { GeminiBody.Block(type: $0.kind, data: $0.data.base64EncodedString(), mimeType: $0.mimeType) }
            let input: GeminiBody.Input = media.isEmpty ? .text(state) : .blocks([GeminiBody.Block(type: "text", text: state)] + media)
            return try JSONEncoder().encode(GeminiBody(model: model, input: input, questions: request.questions))
        } catch {
            throw .invalidRequestStructure(details: "systemone: the cloud request cannot be encoded")
        }
    }

    static func response(_ data: Data, for request: SystemOneRequest, model: String = "", provider: CloudDecisionProvider = .openRouter) -> SystemOneResponse? {
        guard let wire = try? JSONDecoder().decode(Wire.self, from: data) else { return nil }
        var answers: [String: SystemOneAnswer] = [:]
        for (id, question) in request.questions {
            guard let answer = wire.answers[id].flatMap({ Self.answer($0, for: question) }) else { continue }
            answers[id] = SystemOneCore.gated(answer, abstainBelow: request.abstainBelow, minMargin: request.minMargin)
        }
        var usage = SystemOneUsage(prefixTokens: 0, candidateBranches: 0, prefixEvaluations: 0, branchForwardCalls: 0, suffixTokens: 0)
        let tokens = provider == .gemini ? wire.usage?.totalInputTokens.map { Double($0) * CloudDecisionProvider.geminiInputPrice } : nil
        usage.cost = (wire.usage?.cost ?? tokens).flatMap { $0.isFinite && $0 >= 0 ? $0 : nil }
        return SystemOneResponse(model: wire.model ?? model, answers: answers, usage: usage, calibrated: request.calibrated != false)
    }

    static func answer(_ wire: Wire.Answer, for question: SystemOneQuestion) -> SystemOneAnswer? {
        let (keys, descriptions) = SystemOneCore.options(question)
        let unit = 0.0 ... 1.0
        guard [wire.noul, wire.confidence].allSatisfy({ $0.map(unit.contains) ?? true }),
              (wire.probabilities ?? [:]).allSatisfy({ keys.contains($0.key) && unit.contains($0.value) }) else { return nil }
        var answer = SystemOneAnswer()
        answer.confidence = wire.confidence
        switch question.type {
        case .noul:
            guard let noul = wire.noul else { return nil }
            (answer.noul, answer.confidence) = (noul, wire.confidence ?? abs(2 * noul - 1))
        case .choice:
            guard let choice = wire.choice, keys.contains(choice) else { return nil }
            (answer.choice, answer.probabilities) = (choice, wire.probabilities)
        case .score:
            guard let score = wire.score, (0 ... Double(keys.count - 1)).contains(score) else { return nil }
            (answer.score, answer.legend, answer.probabilities) = (score, descriptions, wire.probabilities)
        }
        return answer
    }
}

public struct RemoteDecisionEngine: DecisionEngine {
    struct Secret: Sendable, CustomStringConvertible, CustomDebugStringConvertible, CustomReflectable {
        let value: String
        var description: String { "<redacted>" }
        var debugDescription: String { description }
        var customMirror: Mirror { Mirror(self, children: []) }
    }

    public static let openRouter = CloudDecisionProvider.openRouter.baseURL

    public let modelID: String
    public let provider: CloudDecisionProvider
    public let runsBesideDecode = true
    public var onDevice: (any DecisionEngine)? { nil }
    public var acceptsMedia: Bool { provider == .gemini }
    let endpoint: URL
    let key: Secret
    let session: URLSession

    public init(model: String, apiKey: String, provider: CloudDecisionProvider = .openRouter, baseURL: URL? = nil, timeout: TimeInterval? = nil) throws(BasaltError) {
        guard !apiKey.isEmpty else { throw .authenticationFailed(details: "cloud decisions need \(provider.keyEnvironment)") }
        let timeout = timeout ?? provider.timeout
        guard !model.trimmingCharacters(in: .whitespaces).isEmpty, timeout > 0 else { throw .invalidRequestStructure(details: "cloud decisions need a model and a positive timeout") }
        (modelID, self.provider, endpoint, key) = (model, provider, (baseURL ?? provider.baseURL).appendingPathComponent("decisions"), Secret(value: apiKey))
        let configuration = URLSessionConfiguration.ephemeral
        (configuration.timeoutIntervalForRequest, configuration.timeoutIntervalForResource) = (timeout, timeout)
        session = URLSession(configuration: configuration)
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try SystemOneCore.validate(request)
        let (status, data, retryAfter) = try await send(try RemoteDecisionCore.body(request, model: modelID, provider: provider))
        guard (200 ..< 300).contains(status) else { throw Self.failure(status: status, retryAfter: retryAfter) }
        guard let response = RemoteDecisionCore.response(data, for: request, model: modelID, provider: provider) else { throw Self.failure("malformed response") }
        return response
    }

    func send(_ body: Data) async throws(BasaltError) -> (status: Int, data: Data, retryAfter: Int?) {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if provider == .gemini {
            request.setValue(key.value, forHTTPHeaderField: "x-goog-api-key")
        } else {
            request.setValue("Bearer \(key.value)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = body
        do {
            let (data, response) = try await session.data(for: request, delegate: NoRedirects())
            let http = response as? HTTPURLResponse
            return (http?.statusCode ?? 0, data, http?.value(forHTTPHeaderField: "Retry-After").flatMap { Int($0) })
        } catch {
            guard !Task.isCancelled else { throw .cancelled }
            let code = (error as? URLError)?.code
            throw Self.failure(code == .timedOut ? "timeout" : "network error \(code?.rawValue ?? 0)")
        }
    }

    static func failure(status: Int, retryAfter: Int?) -> BasaltError {
        switch status {
        case 401, 403: .authenticationFailed(details: "cloud decisions: refused, HTTP \(status): check the key")
        case 429: .serverBusy(retryAfterSeconds: retryAfter.map { min(max($0, 1), 3600) } ?? Int(CloudGateCore.defaultPause))
        case 500...: failure("server error, HTTP \(status)")
        default: failure("refused, HTTP \(status)")
        }
    }

    static func failure(_ reason: String) -> BasaltError {
        .modelInferenceError(details: "cloud decisions: \(reason)")
    }
}
