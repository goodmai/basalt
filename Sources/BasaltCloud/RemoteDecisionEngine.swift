import Foundation
import BasaltCore

enum RemoteDecisionCore {
    struct Body: Encodable {
        let model: String
        let state: String
        let questions: [String: SystemOneQuestion]
        let calibrated: Bool?
    }

    struct Wire: Decodable {
        struct Usage: Decodable {
            let cost: Double?
        }

        struct Answer: Decodable {
            let noul: Double?
            let confidence: Double?
            let score: Double?
            let choice: String?
            let probabilities: [String: Double]?
        }

        let model: String
        let answers: [String: Answer]
        let usage: Usage?
    }

    static func body(_ request: SystemOneRequest, model: String) throws(BasaltError) -> Data {
        do {
            return try JSONEncoder().encode(Body(model: model, state: SecretRedactor.redact(request.state), questions: request.questions, calibrated: request.calibrated))
        } catch {
            throw .invalidRequestStructure(details: "systemone: the cloud request cannot be encoded")
        }
    }

    static func response(_ data: Data, for request: SystemOneRequest) -> SystemOneResponse? {
        guard let wire = try? JSONDecoder().decode(Wire.self, from: data) else { return nil }
        var answers: [String: SystemOneAnswer] = [:]
        for (id, question) in request.questions {
            guard let answer = wire.answers[id].flatMap({ Self.answer($0, for: question) }) else { continue }
            answers[id] = SystemOneCore.gated(answer, abstainBelow: request.abstainBelow, minMargin: request.minMargin)
        }
        var usage = SystemOneUsage(prefixTokens: 0, candidateBranches: 0, prefixEvaluations: 0, branchForwardCalls: 0, suffixTokens: 0)
        usage.cost = wire.usage?.cost.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil }
        return SystemOneResponse(model: wire.model, answers: answers, usage: usage, calibrated: request.calibrated != false)
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

    public static let openRouter = URL(string: "https://openrouter.ai/api/alpha")!

    public let modelID: String
    public let runsBesideDecode = true
    public var onDevice: (any DecisionEngine)? { nil }
    let endpoint: URL
    let key: Secret
    let session: URLSession

    public init(model: String, apiKey: String, baseURL: URL = openRouter, timeout: TimeInterval = 3) throws(BasaltError) {
        guard !apiKey.isEmpty else { throw .authenticationFailed(details: "cloud decisions need OPENROUTER_API_KEY") }
        guard !model.trimmingCharacters(in: .whitespaces).isEmpty, timeout > 0 else { throw .invalidRequestStructure(details: "cloud decisions need a model and a positive timeout") }
        (modelID, endpoint, key) = (model, baseURL.appendingPathComponent("decisions"), Secret(value: apiKey))
        let configuration = URLSessionConfiguration.ephemeral
        (configuration.timeoutIntervalForRequest, configuration.timeoutIntervalForResource) = (timeout, timeout)
        session = URLSession(configuration: configuration)
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try SystemOneCore.validate(request)
        let (status, data) = try await send(try RemoteDecisionCore.body(request, model: modelID))
        guard (200 ..< 300).contains(status) else { throw Self.failure(status >= 500 ? "server error, HTTP \(status)" : "refused, HTTP \(status)") }
        guard let response = RemoteDecisionCore.response(data, for: request) else { throw Self.failure("malformed response") }
        return response
    }

    func send(_ body: Data) async throws(BasaltError) -> (status: Int, data: Data) {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(key.value)", forHTTPHeaderField: "Authorization")
        request.httpBody = body
        do {
            let (data, response) = try await session.data(for: request, delegate: NoRedirects())
            return ((response as? HTTPURLResponse)?.statusCode ?? 0, data)
        } catch {
            guard !Task.isCancelled else { throw .cancelled }
            let code = (error as? URLError)?.code
            throw Self.failure(code == .timedOut ? "timeout" : "network error \(code?.rawValue ?? 0)")
        }
    }

    static func failure(_ reason: String) -> BasaltError {
        .modelInferenceError(details: "cloud decisions: \(reason)")
    }
}
