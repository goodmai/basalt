import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct SystemOneController: Sendable {
    let engine: @Sendable () async -> (any DecisionEngine)?
    var log: SessionLogWriter? = nil

    static func decide(
        _ engine: any DecisionEngine, _ request: SystemOneRequest, log: SessionLogWriter?, origin: RequestOrigin, sessionId: String?
    ) async throws(BasaltError) -> SystemOneResponse {
        let start = ContinuousClock.now
        let response = try await engine.decide(request)
        let elapsed = ContinuousClock.now - start
        await log?.logDecision(
            origin: origin, sessionId: sessionId, request: request, response: response,
            latencyMs: Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15)
        return response
    }

    static func request(from data: Data) -> SystemOneRequest? {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let state = root["state"],
              let questions = root["questions"], JSONSerialization.isValidJSONObject(["q": questions]),
              let questionData = try? JSONSerialization.data(withJSONObject: questions),
              let decoded = try? JSONDecoder().decode([String: SystemOneQuestion].self, from: questionData)
        else { return nil }
        let text: String
        if let string = state as? String {
            text = string
        } else if state is [String: Any] || state is [Any],
                  let encoded = try? JSONSerialization.data(withJSONObject: state, options: [.sortedKeys, .withoutEscapingSlashes]) {
            text = String(decoding: encoded, as: UTF8.self)
        } else {
            return nil
        }
        return SystemOneRequest(
            state: text, questions: decoded, calibrated: root["calibrated"] as? Bool,
            abstainBelow: (root["abstain_below"] as? NSNumber)?.doubleValue, minMargin: (root["min_margin"] as? NSNumber)?.doubleValue)
    }

    @Sendable
    func decide(request: Request, context: BasicRequestContext) async throws -> Response {
        guard let engine = await engine() else {
            return ProviderErrorResponse.response(status: .notFound, message: "No decision model is loaded; start basalt serve with --decision-model <OpenJev repo|dir>.", shape: .openAI)
        }
        let data = Data(try await request.body.collect(upTo: ChatPipeline.textBodyLimit).readableBytesView)
        guard let dto = Self.request(from: data) else {
            return ProviderErrorResponse.response(
                status: .badRequest,
                message: "A System One request needs \"state\" (text, object or array) and \"questions\": {id: {\"type\": \"noul\"|\"choice\"|\"score\", \"instructions\": ..., \"criteria\": ...}}.",
                shape: .openAI)
        }
        do {
            var buffer = ByteBuffer()
            let response = try await Self.decide(
                engine, dto, log: log, origin: RequestOrigin(source: "systemone", endpoint: "/v1/systemone", stream: false),
                sessionId: GenerationRequest.sessionId(in: request.headers))
            buffer.writeBytes(try JSONEncoder().encode(response))
            return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/v1/systemone")
        }
    }
}
