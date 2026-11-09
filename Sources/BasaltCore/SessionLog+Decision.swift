import Foundation

extension SessionLogRecord {
    public struct DecisionAnswer: Codable, Sendable, Equatable {
        public let type: String
        public let choice: String?
        public let noul: Double?
        public let score: Double?
        public let confidence: Double?
        public let margin: Double?
        public let abstain: Bool
    }

    public struct Decision: Codable, Sendable {
        public let v: Int
        public let type: String
        public let ts: String
        public let session: String
        public let source: String
        public let endpoint: String
        public let model: String
        public let state: Digest
        public let stateText: String?
        public let questions: [String: DecisionAnswer]
        public let escalate: Bool
        public let calibrated: Bool
        public let prefixTokens: Int
        public let branches: Int
        public let latencyMs: Double
        public let cost: Double?
        public let stages: [SystemOneStage]?

        enum CodingKeys: String, CodingKey {
            case v, type, ts, session, source, endpoint, model, state, questions, escalate, calibrated, branches, cost, stages
            case stateText = "state_text", prefixTokens = "prefix_tokens", latencyMs = "latency_ms"
        }
    }
}

extension SessionLogWriter {
    public func logDecision(origin: RequestOrigin, sessionId: String?, request: SystemOneRequest, response: SystemOneResponse, latencyMs: Double) {
        guard isEnabled else { return }
        let stamped = stamp(Self.sanitized(sessionId) ?? defaultSession)
        let questions = response.answers.reduce(into: [String: SessionLogRecord.DecisionAnswer]()) { result, entry in
            let answer = entry.value
            result[entry.key] = .init(
                type: request.questions[entry.key]?.type.rawValue ?? "", choice: answer.choice, noul: answer.noul, score: answer.score,
                confidence: answer.confidence ?? answer.noul.map { max($0, 1 - $0) }, margin: answer.margin, abstain: answer.abstain ?? false)
        }
        append(SessionLogRecord.Decision(
            v: SessionLogRecord.schemaVersion, type: "decision", ts: stamped.ts, session: stamped.session,
            source: origin.source, endpoint: origin.endpoint, model: response.model, state: .init(request.state),
            stateText: mode == .full ? (redacts ? SecretRedactor.redact(request.state) : request.state) : nil, questions: questions, escalate: response.escalate,
            calibrated: response.calibrated, prefixTokens: response.usage.prefixTokens, branches: response.usage.candidateBranches,
            latencyMs: latencyMs, cost: response.usage.cost, stages: response.usage.stages), session: stamped.session)
    }
}
