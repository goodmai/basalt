import Foundation

extension SessionLogRecord {
    public struct Pipeline: Codable, Sendable {
        public let v: Int
        public let type: String
        public let ts: String
        public let session: String
        public let source: String
        public let endpoint: String
        public let receipt: PipelineReceipt
    }
}

extension SessionLogWriter {
    public func logPipeline(origin: RequestOrigin, sessionId: String?, receipt: PipelineReceipt) {
        guard isEnabled else { return }
        let stamped = stamp(Self.sanitized(sessionId) ?? defaultSession)
        append(SessionLogRecord.Pipeline(
            v: SessionLogRecord.schemaVersion, type: "pipeline", ts: stamped.ts, session: stamped.session,
            source: origin.source, endpoint: origin.endpoint, receipt: receipt), session: stamped.session)
    }
}
