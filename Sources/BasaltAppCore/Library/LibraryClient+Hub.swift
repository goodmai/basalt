import BasaltCore
import Foundation

extension LibraryClient {
    public typealias TelemetryHandler = @Sendable (HubTransferTelemetry) -> Void

    static func liveSearchHub(
        _ query: String,
        limit: Int,
        endpoint: EngineEndpoint,
        session: URLSession
    ) async throws -> [HubModelSuggestion] {
        var components = URLComponents(url: endpoint.base.appending(path: "v1/models/hub"), resolvingAgainstBaseURL: false)
        components?.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "limit", value: String(limit)),
        ]
        var request = endpoint.request("v1/models/hub")
        request.url = components?.url
        let (data, response) = try await session.data(for: request)
        try EngineClient.check(response, data)
        struct Envelope: Decodable { let data: [HubModelSuggestion] }
        return try JSONDecoder().decode(Envelope.self, from: data).data
    }

    static func liveDownloadTelemetry(
        _ id: String,
        endpoint: EngineEndpoint,
        session: URLSession,
        onTelemetry: @escaping TelemetryHandler
    ) async throws {
        guard LibraryReducer.isRepoID(id) else {
            throw EngineError.malformed("not a repo id: \(id)")
        }
        let body = try JSONSerialization.data(withJSONObject: ["model": id, "stream": true])
        var request = endpoint.request("v1/models/download", method: "POST", body: body)
        request.timeoutInterval = 7 * 24 * 3600
        let (bytes, response) = try await session.bytes(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            var errData = Data()
            for try await byte in bytes.prefix(4096) { errData.append(byte) }
            try EngineClient.check(response, errData)
        }
        _ = try await relayHubTransferSSE(bytes, onTelemetry: onTelemetry)
    }

    static func livePublishModel(
        _ req: HubPublishRequest,
        endpoint: EngineEndpoint,
        session: URLSession,
        onTelemetry: @escaping TelemetryHandler
    ) async throws -> HubPublishResult {
        let payload: [String: Any] = [
            "source": req.sourceId,
            "repo": req.repoId,
            "private": req.isPrivate,
            "message": req.commitMessage,
            "stream": true,
        ]
        let body = try JSONSerialization.data(withJSONObject: payload)
        var request = endpoint.request("v1/models/publish", method: "POST", body: body)
        request.timeoutInterval = 7 * 24 * 3600
        let (bytes, response) = try await session.bytes(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            var errData = Data()
            for try await byte in bytes.prefix(4096) { errData.append(byte) }
            try EngineClient.check(response, errData)
        }
        let doneJSON = try await relayHubTransferSSE(bytes, onTelemetry: onTelemetry)
        let repo = (doneJSON["repo"] as? String) ?? req.repoId
        let url = (doneJSON["url"] as? String) ?? "https://huggingface.co/\(repo)"
        let files = (doneJSON["uploaded_files"] as? NSNumber)?.intValue ?? 0
        let total = (doneJSON["total_bytes"] as? NSNumber)?.int64Value ?? 0
        return HubPublishResult(repoId: repo, url: url, uploadedFiles: files, totalBytes: total)
    }

    private static func relayHubTransferSSE(
        _ bytes: URLSession.AsyncBytes,
        onTelemetry: @escaping TelemetryHandler
    ) async throws -> [String: Any] {
        var parser = SSEParser()
        var pending: [UInt8] = []
        for try await byte in bytes {
            pending.append(byte)
            guard byte == UInt8(ascii: "\n") else { continue }
            for event in parser.feed(pending) {
                if let done = try parseHubEvent(event, onTelemetry: onTelemetry) { return done }
            }
            pending.removeAll(keepingCapacity: true)
        }
        if !pending.isEmpty {
            pending.append(UInt8(ascii: "\n"))
            pending.append(UInt8(ascii: "\n"))
            for event in parser.feed(pending) {
                if let done = try parseHubEvent(event, onTelemetry: onTelemetry) { return done }
            }
        }
        for event in parser.feed([UInt8(ascii: "\n")]) {
            if let done = try parseHubEvent(event, onTelemetry: onTelemetry) { return done }
        }
        throw EngineError.transport("hub transfer stream closed before completion")
    }

    private static func parseHubEvent(
        _ event: SSEEvent,
        onTelemetry: @escaping TelemetryHandler
    ) throws -> [String: Any]? {
        let data = Data(event.data.utf8)
        switch event.name {
        case "progress":
            if let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                let done = (json["done"] as? NSNumber)?.int64Value ?? 0
                let total = (json["total"] as? NSNumber)?.int64Value ?? 0
                let bps = (json["bytes_per_second"] as? NSNumber)?.int64Value ?? 0
                let eta = (json["eta_seconds"] as? NSNumber)?.intValue
                onTelemetry(HubTransferTelemetry(done: done, total: total, bytesPerSecond: bps, etaSeconds: eta))
            }
            return nil
        case "done":
            return ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any]) ?? [:]
        case "error":
            if let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                let msg = ((json["error"] as? [String: Any])?["message"] as? String)
                    ?? (json["message"] as? String)
                    ?? "Hub transfer failed"
                throw EngineError.refused(status: 502, message: msg)
            }
            throw EngineError.refused(status: 502, message: event.data)
        default:
            return nil
        }
    }
}
