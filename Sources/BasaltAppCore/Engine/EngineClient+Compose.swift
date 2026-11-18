import BasaltCore
import Foundation

public struct EngineReadiness: Sendable, Equatable, Decodable {
    public var status: String
    public var waiting: [String]

    public init(status: String, waiting: [String] = []) {
        (self.status, self.waiting) = (status, waiting)
    }

    public var isReady: Bool { status == "ready" && waiting.isEmpty }
}

public struct ComposeStreamEvent: Sendable, Equatable, Decodable {
    public var at: Date?
    public var kind: String
    public var model: String?
    public var message: String

    public init(at: Date? = nil, kind: String, model: String? = nil, message: String) {
        (self.at, self.kind, self.model, self.message) = (at, kind, model, message)
    }
}

public struct ComposeApplyCall: Sendable, Equatable, Encodable {
    public var yaml: String
    public var base: String?
    public var variables: [String: String]?
    public var profiles: [String]?

    public init(yaml: String, base: String? = nil, variables: [String: String]? = nil, profiles: [String]? = nil) {
        (self.yaml, self.base, self.variables, self.profiles) = (yaml, base, variables, profiles)
    }
}

public enum HotSwitchPlan: Sendable, Equatable {
    case noop
    case hotLoadModel(String)
    case hotApplyCompose(path: String)
    case relaunch([String])

    public static func plan(from current: [String], to next: [String], isReady: Bool) -> HotSwitchPlan {
        guard current != next else { return .noop }
        guard isReady else { return .relaunch(next) }
        if current.count == 3, next.count == 3,
           current[0] == "serve", current[1] == "--model",
           next[0] == "serve", next[1] == "--model",
           LibraryReducer.isRepoID(next[2]) {
            return .hotLoadModel(next[2])
        }
        if current.count == 3, next.count == 3,
           current[0] == "serve", current[1] == "--compose",
           next[0] == "serve", next[1] == "--compose",
           !next[2].isEmpty {
            return .hotApplyCompose(path: next[2])
        }
        return .relaunch(next)
    }
}

extension EngineClient {
    static func configureCompose(_ client: inout EngineClient, _ endpoint: EngineEndpoint, session: URLSession) {
        client.compose = { try await fetchCompose(endpoint, session: session) }
        client.readiness = { try await fetchReadiness(endpoint, session: session) }
        client.composeEvents = { composeEventStream(endpoint, session: session) }
        client.loadModel = { id in try await loadRemoteModel(id, endpoint, session: session) }
        client.applyCompose = { call in try await applyRemoteCompose(call, endpoint, session: session) }
        configureHub(&client, endpoint, session: session)
    }

    private static func fetchCompose(_ endpoint: EngineEndpoint, session: URLSession) async throws -> ComposeInfo? {
        let (data, response) = try await session.data(for: endpoint.request("v1/compose"))
        if (response as? HTTPURLResponse)?.statusCode == 404 { return nil }
        try check(response, data)
        return try JSONDecoder().decode(ComposeInfo.self, from: data)
    }

    private static func fetchReadiness(_ endpoint: EngineEndpoint, session: URLSession) async throws -> EngineReadiness {
        let (data, response) = try await session.data(for: endpoint.request("v1/health/ready"))
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        if code != 200 && code != 503 { try check(response, data) }
        do {
            return try JSONDecoder().decode(EngineReadiness.self, from: data)
        } catch {
            throw EngineError.malformed("readiness: \(error)")
        }
    }

    private static func loadRemoteModel(_ id: String, _ endpoint: EngineEndpoint, session: URLSession) async throws -> String {
        guard LibraryReducer.isRepoID(id) else { throw EngineError.malformed("not a repo id: \(id)") }
        let body = try JSONSerialization.data(withJSONObject: ["model": id])
        var request = endpoint.request("v1/models/load", method: "POST", body: body)
        request.timeoutInterval = 600
        let (data, response) = try await session.data(for: request)
        try check(response, data)
        let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        return (json?["display_name"] as? String) ?? id
    }

    private static func applyRemoteCompose(_ call: ComposeApplyCall, _ endpoint: EngineEndpoint, session: URLSession) async throws -> ComposeDiff {
        let body = try JSONEncoder().encode(call)
        var request = endpoint.request("v1/compose/apply", method: "POST", body: body)
        request.timeoutInterval = 600
        let (data, response) = try await session.data(for: request)
        try check(response, data)
        do {
            return try JSONDecoder().decode(ComposeDiff.self, from: data)
        } catch {
            throw EngineError.malformed("compose diff: \(error)")
        }
    }
}

extension EngineClient {
    static func configureHub(_ client: inout EngineClient, _ endpoint: EngineEndpoint, session: URLSession) {
        client.downloadModel = { id, progress in
            guard LibraryReducer.isRepoID(id) else { throw EngineError.malformed("not a repo id: \(id)") }
            let body = try JSONSerialization.data(withJSONObject: ["model": id, "stream": true])
            var request = endpoint.request("v1/models/download", method: "POST", body: body)
            request.timeoutInterval = 7 * 24 * 3600
            let (bytes, response) = try await session.bytes(for: request)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                var errData = Data()
                for try await byte in bytes.prefix(4096) { errData.append(byte) }
                try check(response, errData)
            }
            try await relayDownload(bytes, progress: progress)
        }
        client.deleteModel = { id in
            guard LibraryReducer.isRepoID(id) else { throw EngineError.malformed("not a repo id: \(id)") }
            let safe = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
            let encoded = id.addingPercentEncoding(withAllowedCharacters: safe) ?? id
            let base = endpoint.base.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            guard let url = URL(string: "\(base)/v1/models/\(encoded)") else {
                throw EngineError.malformed("invalid model URL for \(id)")
            }
            let (data, response) = try await session.data(for: endpoint.rawRequest(url: url, method: "DELETE"))
            try check(response, data)
        }
    }

    private static func relayDownload(
        _ bytes: URLSession.AsyncBytes,
        progress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws {
        var parser = SSEParser()
        var pending: [UInt8] = []
        for try await byte in bytes {
            pending.append(byte)
            guard byte == UInt8(ascii: "\n") else { continue }
            for event in parser.feed(pending) {
                if try handleDownloadEvent(event, progress: progress) { return }
            }
            pending.removeAll(keepingCapacity: true)
        }
        if !pending.isEmpty {
            pending.append(UInt8(ascii: "\n"))
            pending.append(UInt8(ascii: "\n"))
            for event in parser.feed(pending) {
                if try handleDownloadEvent(event, progress: progress) { return }
            }
        }
        for event in parser.feed([UInt8(ascii: "\n")]) {
            if try handleDownloadEvent(event, progress: progress) { return }
        }
        throw EngineError.transport("model download stream closed before completion")
    }

    private static func handleDownloadEvent(
        _ event: SSEEvent,
        progress: @escaping @Sendable (Int64, Int64) -> Void
    ) throws -> Bool {
        switch event.name {
        case "progress":
            if let json = (try? JSONSerialization.jsonObject(with: Data(event.data.utf8))) as? [String: Any] {
                let done = (json["done"] as? NSNumber)?.int64Value ?? 0
                let total = (json["total"] as? NSNumber)?.int64Value ?? 0
                progress(done, total)
            }
            return false
        case "done":
            return true
        case "error":
            throw refusedError(status: 502, data: Data(event.data.utf8))
        default:
            return false
        }
    }
}

extension EngineClient {
    static func composeEventStream(_ endpoint: EngineEndpoint, session: URLSession) -> AsyncThrowingStream<ComposeStreamEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await session.bytes(for: endpoint.request("v1/compose/events"))
                    if let http = response as? HTTPURLResponse {
                        if http.statusCode == 404 {
                            continuation.finish()
                            return
                        }
                        if http.statusCode != 200 {
                            var data = Data()
                            for try await byte in bytes.prefix(4096) { data.append(byte) }
                            try check(response, data)
                        }
                    }
                    try await relayComposeEvents(bytes, to: continuation)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: EngineError.cancelled)
                } catch let error as EngineError {
                    continuation.finish(throwing: error)
                } catch {
                    continuation.finish(throwing: Task.isCancelled ? EngineError.cancelled : EngineError.transport(error.localizedDescription))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func relayComposeEvents(
        _ bytes: URLSession.AsyncBytes,
        to continuation: AsyncThrowingStream<ComposeStreamEvent, any Error>.Continuation
    ) async throws {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var parser = SSEParser()
        var pending: [UInt8] = []
        for try await byte in bytes {
            pending.append(byte)
            guard byte == UInt8(ascii: "\n") else { continue }
            for event in parser.feed(pending) {
                if let item = try? decoder.decode(ComposeStreamEvent.self, from: Data(event.data.utf8)) {
                    continuation.yield(item)
                }
            }
            pending.removeAll(keepingCapacity: true)
        }
    }
}
