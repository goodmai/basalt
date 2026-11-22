import BasaltCore
import Foundation

public struct EngineClient: Sendable {
    public var chat: @Sendable (ChatRequest) -> AsyncThrowingStream<ChatDelta, any Error>
    public var health: @Sendable () async -> Bool
    public var stats: @Sendable () async throws -> ServerTelemetrySnapshot
    public var pipeline: @Sendable (PipelineCall) async throws -> PipelineResponse
    public var compose: @Sendable () async throws -> ComposeInfo? = { nil }
    public var readiness: @Sendable () async throws -> EngineReadiness = { throw EngineError.notRunning }
    public var composeEvents: @Sendable () -> AsyncThrowingStream<ComposeStreamEvent, any Error> = {
        AsyncThrowingStream { $0.finish(throwing: EngineError.notRunning) }
    }
    public var loadModel: @Sendable (String) async throws -> String = { _ in throw EngineError.notRunning }
    public var applyCompose: @Sendable (ComposeApplyCall) async throws -> ComposeDiff = { _ in throw EngineError.notRunning }
    public var downloadModel: @Sendable (String, @escaping @Sendable (Int64, Int64) -> Void) async throws -> Void = { _, _ in
        throw EngineError.notRunning
    }
    public var deleteModel: @Sendable (String) async throws -> Void = { _ in throw EngineError.notRunning }
    public var transcribeAudio: @Sendable (AudioTranscriptionCall) async throws -> String = { _ in throw EngineError.notRunning }
    public var synthesizeSpeech: @Sendable (SpeechSynthesisCall) async throws -> Data = { _ in throw EngineError.notRunning }
    public var streamImage: @Sendable (ImageGenerationCall) -> AsyncThrowingStream<ImageStreamEvent, any Error> = { _ in
        AsyncThrowingStream { $0.finish(throwing: EngineError.notRunning) }
    }
    public var fetchImageContent: @Sendable (String) async throws -> Data = { _ in throw EngineError.notRunning }

    public init(
        chat: @escaping @Sendable (ChatRequest) -> AsyncThrowingStream<ChatDelta, any Error>,
        health: @escaping @Sendable () async -> Bool,
        stats: @escaping @Sendable () async throws -> ServerTelemetrySnapshot,
        pipeline: @escaping @Sendable (PipelineCall) async throws -> PipelineResponse = { _ in throw EngineError.notRunning }
    ) {
        (self.chat, self.health, self.stats, self.pipeline) = (chat, health, stats, pipeline)
    }
}

public struct EngineEndpoint: Sendable, Equatable {
    public var base: URL
    public var token: String?

    public init(base: URL, token: String? = nil) { (self.base, self.token) = (base, token) }

    public static func loopback(port: Int, token: String?) -> EngineEndpoint {
        EngineEndpoint(base: URL(string: "http://127.0.0.1:\(port)")!, token: token)
    }

    func request(_ path: String, method: String = "GET", body: Data? = nil, contentType: String = "application/json") -> URLRequest {
        var request = URLRequest(url: base.appending(path: path))
        (request.httpMethod, request.httpBody) = (method, body)
        if body != nil { request.setValue(contentType, forHTTPHeaderField: "Content-Type") }
        if let token, !token.isEmpty { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        return request
    }

    func rawRequest(url: URL, method: String = "GET", body: Data? = nil, contentType: String = "application/json") -> URLRequest {
        var request = URLRequest(url: url)
        (request.httpMethod, request.httpBody) = (method, body)
        if body != nil { request.setValue(contentType, forHTTPHeaderField: "Content-Type") }
        if let token, !token.isEmpty { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        return request
    }
}

extension EngineClient {
    public static func http(_ endpoint: EngineEndpoint, session: URLSession = .engine) -> EngineClient {
        var client = EngineClient(
            chat: { request in chatStream(endpoint, request, session: session) },
            health: {
                guard let (_, response) = try? await session.data(for: endpoint.request("api/v1/health")) else { return false }
                return (response as? HTTPURLResponse)?.statusCode == 200
            },
            stats: {
                let (data, response) = try await session.data(for: endpoint.request("api/v1/stats"))
                try check(response, data)
                return try JSONDecoder().decode(ServerTelemetrySnapshot.self, from: data)
            },
            pipeline: { call in
                var request = endpoint.request("v1/pipeline", method: "POST", body: call.body)
                request.timeoutInterval = 900
                let (data, response) = try await session.data(for: request)
                try check(response, data)
                do {
                    return try JSONDecoder().decode(PipelineResponse.self, from: data)
                } catch {
                    throw EngineError.malformed("pipeline answer: \(error)")
                }
            })
        configureCompose(&client, endpoint, session: session)
        configureSpeech(&client, endpoint, session: session)
        configureImages(&client, endpoint, session: session)
        return client
    }

    static func chatStream(_ endpoint: EngineEndpoint, _ request: ChatRequest, session: URLSession) -> AsyncThrowingStream<ChatDelta, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let body = try JSONEncoder().encode(request)
                    var urlRequest = endpoint.request("v1/chat/completions", method: "POST", body: body)
                    urlRequest.timeoutInterval = 900
                    let (bytes, response) = try await session.bytes(for: urlRequest)
                    if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                        var data = Data()
                        for try await byte in bytes.prefix(4096) { data.append(byte) }
                        try check(response, data)
                    }
                    try await relay(bytes, to: continuation)
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

    static func relay(_ bytes: URLSession.AsyncBytes, to continuation: AsyncThrowingStream<ChatDelta, any Error>.Continuation) async throws {
        var parser = SSEParser()
        var pending: [UInt8] = []
        for try await byte in bytes {
            pending.append(byte)
            guard byte == UInt8(ascii: "\n") else { continue }
            for event in parser.feed(pending) {
                switch try ChatDeltaDecoder.decode(event) {
                case .done: return
                case .deltas(let deltas): for delta in deltas { continuation.yield(delta) }
                }
            }
            pending.removeAll(keepingCapacity: true)
        }
        throw Task.isCancelled ? EngineError.cancelled : EngineError.transport("chat stream ended before [DONE]")
    }

    static func check(_ response: URLResponse, _ data: Data) throws(EngineError) {
        guard let http = response as? HTTPURLResponse, http.statusCode != 200 else { return }
        throw refusedError(status: http.statusCode, data: data)
    }

    static func refusedError(status: Int, data: Data) -> EngineError {
        let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        let message = ((json?["error"] as? [String: Any])?["message"] as? String) ?? String(decoding: data.prefix(300), as: UTF8.self)
        return .refused(status: status, message: message)
    }
}

extension URLSession {
    public static let engine: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpMaximumConnectionsPerHost = 32
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 7 * 24 * 3600
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        return URLSession(configuration: configuration)
    }()
}
