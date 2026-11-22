import Foundation
import BasaltCore

public struct CloudAPIConfig: Sendable {
    public let apiKey: String
    public let baseURL: URL
    public let timeout: TimeInterval
    public let maxRetries: Int
    public let extraHeaders: [String: String]

    public init(
        apiKey: String,
        baseURL: URL,
        timeout: TimeInterval = 60.0,
        maxRetries: Int = 3,
        extraHeaders: [String: String] = [:]
    ) {
        self.apiKey = apiKey
        self.baseURL = baseURL
        self.timeout = timeout
        self.maxRetries = maxRetries
        self.extraHeaders = extraHeaders
    }
}

public actor CloudAPIClient: Sendable {
    private let config: CloudAPIConfig
    private let session: URLSession
    private let sleep: @Sendable (Duration) async throws -> Void
    private var requestCount: Int = 0
    private var errorCount: Int = 0

    public init(
        config: CloudAPIConfig, configuration: URLSessionConfiguration = .default,
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) {
        self.config = config
        self.sleep = sleep
        let configuration = configuration.copy() as! URLSessionConfiguration
        configuration.timeoutIntervalForRequest = config.timeout
        configuration.timeoutIntervalForResource = config.timeout * 2
        self.session = URLSession(configuration: configuration)
    }

    public func performRequest<T: Decodable>(
        endpoint: String,
        method: String = "GET",
        body: Data? = nil,
        responseType: T.Type
    ) async throws -> T {
        let request = makeRequest(endpoint: endpoint, method: method, body: body)
        let maxAttempts = max(1, config.maxRetries)
        let noRedirects = NoRedirects()
        var lastError: Error?
        for attempt in 0..<maxAttempts {
            let attemptsLeft = attempt < maxAttempts - 1
            do {
                let (data, response) = try await session.data(for: request, delegate: noRedirects)
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw BasaltError.invalidRequestStructure(details: "Invalid response type")
                }
                if let failure = Self.failure(of: httpResponse, body: data) {
                    if case .modelInferenceError = failure, attemptsLeft {
                        try await backoff(attempt)
                        continue
                    }
                    throw failure
                }
                let decoded = try JSONDecoder().decode(T.self, from: data)
                requestCount += 1
                return decoded
            } catch let error as BasaltError {
                guard case .modelInferenceError = error else {
                    errorCount += 1
                    throw error
                }
                lastError = error
            } catch let error as DecodingError {
                errorCount += 1
                throw error
            } catch {
                if Self.isCancellation(error) { throw error }
                lastError = error
                if attemptsLeft { try await backoff(attempt) }
            }
        }
        errorCount += 1
        throw lastError ?? BasaltError.modelInferenceError(
            details: "Request failed after \(maxAttempts) attempts"
        )
    }

    private static func failure(of response: HTTPURLResponse, body: Data) -> BasaltError? {
        let status = response.statusCode
        guard !(200...299).contains(status) else { return nil }
        let errorBody = String(data: body, encoding: .utf8) ?? "Unknown error"
        switch status {
        case 401:
            return .authenticationFailed(details: "Invalid API key")
        case 429:
            return .rateLimitExceeded(retryAfter: response.value(forHTTPHeaderField: "Retry-After"))
        case 500...599:
            return .modelInferenceError(details: "API error (HTTP \(status)): \(errorBody)")
        default:
            return .invalidRequestStructure(details: "HTTP \(status): \(errorBody)")
        }
    }

    private static func isCancellation(_ error: Error) -> Bool {
        Task.isCancelled || error is CancellationError || (error as? URLError)?.code == .cancelled
    }

    private func backoff(_ attempt: Int) async throws {
        try await sleep(.seconds(1 << min(max(0, attempt), 10)))
    }

    private func makeRequest(endpoint: String, method: String, body: Data?, accept: String? = nil) -> URLRequest {
        var request = URLRequest(url: config.baseURL.appendingPathComponent(endpoint))
        request.httpMethod = method
        if !config.apiKey.isEmpty { request.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization") }
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let accept { request.setValue(accept, forHTTPHeaderField: "Accept") }
        for (key, value) in config.extraHeaders {
            request.setValue(value, forHTTPHeaderField: key)
        }
        if let body {
            request.httpBody = body
        }
        return request
    }

    public func performStreamingRequest<T: Decodable & Sendable>(
        endpoint: String,
        method: String = "POST",
        body: Data? = nil,
        responseType: T.Type
    ) async throws -> AsyncThrowingStream<T, Error> {
        let request = makeRequest(endpoint: endpoint, method: method, body: body, accept: "text/event-stream")
        let (result, response) = try await session.bytes(for: request, delegate: NoRedirects())
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw BasaltError.invalidRequestStructure(details: "Invalid response type")
        }
        
        if !(200...299).contains(httpResponse.statusCode) {
            var errBytes: [UInt8] = []
            errBytes.reserveCapacity(1024)
            do {
                for try await byte in result {
                    errBytes.append(byte)
                    if errBytes.count >= 8192 { break }
                }
            } catch {}
            if let failure = Self.failure(of: httpResponse, body: Data(errBytes)) {
                self.errorCount += 1
                throw failure
            }
        }
        
        self.requestCount += 1
        
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let decoder = JSONDecoder()
                    for try await line in result.lines {
                        guard line.hasPrefix("data:") else { continue }
                        let dataString = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
                        guard !dataString.isEmpty else { continue }
                        if dataString == "[DONE]" {
                            continuation.finish()
                            return
                        }
                        guard let data = dataString.data(using: .utf8) else { continue }
                        let decoded = try decoder.decode(T.self, from: data)
                        continuation.yield(decoded)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func getMetrics() -> (requests: Int, errors: Int) {
        return (requestCount, errorCount)
    }

    public func performProbe<T: Decodable & Sendable>(
        endpoint: String,
        timeout: Duration,
        responseType: T.Type
    ) async throws -> T {
        var request = makeRequest(endpoint: endpoint, method: "GET", body: nil)
        let seconds = Double(timeout.components.seconds) + Double(timeout.components.attoseconds) * 1e-18
        request.timeoutInterval = max(0.01, seconds)
        let probeRequest = request
        let session = self.session
        let (data, response) = try await withThrowingTaskGroup(of: (Data, URLResponse).self) { group in
            group.addTask { try await session.data(for: probeRequest, delegate: NoRedirects()) }
            group.addTask {
                try await Task.sleep(for: timeout)
                throw URLError(.timedOut)
            }
            guard let first = try await group.next() else { throw URLError(.timedOut) }
            group.cancelAll()
            return first
        }
        guard let httpResponse = response as? HTTPURLResponse else {
            errorCount += 1
            throw BasaltError.invalidRequestStructure(details: "Invalid response type")
        }
        if let failure = Self.failure(of: httpResponse, body: data) {
            errorCount += 1
            throw failure
        }
        requestCount += 1
        return try JSONDecoder().decode(T.self, from: data)
    }
}
