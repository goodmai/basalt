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
        var lastError: Error?
        for attempt in 0..<config.maxRetries {
            let attemptsLeft = attempt < config.maxRetries - 1
            do {
                let (data, response) = try await session.data(for: request)
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
            } catch {
                lastError = error
                if attemptsLeft { try await backoff(attempt) }
            }
        }
        errorCount += 1
        throw lastError ?? BasaltError.modelInferenceError(
            details: "Request failed after \(config.maxRetries) attempts"
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

    private func backoff(_ attempt: Int) async throws {
        try await sleep(.seconds(1 << attempt))
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
        let (result, response) = try await session.bytes(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw BasaltError.invalidRequestStructure(details: "Invalid response type")
        }
        
        if let failure = Self.failure(of: httpResponse, body: Data()) {
            self.errorCount += 1
            throw failure
        }
        
        self.requestCount += 1
        
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let decoder = JSONDecoder()
                    for try await line in result.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let dataString = line.dropFirst("data: ".count).trimmingCharacters(in: .whitespaces)
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
        }
    }

    public func getMetrics() -> (requests: Int, errors: Int) {
        return (requestCount, errorCount)
    }
}
