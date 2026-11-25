import Foundation
import BasaltCore

public enum BenchmarkError: Error, Sendable, Equatable, CustomStringConvertible {
    case unknownSuite(String, available: [String])
    case unknownAdapter(String)
    case httpFailure(status: Int, path: String, body: String)
    case invalidResponse(String)
    case budgetExceeded(String)
    case gpuLockBusy(String)
    case gpuLockIO(String)
    case submissionInvalid(target: SubmissionTarget, errors: [String])

    public var description: String {
        switch self {
        case .unknownSuite(let name, let available):
            let list = available.isEmpty ? "none" : available.joined(separator: ", ")
            return "unknown benchmark suite '\(name)' (declared: \(list))"
        case .unknownAdapter(let id):
            return "unknown benchmark adapter '\(id)'"
        case .httpFailure(let status, let path, let body):
            return "HTTP \(status) on \(path): \(body.prefix(240))"
        case .invalidResponse(let detail):
            return "invalid benchmark response: \(detail)"
        case .budgetExceeded(let detail):
            return "benchmark suite budget exceeded: \(detail)"
        case .gpuLockBusy(let path):
            return "GPU lock at \(path) is held by another process"
        case .gpuLockIO(let detail):
            return "GPU lock error: \(detail)"
        case .submissionInvalid(let target, let errors):
            return "submission validation failed for \(target.rawValue): \(errors.joined(separator: "; "))"
        }
    }
}

public struct BenchmarkHTTPRequest: Sendable, Equatable {
    public var method: String
    public var path: String
    public var headers: [String: String]
    public var body: Data?

    public init(method: String = "GET", path: String, headers: [String: String] = [:], body: Data? = nil) {
        (self.method, self.path, self.headers, self.body) = (method, path, headers, body)
    }
}

public struct BenchmarkHTTPResponse: Sendable, Equatable {
    public var statusCode: Int
    public var headers: [String: String]
    public var body: Data
    public var ttftMs: Double?
    public var totalMs: Double

    public init(
        statusCode: Int = 200, headers: [String: String] = [:], body: Data,
        ttftMs: Double? = nil, totalMs: Double = 1.0
    ) {
        (self.statusCode, self.headers, self.body) = (statusCode, headers, body)
        (self.ttftMs, self.totalMs) = (ttftMs, totalMs)
    }
}

public struct BenchmarkEndpoint: Sendable {
    public var baseURL: URL
    public var send: @Sendable (BenchmarkHTTPRequest) async throws -> BenchmarkHTTPResponse

    public init(
        baseURL: URL,
        send: @escaping @Sendable (BenchmarkHTTPRequest) async throws -> BenchmarkHTTPResponse
    ) {
        (self.baseURL, self.send) = (baseURL, send)
    }

    public static func mock(
        baseURL: URL = URL(string: "http://127.0.0.1:18080")!,
        handler: @escaping @Sendable (BenchmarkHTTPRequest) async throws -> BenchmarkHTTPResponse
    ) -> BenchmarkEndpoint {
        BenchmarkEndpoint(baseURL: baseURL, send: handler)
    }

    public static func live(
        baseURL: URL, apiKey: String? = nil, timeoutSeconds: TimeInterval = 600
    ) -> BenchmarkEndpoint {
        BenchmarkEndpoint(baseURL: baseURL) { req in
            let normalized = req.path.hasPrefix("/") ? String(req.path.dropFirst()) : req.path
            let url = baseURL.appendingPathComponent(normalized)
            var urlReq = URLRequest(url: url)
            urlReq.httpMethod = req.method
            urlReq.timeoutInterval = timeoutSeconds
            urlReq.httpBody = req.body
            if req.body != nil { urlReq.setValue("application/json", forHTTPHeaderField: "Content-Type") }
            if let apiKey, !apiKey.isEmpty { urlReq.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
            for (k, v) in req.headers { urlReq.setValue(v, forHTTPHeaderField: k) }
            let start = ContinuousClock.now
            if Self.requestsStreaming(req.body) {
                return try await Self.sendStreaming(urlReq, started: start)
            }
            let (data, response) = try await URLSession.shared.data(for: urlReq)
            let elapsed = start.duration(to: ContinuousClock.now)
            let ms = Double(elapsed.components.seconds) * 1000.0 + Double(elapsed.components.attoseconds) * 1e-15
            let http = response as? HTTPURLResponse
            var headers: [String: String] = [:]
            for (key, value) in http?.allHeaderFields ?? [:] {
                if let k = key as? String, let v = value as? String { headers[k] = v }
            }
            return BenchmarkHTTPResponse(statusCode: http?.statusCode ?? 0, headers: headers, body: data, totalMs: max(0.01, ms))
        }
    }

    private static func requestsStreaming(_ body: Data?) -> Bool {
        guard let body,
              let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let stream = json["stream"] as? Bool else {
            return false
        }
        return stream
    }

    private static func elapsedMs(since start: ContinuousClock.Instant) -> Double {
        let elapsed = start.duration(to: ContinuousClock.now)
        return max(0.01, Double(elapsed.components.seconds) * 1000.0 + Double(elapsed.components.attoseconds) * 1e-15)
    }

    private static func sendStreaming(
        _ urlReq: URLRequest,
        started: ContinuousClock.Instant
    ) async throws -> BenchmarkHTTPResponse {
        let (bytes, response) = try await URLSession.shared.bytes(for: urlReq)
        let http = response as? HTTPURLResponse
        let status = http?.statusCode ?? 0
        var headers: [String: String] = [:]
        for (key, value) in http?.allHeaderFields ?? [:] {
            if let k = key as? String, let v = value as? String { headers[k] = v }
        }
        guard (200 ... 299).contains(status) else {
            var raw = Data()
            for try await byte in bytes { raw.append(byte) }
            return BenchmarkHTTPResponse(statusCode: status, headers: headers, body: raw, totalMs: elapsedMs(since: started))
        }
        var content = ""
        var usage: [String: Any] = [:]
        var ttftMs: Double?
        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }
            let payload = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            if payload.isEmpty || payload == "[DONE]" { continue }
            guard let chunk = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any] else { continue }
            if let chunkUsage = chunk["usage"] as? [String: Any] { usage = chunkUsage }
            let delta = Self.extractStreamDelta(chunk)
            if ttftMs == nil, delta.hasToken { ttftMs = elapsedMs(since: started) }
            if !delta.content.isEmpty { content += delta.content }
        }
        var assembled: [String: Any] = ["choices": [["message": ["role": "assistant", "content": content]]]]
        if !usage.isEmpty { assembled["usage"] = usage }
        let body = try JSONSerialization.data(withJSONObject: assembled, options: [.sortedKeys])
        return BenchmarkHTTPResponse(statusCode: status, headers: headers, body: body, ttftMs: ttftMs, totalMs: elapsedMs(since: started))
    }

    private static func extractStreamDelta(_ chunk: [String: Any]) -> (content: String, hasToken: Bool) {
        guard let choices = chunk["choices"] as? [[String: Any]],
              let delta = choices.first?["delta"] as? [String: Any] else {
            return ("", false)
        }
        let text = delta["content"] as? String ?? ""
        let reasoning = (delta["reasoning_content"] as? String) ?? (delta["reasoning"] as? String) ?? ""
        return (text, !text.isEmpty || !reasoning.isEmpty)
    }

    public func postJSON(_ path: String, payload: Any) async throws -> (json: [String: Any], response: BenchmarkHTTPResponse) {
        let body = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        let reply = try await send(BenchmarkHTTPRequest(method: "POST", path: path, body: body))
        guard (200 ... 299).contains(reply.statusCode) else {
            throw BenchmarkError.httpFailure(status: reply.statusCode, path: path, body: String(decoding: reply.body, as: UTF8.self))
        }
        guard let object = try? JSONSerialization.jsonObject(with: reply.body) as? [String: Any] else {
            throw BenchmarkError.invalidResponse("\(path) did not return a JSON object")
        }
        return (object, reply)
    }

    public func getJSON(_ path: String) async throws -> (json: [String: Any], response: BenchmarkHTTPResponse) {
        let reply = try await send(BenchmarkHTTPRequest(method: "GET", path: path))
        guard (200 ... 299).contains(reply.statusCode) else {
            throw BenchmarkError.httpFailure(status: reply.statusCode, path: path, body: String(decoding: reply.body, as: UTF8.self))
        }
        guard let object = try? JSONSerialization.jsonObject(with: reply.body) as? [String: Any] else {
            throw BenchmarkError.invalidResponse("\(path) did not return a JSON object")
        }
        return (object, reply)
    }
}
