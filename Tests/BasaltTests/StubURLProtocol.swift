import Foundation

/// URLProtocol stub keyed by host, so tests running in parallel each own their responses:
/// give every test its own host (`StubURLProtocol.host()`), register a handler for it, and pass
/// `StubURLProtocol.configuration` to the client under test.
final class StubURLProtocol: URLProtocol {
    typealias Handler = @Sendable (URLRequest, Data?) -> (status: Int, headers: [String: String], body: Data)

    private static let lock = NSLock()
    nonisolated(unsafe) private static var handlers: [String: Handler] = [:]
    nonisolated(unsafe) private static var requests: [String: [(URLRequest, Data?)]] = [:]

    static var configuration: URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        return config
    }

    /// A fresh host with `handler` behind it.
    static func host(_ handler: @escaping Handler) -> String {
        let host = "stub-\(UUID().uuidString.lowercased()).test"
        lock.withLock { handlers[host] = handler }
        return host
    }

    /// Requests seen for `host`, with their bodies.
    static func requests(for host: String) -> [(URLRequest, Data?)] {
        lock.withLock { requests[host] ?? [] }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let host = request.url?.host ?? ""
        let body = request.httpBody ?? request.httpBodyStream.map(Self.read)
        let handler = Self.lock.withLock { () -> Handler? in
            Self.requests[host, default: []].append((request, body))
            return Self.handlers[host]
        }
        guard let handler, let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotFindHost))
            return
        }
        let (status, headers, data) = handler(request, body)
        let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func read(_ stream: InputStream) -> Data {
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while stream.hasBytesAvailable {
            let n = stream.read(&buffer, maxLength: buffer.count)
            if n <= 0 { break }
            data.append(buffer, count: n)
        }
        return data
    }
}
