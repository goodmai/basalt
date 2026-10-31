import Foundation
import Synchronization

final class StubURLProtocol: URLProtocol {
    typealias Handler = @Sendable (URLRequest, Data?) -> (status: Int, headers: [String: String], body: Data)

    private struct State: Sendable {
        var handlers: [String: Handler] = [:]
        var requests: [String: [(URLRequest, Data?)]] = [:]
    }

    private static let state = Mutex(State())

    static var configuration: URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        return config
    }

    static func host(_ handler: @escaping Handler) -> String {
        let host = "stub-\(UUID().uuidString.lowercased()).test"
        state.withLock { $0.handlers[host] = handler }
        return host
    }

    static func requests(for host: String) -> [(URLRequest, Data?)] {
        state.withLock { $0.requests[host] ?? [] }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let host = request.url?.host ?? ""
        let body = request.httpBody ?? request.httpBodyStream.map(Self.read)
        let handler = Self.state.withLock { state -> Handler? in
            state.requests[host, default: []].append((request, body))
            return state.handlers[host]
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
