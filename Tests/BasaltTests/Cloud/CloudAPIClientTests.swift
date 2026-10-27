import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCloud

@Suite("CloudAPIClient over a stub URLProtocol")
struct CloudAPIClientTests {

    private struct Reply: Codable, Sendable, Equatable { let value: String }

    private func client(host: String, maxRetries: Int = 2, headers: [String: String] = [:]) -> CloudAPIClient {
        let config = CloudAPIConfig(apiKey: "sk-test", baseURL: URL(string: "https://\(host)/v1")!,
                                    timeout: 5, maxRetries: maxRetries, extraHeaders: headers)
        return CloudAPIClient(config: config, configuration: StubURLProtocol.configuration)
    }

    private func json(_ s: String) -> Data { Data(s.utf8) }

    @Test("A request carries the key, JSON content type, extra headers, method and body, and decodes the reply")
    func success() async throws {
        let host = StubURLProtocol.host { _, _ in (200, [:], Data(#"{"value": "ok"}"#.utf8)) }
        let api = client(host: host, headers: ["HTTP-Referer": "basalt"])
        let reply = try await api.performRequest(endpoint: "chat", method: "POST",
                                                 body: json(#"{"q":1}"#), responseType: Reply.self)
        #expect(reply == Reply(value: "ok"))

        let (request, body) = try #require(StubURLProtocol.requests(for: host).first)
        #expect(request.url?.path == "/v1/chat")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer sk-test")
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
        #expect(request.value(forHTTPHeaderField: "HTTP-Referer") == "basalt")
        #expect(body == json(#"{"q":1}"#))
        #expect(await api.getMetrics() == (requests: 1, errors: 0))
    }

    @Test("401, 429 and other 4xx map to their errors and are not retried")
    func clientErrorsAreNotRetried() async throws {
        let cases: [(Int, [String: String], (BasaltError) -> Bool)] = [
            (401, [:], { if case .authenticationFailed = $0 { true } else { false } }),
            (429, ["Retry-After": "17"], { if case .rateLimitExceeded(let after) = $0 { after == "17" } else { false } }),
            (400, [:], { if case .invalidRequestStructure(let d) = $0 { d.contains("HTTP 400") && d.contains("bad field") } else { false } }),
        ]
        for (status, headers, matches) in cases {
            let host = StubURLProtocol.host { _, _ in (status, headers, Data("bad field".utf8)) }
            let api = client(host: host, maxRetries: 3)
            do {
                _ = try await api.performRequest(endpoint: "x", responseType: Reply.self)
                Issue.record("HTTP \(status) did not throw")
            } catch let error as BasaltError {
                #expect(matches(error), "HTTP \(status): \(error)")
            }
            #expect(StubURLProtocol.requests(for: host).count == 1, "HTTP \(status) must not be retried")
            #expect(await api.getMetrics() == (requests: 0, errors: 1))
        }
    }

    @Test("A 5xx is retried; it succeeds when the retry does, and fails after maxRetries attempts")
    func serverErrorsAreRetried() async throws {
        let counter = Counter()
        let flaky = StubURLProtocol.host { _, _ in
            counter.next() == 1 ? (503, [:], Data()) : (200, [:], Data(#"{"value": "second"}"#.utf8))
        }
        let reply = try await client(host: flaky).performRequest(endpoint: "x", responseType: Reply.self)
        #expect(reply.value == "second")
        #expect(StubURLProtocol.requests(for: flaky).count == 2)

        let down = StubURLProtocol.host { _, _ in (500, [:], Data("boom".utf8)) }
        let api = client(host: down)
        await #expect {
            _ = try await api.performRequest(endpoint: "x", responseType: Reply.self)
        } throws: { error in
            if case .modelInferenceError(let d) = error as? BasaltError { return d.contains("HTTP 500") && d.contains("boom") }
            return false
        }
        #expect(StubURLProtocol.requests(for: down).count == 2)
        #expect(await api.getMetrics().errors == 1)
    }

    @Test("A streaming request yields each data line until [DONE] and skips comments and blank lines")
    func streaming() async throws {
        let sse = ": keep-alive\n\ndata: {\"value\": \"a\"}\n\nevent: x\ndata: {\"value\": \"b\"}\n\ndata: [DONE]\n\ndata: {\"value\": \"after\"}\n\n"
        let host = StubURLProtocol.host { _, _ in (200, ["Content-Type": "text/event-stream"], Data(sse.utf8)) }
        let stream = try await client(host: host).performStreamingRequest(endpoint: "chat", body: json("{}"), responseType: Reply.self)
        var values: [String] = []
        for try await reply in stream { values.append(reply.value) }
        #expect(values == ["a", "b"])
        let request = try #require(StubURLProtocol.requests(for: host).first?.0)
        #expect(request.value(forHTTPHeaderField: "Accept") == "text/event-stream")
        #expect(request.httpMethod == "POST")
    }

    @Test("A streaming request with a non-2xx status throws before yielding")
    func streamingError() async throws {
        let host = StubURLProtocol.host { _, _ in (502, [:], Data()) }
        let api = client(host: host)
        await #expect(throws: BasaltError.self) {
            _ = try await api.performStreamingRequest(endpoint: "chat", responseType: Reply.self)
        }
        #expect(await api.getMetrics() == (requests: 0, errors: 1))
    }

    @Test("A keyless endpoint (a self-hosted OpenAI-compatible server) gets no Authorization header")
    func keylessRequestHasNoAuthorization() async throws {
        let host = StubURLProtocol.host { _, _ in (200, [:], Data(#"{"value": "ok"}"#.utf8)) }
        let config = CloudAPIConfig(apiKey: "", baseURL: URL(string: "https://\(host)/v1")!, timeout: 5, maxRetries: 1)
        _ = try await CloudAPIClient(config: config, configuration: StubURLProtocol.configuration)
            .performRequest(endpoint: "chat", method: "POST", body: json("{}"), responseType: Reply.self)
        let (request, _) = try #require(StubURLProtocol.requests(for: host).first)
        #expect(request.value(forHTTPHeaderField: "Authorization") == nil)
    }
}

private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func next() -> Int { lock.withLock { value += 1; return value } }
}
