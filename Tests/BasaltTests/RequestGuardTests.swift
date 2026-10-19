import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltServer
@testable import BasaltCLI

@Suite("Host, Origin and Content-Type checks on the real router (#72)")
struct RequestGuardTests {

    private static let host = HTTPField.Name("Host")!
    private static let messages = #"{"max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}"#

    private func app(host: String = "127.0.0.1", apiKey: String? = nil, origins: [String] = []) async throws
        -> Application<RouterResponder<BasicRequestContext>> {
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine(), maxTokens: 65536, telemetry: TelemetryTracker())
        try await orch.loadModel(path: "mock")
        let config = try ServerConfig(modelPath: "org/mock-model", host: host, apiKey: apiKey, allowedOrigins: origins)
        return Application(router: RESTServer(orchestrator: orch, config: config).makeRouter())
    }

    private func post(
        _ client: some TestClientProtocol, _ uri: String = "/v1/messages", _ body: String = messages,
        headers: HTTPFields = [.contentType: "application/json"]
    ) async throws -> HTTPResponse.Status {
        try await client.execute(uri: uri, method: .post, headers: headers, body: ByteBuffer(string: body)) { $0.status }
    }

    @Test("Loopback bind: a foreign Host is refused with 403, with or without Origin (DNS rebinding)")
    func foreignHostOnLoopback() async throws {
        try await app().test(.router) { (client) async throws in
            for value in ["evil.example", "attacker.example:8080", "127.0.0.1.evil.example", "localhost.", "0.0.0.0:8080"] {
                var headers: HTTPFields = [.contentType: "application/json"]
                headers[Self.host] = value
                #expect(try await post(client, headers: headers) == .forbidden, "\(value)")
                let status = try await client.execute(uri: "/v1/models", method: .get, headers: [Self.host: value]) { $0.status }
                #expect(status == .forbidden, "a rebound page must not read GET responses either: \(value)")
            }
            for value in ["localhost", "localhost:8080", "127.0.0.1:8080", "[::1]:8080", "LOCALHOST:1"] {
                var headers: HTTPFields = [.contentType: "application/json"]
                headers[Self.host] = value
                #expect(try await post(client, headers: headers) == .ok, "\(value)")
            }
        }
    }

    @Test("A cross-origin page is refused with 403; loopback origins on the server's own port and allow-listed ones pass")
    func originCheck() async throws {
        try await app(origins: ["https://app.example:3000", "http://localhost:5173"]).test(.router) { (client) async throws in
            for origin in ["https://evil.example", "http://evil.example:8080", "null", "file://", "http://localhost.evil.example",
                           "https://app.example", "http://app.example:3000", "https://app.example:3000/path",
                           "http://localhost:3000", "http://127.0.0.1:8888", "http://[::1]:9000", "https://localhost", "http://localhost"] {
                #expect(try await post(client, headers: [.contentType: "application/json", .origin: origin]) == .forbidden, "\(origin)")
            }
            for origin in ["http://localhost:8080", "http://127.0.0.1:8080", "https://localhost:8080", "http://[::1]:8080",
                           "https://app.example:3000", "HTTPS://APP.example:3000", "http://localhost:5173"] {
                #expect(try await post(client, headers: [.contentType: "application/json", .origin: origin]) == .ok, "\(origin)")
            }
        }
    }

    @Test("POST needs Content-Type application/json: a CORS simple request gets 415")
    func contentType() async throws {
        try await app().test(.router) { (client) async throws in
            for uri in ["/v1/messages", "/v1/chat/completions", "/v1/responses", "/v1/generate",
                        "/api/v1/generate", "/api/v1/generate/stream", "/v1/models/load"] {
                for type in ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"] {
                    #expect(try await post(client, uri, headers: [.contentType: type]) == .unsupportedMediaType, "\(uri) \(type)")
                }
                #expect(try await post(client, uri, headers: [:]) == .unsupportedMediaType, "\(uri) without Content-Type")
            }
            for type in ["application/json", "application/json; charset=utf-8", "Application/JSON"] {
                #expect(try await post(client, headers: [.contentType: type]) == .ok, "\(type)")
            }
            let status = try await client.execute(uri: "/v1/models", method: .get) { $0.status }
            #expect(status == .ok, "GET carries no body and needs no Content-Type")
        }
    }

    @Test("curl-style clients (no Origin, JSON, loopback Host) are unaffected")
    func curlStyle() async throws {
        try await app().test(.router) { (client) async throws in
            let headers: HTTPFields = [.contentType: "application/json", Self.host: "localhost:8080"]
            #expect(try await post(client, headers: headers) == .ok)
            #expect(try await post(client, "/v1/chat/completions", #"{"messages": [{"role": "user", "content": "hi"}]}"#, headers: headers) == .ok)
            #expect(try await post(client, "/v1/responses", #"{"input": "hi"}"#, headers: headers) == .ok)
        }
    }

    @Test("Off loopback: the bound host and loopback names pass; a wildcard bind accepts any Host (the key is mandatory there)")
    func offLoopbackHosts() async throws {
        let auth: HTTPFields = [.contentType: "application/json", .authorization: "Bearer k-1"]
        try await app(host: "192.168.1.5", apiKey: "k-1").test(.router) { (client) async throws in
            for (value, expected) in [("192.168.1.5:8080", HTTPResponse.Status.ok), ("localhost:8080", .ok), ("evil.example", .forbidden)] {
                var headers = auth
                headers[Self.host] = value
                #expect(try await post(client, headers: headers) == expected, "\(value)")
            }
        }
        try await app(host: "0.0.0.0", apiKey: "k-1").test(.router) { (client) async throws in
            var headers = auth
            headers[Self.host] = "studio.lan:8080"
            #expect(try await post(client, headers: headers) == .ok)
            #expect(try await post(client, headers: [.contentType: "application/json", .origin: "https://evil.example"]) == .forbidden,
                    "the guard answers before the authenticator")
            #expect(try await post(client, headers: [.contentType: "application/json"]) == .unauthorized)
        }
        #expect(RequestGuardMiddleware.hostAllowlist(bindHost: "::") == nil)
        #expect(RequestGuardMiddleware.hostAllowlist(bindHost: "fd00::5")?.contains("[fd00::5]") == true)
    }

    @Test("allowed_origins from the config and --allow-origin are merged; a malformed origin is a startup error")
    func allowedOriginsConfig() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("config.json")
        try Data(#"{"allowed_origins": ["https://a.example"]}"#.utf8).write(to: file)
        let network = try ServeCommand.resolveNetwork(cliPort: nil, cliHost: nil, cliApiKey: nil,
                                                      cliAllowedOrigins: ["http://b.example:3000"], configFile: file.path)
        #expect(Set(network.allowedOrigins) == ["http://b.example:3000", "https://a.example"])
        #expect(try ServeCommand.parse(["--allow-origin", "https://x.example", "--allow-origin", "http://y:1"]).allowOrigins.count == 2)
        for bad in ["a.example", "https://a.example/app", "ftp://a.example", "*"] {
            #expect(throws: (any Error).self, "\(bad)") {
                _ = try ServeCommand.resolveNetwork(cliPort: nil, cliHost: nil, cliApiKey: nil, cliAllowedOrigins: [bad], configFile: nil)
            }
        }
    }

    @Test("On port 80 a loopback Origin without a port is the server's own; 443 is not")
    func originDefaultPort() async throws {
        let guardOn80 = RequestGuardMiddleware(bindHost: "127.0.0.1", bindPort: 80)
        func allows(_ origin: String) -> Bool {
            guardOn80.allowsOrigin(Request(head: .init(method: .post, scheme: "http", authority: "localhost", path: "/v1/messages",
                                                       headerFields: [.origin: origin]),
                                           body: .init(buffer: ByteBuffer())))
        }
        #expect(allows("http://localhost"))
        #expect(allows("http://127.0.0.1:80"))
        #expect(!allows("https://localhost"))
        #expect(!allows("http://localhost:8080"))
    }

    @Test("A second Host header that is foreign, or two Origin headers, are refused with 403")
    func duplicateHeaders() async throws {
        try await app().test(.router) { (client) async throws in
            var hosts: HTTPFields = [.contentType: "application/json"]
            hosts.append(HTTPField(name: Self.host, value: "localhost:8080"))
            hosts.append(HTTPField(name: Self.host, value: "evil.example"))
            #expect(try await post(client, headers: hosts) == .forbidden)
            var origins: HTTPFields = [.contentType: "application/json"]
            origins.append(HTTPField(name: .origin, value: "http://localhost:8080"))
            origins.append(HTTPField(name: .origin, value: "http://localhost:8080"))
            #expect(try await post(client, headers: origins) == .forbidden)
        }
    }

    @Test("A request with no Host at all passes the Host check; an empty Host does not")
    func missingHost() {
        let guardMiddleware = RequestGuardMiddleware(bindHost: "127.0.0.1", bindPort: 8080)
        func request(authority: String?, headers: HTTPFields = [:]) -> Request {
            Request(head: .init(method: .get, scheme: "http", authority: authority, path: "/v1/models", headerFields: headers),
                    body: .init(buffer: ByteBuffer()))
        }
        #expect(guardMiddleware.allowsHost(request(authority: nil)), "HTTP/1.0 without Host: no browser sends this")
        #expect(!guardMiddleware.allowsHost(request(authority: "")))
        #expect(!guardMiddleware.allowsHost(request(authority: nil, headers: [Self.host: "evil.example"])))
    }

    @Test("OPTIONS preflights are never answered with Access-Control-Allow-Origin")
    func noCORSPreflight() async throws {
        try await app().test(.router) { (client) async throws in
            for origin in ["https://evil.example", "http://localhost:8080"] {
                let headers: HTTPFields = [.origin: origin, .accessControlRequestMethod: "POST",
                                           .accessControlRequestHeaders: "content-type"]
                for uri in ["/v1/messages", "/v1/models/load"] {
                    try await client.execute(uri: uri, method: .options, headers: headers) { res in
                        #expect(res.headers[.accessControlAllowOrigin] == nil, "\(uri) \(origin)")
                        #expect(res.status != .ok && res.status != .noContent, "\(uri) \(origin): \(res.status)")
                    }
                }
            }
        }
    }
}
