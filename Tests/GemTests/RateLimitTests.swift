import Testing
import Foundation
import Hummingbird
import HTTPTypes
import HummingbirdTesting
@testable import GemCore

@Suite("Rate limit (token bucket) and the REST perimeter")
struct RateLimitTests {

    @Test("A bucket spends its capacity, then refills at its rate")
    func testBucketArithmetic() throws {
        var bucket = TokenBucket(capacity: 2, refillPerSecond: 0.5, now: 0)
        #expect(bucket.take(now: 0) == nil)
        #expect(bucket.take(now: 0) == nil)
        #expect(bucket.take(now: 0) == 2, "empty: one token is 1 / 0.5 s away")
        #expect(bucket.take(now: 1) == 1, "half a token refilled")
        #expect(bucket.take(now: 2) == nil)
        #expect(bucket.take(now: 1000) == nil)
        #expect(bucket.tokens <= bucket.capacity - 1, "refill never exceeds capacity")
        // A clock that steps back neither mints tokens nor rewinds the refill point.
        #expect(bucket.take(now: 0) == nil)
        #expect(bucket.take(now: 0) != nil)
    }

    @Test("Off loopback the limit is on by default; 0 turns it off; loopback is opt-in")
    func testResolveRateLimit() throws {
        #expect(ServerConfig.resolveRateLimit(host: "0.0.0.0", requested: nil) == ServerConfig.defaultRateLimitPerMinute)
        #expect(ServerConfig.resolveRateLimit(host: "0.0.0.0", requested: 0) == nil)
        #expect(ServerConfig.resolveRateLimit(host: "0.0.0.0", requested: 30) == 30)
        #expect(ServerConfig.resolveRateLimit(host: "127.0.0.1", requested: nil) == nil)
        #expect(ServerConfig.resolveRateLimit(host: "localhost", requested: 10) == 10)
        #expect(try ServerConfig(modelPath: "p", host: "0.0.0.0", apiKey: "k").rateLimitPerMinute == 120)
    }

    private func router(rateLimit: Int) throws -> Router<BasicRequestContext> {
        let config = try ServerConfig(modelPath: "stub", host: "0.0.0.0", apiKey: "k-123", rateLimitPerMinute: rateLimit)
        return RESTServer(orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()), config: config).makeRouter()
    }

    @Test("The real router: 401 without the key, 429 past the limit with Retry-After, health exempt")
    func testRouterRateLimit() async throws {
        let app = Application(router: try router(rateLimit: 2))
        try await app.test(.router) { client in
            let auth: HTTPFields = [.authorization: "Bearer k-123"]
            // Unauthenticated requests are refused before the limiter and spend nothing.
            for _ in 0..<5 {
                let status = try await client.execute(uri: "/v1/models", method: .get) { $0.status }
                #expect(status == .unauthorized)
            }
            for _ in 0..<2 {
                let status = try await client.execute(uri: "/v1/models", method: .get, headers: auth) { $0.status }
                #expect(status == .ok)
            }
            let (status, retryAfter) = try await client.execute(uri: "/v1/models", method: .get, headers: auth) {
                ($0.status, $0.headers[.retryAfter])
            }
            #expect(status == .tooManyRequests)
            #expect(retryAfter.flatMap(Int.init).map { $0 >= 1 } == true)
            let health = try await client.execute(uri: "/api/v1/health", method: .get) { $0.status }
            #expect(health == .ok)
        }
    }

    @Test("Without a configured limit the router never answers 429")
    func testNoLimitNo429() async throws {
        let app = Application(router: try router(rateLimit: 0))
        try await app.test(.router) { client in
            for _ in 0..<200 {
                let status = try await client.execute(uri: "/v1/models", method: .get,
                                                      headers: [.authorization: "Bearer k-123"]) { $0.status }
                #expect(status == .ok)
            }
        }
    }

    @Test("No stub file whose whole content is a `// removed` comment is left in Sources")
    func testNoRemovedStubs() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources")
        let files = FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" } ?? []
        #expect(!files.isEmpty)
        let stubs = try files.filter { url in
            let code = try String(contentsOf: url, encoding: .utf8)
                .split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty && !$0.hasPrefix("//") }
            return code.isEmpty
        }
        #expect(stubs.isEmpty, "\(stubs.map(\.lastPathComponent))")
    }
}
