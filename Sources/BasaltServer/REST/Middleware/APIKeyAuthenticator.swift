import Foundation
import Hummingbird
import HTTPTypes
import CryptoKit
import BasaltCore

/// Middleware that enforces API key authentication for incoming REST requests
/// when an API key is configured on the server.
public struct APIKeyAuthenticator<Context: RequestContext>: RouterMiddleware {
    public let apiKey: String
    public let virtualKeys: [VirtualKey]
    static var budgetKeyHeader: HTTPField.Name { HTTPField.Name("x-basalt-budget-key")! }
    private let virtualKeyDigests: [(digest: SHA256.Digest, id: String)]

    /// Health check and API documentation stay reachable without a key and outside the rate limit.
    static var publicPaths: Set<String> { ["/api/v1/health", "/swagger", "/api/v1/openapi.json", "/"] }
    private let apiKeyDigest: SHA256.Digest

    public init(apiKey: String, virtualKeys: [VirtualKey] = []) {
        self.apiKey = apiKey
        self.virtualKeys = virtualKeys
        self.apiKeyDigest = SHA256.hash(data: Data(apiKey.utf8))
        self.virtualKeyDigests = virtualKeys.map { (SHA256.hash(data: Data($0.key.utf8)), $0.id) }
    }

    static func request(_ request: Request, budgetKey: String?) -> Request {
        guard budgetKey != nil || request.headers[budgetKeyHeader] != nil else { return request }
        var head = request.head
        head.headerFields[budgetKeyHeader] = budgetKey
        return Request(head: head, body: request.body)
    }

    func virtualKeyId(for candidate: String) -> String? {
        let digest = SHA256.hash(data: Data(candidate.utf8))
        return virtualKeyDigests.reduce(nil) { found, entry in
            let equal = zip(digest, entry.digest).reduce(UInt8(0)) { $0 | ($1.0 ^ $1.1) } == 0
            return found ?? (equal ? entry.id : nil)
        }
    }

    /// Compares SHA-256 digests with a full XOR fold, so the time spent does not depend
    /// on how many leading bytes of the candidate match the key.
    func matches(_ candidate: String) -> Bool {
        let digest = SHA256.hash(data: Data(candidate.utf8))
        return zip(digest, apiKeyDigest).reduce(UInt8(0)) { $0 | ($1.0 ^ $1.1) } == 0
    }

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let request = Self.request(request, budgetKey: nil)
        if Self.publicPaths.contains(request.uri.path) {
            return try await next(request, context)
        }

        var candidates: [String] = []
        if let authHeader = request.headers[.authorization] {
            let parts = authHeader.split(separator: " ", maxSplits: 1)
            if parts.count == 2 && parts[0].lowercased() == "bearer" {
                candidates.append(String(parts[1]))
            }
        }
        if let headerName = HTTPField.Name("x-api-key"), let customKey = request.headers[values: headerName].first {
            candidates.append(customKey)
        }
        for candidate in candidates where !candidate.isEmpty {
            if !apiKey.isEmpty, matches(candidate) {
                return try await next(request, context)
            }
            if let id = virtualKeyId(for: candidate) {
                return try await next(Self.request(request, budgetKey: id), context)
            }
        }

        // Unauthorized
        let json = "{\"error\": \"Unauthorized: missing or invalid API key.\", \"code\": 401}"
        var buffer = ByteBuffer()
        buffer.writeString(json)
        return Response(
            status: .unauthorized,
            headers: [.contentType: "application/json", .wwwAuthenticate: "Bearer"],
            body: ResponseBody(byteBuffer: buffer)
        )
    }
}
