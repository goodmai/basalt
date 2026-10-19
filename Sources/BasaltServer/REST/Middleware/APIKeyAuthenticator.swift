import Foundation
import Hummingbird
import HTTPTypes
import CryptoKit

/// Middleware that enforces API key authentication for incoming REST requests
/// when an API key is configured on the server.
public struct APIKeyAuthenticator<Context: RequestContext>: RouterMiddleware {
    public let apiKey: String

    /// Health check and API documentation stay reachable without a key and outside the rate limit.
    static var publicPaths: Set<String> { ["/api/v1/health", "/swagger", "/api/v1/openapi.json", "/"] }
    private let apiKeyDigest: SHA256.Digest

    public init(apiKey: String) {
        self.apiKey = apiKey
        self.apiKeyDigest = SHA256.hash(data: Data(apiKey.utf8))
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
        if Self.publicPaths.contains(request.uri.path) {
            return try await next(request, context)
        }

        // 1. Check standard Authorization: Bearer <token>
        if let authHeader = request.headers[.authorization] {
            let parts = authHeader.split(separator: " ", maxSplits: 1)
            if parts.count == 2 && parts[0].lowercased() == "bearer" {
                let token = String(parts[1])
                if matches(token) {
                    return try await next(request, context)
                }
            }
        }

        // 2. Check x-api-key header
        if let headerName = HTTPField.Name("x-api-key"),
           let customKey = request.headers[values: headerName].first,
           matches(customKey) {
            return try await next(request, context)
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
