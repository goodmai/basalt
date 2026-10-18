import Foundation
import Hummingbird
import HTTPTypes

/// Middleware that enforces API key authentication for incoming REST requests
/// when an API key is configured on the server.
public struct APIKeyAuthenticator<Context: RequestContext>: RouterMiddleware {
    public let apiKey: String

    public init(apiKey: String) {
        self.apiKey = apiKey
    }

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let path = request.uri.path

        // Health check and API documentation endpoints remain publicly accessible.
        if path == "/api/v1/health" || path == "/swagger" || path == "/api/v1/openapi.json" || path == "/" {
            return try await next(request, context)
        }

        // 1. Check standard Authorization: Bearer <token>
        if let authHeader = request.headers[.authorization] {
            let parts = authHeader.split(separator: " ", maxSplits: 1)
            if parts.count == 2 && parts[0].lowercased() == "bearer" {
                let token = String(parts[1])
                if token == apiKey {
                    return try await next(request, context)
                }
            }
        }

        // 2. Check x-api-key header
        if let headerName = HTTPField.Name("x-api-key"),
           let customKey = request.headers[values: headerName].first,
           customKey == apiKey {
            return try await next(request, context)
        }

        // Unauthorized
        let json = "{\"error\": \"Unauthorized: missing or invalid API key.\", \"code\": 401}"
        var buffer = ByteBuffer()
        buffer.writeString(json)
        return Response(
            status: .unauthorized,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: buffer)
        )
    }
}
