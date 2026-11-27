import Foundation
import Hummingbird
import HTTPTypes
import CryptoKit
import BasaltCore

public struct APIKeyAuthenticator<Context: RequestContext>: RouterMiddleware {
    public let apiKey: String
    public let virtualKeys: [VirtualKey]
    static var budgetKeyHeader: HTTPField.Name { HTTPField.Name("x-basalt-budget-key")! }
    static var rateKeyHeader: HTTPField.Name { HTTPField.Name("x-basalt-rate-key")! }
    private let virtualKeyDigests: [(digest: SHA256.Digest, id: String)]

    static var publicPaths: Set<String> {
        ["/api/v1/health", "/v1/health/live", "/v1/health/ready", "/swagger", "/api/v1/openapi.json", "/"]
    }
    private let apiKeyDigest: SHA256.Digest

    public init(apiKey: String, virtualKeys: [VirtualKey] = []) {
        self.apiKey = apiKey
        self.virtualKeys = virtualKeys
        self.apiKeyDigest = SHA256.hash(data: Data(apiKey.utf8))
        self.virtualKeyDigests = virtualKeys.map { (SHA256.hash(data: Data($0.key.utf8)), $0.id) }
    }

    static func normalizedPath(_ path: String) -> String {
        let joined = path.split(separator: "/").joined(separator: "/")
        return joined.isEmpty ? "/" : "/\(joined)"
    }

    private static var adminExactPaths: Set<String> {
        [
            "/v1/models/load", "/v1/models/download", "/v1/models/publish",
            "/v1/compose", "/v1/mcp/approvals",
            "/v1/xray/depth", "/v1/xray/outliers", "/v1/xray/causal",
            "/api/v1/xray/depth", "/api/v1/xray/outliers", "/api/v1/xray/causal",
        ]
    }

    private static var adminPrefixes: [String] {
        ["/v1/models/convert", "/api/v1/models/convert", "/v1/compose/", "/v1/mcp/approvals/"]
    }

    static func isAdminRequest(_ request: Request) -> Bool {
        let path = normalizedPath(request.uri.path)
        if request.method == .delete && (path == "/v1/models" || path.hasPrefix("/v1/models/")) { return true }
        if adminExactPaths.contains(path) { return true }
        return adminPrefixes.contains { path.hasPrefix($0) }
    }

    static func request(_ request: Request, budgetKey: String?, rateKey: String? = nil) -> Request {
        guard budgetKey != nil || rateKey != nil || request.headers[budgetKeyHeader] != nil || request.headers[rateKeyHeader] != nil else {
            return request
        }
        var head = request.head
        head.headerFields[budgetKeyHeader] = budgetKey
        head.headerFields[rateKeyHeader] = rateKey
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
        if Self.publicPaths.contains(Self.normalizedPath(request.uri.path)) {
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
                return try await next(Self.request(request, budgetKey: nil, rateKey: "master"), context)
            }
            if let id = virtualKeyId(for: candidate) {
                if Self.isAdminRequest(request) {
                    return ProviderErrorResponse.response(
                        status: .forbidden,
                        message: "Forbidden: virtual keys cannot access admin or control routes.",
                        path: request.uri.path
                    )
                }
                return try await next(Self.request(request, budgetKey: id, rateKey: "vk:\(id)"), context)
            }
        }

        return ProviderErrorResponse.response(
            status: .unauthorized,
            message: "Unauthorized: missing or invalid API key.",
            path: request.uri.path
        )
    }
}
