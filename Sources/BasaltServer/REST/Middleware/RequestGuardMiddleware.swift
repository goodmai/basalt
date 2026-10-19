import Foundation
import Hummingbird
import HTTPTypes

public struct RequestGuardMiddleware: RouterMiddleware {
    public typealias Context = BasicRequestContext

    static let loopbackHosts: Set<String> = ["localhost", "127.0.0.1", "[::1]"]
    static let wildcardBinds: Set<String> = ["0.0.0.0", "::", "[::]"]
    private static let hostField = HTTPField.Name("Host")!

    let allowedHosts: Set<String>?
    let allowedOrigins: Set<String>
    let bindPort: Int

    public init(bindHost: String, bindPort: Int, allowedOrigins: [String] = []) {
        self.allowedHosts = Self.hostAllowlist(bindHost: bindHost)
        self.bindPort = bindPort
        self.allowedOrigins = Set(allowedOrigins.compactMap { Self.parseOrigin($0)?.origin })
    }

    static func hostAllowlist(bindHost: String) -> Set<String>? {
        let host = bindHost.trimmingCharacters(in: .whitespaces).lowercased()
        if wildcardBinds.contains(host) { return nil }
        return loopbackHosts.union([bracketed(host)])
    }

    private static func bracketed(_ host: String) -> String {
        host.contains(":") && !host.hasPrefix("[") ? "[\(host)]" : host
    }

    static func hostName(ofAuthority authority: String) -> String {
        let lower = authority.lowercased()
        if lower.hasPrefix("["), let end = lower.firstIndex(of: "]") {
            return String(lower[...end])
        }
        return String(lower.prefix { $0 != ":" })
    }

    package static func parseOrigin(_ raw: String) -> (origin: String, host: String, port: Int)? {
        guard !raw.contains("*"), let components = URLComponents(string: raw.trimmingCharacters(in: .whitespaces)),
              let scheme = components.scheme?.lowercased(), scheme == "http" || scheme == "https",
              let rawHost = components.host, !rawHost.isEmpty,
              components.path.isEmpty, components.query == nil, components.fragment == nil,
              components.user == nil, components.password == nil
        else { return nil }
        let host = bracketed(rawHost.lowercased())
        let port = components.port ?? (scheme == "https" ? 443 : 80)
        return ("\(scheme)://\(host):\(port)", host, port)
    }

    func allowsHost(_ request: Request) -> Bool {
        guard let allowedHosts else { return true }
        let authorities = [request.head.authority].compactMap { $0 } + request.headers[values: Self.hostField]
        return authorities.allSatisfy { allowedHosts.contains(Self.hostName(ofAuthority: $0)) }
    }

    func allowsOrigin(_ request: Request) -> Bool {
        let values = request.headers[values: .origin]
        guard !values.isEmpty else { return true }
        guard values.count == 1, let parsed = Self.parseOrigin(values[0]) else { return false }
        return (Self.loopbackHosts.contains(parsed.host) && parsed.port == bindPort) || allowedOrigins.contains(parsed.origin)
    }

    static func isJSON(_ request: Request) -> Bool {
        guard let header = request.headers[.contentType], let type = MediaType(from: header) else { return false }
        return type.isType(.applicationJson)
    }

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        guard allowsHost(request) else {
            return Self.reject(.forbidden, "Forbidden: Host header is not allowed for this server.")
        }
        guard allowsOrigin(request) else {
            return Self.reject(.forbidden, "Forbidden: cross-origin requests are not allowed (see allowed_origins).")
        }
        if request.method == .post, !Self.isJSON(request) {
            return Self.reject(.unsupportedMediaType, "Content-Type must be application/json.")
        }
        return try await next(request, context)
    }

    private static func reject(_ status: HTTPResponse.Status, _ message: String) -> Response {
        let body = (try? JSONSerialization.data(withJSONObject: ["error": message, "code": status.code])) ?? Data()
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(bytes: body))
        )
    }
}
