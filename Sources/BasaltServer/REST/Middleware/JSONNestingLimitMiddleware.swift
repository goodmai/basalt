import Foundation
import Hummingbird
import BasaltCore

public struct JSONNestingLimitMiddleware<Context: RequestContext>: RouterMiddleware {
    public static var depthLimit: Int { 128 }
    static var bodyLimit: Int { 4 * 1024 * 1024 }

    public init() {}

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        guard request.method == .post || request.method == .put else { return try await next(request, context) }
        var request = request
        let body = try await request.collectBody(upTo: Self.bodyLimit)
        let bytes = Data(body.readableBytesView)
        guard !bytes.contains(0) else { return Self.badRequest("the request body is not UTF-8 JSON") }
        guard JSONNesting.depth(of: bytes) <= Self.depthLimit else {
            return Self.badRequest("the request body nests deeper than \(Self.depthLimit) JSON levels")
        }
        return try await next(request, context)
    }

    static func badRequest(_ message: String) -> Response {
        var buffer = ByteBuffer()
        buffer.writeString("{\"error\": {\"message\": \"\(message)\", \"code\": 400}}")
        return Response(status: .badRequest, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
    }
}
