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
        guard JSONNesting.depth(of: Data(body.readableBytesView)) <= Self.depthLimit else {
            var buffer = ByteBuffer()
            buffer.writeString("{\"error\": {\"message\": \"the request body nests deeper than \(Self.depthLimit) JSON levels\", \"code\": 400}}")
            return Response(status: .badRequest, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
        }
        return try await next(request, context)
    }
}
