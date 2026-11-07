import Foundation
import Hummingbird
import BasaltCore

public struct JSONNestingLimitMiddleware<Context: RequestContext>: RouterMiddleware {
    public static var depthLimit: Int { 128 }
    static var bodyLimit: Int { ChatPipeline.bodyLimit }

    let zeroCopyBody: Bool

    public init(zeroCopyBody: Bool = false) {
        self.zeroCopyBody = zeroCopyBody
    }

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        guard request.method == .post || request.method == .put, request.uri.path != TranscriptionsController.path else {
            return try await next(request, context)
        }
        var request = request
        let bytes = ChatPipeline.data(of: try await request.collectBody(upTo: Self.bodyLimit), zeroCopy: zeroCopyBody)
        guard !JSONNesting.containsZero(bytes) else { return Self.badRequest("the request body is not UTF-8 JSON", path: request.uri.path) }
        guard JSONNesting.depth(of: bytes) <= Self.depthLimit else {
            return Self.badRequest("the request body nests deeper than \(Self.depthLimit) JSON levels", path: request.uri.path)
        }
        return try await next(request, context)
    }

    static func badRequest(_ message: String, path: String = "/v1/chat/completions") -> Response {
        ProviderErrorResponse.response(status: .badRequest, message: message, path: path)
    }
}
