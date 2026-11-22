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
        guard request.method == .post || request.method == .put,
              request.uri.path != TranscriptionsController.path,
              !RequestGuardMiddleware.has(.multipartForm, request) else {
            return try await next(request, context)
        }
        var request = request
        let buffer: ByteBuffer
        do {
            buffer = try await request.collectBody(upTo: Self.bodyLimit(for: request.uri.path))
        } catch let error as HTTPError where error.status == .contentTooLarge {
            return ProviderErrorResponse.response(status: .contentTooLarge, message: "Request body too large.", path: request.uri.path)
        }
        let bytes = buffer.readableBytesSpan
        guard !JSONNesting.containsZero(bytes) else { return Self.badRequest("the request body is not UTF-8 JSON", path: request.uri.path) }
        guard JSONNesting.depth(of: bytes) <= Self.depthLimit else {
            return Self.badRequest("the request body nests deeper than \(Self.depthLimit) JSON levels", path: request.uri.path)
        }
        return try await next(request, context)
    }

    private static let mediaBodyPaths: Set<String> = [
        "/v1/chat/completions", "/api/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/ocr"
    ]

    static func bodyLimit(for path: String) -> Int {
        if path.hasPrefix("/v1/images/") { return ImagesController.bodyLimit }
        if path.hasPrefix("/v1/pipeline") || path == "/v1/systemone" { return PipelineController.bodyLimit }
        if mediaBodyPaths.contains(path) { return Self.bodyLimit }
        return 4 * 1024 * 1024
    }

    static func badRequest(_ message: String, path: String = "/v1/chat/completions") -> Response {
        ProviderErrorResponse.response(status: .badRequest, message: message, path: path)
    }
}
