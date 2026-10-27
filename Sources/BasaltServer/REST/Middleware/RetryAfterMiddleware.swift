import Foundation
import Hummingbird
import HTTPTypes

struct RetryAfterMiddleware<Context: RequestContext>: RouterMiddleware {
    static var defaultSeconds: String { "1" }

    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        var response = try await next(request, context)
        if response.status == .serviceUnavailable, response.headers[.retryAfter] == nil {
            response.headers[.retryAfter] = Self.defaultSeconds
        }
        return response
    }
}
