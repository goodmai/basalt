import Foundation
import Hummingbird
import NIOCore

/// Простая реализация In-Memory Rate Limiter для защиты от CWE-400 (Denial of Service).
public actor RateLimiter {
    private var requests: [String: [Date]] = [:]
    public let maxRequests: Int
    public let window: TimeInterval
    
    public init(maxRequests: Int = 50, window: TimeInterval = 60) {
        self.maxRequests = maxRequests
        self.window = window
    }
    
    public func checkLimit(clientId: String) throws {
        let now = Date()
        let windowStart = now.addingTimeInterval(-window)
        
        var clientRequests = requests[clientId] ?? []
        clientRequests = clientRequests.filter { $0 > windowStart }
        
        guard clientRequests.count < maxRequests else {
            throw GemmaServerError.rateLimitExceeded(retryAfter: nil)
        }
        
        clientRequests.append(now)
        requests[clientId] = clientRequests
    }
}

public struct RateLimitMiddleware<Context: RequestContext>: RouterMiddleware, Sendable {
    let rateLimiter: RateLimiter
    
    public init(rateLimiter: RateLimiter) {
        self.rateLimiter = rateLimiter
    }
    
    public func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        // Идентификация клиента по IP адресу. Если за балансировщиком, нужно парсить X-Forwarded-For
        // Hummingbird 2 предоставляет доступ к каналу через контекст или request
        let ip = "unknown" // request.channel?.remoteAddress?.description ?? "unknown" 
        // В Hummingbird 2 remote address может быть сложнее получить, 
        // поэтому мы можем использовать JWT 'sub' если доступно, или IP как fallback.
        
        // Попытка получить subject из контекста, если запрос аутентифицирован
        var clientId = ip
        if let gemmaContext = context as? GemmaRequestContext, 
           let subject = try? gemmaContext.requireIdentity() {
            clientId = subject.username
        }
        
        do {
            try await rateLimiter.checkLimit(clientId: clientId)
        } catch {
            return Response(
                status: .tooManyRequests,
                headers: [.contentType: "application/json"],
                body: ResponseBody(byteBuffer: ByteBuffer(string: "{\"error\": \"Rate limit exceeded\", \"code\": 429}"))
            )
        }
        
        return try await next(request, context)
    }
}
