import Foundation
import Hummingbird
import HTTPTypes
import Synchronization

/// Token bucket: `capacity` tokens, refilled continuously at `refillPerSecond`.
/// A value type with an explicit clock so the arithmetic is testable without sleeping.
struct TokenBucket: Sendable {
    let capacity: Double
    let refillPerSecond: Double
    private(set) var tokens: Double
    private var last: Double

    init(capacity: Double, refillPerSecond: Double, now: Double) {
        self.capacity = capacity
        self.refillPerSecond = refillPerSecond
        self.tokens = capacity
        self.last = now
    }

    /// Takes one token. Returns nil on success, or the seconds until one token is available.
    mutating func take(now: Double) -> Double? {
        tokens = min(capacity, tokens + max(0, now - last) * refillPerSecond)
        last = max(last, now)
        if tokens >= 1 {
            tokens -= 1
            return nil
        }
        return (1 - tokens) / refillPerSecond
    }
}

/// Rejects requests with 429 once the server-wide budget of `requestsPerMinute` is spent.
/// Mounted after the authenticator, so unauthenticated traffic is refused with 401 before
/// it can drain the budget of the key holder.
// ponytail: one bucket for the whole server — it guards the single GPU, not fairness between
// clients; per-key or per-address buckets if several clients ever share one server.
public struct RateLimitMiddleware<Context: RequestContext>: RouterMiddleware {
    // Mutex is ~Copyable; a class reference keeps the middleware copyable and the bucket shared.
    private final class SharedBucket: Sendable {
        let mutex: Mutex<TokenBucket>
        init(_ bucket: TokenBucket) { mutex = Mutex(bucket) }
    }
    private let bucket: SharedBucket
    private let clock: @Sendable () -> Double

    public init(requestsPerMinute: Int, clock: @escaping @Sendable () -> Double = {
        ProcessInfo.processInfo.systemUptime
    }) {
        let perMinute = Double(max(1, requestsPerMinute))
        self.clock = clock
        self.bucket = SharedBucket(TokenBucket(capacity: perMinute, refillPerSecond: perMinute / 60, now: clock()))
    }

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        if APIKeyAuthenticator<Context>.publicPaths.contains(request.uri.path) {
            return try await next(request, context)
        }
        let now = clock()
        guard let wait = bucket.mutex.withLock({ $0.take(now: now) }) else {
            return try await next(request, context)
        }
        var buffer = ByteBuffer()
        buffer.writeString("{\"error\": \"Rate limit exceeded.\", \"code\": 429}")
        return Response(
            status: .tooManyRequests,
            headers: [.contentType: "application/json", .retryAfter: String(Int(wait.rounded(.up)))],
            body: ResponseBody(byteBuffer: buffer)
        )
    }
}
