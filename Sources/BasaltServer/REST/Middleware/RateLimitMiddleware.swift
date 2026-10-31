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

final class RateLimiterState: Sendable {
    struct Entry: Sendable {
        var bucket: TokenBucket
        var lastAccess: Double
    }
    let mutex: Mutex<[String: Entry]>
    let capacity: Double
    let refillPerSecond: Double
    let maxEntries: Int

    init(capacity: Double, refillPerSecond: Double, maxEntries: Int = 4096) {
        self.capacity = capacity
        self.refillPerSecond = refillPerSecond
        self.maxEntries = maxEntries
        self.mutex = Mutex([:])
    }

    func take(clientKey: String, now: Double) -> Double? {
        mutex.withLock { entries in
            if entries[clientKey] == nil && entries.count >= maxEntries {
                entries = entries.filter { _, entry in
                    now - entry.lastAccess < 60.0
                }
                if entries.count >= maxEntries {
                    let sortedKeys = entries.sorted { $0.value.lastAccess < $1.value.lastAccess }
                    let dropCount = max(1, entries.count - (maxEntries * 3 / 4))
                    for i in 0..<dropCount {
                        entries.removeValue(forKey: sortedKeys[i].key)
                    }
                }
            }

            var entry = entries[clientKey] ?? Entry(
                bucket: TokenBucket(capacity: capacity, refillPerSecond: refillPerSecond, now: now),
                lastAccess: now
            )
            entry.lastAccess = now
            let wait = entry.bucket.take(now: now)
            entries[clientKey] = entry
            return wait
        }
    }

    var clientCount: Int {
        mutex.withLock { $0.count }
    }
}

/// Rejects requests with 429 once the server-wide budget of `requestsPerMinute` is spent.
/// Mounted after the authenticator, so unauthenticated traffic is refused with 401 before
/// it can drain the budget of the key holder.
public struct RateLimitMiddleware<Context: RequestContext>: RouterMiddleware {
    private let limiter: RateLimiterState
    private let clock: @Sendable () -> Double

    public init(
        requestsPerMinute: Int,
        burst: Int? = nil,
        maxClients: Int = 4096,
        clock: @escaping @Sendable () -> Double = {
            ProcessInfo.processInfo.systemUptime
        }
    ) {
        let perMinute = Double(max(1, requestsPerMinute))
        let burstCapacity = Double(max(1, burst ?? max(1, min(requestsPerMinute, 10))))
        self.clock = clock
        self.limiter = RateLimiterState(
            capacity: burstCapacity,
            refillPerSecond: perMinute / 60,
            maxEntries: maxClients
        )
    }

    package var clientCount: Int {
        limiter.clientCount
    }

    private func clientKey(from request: Request) -> String {
        if let budgetKey = request.headers[APIKeyAuthenticator<Context>.budgetKeyHeader] {
            return budgetKey
        }
        if let authHeader = request.headers[.authorization] {
            let parts = authHeader.split(separator: " ", maxSplits: 1)
            if parts.count == 2 && parts[0].lowercased() == "bearer" {
                return String(parts[1])
            }
        }
        if let customKey = request.headers[values: HTTPField.Name("x-api-key")!].first {
            return customKey
        }
        if let forwarded = request.headers[values: HTTPField.Name("x-forwarded-for")!].first {
            return forwarded.split(separator: ",").first?.trimmingCharacters(in: .whitespaces) ?? forwarded
        }
        return "127.0.0.1"
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
        let key = clientKey(from: request)
        guard let wait = limiter.take(clientKey: key, now: now) else {
            return try await next(request, context)
        }
        return ProviderErrorResponse.response(
            status: .tooManyRequests,
            message: "Rate limit exceeded.",
            path: request.uri.path,
            retryAfter: String(max(1, Int(wait.rounded(.up))))
        )
    }
}
