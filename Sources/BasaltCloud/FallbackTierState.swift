import Foundation
import BasaltCore

public struct FallbackTier: Sendable {
    public let id: String
    public let client: OpenRouterClient
    public let baseURL: URL
    public let model: String
    public let isFree: Bool

    public init(id: String? = nil, client: OpenRouterClient, baseURL: URL, model: String, isFree: Bool = false) {
        self.id = id ?? "\(baseURL.absoluteString)#\(model)"
        self.client = client
        self.baseURL = baseURL
        self.model = model
        self.isFree = isFree
    }
}

public struct FallbackTierProbeConfig: Sendable {
    public let timeout: Duration
    public let healthyTTL: Duration
    public let baseBackoffSeconds: Int
    public let maxBackoffSeconds: Int
    public let now: @Sendable () -> ContinuousClock.Instant

    public init(
        timeout: Duration = .milliseconds(200),
        healthyTTL: Duration = .seconds(30),
        baseBackoffSeconds: Int = 1,
        maxBackoffSeconds: Int = 60,
        now: @escaping @Sendable () -> ContinuousClock.Instant = { .now }
    ) {
        self.timeout = timeout
        self.healthyTTL = healthyTTL
        self.baseBackoffSeconds = max(1, baseBackoffSeconds)
        self.maxBackoffSeconds = max(1, maxBackoffSeconds)
        self.now = now
    }
}

enum FallbackTierVerdict: Sendable, Equatable {
    case healthy
    case unhealthy
    case needsProbe
}

enum FallbackTierEvent: Sendable {
    case probeStarted(String)
    case probed(id: String, generation: Int, healthy: Bool, at: ContinuousClock.Instant)
    case failed(id: String, at: ContinuousClock.Instant)
    case sessionBound(sessionId: String?, tierId: String)
}

struct FallbackTierState: Sendable {
    struct HealthEntry: Sendable, Equatable {
        let healthy: Bool
        let failures: Int
        let validUntil: ContinuousClock.Instant
    }

    static let maxTrackedSessions = 4_096

    let tiers: [FallbackTier]
    let probeConfig: FallbackTierProbeConfig
    private(set) var probesStarted: [String: Int] = [:]
    private(set) var probesApplied: [String: Int] = [:]
    private(set) var health: [String: HealthEntry] = [:]
    private(set) var sessionTiers: [String: String] = [:]

    init(tiers: [FallbackTier] = [], probeConfig: FallbackTierProbeConfig = FallbackTierProbeConfig()) {
        self.tiers = tiers
        self.probeConfig = probeConfig
    }

    var isMultiTier: Bool { !tiers.isEmpty }

    func verdict(for id: String, at instant: ContinuousClock.Instant) -> FallbackTierVerdict {
        guard let entry = health[id], instant < entry.validUntil else { return .needsProbe }
        return entry.healthy ? .healthy : .unhealthy
    }

    func orderedTiers(sessionId: String?) -> [FallbackTier] {
        guard let sessionId, let stickyId = sessionTiers[sessionId],
              let index = tiers.firstIndex(where: { $0.id == stickyId }), index > 0
        else { return tiers }
        var ordered = [tiers[index]]
        ordered.reserveCapacity(tiers.count)
        for (offset, tier) in tiers.enumerated() where offset != index {
            ordered.append(tier)
        }
        return ordered
    }

    static func reduce(_ state: FallbackTierState, _ event: FallbackTierEvent) -> FallbackTierState {
        var next = state
        switch event {
        case .probeStarted(let id):
            next.probesStarted[id, default: 0] += 1
        case .probed(let id, let generation, let healthy, let instant):
            guard generation > (state.probesApplied[id] ?? 0) else { return state }
            next.probesApplied[id] = generation
            next.health[id] = next.entry(for: id, healthy: healthy, at: instant)
        case .failed(let id, let instant):
            let generation = (state.probesStarted[id] ?? 0) + 1
            next.probesStarted[id] = generation
            next.probesApplied[id] = generation
            next.health[id] = next.entry(for: id, healthy: false, at: instant)
        case .sessionBound(let sessionId, let tierId):
            guard let sessionId, !sessionId.isEmpty else { return state }
            if next.sessionTiers[sessionId] == nil, next.sessionTiers.count >= Self.maxTrackedSessions {
                next.sessionTiers.removeAll()
            }
            next.sessionTiers[sessionId] = tierId
        }
        return next
    }

    private func entry(for id: String, healthy: Bool, at instant: ContinuousClock.Instant) -> HealthEntry {
        if healthy {
            return HealthEntry(healthy: true, failures: 0, validUntil: instant.advanced(by: probeConfig.healthyTTL))
        }
        let failures = (health[id]?.failures ?? 0) + 1
        let seconds = min(probeConfig.maxBackoffSeconds, probeConfig.baseBackoffSeconds * (1 << min(failures - 1, 6)))
        return HealthEntry(healthy: false, failures: failures, validUntil: instant.advanced(by: .seconds(seconds)))
    }
}

extension ModelRouter.Options {
    public func withFallbackTiers(
        _ tiers: [FallbackTier],
        probe: FallbackTierProbeConfig = FallbackTierProbeConfig()
    ) -> ModelRouter.Options {
        var copy = self
        copy.fallbackTiers = tiers
        copy.probeConfig = probe
        return copy
    }
}
