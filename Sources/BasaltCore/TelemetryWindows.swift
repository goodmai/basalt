import Foundation

public struct TelemetrySample: Sendable, Equatable {
    public var at: Date
    public var ttft: Double
    public var tokensPerSecond: Double
    public var completionTokens: Int
    public var promptTokens: Int
    public var cachedPromptTokens = 0
    public var draftTokens = 0
    public var acceptedDraftTokens = 0

    public init(at: Date, ttft: Double, tokensPerSecond: Double, completionTokens: Int, promptTokens: Int) {
        (self.at, self.ttft, self.tokensPerSecond, self.completionTokens, self.promptTokens) = (at, ttft, tokensPerSecond, completionTokens, promptTokens)
    }
}

public struct TelemetryWindow: Codable, Sendable, Equatable {
    public var seconds: Int
    public var requests: Int
    public var tokens: Int
    public var ttftP50: Double?
    public var ttftP95: Double?
    public var tokensPerSecondP50: Double?
    public var tokensPerSecondP95: Double?
    public var promptCacheHitRatio: Double?
    public var draftAcceptance: Double?

    enum CodingKeys: String, CodingKey {
        case seconds, requests, tokens
        case ttftP50 = "ttft_p50", ttftP95 = "ttft_p95", tokensPerSecondP50 = "tokens_per_second_p50", tokensPerSecondP95 = "tokens_per_second_p95"
        case promptCacheHitRatio = "prompt_cache_hit_ratio", draftAcceptance = "draft_acceptance"
    }
}

public struct TelemetryWindows: Sendable, Equatable {
    public static let spans = [60, 300, 3600]
    public static let capacity = 20_000
    public private(set) var samples: [TelemetrySample] = []

    public init() {}

    public mutating func record(_ sample: TelemetrySample) {
        samples.append(sample)
        let horizon = sample.at.addingTimeInterval(-Double(Self.spans.max() ?? 3600))
        if let keep = samples.firstIndex(where: { $0.at >= horizon }), keep > 0 { samples.removeFirst(keep) }
        if samples.count > Self.capacity { samples.removeFirst(samples.count - Self.capacity) }
    }

    public func window(seconds: Int, now: Date) -> TelemetryWindow {
        let since = now.addingTimeInterval(-Double(seconds))
        let recent = samples.filter { $0.at >= since && $0.at <= now }
        let prompt = recent.reduce(0) { $0 + $1.promptTokens }
        let drafts = recent.reduce(0) { $0 + $1.draftTokens }
        return TelemetryWindow(
            seconds: seconds, requests: recent.count, tokens: recent.reduce(0) { $0 + $1.completionTokens },
            ttftP50: Self.percentile(recent.map(\.ttft), 0.5), ttftP95: Self.percentile(recent.map(\.ttft), 0.95),
            tokensPerSecondP50: Self.percentile(recent.map(\.tokensPerSecond), 0.5), tokensPerSecondP95: Self.percentile(recent.map(\.tokensPerSecond), 0.95),
            promptCacheHitRatio: prompt > 0 ? Double(recent.reduce(0) { $0 + $1.cachedPromptTokens }) / Double(prompt) : nil,
            draftAcceptance: drafts > 0 ? Double(recent.reduce(0) { $0 + $1.acceptedDraftTokens }) / Double(drafts) : nil)
    }

    public func windows(now: Date) -> [TelemetryWindow] { Self.spans.map { window(seconds: $0, now: now) } }

    public static func percentile(_ values: [Double], _ quantile: Double) -> Double? {
        let sorted = values.filter(\.isFinite).sorted()
        guard !sorted.isEmpty else { return nil }
        let rank = Int((quantile * Double(sorted.count - 1)).rounded(.up))
        return sorted[min(max(rank, 0), sorted.count - 1)]
    }
}

public struct DecisionTelemetry: Codable, Sendable, Equatable {
    public static let bins = 10
    public var requests = 0
    public var answers = 0
    public var abstains = 0
    public var escalations = 0
    public var cost = 0.0
    public var latencyMs = 0.0
    public var confidence = [Int](repeating: 0, count: DecisionTelemetry.bins)

    public init() {}

    public mutating func record(_ response: SystemOneResponse, latencyMs: Double) {
        requests += 1
        self.latencyMs += latencyMs
        cost += response.usage.cost ?? 0
        if response.escalate { escalations += 1 }
        for answer in response.answers.values {
            answer.abstain == true ? (abstains += 1) : (answers += 1)
            if let value = answer.confidence, value.isFinite { confidence[min(Self.bins - 1, max(0, Int(value * Double(Self.bins))))] += 1 }
        }
    }

    enum CodingKeys: String, CodingKey {
        case requests, answers, abstains, escalations, cost, confidence
        case latencyMs = "latency_ms"
    }
}
