import Foundation

public enum DegenerateOutput: String, Codable, Sendable {
    case repeatedToken
    case emptyAfterThinking
    case blankAnswer
}

public enum DraftSource: String, Codable, Sendable {
    case mtp
    case lookup
}

enum DriftMetric: String, Sendable {
    case decodeTokensPerSecond
    case mtpAcceptance
    case lookupAcceptance

    static let minimumDecodeTokens = 32

    static func samples(_ response: GenerationResponse) -> [(DriftMetric, Double)] {
        guard response.finishReason != .error else { return [] }
        var samples: [(DriftMetric, Double)] = []
        if response.completionTokens >= minimumDecodeTokens, response.tokensPerSecond > 0 {
            samples.append((.decodeTokensPerSecond, response.tokensPerSecond))
        }
        if let source = response.draftSource, let drafted = response.draftTokens, drafted > 0 {
            let metric: DriftMetric = switch source {
            case .mtp: .mtpAcceptance
            case .lookup: .lookupAcceptance
            }
            samples.append((metric, Double(response.acceptedDraftTokens ?? 0) / Double(drafted)))
        }
        return samples
    }
}

public struct DriftWatch: Codable, Sendable, Equatable {
    public static let window = 8
    public static let warmup = 8
    public static let floor = 0.7
    public static let smoothing = 0.1

    public private(set) var baseline: Double?
    public private(set) var baselineSamples = 0
    public private(set) var recent: [Double] = []
    public private(set) var drifting = false
    public private(set) var alerts = 0

    public init() {}

    public var recentMean: Double { recent.isEmpty ? 0 : recent.reduce(0, +) / Double(recent.count) }

    public mutating func observe(_ value: Double) -> Bool {
        recent.append(value)
        if recent.count > Self.window {
            let old = recent.removeFirst()
            baseline = baseline.map { $0 + Self.smoothing * (old - $0) } ?? old
            baselineSamples += 1
        }
        let dropped = recent.count == Self.window && baselineSamples >= Self.warmup
            && recentMean < Self.floor * (baseline ?? 0)
        let wasDrifting = drifting
        drifting = dropped
        guard dropped, !wasDrifting else { return false }
        alerts += 1
        return true
    }
}

public struct CanaryTelemetry: Codable, Sendable, Equatable {
    public private(set) var passed = 0
    public private(set) var failed = 0
    public private(set) var lastPassed: Bool?
    public private(set) var lastRun: Date?

    public init() {}

    public mutating func record(passed ok: Bool, at date: Date) {
        if ok { passed += 1 } else { failed += 1 }
        lastPassed = ok
        lastRun = date
    }
}

public struct BelnapDistribution: Sendable, Codable, Equatable {
    public var neither: Int
    public var trueCount: Int
    public var falseCount: Int
    public var both: Int

    public init(neither: Int = 0, trueCount: Int = 0, falseCount: Int = 0, both: Int = 0) {
        self.neither = max(0, neither)
        self.trueCount = max(0, trueCount)
        self.falseCount = max(0, falseCount)
        self.both = max(0, both)
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        neither = max(0, try container.decodeIfPresent(Int.self, forKey: .neither) ?? 0)
        trueCount = max(0, try container.decodeIfPresent(Int.self, forKey: .trueCount) ?? 0)
        falseCount = max(0, try container.decodeIfPresent(Int.self, forKey: .falseCount) ?? 0)
        both = max(0, try container.decodeIfPresent(Int.self, forKey: .both) ?? 0)
    }

    public var total: Int { neither + trueCount + falseCount + both }
    public var isEmpty: Bool { total == 0 }

    public subscript(state: BelnapState) -> Int {
        switch state {
        case .neither: neither
        case .true: trueCount
        case .false: falseCount
        case .both: both
        }
    }

    public mutating func record(_ state: BelnapState) {
        switch state {
        case .neither: neither += 1
        case .true: trueCount += 1
        case .false: falseCount += 1
        case .both: both += 1
        }
    }

    enum CodingKeys: String, CodingKey {
        case neither
        case trueCount = "true"
        case falseCount = "false"
        case both
    }
}

public struct HealthTelemetry: Codable, Sendable, Equatable {
    public private(set) var degenerateOutputs: [String: Int] = [:]
    public private(set) var drift: [String: [String: DriftWatch]] = [:]
    public var canary: CanaryTelemetry?
    public private(set) var belnap = BelnapDistribution()

    public init() {}

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        degenerateOutputs = try container.decodeIfPresent([String: Int].self, forKey: .degenerateOutputs) ?? [:]
        drift = try container.decodeIfPresent([String: [String: DriftWatch]].self, forKey: .drift) ?? [:]
        canary = try container.decodeIfPresent(CanaryTelemetry.self, forKey: .canary)
        belnap = try container.decodeIfPresent(BelnapDistribution.self, forKey: .belnap) ?? BelnapDistribution()
    }

    enum CodingKeys: String, CodingKey {
        case degenerateOutputs, drift, canary, belnap
    }

    public mutating func record(_ response: GenerationResponse, model: String) -> [String] {
        if let confidence = response.confidence {
            belnap.record(confidence.belnapState())
        }
        var warnings: [String] = []
        if let kind = response.degenerate {
            degenerateOutputs[kind.rawValue, default: 0] += 1
            warnings.append("Degenerate output from \(model): \(kind.rawValue)")
        }
        for (metric, value) in DriftMetric.samples(response) {
            guard drift[model, default: [:]][metric.rawValue, default: DriftWatch()].observe(value),
                  let watch = drift[model]?[metric.rawValue], let baseline = watch.baseline else { continue }
            warnings.append("\(metric.rawValue) on \(model) dropped to \(String(format: "%.3g", watch.recentMean)) over the last "
                + "\(DriftWatch.window) requests, below \(Int(DriftWatch.floor * 100))% of its baseline \(String(format: "%.3g", baseline))")
        }
        return warnings
    }
}
