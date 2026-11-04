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

public struct HealthTelemetry: Codable, Sendable, Equatable {
    public private(set) var degenerateOutputs: [String: Int] = [:]
    public private(set) var drift: [String: [String: DriftWatch]] = [:]
    public var canary: CanaryTelemetry?

    public init() {}

    public mutating func record(_ response: GenerationResponse, model: String) -> [String] {
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
