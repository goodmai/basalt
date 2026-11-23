import Foundation

public enum ComposeScalar: Sendable, Equatable, Codable, CustomStringConvertible {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let flag = try? container.decode(Bool.self) {
            self = .bool(flag)
        } else if let whole = try? container.decode(Int.self) {
            self = .int(whole)
        } else if let real = try? container.decode(Double.self) {
            self = .double(real)
        } else {
            self = .string(try container.decode(String.self))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        }
    }

    public var description: String {
        switch self {
        case .string(let value): value
        case .int(let value): String(value)
        case .double(let value): String(value)
        case .bool(let value): String(value)
        }
    }

    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    public var intValue: Int? {
        switch self {
        case .int(let value): return value
        case .string(let text): return Int(text)
        default: return nil
        }
    }

    public var doubleValue: Double? {
        switch self {
        case .double(let value): return value
        case .int(let value): return Double(value)
        case .string(let text): return Double(text)
        case .bool: return nil
        }
    }

    public var boolValue: Bool? {
        switch self {
        case .bool(let value): return value
        case .string("true"): return true
        case .string("false"): return false
        default: return nil
        }
    }
}

extension ComposeScalar: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral, ExpressibleByBooleanLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int) { self = .int(value) }
    public init(floatLiteral value: Double) { self = .double(value) }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
}

public enum BenchmarkAdapterID {
    public static let tierANative: Set<String> = [
        "basalt:throughput", "basalt:perplexity", "basalt:system-one",
        "basalt:passkey", "basalt:mom", "basalt:harness",
    ]

    public static let pureSwift: Set<String> = [
        "arc-agi", "arc-agi-2", "basalt:translation",
    ]

    public static let externalHarnesses: Set<String> = [
        "terminal-bench", "swe-bench", "lm-eval", "livecodebench",
        "bfcl", "inspect-ai", "tau-bench", "gaia", "evalplus",
    ]

    public static let known: Set<String> = tierANative.union(pureSwift).union(externalHarnesses)
}

public struct BenchmarkDefaultsSpec: Sendable, Equatable, Codable {
    public var seed: UInt64?
    public var temperature: Double?
    public var gpuLock: Bool?
    public var outputDir: String?

    public init(seed: UInt64? = nil, temperature: Double? = nil, gpuLock: Bool? = nil, outputDir: String? = nil) {
        (self.seed, self.temperature, self.gpuLock, self.outputDir) = (seed, temperature, gpuLock, outputDir)
    }

    enum CodingKeys: String, CodingKey {
        case seed, temperature
        case gpuLock = "gpu_lock"
        case outputDir = "output_dir"
    }
}

public struct BenchmarkGatesSpec: Sendable, Equatable, Codable {
    public var minDecodeTPS: Double?
    public var maxPeakGB: Double?
    public var minAccuracy: Double?
    public var maxP50Ms: Double?
    public var maxPPL: Double?
    public var maxECE: Double?

    public init(
        minDecodeTPS: Double? = nil, maxPeakGB: Double? = nil, minAccuracy: Double? = nil,
        maxP50Ms: Double? = nil, maxPPL: Double? = nil, maxECE: Double? = nil
    ) {
        (self.minDecodeTPS, self.maxPeakGB, self.minAccuracy) = (minDecodeTPS, maxPeakGB, minAccuracy)
        (self.maxP50Ms, self.maxPPL, self.maxECE) = (maxP50Ms, maxPPL, maxECE)
    }

    enum CodingKeys: String, CodingKey {
        case minDecodeTPS = "min_decode_tps"
        case maxPeakGB = "max_peak_gb"
        case minAccuracy = "min_accuracy"
        case maxP50Ms = "max_p50_ms"
        case maxPPL = "max_ppl"
        case maxECE = "max_ece"
    }
}

public struct BenchmarkSubmitSpec: Sendable, Equatable, Codable {
    public var target: SubmissionTarget
    public var apiKeyEnv: String?
    public var endpoint: String?
    public var subset: String?
    public var split: String?

    public init(
        target: SubmissionTarget, apiKeyEnv: String? = nil, endpoint: String? = nil,
        subset: String? = nil, split: String? = nil
    ) {
        (self.target, self.apiKeyEnv, self.endpoint, self.subset, self.split) = (target, apiKeyEnv, endpoint, subset, split)
    }

    enum CodingKeys: String, CodingKey {
        case target, endpoint, subset, split
        case apiKeyEnv = "api_key_env"
    }
}

public struct BenchmarkTaskSpec: Sendable, Equatable, Codable {
    public var adapter: String
    public var dataset: String?
    public var target: String
    public var params: [String: ComposeScalar]
    public var gates: BenchmarkGatesSpec?
    public var submit: BenchmarkSubmitSpec?

    public init(
        adapter: String, dataset: String? = nil, target: String = "pipeline",
        params: [String: ComposeScalar] = [:], gates: BenchmarkGatesSpec? = nil, submit: BenchmarkSubmitSpec? = nil
    ) {
        (self.adapter, self.dataset, self.target) = (adapter, dataset, target)
        (self.params, self.gates, self.submit) = (params, gates, submit)
    }
}

public struct BenchmarkBudgetSpec: Sendable, Equatable, Codable {
    public var usdMax: Double?
    public var maxMinutes: Double?

    public init(usdMax: Double? = nil, maxMinutes: Double? = nil) {
        (self.usdMax, self.maxMinutes) = (usdMax, maxMinutes)
    }

    enum CodingKeys: String, CodingKey {
        case usdMax = "usd_max"
        case maxMinutes = "max_minutes"
    }
}

public struct BenchmarkSuiteSpec: Sendable, Equatable, Codable {
    public var profiles: [String]
    public var budget: BenchmarkBudgetSpec?
    public var tasks: [BenchmarkTaskSpec]

    public init(profiles: [String] = [], budget: BenchmarkBudgetSpec? = nil, tasks: [BenchmarkTaskSpec] = []) {
        (self.profiles, self.budget, self.tasks) = (profiles, budget, tasks)
    }

    enum CodingKeys: String, CodingKey {
        case profiles, budget, tasks
    }

    public init(from decoder: any Decoder) throws {
        if let single = try? decoder.singleValueContainer(), let list = try? single.decode([BenchmarkTaskSpec].self) {
            (self.profiles, self.budget, self.tasks) = ([], nil, list)
            return
        }
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.profiles = try container.decodeIfPresent([String].self, forKey: .profiles) ?? []
        self.budget = try container.decodeIfPresent(BenchmarkBudgetSpec.self, forKey: .budget)
        self.tasks = try container.decode([BenchmarkTaskSpec].self, forKey: .tasks)
    }
}

public struct ComposeBenchmarksSpec: Sendable, Equatable, Codable {
    public var defaults: BenchmarkDefaultsSpec?
    public var suites: [String: BenchmarkSuiteSpec]

    public init(defaults: BenchmarkDefaultsSpec? = nil, suites: [String: BenchmarkSuiteSpec] = [:]) {
        (self.defaults, self.suites) = (defaults, suites)
    }
}
