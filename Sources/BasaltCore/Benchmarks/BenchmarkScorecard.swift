import Foundation

public struct BenchmarkGateResult: Sendable, Equatable, Codable {
    public var gate: String
    public var comparator: String
    public var threshold: Double
    public var observed: Double?
    public var margin: Double?
    public var passed: Bool

    public init(
        gate: String, comparator: String, threshold: Double,
        observed: Double?, margin: Double?, passed: Bool
    ) {
        (self.gate, self.comparator, self.threshold) = (gate, comparator, threshold)
        (self.observed, self.margin, self.passed) = (observed, margin, passed)
    }
}

public struct BenchmarkScorecard: Sendable, Equatable, Codable {
    public var metrics: [String: Double]
    public var gates: [BenchmarkGateResult]
    public var totalSamples: Int
    public var passedSamples: Int
    public var totalPromptTokens: Int
    public var totalCompletionTokens: Int
    public var totalCostUSD: Double
    public var passed: Bool

    public init(
        metrics: [String: Double] = [:], gates: [BenchmarkGateResult] = [],
        totalSamples: Int = 0, passedSamples: Int = 0, passed: Bool = true
    ) {
        self.metrics = metrics
        self.gates = gates
        self.totalSamples = totalSamples
        self.passedSamples = passedSamples
        self.totalPromptTokens = 0
        self.totalCompletionTokens = 0
        self.totalCostUSD = 0
        self.passed = passed
    }

    enum CodingKeys: String, CodingKey {
        case metrics, gates
        case totalSamples = "total_samples"
        case passedSamples = "passed_samples"
        case totalPromptTokens = "total_prompt_tokens"
        case totalCompletionTokens = "total_completion_tokens"
        case totalCostUSD = "total_cost_usd"
        case passed
    }

    public static func evaluate(
        metrics: [String: Double],
        gates: BenchmarkGatesSpec?,
        samples: [BenchmarkSampleRecord],
        extraCostUSD: Double = 0
    ) -> BenchmarkScorecard {
        let gateResults = Self.checkGates(metrics: metrics, spec: gates)
        var card = BenchmarkScorecard(
            metrics: metrics,
            gates: gateResults,
            totalSamples: samples.count,
            passedSamples: samples.filter(\.passed).count,
            passed: gateResults.allSatisfy(\.passed)
        )
        card.totalPromptTokens = samples.reduce(0) { $0 + $1.promptTokens }
        card.totalCompletionTokens = samples.reduce(0) { $0 + $1.completionTokens }
        card.totalCostUSD = samples.reduce(extraCostUSD) { $0 + $1.costUSD }
        return card
    }

    static func checkGates(metrics: [String: Double], spec: BenchmarkGatesSpec?) -> [BenchmarkGateResult] {
        guard let spec else { return [] }
        let specs: [(name: String, key: String, comparator: String, limit: Double?)] = [
            ("min_decode_tps", "decode_tps", ">=", spec.minDecodeTPS),
            ("max_peak_gb", "peak_gb", "<=", spec.maxPeakGB),
            ("min_accuracy", "accuracy", ">=", spec.minAccuracy),
            ("max_p50_ms", "p50_ms", "<=", spec.maxP50Ms),
            ("max_ppl", "ppl", "<=", spec.maxPPL),
            ("max_ece", "ece", "<=", spec.maxECE),
        ]
        var out: [BenchmarkGateResult] = []
        for entry in specs {
            guard let threshold = entry.limit else { continue }
            guard let observed = metrics[entry.key], observed.isFinite else {
                out.append(BenchmarkGateResult(
                    gate: entry.name, comparator: entry.comparator, threshold: threshold,
                    observed: nil, margin: nil, passed: false
                ))
                continue
            }
            let margin = entry.comparator == ">=" ? observed - threshold : threshold - observed
            out.append(BenchmarkGateResult(
                gate: entry.name, comparator: entry.comparator, threshold: threshold,
                observed: observed, margin: margin, passed: margin >= 0
            ))
        }
        return out
    }
}

public struct BenchmarkRunArtifact: Sendable, Equatable, Codable {
    public var manifest: BenchmarkManifest
    public var provenance: BasaltProvenance
    public var scorecard: BenchmarkScorecard
    public var samples: [BenchmarkSampleRecord]

    public init(
        manifest: BenchmarkManifest,
        provenance: BasaltProvenance,
        scorecard: BenchmarkScorecard,
        samples: [BenchmarkSampleRecord]
    ) {
        (self.manifest, self.provenance, self.scorecard, self.samples) = (manifest, provenance, scorecard, samples)
    }
}
