import Foundation
import BasaltCore

public enum BenchmarkSubmitter {
    @discardableResult
    public static func exportDryRun(
        target: SubmissionTarget,
        artifact: BenchmarkRunArtifact,
        submit: BenchmarkSubmitSpec? = nil,
        outputDirectory: URL
    ) throws -> SubmissionPlan {
        let plan = SubmissionFormatter.build(target: target, artifact: artifact, submit: submit)
        guard plan.isValid else {
            throw BenchmarkError.submissionInvalid(target: target, errors: plan.validationErrors)
        }
        try withTaskCancellationShield {
            let fm = FileManager.default
            try fm.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
            for file in plan.files {
                let destination = outputDirectory.appendingPathComponent(file.relativePath)
                try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                try file.data.write(to: destination, options: .atomic)
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            let planSummary: [String: Any] = [
                "target": plan.target.rawValue,
                "dry_run": true,
                "files": plan.files.map(\.relativePath),
                "cli_command": plan.cliCommand ?? [],
                "http_endpoint": plan.httpEndpoint ?? "",
                "required_env_var": plan.requiredEnvVar ?? "",
            ]
            let summaryData = try JSONSerialization.data(
                withJSONObject: planSummary,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            )
            try summaryData.write(to: outputDirectory.appendingPathComponent("submission_plan.json"), options: .atomic)
        }
        return plan
    }
}

public struct BenchmarkMetricDelta: Sendable, Equatable, Codable {
    public var metric: String
    public var baseline: Double
    public var candidate: Double
    public var absoluteDelta: Double
    public var percentDelta: Double
    public var improved: Bool
    public var regressed: Bool

    public init(
        metric: String, baseline: Double, candidate: Double,
        improved: Bool, regressed: Bool
    ) {
        self.metric = metric
        let safeBase = baseline.isFinite ? baseline : 0.0
        let safeCand = candidate.isFinite ? candidate : 0.0
        (self.baseline, self.candidate) = (safeBase, safeCand)
        let delta = safeCand - safeBase
        let denom = abs(safeBase) > 1e-9 ? abs(safeBase) : 1.0
        let pct = (delta / denom) * 100.0
        (self.absoluteDelta, self.percentDelta) = (delta.isFinite ? delta : 0.0, pct.isFinite ? pct : 0.0)
        (self.improved, self.regressed) = (improved, regressed)
    }
}

public struct BenchmarkComparisonReport: Sendable, Equatable, Codable {
    public var baselineRunID: String
    public var candidateRunID: String
    public var composeChanged: Bool
    public var hardwareMatched: Bool
    public var deltas: [BenchmarkMetricDelta]
    public var sampleRegressions: [String]
    public var sampleImprovements: [String]
    public var hasRegressions: Bool

    public init(
        baselineRunID: String,
        candidateRunID: String,
        composeChanged: Bool = false,
        hardwareMatched: Bool = true,
        deltas: [BenchmarkMetricDelta] = []
    ) {
        self.baselineRunID = baselineRunID
        self.candidateRunID = candidateRunID
        self.composeChanged = composeChanged
        self.hardwareMatched = hardwareMatched
        self.deltas = deltas
        self.sampleRegressions = []
        self.sampleImprovements = []
        self.hasRegressions = deltas.contains(where: \.regressed)
    }

    public var markdownTable: String {
        var lines = [
            "Benchmark Comparison: \(baselineRunID) -> \(candidateRunID)",
            "Hardware match: \(hardwareMatched ? "yes" : "no") | Compose SHA changed: \(composeChanged ? "yes" : "no")",
            "",
            "| Metric | Baseline | Candidate | Delta | % | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in deltas {
            let status = row.regressed ? "REGRESSED" : (row.improved ? "IMPROVED" : "SAME")
            lines.append(String(
                format: "| %@ | %.4f | %.4f | %+.4f | %+.2f%% | %@ |",
                row.metric, row.baseline, row.candidate, row.absoluteDelta, row.percentDelta, status
            ))
        }
        if !sampleRegressions.isEmpty {
            lines.append("")
            lines.append("Sample regressions (\(sampleRegressions.count)): \(sampleRegressions.joined(separator: ", "))")
        }
        if !sampleImprovements.isEmpty {
            lines.append("Sample improvements (\(sampleImprovements.count)): \(sampleImprovements.joined(separator: ", "))")
        }
        return lines.joined(separator: "\n")
    }
}

public enum BenchmarkComparator {
    static let lowerIsBetter: Set<String> = [
        "ttft_ms", "p50_ms", "p95_ms", "peak_gb", "ppl", "ece",
        "cost_usd", "cost_per_1k_usd", "abstain_rate", "metricx24",
    ]

    private static func metricDelta(
        key: String, oldVal: Double, newVal: Double, relativeTolerance: Double
    ) -> BenchmarkMetricDelta {
        guard oldVal.isFinite && newVal.isFinite else {
            return BenchmarkMetricDelta(
                metric: key, baseline: oldVal, candidate: newVal,
                improved: !oldVal.isFinite && newVal.isFinite,
                regressed: oldVal.isFinite && !newVal.isFinite
            )
        }
        let absDelta = newVal - oldVal
        let denom = abs(oldVal) > 1e-9 ? abs(oldVal) : 1.0
        let significant = abs(absDelta / denom) > relativeTolerance
        let lowerBetter = lowerIsBetter.contains(key)
        let improved = significant && (lowerBetter ? absDelta < 0 : absDelta > 0)
        let regressed = significant && (lowerBetter ? absDelta > 0 : absDelta < 0)
        return BenchmarkMetricDelta(
            metric: key, baseline: oldVal, candidate: newVal,
            improved: improved, regressed: regressed
        )
    }

    public static func compare(
        baseline: BenchmarkRunArtifact,
        candidate: BenchmarkRunArtifact,
        relativeTolerance: Double = 0.02
    ) -> BenchmarkComparisonReport {
        let sharedKeys = Set(baseline.scorecard.metrics.keys).intersection(candidate.scorecard.metrics.keys).sorted()
        let deltas = sharedKeys.compactMap { key -> BenchmarkMetricDelta? in
            guard let oldVal = baseline.scorecard.metrics[key],
                  let newVal = candidate.scorecard.metrics[key] else { return nil }
            return metricDelta(key: key, oldVal: oldVal, newVal: newVal, relativeTolerance: relativeTolerance)
        }
        let baseByID = Dictionary(baseline.samples.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
        var sampleRegressions: [String] = []
        var sampleImprovements: [String] = []
        for cand in candidate.samples {
            guard let prev = baseByID[cand.id] else { continue }
            if prev.passed && !cand.passed {
                sampleRegressions.append(cand.id)
            } else if !prev.passed && cand.passed {
                sampleImprovements.append(cand.id)
            }
        }
        var report = BenchmarkComparisonReport(
            baselineRunID: baseline.manifest.runID,
            candidateRunID: candidate.manifest.runID,
            composeChanged: baseline.provenance.composeSHA256 != candidate.provenance.composeSHA256,
            hardwareMatched: baseline.provenance.chip == candidate.provenance.chip
                && baseline.provenance.unifiedMemoryGB == candidate.provenance.unifiedMemoryGB,
            deltas: deltas
        )
        report.sampleRegressions = sampleRegressions.sorted()
        report.sampleImprovements = sampleImprovements.sorted()
        report.hasRegressions = !report.sampleRegressions.isEmpty || report.deltas.contains(where: \.regressed)
        return report
    }
}
