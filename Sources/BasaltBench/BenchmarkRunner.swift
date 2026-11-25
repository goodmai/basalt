import Foundation
import BasaltCore

public struct BenchmarkSuiteRunRequest: Sendable {
    public var spec: ComposeSpec
    public var composeYAML: String
    public var suiteName: String
    public var endpoint: BenchmarkEndpoint
    public var runID: String
    public var outputDirectory: URL?
    public var gpuLockPath: String
    public var gitCommit: String

    public init(
        spec: ComposeSpec,
        composeYAML: String,
        suiteName: String,
        endpoint: BenchmarkEndpoint,
        runID: String = "run-\(UUID().uuidString.prefix(8).lowercased())",
        outputDirectory: URL? = nil
    ) {
        self.spec = spec
        self.composeYAML = composeYAML
        self.suiteName = suiteName
        self.endpoint = endpoint
        self.runID = runID
        self.outputDirectory = outputDirectory
        self.gpuLockPath = GPULock.defaultPath
        self.gitCommit = "HEAD"
    }
}

public struct BenchmarkRunner: Sendable {
    public let registry: BenchmarkAdapterRegistry

    public init(registry: BenchmarkAdapterRegistry = .standard) {
        self.registry = registry
    }

    public func run(_ request: BenchmarkSuiteRunRequest) async throws -> BenchmarkRunArtifact {
        let suites = request.spec.benchmarks?.suites ?? [:]
        guard let suite = suites[request.suiteName] else {
            throw BenchmarkError.unknownSuite(request.suiteName, available: suites.keys.sorted())
        }
        let defaults = request.spec.benchmarks?.defaults ?? BenchmarkDefaultsSpec()
        let needsLock = defaults.gpuLock ?? false
        if needsLock {
            let lockPath = request.gpuLockPath
            return try await GPULock.withLock(path: lockPath) {
                try await executeSuite(request, suite: suite, defaults: defaults, lockHeld: true)
            }
        }
        return try await executeSuite(request, suite: suite, defaults: defaults, lockHeld: false)
    }

    func executeSuite(
        _ request: BenchmarkSuiteRunRequest,
        suite: BenchmarkSuiteSpec,
        defaults: BenchmarkDefaultsSpec,
        lockHeld: Bool
    ) async throws -> BenchmarkRunArtifact {
        let started = ContinuousClock.now
        var allSamples: [BenchmarkSampleRecord] = []
        var mergedMetrics: [String: Double] = [:]
        var metricSums: [String: Double] = [:]
        var metricCounts: [String: Int] = [:]
        var allGateResults: [BenchmarkGateResult] = []
        var totalExtraCost = 0.0
        for (index, task) in suite.tasks.enumerated() {
            try Task.checkCancellation()
            try checkSuiteBudget(suite.budget, started: started, samples: allSamples, extraCost: totalExtraCost)
            guard let adapter = registry.adapter(for: task.adapter) else { throw BenchmarkError.unknownAdapter(task.adapter) }
            let context = BenchmarkTaskContext(spec: request.spec, suiteName: request.suiteName, task: task, defaults: defaults, endpoint: request.endpoint)
            let output = try await adapter.run(context: context)
            allSamples += Self.taggedSamples(output.samples, index: index, multiTask: suite.tasks.count > 1)
            totalExtraCost += output.extraCostUSD
            Self.mergeMetrics(output.metrics, index: index, multiTask: suite.tasks.count > 1, sums: &metricSums, counts: &metricCounts, into: &mergedMetrics)
            let taskCard = BenchmarkScorecard.evaluate(metrics: output.metrics, gates: task.gates, samples: output.samples, extraCostUSD: output.extraCostUSD)
            allGateResults += taskCard.gates.map { gate in
                var tagged = gate
                if suite.tasks.count > 1 { tagged.gate = "task[\(index)].\(gate.gate)" }
                return tagged
            }
        }
        try checkSuiteBudget(suite.budget, started: started, samples: allSamples, extraCost: totalExtraCost)
        let elapsed = started.duration(to: ContinuousClock.now)
        let wallSeconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
        let summary = SuiteAccumulation(
            samples: allSamples, metrics: mergedMetrics, gates: allGateResults,
            extraCost: totalExtraCost, wallSeconds: wallSeconds, lockHeld: lockHeld
        )
        let artifact = buildArtifact(request: request, suite: suite, defaults: defaults, summary: summary)
        if let dir = request.outputDirectory {
            guard !request.runID.isEmpty,
                  !request.runID.contains("/"),
                  !request.runID.contains("\\"),
                  !request.runID.contains("..") else {
                throw BenchmarkError.invalidResponse("runID must be a single safe path component, got '\(request.runID)'")
            }
            try Self.writeArtifact(artifact, to: dir.appendingPathComponent(request.runID, isDirectory: true))
        }
        return artifact
    }

    struct SuiteAccumulation {
        var samples: [BenchmarkSampleRecord]
        var metrics: [String: Double]
        var gates: [BenchmarkGateResult]
        var extraCost: Double
        var wallSeconds: Double
        var lockHeld: Bool
    }

    func checkSuiteBudget(
        _ budget: BenchmarkBudgetSpec?,
        started: ContinuousClock.Instant,
        samples: [BenchmarkSampleRecord],
        extraCost: Double
    ) throws(BenchmarkError) {
        guard let budget else { return }
        let spend = samples.reduce(extraCost) { $0 + $1.costUSD }
        if let usdMax = budget.usdMax, spend > usdMax {
            throw .budgetExceeded(String(format: "spend $%.4f exceeded usd_max $%.4f", spend, usdMax))
        }
        if let maxMinutes = budget.maxMinutes {
            let elapsed = started.duration(to: ContinuousClock.now)
            let minutes = (Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18) / 60.0
            if minutes > maxMinutes {
                throw .budgetExceeded(String(format: "elapsed %.2f min exceeded max_minutes %.2f", minutes, maxMinutes))
            }
        }
    }

    func buildArtifact(
        request: BenchmarkSuiteRunRequest,
        suite: BenchmarkSuiteSpec,
        defaults: BenchmarkDefaultsSpec,
        summary: SuiteAccumulation
    ) -> BenchmarkRunArtifact {
        var finalMetrics = summary.metrics
        if !summary.samples.isEmpty {
            let passRate = Double(summary.samples.filter(\.passed).count) / Double(summary.samples.count)
            finalMetrics["pass_rate"] = passRate
            if suite.tasks.count > 1, finalMetrics["accuracy"] != nil {
                finalMetrics["accuracy"] = passRate
            }
            finalMetrics["p50_ms"] = BenchmarkStats.percentile(summary.samples.map(\.latencyMs), 0.5)
        }
        var scorecard = BenchmarkScorecard.evaluate(
            metrics: finalMetrics, gates: nil, samples: summary.samples, extraCostUSD: summary.extraCost
        )
        scorecard.gates = summary.gates
        scorecard.passed = summary.gates.isEmpty
            ? (!summary.samples.isEmpty && summary.samples.allSatisfy(\.passed))
            : summary.gates.allSatisfy(\.passed)
        var manifest = BenchmarkManifest(
            runID: request.runID, composeName: request.spec.name, suite: request.suiteName,
            seed: defaults.seed ?? 42, temperature: defaults.temperature ?? 0.0,
            endpoint: request.endpoint.baseURL.absoluteString
        )
        manifest.adapters = suite.tasks.map(\.adapter)
        manifest.targets = suite.tasks.map(\.target)
        let ramGB = Int(ProcessInfo.processInfo.physicalMemory / 1_073_741_824)
        var provenance = BasaltProvenance(
            gitCommit: request.gitCommit,
            composeSHA256: BasaltProvenance.sha256Hex(request.composeYAML),
            chip: SystemProfiler.detectChipModel(),
            unifiedMemoryGB: ramGB
        )
        for (slot, model) in request.spec.models {
            switch model.source {
            case .local(let repo): provenance.slotDigests[slot] = BasaltProvenance.sha256Hex(repo)
            case .cloud(let provider, let id): provenance.slotDigests[slot] = "cloud:\(provider)/\(id)"
            }
        }
        provenance.gpuLockAcquired = summary.lockHeld
        provenance.wallClockSeconds = summary.wallSeconds
        provenance.cloudSpendUSD = scorecard.totalCostUSD
        return BenchmarkRunArtifact(manifest: manifest, provenance: provenance, scorecard: scorecard, samples: summary.samples)
    }

    public static func writeArtifact(_ artifact: BenchmarkRunArtifact, to directory: URL) throws {
        try withTaskCancellationShield {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            let data = try encoder.encode(artifact)
            try data.write(to: directory.appendingPathComponent("artifact.json"), options: .atomic)
        }
    }
}

private extension BenchmarkRunner {
    static func taggedSamples(_ samples: [BenchmarkSampleRecord], index: Int, multiTask: Bool) -> [BenchmarkSampleRecord] {
        guard multiTask else { return samples }
        return samples.map { sample in
            var tagged = sample
            tagged.id = "task[\(index)].\(sample.id)"
            return tagged
        }
    }

    static func mergeMetrics(
        _ metrics: [String: Double],
        index: Int,
        multiTask: Bool,
        sums: inout [String: Double],
        counts: inout [String: Int],
        into merged: inout [String: Double]
    ) {
        for (key, value) in metrics {
            if multiTask { merged["task[\(index)].\(key)"] = value }
            sums[key, default: 0] += value
            counts[key, default: 0] += 1
            merged[key] = sums[key, default: 0] / Double(counts[key, default: 1])
        }
    }
}
