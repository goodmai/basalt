import Foundation

public struct MetalKernelBenchHost: Codable, Equatable, Sendable {
    public let chip: String
    public let osVersion: String
    public let memoryGB: Int
    public let loadAvg1m: Double

    public init(chip: String, osVersion: String, memoryGB: Int, loadAvg1m: Double) {
        self.chip = chip
        self.osVersion = osVersion
        self.memoryGB = memoryGB
        self.loadAvg1m = loadAvg1m
    }
}

public struct MetalKernelBenchRunConfig: Codable, Equatable, Sendable {
    public let profile: String
    public let iterations: Int
    public let callsPerSample: Int

    public init(profile: String, iterations: Int, callsPerSample: Int) {
        self.profile = profile
        self.iterations = iterations
        self.callsPerSample = callsPerSample
    }
}

public struct MetalKernelBenchMetadata: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 3

    public let schemaVersion: Int
    public let commit: String
    public let dirty: Bool
    public let buildConfig: String
    public let host: MetalKernelBenchHost
    public let runConfig: MetalKernelBenchRunConfig

    public init(
        schemaVersion: Int = MetalKernelBenchMetadata.currentSchemaVersion,
        commit: String,
        dirty: Bool = false,
        buildConfig: String,
        host: MetalKernelBenchHost,
        runConfig: MetalKernelBenchRunConfig
    ) {
        self.schemaVersion = schemaVersion
        self.commit = commit
        self.dirty = dirty
        self.buildConfig = buildConfig
        self.host = host
        self.runConfig = runConfig
    }
}

public struct MetalKernelBenchTiming: Codable, Equatable, Sendable {
    public let medianMs: Double
    public let iqrMs: Double
    public let effectiveGBps: Double
    public let passMediansMs: [Double]

    public var relativeIQRPct: Double {
        guard medianMs.isFinite, medianMs > 0, iqrMs.isFinite, iqrMs >= 0 else { return .infinity }
        return (iqrMs / medianMs) * 100.0
    }

    public init(medianMs: Double, iqrMs: Double, effectiveGBps: Double, passMediansMs: [Double] = []) {
        self.medianMs = medianMs
        self.iqrMs = iqrMs
        self.effectiveGBps = effectiveGBps
        self.passMediansMs = passMediansMs
    }
}

public struct MetalKernelBenchParity: Codable, Equatable, Sendable {
    public let maxAbsDiff: Double
    public let tolerance: Double
    public let passed: Bool

    public init(maxAbsDiff: Double, tolerance: Double, passed: Bool) {
        self.maxAbsDiff = maxAbsDiff
        self.tolerance = tolerance
        self.passed = passed
    }
}

public enum MetalKernelBenchGate: Codable, Equatable, Sendable {
    case gated
    case reportOnly(String)

    public var isGated: Bool {
        if case .gated = self { return true }
        return false
    }

    public var reason: String? {
        if case .reportOnly(let text) = self { return text }
        return nil
    }
}

public struct MetalKernelBenchRow: Codable, Equatable, Sendable {
    public let name: String
    public let family: String
    public let shape: String
    public let timing: MetalKernelBenchTiming
    public let parity: MetalKernelBenchParity
    public let gate: MetalKernelBenchGate

    public init(
        name: String,
        family: String,
        shape: String,
        timing: MetalKernelBenchTiming,
        parity: MetalKernelBenchParity,
        gate: MetalKernelBenchGate = .gated
    ) {
        self.name = name
        self.family = family
        self.shape = shape
        self.timing = timing
        self.parity = parity
        self.gate = gate
    }
}

public struct MetalKernelBenchReport: Codable, Equatable, Sendable {
    public let metadata: MetalKernelBenchMetadata
    public let rows: [MetalKernelBenchRow]

    public var allParityPassed: Bool {
        rows.allSatisfy(\.parity.passed)
    }

    public init(metadata: MetalKernelBenchMetadata, rows: [MetalKernelBenchRow]) {
        self.metadata = metadata
        self.rows = rows
    }
}

public enum MetalKernelBenchStatus: Codable, Equatable, Sendable {
    case ok
    case reportOnly(String)
    case regressed
    case parityFail

    public var label: String {
        switch self {
        case .ok: return "OK"
        case .reportOnly(let reason): return "REPORT_ONLY (\(reason))"
        case .regressed: return "REGRESSED"
        case .parityFail: return "PARITY_FAIL"
        }
    }
}

public struct MetalKernelBenchDelta: Codable, Equatable, Sendable {
    public let name: String
    public let shape: String
    public let baselineMs: Double
    public let candidateMs: Double
    public let effectiveThresholdPct: Double
    public let status: MetalKernelBenchStatus

    public var deltaPct: Double {
        baselineMs > 0 ? ((candidateMs - baselineMs) / baselineMs) * 100.0 : .nan
    }

    public var regressed: Bool { status == .regressed }
    public var parityPassed: Bool { status != .parityFail }

    public init(
        name: String,
        shape: String,
        baselineMs: Double,
        candidateMs: Double,
        effectiveThresholdPct: Double,
        status: MetalKernelBenchStatus
    ) {
        self.name = name
        self.shape = shape
        self.baselineMs = baselineMs
        self.candidateMs = candidateMs
        self.effectiveThresholdPct = effectiveThresholdPct
        self.status = status
    }
}

public struct MetalKernelBenchComparison: Codable, Equatable, Sendable {
    public let maxRegressionPct: Double
    public let spreadMultiplier: Double
    public let errors: [String]
    public let deltas: [MetalKernelBenchDelta]

    public var hasFailure: Bool {
        !errors.isEmpty || deltas.contains { $0.status == .regressed || $0.status == .parityFail }
    }

    public init(
        maxRegressionPct: Double,
        spreadMultiplier: Double,
        errors: [String],
        deltas: [MetalKernelBenchDelta]
    ) {
        self.maxRegressionPct = maxRegressionPct
        self.spreadMultiplier = spreadMultiplier
        self.errors = errors
        self.deltas = deltas
    }
}

public enum MetalKernelBenchMath {
    public static func medianAndIQR(_ samples: [Double]) -> (median: Double, iqr: Double) {
        guard !samples.isEmpty, samples.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            return (.nan, .nan)
        }
        let sorted = samples.sorted()
        let med = percentile(sorted, 0.5)
        let q1 = percentile(sorted, 0.25)
        let q3 = percentile(sorted, 0.75)
        return (med, max(0, q3 - q1))
    }

    private static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard sorted.count > 1 else { return sorted[0] }
        let idx = p * Double(sorted.count - 1)
        let lo = Int(floor(idx))
        let hi = Int(ceil(idx))
        let frac = idx - Double(lo)
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }

    public static func effectiveGBps(bytes: Double, medianMs: Double) -> Double {
        guard bytes > 0, medianMs.isFinite, medianMs > 0 else { return 0 }
        return (bytes / 1e9) / (medianMs / 1e3)
    }

    public static func medianCI95Pct(relativeIQRPct: Double, iterations: Int) -> Double {
        guard relativeIQRPct.isFinite, relativeIQRPct >= 0 else { return .infinity }
        return 1.96 * 1.253314 * relativeIQRPct / (1.349 * sqrt(Double(max(1, iterations))))
    }

    public static func compare(
        baseline: MetalKernelBenchReport,
        candidate: MetalKernelBenchReport,
        maxRegressionPct: Double = 2.0,
        spreadMultiplier: Double = 1.5,
        allowDebug: Bool = false,
        allowDirty: Bool = false
    ) -> MetalKernelBenchComparison {
        var errors = validateMetadata(
            baseline: baseline.metadata,
            candidate: candidate.metadata,
            maxRegressionPct: maxRegressionPct,
            spreadMultiplier: spreadMultiplier,
            allowDebug: allowDebug,
            allowDirty: allowDirty
        )
        let (baseMap, baseDups) = indexRows(baseline.rows, label: "baseline")
        let (candMap, candDups) = indexRows(candidate.rows, label: "candidate")
        errors.append(contentsOf: baseDups)
        errors.append(contentsOf: candDups)
        errors.append(contentsOf: validateRowSets(baseMap: baseMap, candMap: candMap))

        let iters = (baseline.metadata.runConfig.iterations, candidate.metadata.runConfig.iterations)
        var deltas: [MetalKernelBenchDelta] = []
        for cand in candidate.rows {
            guard let base = baseMap[cand.name] else { continue }
            let (delta, rowErrors) = compareRow(
                base: base,
                cand: cand,
                iterations: iters,
                maxRegressionPct: maxRegressionPct,
                spreadMultiplier: spreadMultiplier
            )
            errors.append(contentsOf: rowErrors)
            if let delta { deltas.append(delta) }
        }
        return MetalKernelBenchComparison(
            maxRegressionPct: maxRegressionPct,
            spreadMultiplier: spreadMultiplier,
            errors: errors,
            deltas: deltas
        )
    }

    private static func validateMetadata(
        baseline: MetalKernelBenchMetadata,
        candidate: MetalKernelBenchMetadata,
        maxRegressionPct: Double,
        spreadMultiplier: Double,
        allowDebug: Bool,
        allowDirty: Bool
    ) -> [String] {
        var errors: [String] = []
        if !maxRegressionPct.isFinite || maxRegressionPct < 0 {
            errors.append("Invalid maxRegressionPct: \(maxRegressionPct)")
        }
        if !spreadMultiplier.isFinite || spreadMultiplier < 0 {
            errors.append("Invalid spreadMultiplier: \(spreadMultiplier)")
        }
        if baseline.schemaVersion != candidate.schemaVersion {
            errors.append("Schema mismatch: baseline v\(baseline.schemaVersion) vs candidate v\(candidate.schemaVersion)")
        }
        if !allowDebug && (baseline.buildConfig != "release" || candidate.buildConfig != "release") {
            errors.append("Release build required (baseline=\(baseline.buildConfig), candidate=\(candidate.buildConfig))")
        }
        if !allowDirty && (baseline.dirty || candidate.dirty) {
            errors.append("Dirty working tree rejected (baseline.dirty=\(baseline.dirty), candidate.dirty=\(candidate.dirty))")
        }
        if baseline.buildConfig != candidate.buildConfig {
            errors.append("Build config mismatch: \(baseline.buildConfig) vs \(candidate.buildConfig)")
        }
        if baseline.host.chip != candidate.host.chip {
            errors.append("Chip mismatch: \(baseline.host.chip) vs \(candidate.host.chip)")
        }
        if baseline.runConfig.profile != candidate.runConfig.profile {
            errors.append("Profile mismatch: \(baseline.runConfig.profile) vs \(candidate.runConfig.profile)")
        }
        return errors
    }

    private static func indexRows(
        _ rows: [MetalKernelBenchRow],
        label: String
    ) -> ([String: MetalKernelBenchRow], [String]) {
        var map: [String: MetalKernelBenchRow] = [:]
        var errors: [String] = []
        for row in rows {
            if map[row.name] != nil {
                errors.append("Duplicate kernel '\(row.name)' in \(label)")
            } else {
                map[row.name] = row
            }
        }
        return (map, errors)
    }

    private static func validateRowSets(
        baseMap: [String: MetalKernelBenchRow],
        candMap: [String: MetalKernelBenchRow]
    ) -> [String] {
        var errors: [String] = []
        let missing = Set(baseMap.keys).subtracting(candMap.keys).sorted()
        let extra = Set(candMap.keys).subtracting(baseMap.keys).sorted()
        if !missing.isEmpty {
            errors.append("Missing kernels in candidate: \(missing.joined(separator: ", "))")
        }
        if !extra.isEmpty {
            errors.append("Unexpected kernels in candidate: \(extra.joined(separator: ", "))")
        }
        return errors
    }

    private static func compareRow(
        base: MetalKernelBenchRow,
        cand: MetalKernelBenchRow,
        iterations: (Int, Int),
        maxRegressionPct: Double,
        spreadMultiplier: Double
    ) -> (MetalKernelBenchDelta?, [String]) {
        var errors: [String] = []
        if base.shape != cand.shape {
            errors.append("Shape mismatch for '\(cand.name)': \(base.shape) vs \(cand.shape)")
        }
        if base.gate.isGated != cand.gate.isGated {
            errors.append("Gate policy mismatch for '\(cand.name)': baseline=\(base.gate.isGated) vs candidate=\(cand.gate.isGated)")
        }
        let bt = base.timing
        let ct = cand.timing
        let validBase = bt.medianMs.isFinite && bt.medianMs > 0 && bt.iqrMs.isFinite && bt.iqrMs >= 0
        let validCand = ct.medianMs.isFinite && ct.medianMs > 0 && ct.iqrMs.isFinite && ct.iqrMs >= 0
        guard validBase, validCand else {
            errors.append("Invalid timing for '\(cand.name)': baseline=\(bt.medianMs)ms candidate=\(ct.medianMs)ms")
            return (nil, errors)
        }
        _ = iterations
        let deltaPct = ((ct.medianMs - bt.medianMs) / bt.medianMs) * 100.0
        let interPassSpread = max(bt.relativeIQRPct, ct.relativeIQRPct)
        let effectiveThreshold = max(maxRegressionPct, spreadMultiplier * interPassSpread)
        let status: MetalKernelBenchStatus
        if !(base.parity.passed && cand.parity.passed) {
            status = .parityFail
        } else if case .reportOnly(let reason) = cand.gate {
            status = .reportOnly(reason)
        } else if deltaPct > effectiveThreshold {
            status = .regressed
        } else {
            status = .ok
        }
        let delta = MetalKernelBenchDelta(
            name: cand.name,
            shape: cand.shape,
            baselineMs: bt.medianMs,
            candidateMs: ct.medianMs,
            effectiveThresholdPct: effectiveThreshold,
            status: status
        )
        return (delta, errors)
    }
    public static func decodeReport(from data: Data) throws -> MetalKernelBenchReport {
        struct SchemaProbe: Decodable {
            struct Meta: Decodable { let schemaVersion: Int }
            let metadata: Meta
        }
        if let probe = try? JSONDecoder().decode(SchemaProbe.self, from: data),
           probe.metadata.schemaVersion != MetalKernelBenchMetadata.currentSchemaVersion {
            throw MetalKernelBenchDecodeError.unsupportedSchema(
                found: probe.metadata.schemaVersion,
                expected: MetalKernelBenchMetadata.currentSchemaVersion
            )
        }
        return try JSONDecoder().decode(MetalKernelBenchReport.self, from: data)
    }
}

public enum MetalKernelBenchDecodeError: Error, LocalizedError, Equatable, Sendable {
    case unsupportedSchema(found: Int, expected: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedSchema(let found, let expected):
            return "Unsupported baseline schema v\(found); expected v\(expected). Regenerate baseline JSON with current PerformanceBenchmark."
        }
    }
}

public extension MetalKernelBenchMath {
    static func renderMarkdown(_ report: MetalKernelBenchReport) -> String {
        let m = report.metadata
        let h = m.host
        let r = m.runConfig
        let dirtyTag = m.dirty ? " (dirty)" : ""
        var lines: [String] = [
            "### Metal Kernel Microbenchmark (\(h.chip), \(h.memoryGB) GB, \(m.buildConfig), commit `\(m.commit)`\(dirtyTag), profile=\(r.profile), iters=\(r.iterations)×\(r.callsPerSample), load=\(String(format: "%.2f", h.loadAvg1m)))",
            "",
            "| Kernel | Family | Shape | Median (ms) | IQR (ms) | Rel IQR | Eff GB/s | Max Diff | Parity |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in report.rows {
            let status = row.parity.passed ? "PASS" : "FAIL"
            lines.append(
                String(
                    format: "| `%@` | %@ | `%@` | %.4f | %.4f | %.2f%% | %.1f | %.2e | %@ |",
                    row.name, row.family, row.shape, row.timing.medianMs, row.timing.iqrMs,
                    row.timing.relativeIQRPct, row.timing.effectiveGBps, row.parity.maxAbsDiff, status
                )
            )
        }
        return lines.joined(separator: "\n")
    }

    static func renderComparisonMarkdown(_ comparison: MetalKernelBenchComparison) -> String {
        var lines: [String] = [
            String(
                format: "### Metal Kernel Before/After Comparison (base threshold: +%.1f%%, spread×%.1f)",
                comparison.maxRegressionPct, comparison.spreadMultiplier
            ),
            "",
        ]
        if !comparison.errors.isEmpty {
            lines.append("**Validation errors:**")
            for err in comparison.errors {
                lines.append("- \(err)")
            }
            lines.append("")
        }
        lines.append("| Kernel | Shape | Baseline (ms) | Candidate (ms) | Delta | Gate | Parity | Status |")
        lines.append("|---|---|---:|---:|---:|---:|---|---|")
        for row in comparison.deltas {
            let sign = row.deltaPct >= 0 ? "+" : ""
            let parity = row.parityPassed ? "PASS" : "FAIL"
            lines.append(
                String(
                    format: "| `%@` | `%@` | %.4f | %.4f | %@%.2f%% | +%.2f%% | %@ | %@ |",
                    row.name, row.shape, row.baselineMs, row.candidateMs,
                    sign, row.deltaPct, row.effectiveThresholdPct, parity, row.status.label
                )
            )
        }
        return lines.joined(separator: "\n")
    }
}
