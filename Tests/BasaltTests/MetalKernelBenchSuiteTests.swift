import BasaltCore
import BasaltInference
import Foundation
import Testing

@Suite("MetalKernelBenchSuiteTests (#878)", .serialized)
struct MetalKernelBenchSuiteTests {
    private static func makeMeta(
        schemaVersion: Int = MetalKernelBenchMetadata.currentSchemaVersion,
        dirty: Bool = false,
        buildConfig: String = "release",
        chip: String = "Apple M4 Pro",
        profile: String = "production"
    ) -> MetalKernelBenchMetadata {
        MetalKernelBenchMetadata(
            schemaVersion: schemaVersion,
            commit: "abc123def456",
            dirty: dirty,
            buildConfig: buildConfig,
            host: MetalKernelBenchHost(
                chip: chip,
                osVersion: "macOS 15.4",
                memoryGB: 24,
                loadAvg1m: 1.25
            ),
            runConfig: MetalKernelBenchRunConfig(
                profile: profile,
                iterations: 9,
                callsPerSample: 16
            )
        )
    }

    private static func makeRow(
        name: String,
        shape: String = "M=1 K=5120 N=17408",
        medianMs: Double = 0.25,
        iqrMs: Double = 0.002,
        parityPassed: Bool = true,
        gate: MetalKernelBenchGate = .gated
    ) -> MetalKernelBenchRow {
        MetalKernelBenchRow(
            name: name,
            family: "NF4Metal",
            shape: shape,
            timing: MetalKernelBenchTiming(medianMs: medianMs, iqrMs: iqrMs, effectiveGBps: 200.0),
            parity: MetalKernelBenchParity(maxAbsDiff: 1e-4, tolerance: 2e-2, passed: parityPassed),
            gate: gate
        )
    }

    @Test("medianAndIQR handles valid samples and rejects empty, zero, or NaN samples")
    func medianAndIQREdgeCases() {
        let stats = MetalKernelBenchMath.medianAndIQR([5.0, 1.0, 3.0, 2.0, 4.0])
        #expect(abs(stats.median - 3.0) < 1e-9)
        #expect(abs(stats.iqr - 2.0) < 1e-9)
        #expect(abs(MetalKernelBenchMath.effectiveGBps(bytes: 1_000_000_000, medianMs: 10.0) - 100.0) < 1e-9)

        #expect(MetalKernelBenchMath.medianAndIQR([]).median.isNaN)
        #expect(MetalKernelBenchMath.medianAndIQR([1.0, 0.0, 2.0]).median.isNaN)
        #expect(MetalKernelBenchMath.medianAndIQR([1.0, .nan, 2.0]).median.isNaN)
        #expect(MetalKernelBenchMath.medianAndIQR([1.0, .infinity, 2.0]).median.isNaN)
    }

    @Test("compare enforces spread-aware latency gate, parity failures, and valid thresholds")
    func compareLatencyAndParity() {
        let base = MetalKernelBenchReport(
            metadata: Self.makeMeta(),
            rows: [
                Self.makeRow(name: "nf4_m8", medianMs: 1.0, iqrMs: 0.005),
                Self.makeRow(name: "iq_q4K_m1", medianMs: 1.0, iqrMs: 0.005),
            ]
        )
        let regressed = MetalKernelBenchReport(
            metadata: Self.makeMeta(),
            rows: [
                Self.makeRow(name: "nf4_m8", medianMs: 1.01, iqrMs: 0.005),
                Self.makeRow(name: "iq_q4K_m1", medianMs: 1.04, iqrMs: 0.005),
            ]
        )
        let cmp = MetalKernelBenchMath.compare(baseline: base, candidate: regressed, maxRegressionPct: 2.0)
        #expect(cmp.errors.isEmpty)
        #expect(cmp.hasFailure)
        #expect(cmp.deltas[0].regressed == false)
        #expect(cmp.deltas[1].regressed == true)

        let parityBroken = MetalKernelBenchReport(
            metadata: Self.makeMeta(),
            rows: [
                Self.makeRow(name: "nf4_m8", medianMs: 1.0, iqrMs: 0.005, parityPassed: false),
                Self.makeRow(name: "iq_q4K_m1", medianMs: 1.0, iqrMs: 0.005),
            ]
        )
        let parityCmp = MetalKernelBenchMath.compare(baseline: base, candidate: parityBroken, maxRegressionPct: 2.0)
        #expect(parityCmp.hasFailure)
        #expect(parityCmp.deltas[0].parityPassed == false)

        let nanThresh = MetalKernelBenchMath.compare(baseline: base, candidate: base, maxRegressionPct: .nan)
        #expect(nanThresh.hasFailure)
        let negThresh = MetalKernelBenchMath.compare(baseline: base, candidate: base, maxRegressionPct: -1.0)
        #expect(negThresh.hasFailure)

        let reportOnlyBase = MetalKernelBenchReport(
            metadata: Self.makeMeta(),
            rows: [Self.makeRow(name: "flint_router_40b", medianMs: 1.0, iqrMs: 0.005, gate: .reportOnly("sub-10us dispatch"))]
        )
        let reportOnlySlow = MetalKernelBenchReport(
            metadata: Self.makeMeta(),
            rows: [Self.makeRow(name: "flint_router_40b", medianMs: 1.2, iqrMs: 0.005, gate: .reportOnly("sub-10us dispatch"))]
        )
        let roCmp = MetalKernelBenchMath.compare(baseline: reportOnlyBase, candidate: reportOnlySlow)
        #expect(!roCmp.hasFailure)
        #expect(roCmp.deltas[0].status == .reportOnly("sub-10us dispatch"))
        #expect(MetalKernelBenchMath.renderComparisonMarkdown(roCmp).contains("REPORT_ONLY (sub-10us dispatch)"))

        let gateMismatch = MetalKernelBenchReport(
            metadata: Self.makeMeta(),
            rows: [Self.makeRow(name: "flint_router_40b", medianMs: 1.0, iqrMs: 0.005, gate: .gated)]
        )
        #expect(MetalKernelBenchMath.compare(baseline: reportOnlyBase, candidate: gateMismatch).hasFailure)

        let reportOnlyBadParity = MetalKernelBenchReport(
            metadata: Self.makeMeta(),
            rows: [Self.makeRow(name: "flint_router_40b", medianMs: 1.0, iqrMs: 0.005, parityPassed: false, gate: .reportOnly("sub-10us dispatch"))]
        )
        let roParityCmp = MetalKernelBenchMath.compare(baseline: reportOnlyBase, candidate: reportOnlyBadParity)
        #expect(roParityCmp.hasFailure)
        #expect(roParityCmp.deltas[0].status == .parityFail)
    }

    @Test("compare rejects debug builds, metadata mismatches, missing/duplicate kernels, and zero/NaN rows")
    func compareValidationGuards() {
        let rowA = Self.makeRow(name: "nf4_m1")
        let rowB = Self.makeRow(name: "nf4_m8")
        let base = MetalKernelBenchReport(metadata: Self.makeMeta(), rows: [rowA, rowB])

        let debugReport = MetalKernelBenchReport(metadata: Self.makeMeta(buildConfig: "debug"), rows: [rowA, rowB])
        #expect(MetalKernelBenchMath.compare(baseline: debugReport, candidate: debugReport).hasFailure)
        #expect(!MetalKernelBenchMath.compare(baseline: debugReport, candidate: debugReport, allowDebug: true).hasFailure)

        let dirtyReport = MetalKernelBenchReport(metadata: Self.makeMeta(dirty: true), rows: [rowA, rowB])
        #expect(MetalKernelBenchMath.compare(baseline: base, candidate: dirtyReport).hasFailure)
        #expect(!MetalKernelBenchMath.compare(baseline: base, candidate: dirtyReport, allowDirty: true).hasFailure)

        let otherChip = MetalKernelBenchReport(metadata: Self.makeMeta(chip: "Apple M3 Max"), rows: [rowA, rowB])
        #expect(MetalKernelBenchMath.compare(baseline: base, candidate: otherChip).hasFailure)

        let otherSchema = MetalKernelBenchReport(metadata: Self.makeMeta(schemaVersion: 1), rows: [rowA, rowB])
        #expect(MetalKernelBenchMath.compare(baseline: base, candidate: otherSchema).hasFailure)

        let missingKernel = MetalKernelBenchReport(metadata: Self.makeMeta(), rows: [rowA])
        #expect(MetalKernelBenchMath.compare(baseline: base, candidate: missingKernel).hasFailure)

        let duplicateKernel = MetalKernelBenchReport(metadata: Self.makeMeta(), rows: [rowA, rowA])
        #expect(MetalKernelBenchMath.compare(baseline: duplicateKernel, candidate: base).hasFailure)

        let shapeChanged = MetalKernelBenchReport(
            metadata: Self.makeMeta(),
            rows: [Self.makeRow(name: "nf4_m1", shape: "M=1 K=1024 N=1024"), rowB]
        )
        #expect(MetalKernelBenchMath.compare(baseline: base, candidate: shapeChanged).hasFailure)

        let zeroRow = MetalKernelBenchReport(
            metadata: Self.makeMeta(),
            rows: [Self.makeRow(name: "nf4_m1", medianMs: 0.0), rowB]
        )
        #expect(MetalKernelBenchMath.compare(baseline: base, candidate: zeroRow).hasFailure)
    }

    @Test("Report JSON round-trip and Markdown rendering include metadata and gate columns")
    func jsonAndMarkdownRendering() throws {
        let report = MetalKernelBenchReport(
            metadata: Self.makeMeta(),
            rows: [Self.makeRow(name: "nf4_m8", medianMs: 0.312, iqrMs: 0.004)]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let decoded = try JSONDecoder().decode(MetalKernelBenchReport.self, from: try encoder.encode(report))
        #expect(decoded == report)

        let md = MetalKernelBenchMath.renderMarkdown(decoded)
        #expect(md.contains("Apple M4 Pro"))
        #expect(md.contains("abc123def456"))
        #expect(md.contains("`nf4_m8`"))

        let dirtyReport = MetalKernelBenchReport(metadata: Self.makeMeta(dirty: true), rows: report.rows)
        #expect(MetalKernelBenchMath.renderMarkdown(dirtyReport).contains("(dirty)"))

        let legacyJSON = Data(#"{"metadata":{"schemaVersion":2},"rows":[]}"#.utf8)
        #expect(throws: MetalKernelBenchDecodeError.unsupportedSchema(found: 2, expected: MetalKernelBenchMetadata.currentSchemaVersion)) {
            _ = try MetalKernelBenchMath.decodeReport(from: legacyJSON)
        }

        let cmp = MetalKernelBenchMath.compare(baseline: report, candidate: decoded)
        let cmpMd = MetalKernelBenchMath.renderComparisonMarkdown(cmp)
        #expect(cmpMd.contains("`nf4_m8`"))
        #expect(cmpMd.contains("OK"))
    }

    @Test("MetalKernelBenchRunner smoke suite passes #228 parity across all 15 kernels including exact router indices")
    func runnerSmokeEndToEnd() {
        let meta = Self.makeMeta(buildConfig: "debug", profile: "smoke")
        let report = MetalKernelBenchRunner.runAll(profile: .smoke, iterations: 3, warmup: 1, metadata: meta)
        #expect(report.rows.count == 15)
        for row in report.rows {
            #expect(row.timing.medianMs.isFinite && row.timing.medianMs > 0, "Expected positive medianMs for \(row.name)")
            #expect(row.timing.iqrMs.isFinite && row.timing.iqrMs >= 0, "Expected finite iqrMs for \(row.name)")
            #expect(row.parity.passed, "Parity failed for \(row.name): diff=\(row.parity.maxAbsDiff) tol=\(row.parity.tolerance)")
        }
        #expect(report.allParityPassed)
    }
}
