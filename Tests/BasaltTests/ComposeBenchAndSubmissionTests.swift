import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltBench
@testable import BasaltCLI

@Suite("compose v2 benchmarks: YAML parsing, shorthand and full suites, schema coverage, plan output, and secret hygiene (#492)")
struct ComposeBenchYAMLTests {
    static let validV2 = """
        version: 2
        name: bench-desk
        profiles:
          big:
            when: { memory_gb: { over: 32 } }
        models:
          chat:
            source: mlx-community/gemma-4-12B-it-qat-4bit
          candidate:
            source: mlx-community/Qwen3-4B-4bit
            keep: on-demand
        pipeline:
          decide:
            cascade: [chat]
          answer: chat
        pipelines:
          fast:
            answer: chat
        benchmarks:
          defaults:
            seed: 123
            temperature: 0.0
            gpu_lock: true
            output_dir: ./bench-runs
          smoke:
            - adapter: basalt:throughput
              target: model:candidate
              params: { iterations: 2, max_tokens: 32 }
              gates: { min_decode_tps: 20, max_peak_gb: 18 }
          frontier:
            profiles: [big]
            budget: { usd_max: 2.5, max_minutes: 15 }
            tasks:
              - adapter: arc-agi-2
                dataset: ./arc.json
                target: pipeline:fast
                gates: { min_accuracy: 0.25 }
                submit:
                  target: arc-prize
                  api_key_env: ARC_PRIZE_API_KEY
                  subset: public-eval
                  split: test
        """

    @Test("Shorthand list suites, profile-gated full suites, unused-slot exemption, and compose plan benchmark lines parse cleanly")
    func parsesShorthandAndFullSuites() throws {
        let folder = URL(fileURLWithPath: "/tmp/compose-bench-spec")
        let spec64 = try ComposeFile.parse(Self.validV2, file: "b.yaml", context: ComposeContext(memoryGB: 64, base: folder))
        #expect(spec64.benchmarks?.defaults?.seed == 123)
        #expect(spec64.benchmarks?.defaults?.gpuLock == true)
        #expect(spec64.benchmarks?.defaults?.outputDir == folder.appending(path: "bench-runs").standardizedFileURL.path)
        #expect(Set(spec64.benchmarks?.suites.keys.map { $0 } ?? []) == ["smoke", "frontier"])
        #expect(spec64.benchmarks?.suites["smoke"]?.tasks.first?.params["iterations"] == .int(2))
        #expect(spec64.benchmarks?.suites["frontier"]?.budget?.usdMax == 2.5)
        #expect(spec64.benchmarks?.suites["frontier"]?.tasks.first?.dataset == folder.appending(path: "arc.json").standardizedFileURL.path)
        #expect(spec64.benchmarks?.suites["frontier"]?.tasks.first?.submit?.target == .arcPrize)
        #expect(spec64.benchmarkLines.count == 2)
        let spec16 = try ComposeFile.parse(Self.validV2, file: "b.yaml", context: ComposeContext(memoryGB: 16))
        #expect(Set(spec16.benchmarks?.suites.keys.map { $0 } ?? []) == ["smoke"])
    }

    @Test("Explicit benchmarks.suites mapping parses identically to shorthand suite keys")
    func parsesExplicitSuitesMapping() throws {
        let yaml = """
            version: 2
            name: explicit-suites
            models:
              chat: { source: mlx-community/gemma-4-12B-it-qat-4bit }
            pipeline:
              answer: chat
            benchmarks:
              defaults: { seed: 7 }
              suites:
                nightly:
                  budget: { usd_max: 1.0 }
                  tasks:
                    - adapter: basalt:system-one
                      target: pipeline
            """
        let spec = try ComposeFile.parse(yaml, file: "e.yaml")
        #expect(spec.benchmarks?.defaults?.seed == 7)
        #expect(spec.benchmarks?.suites["nightly"]?.tasks.first?.adapter == "basalt:system-one")
    }

    static func refusal(_ yaml: String) -> String {
        do {
            _ = try ComposeFile.parse(yaml, file: "bad.yaml")
            return ""
        } catch {
            return error.description
        }
    }

    @Test("Rejects version 1, unknown adapters, undeclared slots or pipelines, raw secret keys, and leaked API tokens", arguments: [
        (
            """
            version: 1
            name: v1-bench
            models:
              chat: { source: mlx-community/gemma-4-12B-it-qat-4bit }
            pipeline: { answer: chat }
            benchmarks:
              smoke:
                - adapter: basalt:throughput
            """,
            "needs version: 2"
        ),
        (
            """
            version: 2
            name: bad-adapter
            models:
              chat: { source: mlx-community/gemma-4-12B-it-qat-4bit }
            pipeline: { answer: chat }
            benchmarks:
              smoke:
                - adapter: unknown:bench
            """,
            "unknown benchmark adapter"
        ),
        (
            """
            version: 2
            name: missing-slot
            models:
              chat: { source: mlx-community/gemma-4-12B-it-qat-4bit }
            pipeline: { answer: chat }
            benchmarks:
              smoke:
                - adapter: basalt:throughput
                  target: model:ghost
            """,
            "model ghost is not declared under models"
        ),
        (
            """
            version: 2
            name: missing-pipeline
            models:
              chat: { source: mlx-community/gemma-4-12B-it-qat-4bit }
            pipeline: { answer: chat }
            benchmarks:
              smoke:
                - adapter: basalt:mom
                  target: pipeline:nonexistent
            """,
            "pipeline nonexistent is not declared under pipelines"
        ),
        (
            """
            version: 2
            name: leaked-sk
            models:
              chat: { source: mlx-community/gemma-4-12B-it-qat-4bit }
            pipeline: { answer: chat }
            benchmarks:
              smoke:
                - adapter: arc-agi
                  submit: { target: arc-prize, api_key_env: sk-123456 }
            """,
            "secret"
        ),
        (
            """
            version: 2
            name: leaked-hf
            models:
              chat: { source: mlx-community/gemma-4-12B-it-qat-4bit }
            pipeline: { answer: chat }
            benchmarks:
              smoke:
                - adapter: lm-eval
                  submit: { target: hf-leaderboard, api_key_env: hf_secretToken99 }
            """,
            "raw secret token"
        ),
        (
            """
            version: 2
            name: raw-api-key-field
            models:
              chat: { source: mlx-community/gemma-4-12B-it-qat-4bit }
            pipeline: { answer: chat }
            benchmarks:
              smoke:
                - adapter: arc-agi
                  submit: { target: arc-prize, api_key: mysecret }
            """,
            "secrets never go in a compose file"
        ),
    ])
    func rejectsInvalidConfigs(_ yaml: String, _ expectedReason: String) {
        let message = Self.refusal(yaml)
        #expect(message.contains(expectedReason), "Expected '\(expectedReason)' in '\(message)'")
        #expect(message.contains("bad.yaml:"), "Expected file and line number in '\(message)'")
    }
}

@Suite("ARC-AGI pure-Swift grid parser, 2-attempt scorer, and ARC-Prize validator (#492)")
struct ARCAGIGridAndScoringTests {
    @Test("Extracts two 2D integer grids from reasoning traces, fenced JSON objects, and fenced arrays")
    func extractsGridsAndScoresTwoAttempts() {
        let reply = """
            <think>Flipping 0 and 1 across the 2x2 matrix.</think>
            ```json
            {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[1, 1], [0, 1]]}
            ```
            """
        let attempts = ARCGridParser.extractAttemptGrids(from: reply)
        #expect(attempts.count == 2)
        let hit = ARCGridParser.scoreAttempts(attempts, expected: [[1, 1], [0, 1]])
        #expect(hit.passed && hit.score == 1.0)
        let miss = ARCGridParser.scoreAttempts(attempts, expected: [[9, 9], [9, 9]])
        #expect(!miss.passed && miss.score == 0.0)
    }

    @Test("ARC-Prize submission validator rejects ragged rows and out-of-range cell integers", arguments: [
        ([[1, 2], [3]], "ragged row"),
        ([[10, 0], [0, 1]], "outside 0...9"),
        ([[-1, 2], [3, 4]], "outside 0...9"),
        ([[Int]](), "grid height"),
    ])
    func rejectsMalformedARCGrids(_ badGrid: [[Int]], _ reason: String) {
        var sample = BenchmarkSampleRecord(id: "task-bad", adapter: "arc-agi", answer: "[]", score: 0)
        sample.gridAttempts = [badGrid, [[0, 1], [1, 0]]]
        let artifact = SubmissionTestFixture.makeArtifact(samples: [sample])
        let plan = SubmissionFormatter.build(target: .arcPrize, artifact: artifact)
        #expect(!plan.isValid)
        #expect(plan.validationErrors.contains { $0.contains(reason) })
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("arc-bad-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }
        #expect(throws: BenchmarkError.self) {
            try BenchmarkSubmitter.exportDryRun(target: .arcPrize, artifact: artifact, outputDirectory: tempDir)
        }
    }
}

enum SubmissionTestFixture {
    static func makeRichSample() -> BenchmarkSampleRecord {
        var sample = BenchmarkSampleRecord(
            id: "django__django-11099", adapter: "arc-agi", model: "mlx-community/gemma-4-12B-it-qat-4bit",
            prompt: "Solve task", answer: "Reasoning step.\nFINAL ANSWER: 42", score: 1.0
        )
        sample.expected = "42"
        sample.category = "multiple"
        sample.reasoningTrace = "Checked boundary conditions and derived 42."
        sample.codeCompletions = ["def solve(x):\n    return x + 1\n"]
        sample.modelPatch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x=0\n+x=1\n"
        sample.gridAttempts = [[[1, 0], [0, 1]], [[1, 1], [0, 1]]]
        sample.toolCalls = [BenchmarkToolCallRecord(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Zurich\"}")]
        sample.trajectory = [
            BenchmarkTrajectoryStep(role: "user", content: "Solve task"),
            BenchmarkTrajectoryStep(role: "assistant", content: "FINAL ANSWER: 42"),
        ]
        sample.latencyMs = 142.5
        sample.ttftMs = 28.0
        sample.promptTokens = 120
        sample.completionTokens = 48
        sample.decodeTPS = 54.2
        sample.costUSD = 0.0012
        sample.passed = true
        return sample
    }

    static let defaultMetrics: [String: Double] = [
        "decode_tps": 54.2, "peak_gb": 14.2, "accuracy": 1.0,
        "p50_ms": 142.5, "ppl": 5.4, "ece": 0.03,
    ]

    static func makeArtifact(
        runID: String = "run-fixture-1",
        metrics: [String: Double] = defaultMetrics,
        gates: BenchmarkGatesSpec? = nil,
        samples: [BenchmarkSampleRecord]? = nil
    ) -> BenchmarkRunArtifact {
        let items = samples ?? [makeRichSample()]
        let manifest = BenchmarkManifest(runID: runID, composeName: "bench-desk", suite: "smoke", seed: 42, temperature: 0.0)
        var provenance = BasaltProvenance(
            gitCommit: "64dd68e", composeSHA256: BasaltProvenance.sha256Hex("version: 2\nname: bench-desk\n"),
            chip: "M4 Pro", unifiedMemoryGB: 24, timestampISO8601: "2026-11-15T12:00:00Z"
        )
        provenance.wallClockSeconds = 3.5
        let card = BenchmarkScorecard.evaluate(metrics: metrics, gates: gates, samples: items)
        return BenchmarkRunArtifact(manifest: manifest, provenance: provenance, scorecard: card, samples: items)
    }

    static func expectedPrimaryFile(for target: SubmissionTarget) -> String {
        switch target {
        case .arcPrize: "submission.json"
        case .terminalBench: "agent_metadata.json"
        case .sweBench: "preds.json"
        case .hfLeaderboard: "samples.jsonl"
        case .liveCodeBench: "custom_outputs.json"
        case .bfcl: "data_overall.csv"
        case .inspectAI: "header.json"
        case .tauBench: "results.json"
        case .gaia: "validation.jsonl"
        case .weaveOTel: "otel_eval_export.json"
        }
    }
}

@Suite("Top-10 leaderboard submission formatters and dry-run bundle exporter (#492)")
struct SubmissionFormatterAllTargetsTests {
    @Test("Every SubmissionTarget exports valid, non-empty required files, basalt_provenance.json, and a dry-run plan", arguments: SubmissionTarget.allCases)
    func exportsAllTenTargets(target: SubmissionTarget) throws {
        #expect(SubmissionTarget.allCases.count == 10)
        let artifact = SubmissionTestFixture.makeArtifact()
        let submitSpec = BenchmarkSubmitSpec(target: target, apiKeyEnv: "BENCH_API_KEY", subset: "verified", split: "validation")
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("submit-\(target.rawValue)-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let plan = try BenchmarkSubmitter.exportDryRun(
            target: target, artifact: artifact, submit: submitSpec, outputDirectory: tempDir
        )
        #expect(plan.isValid)
        #expect(plan.cliCommand?.isEmpty == false || plan.httpEndpoint?.isEmpty == false)
        let primary = SubmissionTestFixture.expectedPrimaryFile(for: target)
        #expect(plan.files.contains { $0.relativePath == primary && !$0.data.isEmpty })
        #expect(plan.files.contains { $0.relativePath == "basalt_provenance.json" && !$0.data.isEmpty })

        for file in plan.files {
            let diskURL = tempDir.appendingPathComponent(file.relativePath)
            let onDisk = try Data(contentsOf: diskURL)
            #expect(!onDisk.isEmpty && onDisk == file.data)
        }
        let summaryURL = tempDir.appendingPathComponent("submission_plan.json")
        #expect(FileManager.default.fileExists(atPath: summaryURL.path))
    }
}

@Suite("BenchmarkScorecard gate checks and BenchmarkComparator regression diffing (#492)")
struct BenchmarkScorecardAndComparatorTests {
    @Test("Scorecard passes when all six gates hold and fails with negative margin when decode TPS or peak GB violates threshold")
    func evaluatesGatesAndMargins() {
        let strictGates = BenchmarkGatesSpec(
            minDecodeTPS: 40.0, maxPeakGB: 18.0, minAccuracy: 0.9,
            maxP50Ms: 250.0, maxPPL: 8.0, maxECE: 0.08
        )
        let passing = SubmissionTestFixture.makeArtifact(gates: strictGates)
        #expect(passing.scorecard.passed)
        #expect(passing.scorecard.gates.count == 6)
        #expect(passing.scorecard.gates.allSatisfy { ($0.margin ?? -1) >= 0 })

        let failingMetrics = ["decode_tps": 22.0, "peak_gb": 21.5, "accuracy": 1.0, "p50_ms": 140.0, "ppl": 5.0, "ece": 0.02]
        let failing = SubmissionTestFixture.makeArtifact(metrics: failingMetrics, gates: strictGates)
        #expect(!failing.scorecard.passed)
        let failedNames = Set(failing.scorecard.gates.filter { !$0.passed }.map(\.gate))
        #expect(failedNames == ["min_decode_tps", "max_peak_gb"])
    }

    @Test("BenchmarkComparator flags decode_tps drop and peak_gb spike as regressions while celebrating p50_ms speedup")
    func comparesBaselineAndCandidateArtifacts() {
        let baseline = SubmissionTestFixture.makeArtifact(
            runID: "base-1",
            metrics: ["decode_tps": 60.0, "peak_gb": 14.0, "p50_ms": 200.0, "accuracy": 1.0]
        )
        var failedSample = SubmissionTestFixture.makeRichSample()
        failedSample.passed = false
        failedSample.score = 0.0
        let candidate = SubmissionTestFixture.makeArtifact(
            runID: "cand-2",
            metrics: ["decode_tps": 45.0, "peak_gb": 17.5, "p50_ms": 150.0, "accuracy": 0.0],
            samples: [failedSample]
        )
        let report = BenchmarkComparator.compare(baseline: baseline, candidate: candidate)
        #expect(report.hasRegressions && report.hardwareMatched && !report.composeChanged)
        #expect(report.sampleRegressions == ["django__django-11099"])
        let byMetric = Dictionary(uniqueKeysWithValues: report.deltas.map { ($0.metric, $0) })
        #expect(byMetric["decode_tps"]?.regressed == true)
        #expect(byMetric["peak_gb"]?.regressed == true)
        #expect(byMetric["p50_ms"]?.improved == true)
        #expect(report.markdownTable.contains("REGRESSED") && report.markdownTable.contains("IMPROVED"))
    }
}

@Suite("End-to-end Tier A native adapters, ARC-AGI runner, GPULock contention, and CLI commands (#492)")
struct NativeAdapterAndGPULockTests {
    static let suiteYAML = """
        version: 2
        name: e2e-mock-suite
        models:
          chat: { source: mlx-community/gemma-4-12B-it-qat-4bit }
        pipeline:
          decide: { cascade: [chat] }
          answer: chat
        benchmarks:
          defaults:
            seed: 99
            temperature: 0.0
            gpu_lock: true
          smoke:
            - adapter: basalt:throughput
              target: model:chat
              params: { iterations: 2, max_tokens: 32 }
              gates: { min_decode_tps: 10.0, max_peak_gb: 20.0 }
            - adapter: basalt:system-one
              target: pipeline
              gates: { min_accuracy: 0.9, max_ece: 0.25 }
            - adapter: arc-agi
              target: model:chat
              gates: { min_accuracy: 1.0 }
        """

    static func makeMockEndpoint() -> BenchmarkEndpoint {
        BenchmarkEndpoint.mock { request in
            switch request.path {
            case "/v1/telemetry":
                let body = try JSONSerialization.data(withJSONObject: ["gpuPeakBytes": 12_500_000_000.0])
                return BenchmarkHTTPResponse(body: body, totalMs: 1.0)
            case "/v1/decide":
                let reqJSON = (request.body.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }) ?? [:]
                let state = (reqJSON["state"] as? String) ?? ""
                let choice = state.contains("Swift 6") ? "code" : "chat"
                let payload: [String: Any] = [
                    "model": "self",
                    "answers": ["route": ["choice": choice, "confidence": 0.95, "abstain": false]],
                    "usage": ["prefix_tokens": 24, "cost": 0.0],
                ]
                let body = try JSONSerialization.data(withJSONObject: payload)
                return BenchmarkHTTPResponse(body: body, totalMs: 8.0)
            default:
                let reqJSON = (request.body.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }) ?? [:]
                let msgs = (reqJSON["messages"] as? [[String: Any]]) ?? []
                let prompt = (msgs.first?["content"] as? String) ?? ""
                let replyText = prompt.contains("ARC-AGI")
                    ? "{\"attempt_1\": [[1, 1], [0, 1]], \"attempt_2\": [[0, 0], [1, 1]]}"
                    : "Apple Silicon unifies CPU and Metal GPU memory."
                let payload: [String: Any] = [
                    "choices": [["message": ["role": "assistant", "content": replyText]]],
                    "usage": ["prompt_tokens": 32, "completion_tokens": 24],
                    "decode_tps": 64.0,
                    "ttft_ms": 18.0,
                ]
                let body = try JSONSerialization.data(withJSONObject: payload)
                return BenchmarkHTTPResponse(body: body, ttftMs: 18.0, totalMs: 25.0)
            }
        }
    }

    @Test("BenchmarkRunner runs basalt:throughput, basalt:system-one, and arc-agi end-to-end against mock endpoint under GPULock")
    func runsNativeAndARCAdaptersEndToEnd() async throws {
        let spec = try ComposeFile.parse(Self.suiteYAML, file: "mock.yaml")
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("bench-run-\(UUID().uuidString)")
        let lockPath = tempDir.appendingPathComponent("gpu.lock").path
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        var request = BenchmarkSuiteRunRequest(
            spec: spec, composeYAML: Self.suiteYAML, suiteName: "smoke",
            endpoint: Self.makeMockEndpoint(), runID: "run-e2e", outputDirectory: tempDir
        )
        request.gpuLockPath = lockPath
        request.gitCommit = "64dd68e"

        let artifact = try await BenchmarkRunner().run(request)
        #expect(artifact.scorecard.passed)
        #expect(artifact.provenance.gpuLockAcquired)
        #expect(artifact.samples.count == 5)
        #expect(artifact.scorecard.metrics["decode_tps"] == 64.0)
        #expect(artifact.scorecard.metrics["peak_gb"] == 12.5)
        #expect(artifact.scorecard.metrics["accuracy"] == 1.0)
        let savedURL = tempDir.appendingPathComponent("run-e2e/artifact.json")
        let decoded = try JSONDecoder().decode(BenchmarkRunArtifact.self, from: Data(contentsOf: savedURL))
        #expect(decoded.manifest.runID == "run-e2e" && decoded.provenance.composeSHA256 == artifact.provenance.composeSHA256)
    }

    @Test("GPULock refuses concurrent non-blocking acquisition while held and succeeds once released; CLI commands parse")
    func gpuLockContentionAndCLIParsing() throws {
        let lockFile = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-lock-\(UUID().uuidString)").path
        defer { try? FileManager.default.removeItem(atPath: lockFile) }

        let lease = try GPULock.tryAcquire(path: lockFile)
        let firstHeld = lease.isHeld
        #expect(firstHeld)
        #expect(throws: BenchmarkError.self) {
            _ = try GPULock.tryAcquire(path: lockFile)
        }
        lease.release()
        var second = try GPULock.tryAcquire(path: lockFile)
        let secondHeld = second.isHeld
        #expect(secondHeld)
        second.close()

        let composeBench = try ComposeCommand.Bench.parse(["-f", "basalt.yaml", "--suite", "smoke", "--submit", "arc-prize,weave-otel", "--submit-dry-run"])
        #expect(composeBench.suite == "smoke" && composeBench.submitDryRun)
        let benchSubmit = try BenchCommand.Submit.parse(["artifact.json", "--target", "swe-bench", "--dry-run"])
        #expect(benchSubmit.target == "swe-bench" && benchSubmit.dryRun)
        #expect(BenchmarkAdapterRegistry.standard.allDescriptors.count == BenchmarkAdapterID.known.count)
    }
}
