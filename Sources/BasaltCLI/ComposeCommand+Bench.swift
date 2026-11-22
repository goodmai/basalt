import Foundation
import ArgumentParser
import BasaltCore
import BasaltBench

extension ComposeCommand {
    struct Bench: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "bench",
            abstract: "Run a declarative benchmark suite from a compose v2 file, check gates, and optionally export dry-run leaderboard submission bundles (#492)"
        )

        @OptionGroup var source: Source

        @Option(name: .customLong("suite"), help: "Name of the benchmark suite under benchmarks")
        var suite: String

        @Option(name: .customLong("endpoint"), help: "Base URL of a running basalt server (defaults to the running compose system or http://127.0.0.1:8080)")
        var endpoint: String?

        @Option(name: .customLong("submit"), help: "Submission target(s), comma-separated or repeatable")
        var submit: [String] = []

        @Flag(name: .customLong("submit-dry-run"), help: "Validate and write leaderboard submission bundles locally with zero network egress")
        var submitDryRun = false

        @Option(name: .customLong("run-id"), help: "Explicit run identifier for artifacts and submission bundles")
        var runID: String?

        @Option(name: .customLong("output-dir"), help: "Directory where run artifacts and submission bundles are written")
        var outputDir: String?

        func run() async throws {
            let (spec, text, _, _) = try source.load()
            let baseURL = try resolveBaseURL(spec: spec)
            let resolvedRunID = runID ?? "bench-\(suite)-\(Int(Date().timeIntervalSince1970))"
            try Self.validateRunID(resolvedRunID)
            let rawOut = outputDir ?? spec.benchmarks?.defaults?.outputDir ?? ".basalt/bench"
            let rootOut = URL(
                fileURLWithPath: (rawOut as NSString).expandingTildeInPath,
                isDirectory: true
            )
            let targets = try resolveSubmissionTargets(spec: spec)
            var request = BenchmarkSuiteRunRequest(
                spec: spec,
                composeYAML: text,
                suiteName: suite,
                endpoint: BenchmarkEndpoint.live(
                    baseURL: baseURL,
                    apiKey: ProcessInfo.processInfo.environment["BASALT_API_KEY"]
                ),
                runID: resolvedRunID,
                outputDirectory: rootOut
            )
            request.gitCommit = Self.currentGitCommit()
            let artifact = try await BenchmarkRunner().run(request)
            printSummary(artifact, rootOut: rootOut)
            if submitDryRun || !targets.isEmpty {
                try exportSubmissions(targets: targets, spec: spec, artifact: artifact, rootOut: rootOut)
            }
            if !artifact.scorecard.passed {
                throw ExitCode.failure
            }
        }

        func resolveBaseURL(spec: ComposeSpec) throws -> URL {
            if let endpoint {
                guard let url = URL(string: endpoint) else {
                    throw ValidationError("invalid --endpoint URL: \(endpoint)")
                }
                return url
            }
            if let running = ComposeRecord.running(spec.name),
               let url = URL(string: "http://127.0.0.1:\(running.port)") {
                return url
            }
            return URL(string: "http://127.0.0.1:8080")!
        }

        func resolveSubmissionTargets(spec: ComposeSpec) throws -> [SubmissionTarget] {
            let explicit = submit.flatMap { $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } }.filter { !$0.isEmpty }
            if !explicit.isEmpty {
                return try explicit.map { raw in
                    guard let parsed = SubmissionTarget(rawValue: raw) else {
                        let allowed = SubmissionTarget.allCases.map(\.rawValue).joined(separator: ", ")
                        throw ValidationError("unknown --submit target '\(raw)'; expected one of \(allowed)")
                    }
                    return parsed
                }
            }
            guard submitDryRun, let suiteSpec = spec.benchmarks?.suites[suite] else { return [] }
            var seen: [SubmissionTarget] = []
            for task in suiteSpec.tasks {
                if let t = task.submit?.target, !seen.contains(t) { seen.append(t) }
            }
            return seen
        }

        func exportSubmissions(
            targets: [SubmissionTarget], spec: ComposeSpec,
            artifact: BenchmarkRunArtifact, rootOut: URL
        ) throws {
            let suiteSpec = spec.benchmarks?.suites[suite]
            let runDir = rootOut.appendingPathComponent(artifact.manifest.runID, isDirectory: true)
            for target in targets {
                let matchingSpec = suiteSpec?.tasks.compactMap(\.submit).first { $0.target == target }
                let bundleDir = runDir.appendingPathComponent("submissions/\(target.rawValue)", isDirectory: true)
                let plan = try BenchmarkSubmitter.exportDryRun(
                    target: target, artifact: artifact, submit: matchingSpec, outputDirectory: bundleDir
                )
                print("submission [dry-run] \(target.rawValue) -> \(bundleDir.path)")
                for file in plan.files { print("  file: \(file.relativePath) (\(file.data.count) bytes)") }
                if let cli = plan.cliCommand { print("  cli: \(cli.joined(separator: " "))") }
                if let http = plan.httpEndpoint { print("  endpoint: \(http)") }
            }
        }

        static func validateRunID(_ id: String) throws {
            guard !id.isEmpty, id != ".", id != "..",
                  !id.contains("/"), !id.contains("\\"), !id.contains("\0") else {
                throw ValidationError("Invalid run id '\(id)': must be a single path component without '/', '\\', or '..'")
            }
        }

        func printSummary(_ artifact: BenchmarkRunArtifact, rootOut: URL) {
            let card = artifact.scorecard
            let artifactPath = rootOut.appendingPathComponent("\(artifact.manifest.runID)/artifact.json").path
            print("compose \(artifact.manifest.composeName) suite \(artifact.manifest.suite): \(card.passed ? "PASSED" : "FAILED") (\(card.passedSamples)/\(card.totalSamples) samples)")
            for key in card.metrics.keys.sorted() {
                if let val = card.metrics[key] { print(String(format: "  metric %@ = %.4f", key, val)) }
            }
            for gate in card.gates {
                let obs = gate.observed.map { String(format: "%.4f", $0) } ?? "missing"
                print("  gate \(gate.gate) (\(gate.comparator) \(gate.threshold)): observed \(obs) -> \(gate.passed ? "PASS" : "FAIL")")
            }
            print("  artifact: \(artifactPath)")
        }

        static func currentGitCommit() -> String {
            guard let head = try? String(contentsOfFile: ".git/HEAD", encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines) else {
                return "unknown"
            }
            if head.hasPrefix("ref: ") {
                let refPath = ".git/" + head.dropFirst("ref: ".count)
                if let sha = try? String(contentsOfFile: refPath, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines), !sha.isEmpty {
                    return sha
                }
            }
            return head
        }
    }
}
