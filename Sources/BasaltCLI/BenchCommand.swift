import Foundation
import ArgumentParser
import BasaltCore
import BasaltBench

struct BenchCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Inspect benchmark adapters, compare run artifacts, and export verified leaderboard submission bundles (#492)",
        subcommands: [List.self, Compare.self, Submit.self]
    )

    struct List: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "list",
            abstract: "List registered benchmark adapters and leaderboard submission targets"
        )

        @Flag(name: .customLong("json"), help: "Emit adapters and submission targets as JSON")
        var json = false

        func run() async throws {
            let descriptors = BenchmarkAdapterRegistry.standard.allDescriptors
            let targets = SubmissionTarget.allCases.map(\.rawValue)
            if json {
                let payload: [String: Any] = [
                    "adapters": descriptors.map { [
                        "id": $0.id,
                        "tier": $0.tier.rawValue,
                        "summary": $0.summary,
                        "default_submission_target": $0.defaultSubmissionTarget?.rawValue ?? "",
                    ] },
                    "submission_targets": targets,
                ]
                let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
                print(String(decoding: data, as: UTF8.self))
                return
            }
            print("Adapters (\(descriptors.count)):")
            for item in descriptors {
                let id = item.id.padding(toLength: 20, withPad: " ", startingAt: 0)
                let tier = item.tier.rawValue.padding(toLength: 16, withPad: " ", startingAt: 0)
                print("  \(id) \(tier) \(item.summary)")
            }
            print("\nSubmission targets (\(targets.count)): \(targets.joined(separator: ", "))")
        }
    }

    struct Compare: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "compare",
            abstract: "Compare two BenchmarkRunArtifact JSON files and report metric deltas and sample regressions"
        )

        @Argument(help: "Path to baseline artifact.json")
        var baselinePath: String

        @Argument(help: "Path to candidate artifact.json")
        var candidatePath: String

        @Option(name: .customLong("tolerance"), help: "Relative change tolerance before flagging improvement/regression (default: 0.02)")
        var tolerance: Double = 0.02

        @Flag(name: .customLong("json"), help: "Print comparison report as JSON")
        var json = false

        @Flag(name: .customLong("fail-on-regression"), help: "Exit non-zero when any metric or sample regresses")
        var failOnRegression = false

        func run() async throws {
            let baseline = try BenchCommand.loadArtifact(baselinePath)
            let candidate = try BenchCommand.loadArtifact(candidatePath)
            let report = BenchmarkComparator.compare(
                baseline: baseline, candidate: candidate, relativeTolerance: tolerance
            )
            if json {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                print(String(decoding: try encoder.encode(report), as: UTF8.self))
            } else {
                print(report.markdownTable)
            }
            if failOnRegression, report.hasRegressions {
                throw ExitCode.failure
            }
        }
    }

    struct Submit: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "submit",
            abstract: "Validate a BenchmarkRunArtifact and export a dry-run leaderboard submission bundle"
        )

        @Argument(help: "Path to artifact.json")
        var artifactPath: String

        @Option(name: .customLong("target"), help: "Submission target (\(SubmissionTarget.allCases.map(\.rawValue).joined(separator: ", ")))")
        var target: String

        @Option(name: .customLong("output-dir"), help: "Destination directory for the submission bundle")
        var outputDir: String?

        @Option(name: .customLong("api-key-env"), help: "Environment variable name holding the target credential")
        var apiKeyEnv: String?

        @Option(name: .customLong("endpoint"), help: "Optional custom submission endpoint URL")
        var endpoint: String?

        @Option(name: .customLong("subset"), help: "Optional benchmark subset or repository identifier")
        var subset: String?

        @Option(name: .customLong("split"), help: "Optional dataset split (e.g. validation or test)")
        var split: String?

        @Flag(name: .customLong("dry-run"), help: "Export verified submission files and print upload plan without network egress")
        var dryRun = false

        func run() async throws {
            guard let parsedTarget = SubmissionTarget(rawValue: target) else {
                let allowed = SubmissionTarget.allCases.map(\.rawValue).joined(separator: ", ")
                throw ValidationError("unknown --target '\(target)'; expected one of \(allowed)")
            }
            if let apiKeyEnv, let issue = BenchmarkSecretGuard.apiKeyEnvIssue(apiKeyEnv, path: "--api-key-env") {
                throw ValidationError(issue.message)
            }
            let artifact = try BenchCommand.loadArtifact(artifactPath)
            let submitSpec = BenchmarkSubmitSpec(
                target: parsedTarget, apiKeyEnv: apiKeyEnv,
                endpoint: endpoint, subset: subset, split: split
            )
            let defaultOut = URL(fileURLWithPath: artifactPath).deletingLastPathComponent()
                .appendingPathComponent("submissions/\(parsedTarget.rawValue)", isDirectory: true)
            let destination = outputDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath, isDirectory: true) } ?? defaultOut
            let plan = try BenchmarkSubmitter.exportDryRun(
                target: parsedTarget, artifact: artifact, submit: submitSpec, outputDirectory: destination
            )
            _ = dryRun
            print("exported \(parsedTarget.rawValue) dry-run bundle to \(destination.path)")
            for file in plan.files {
                print("  \(file.relativePath) (\(file.data.count) bytes)")
            }
            if let cli = plan.cliCommand {
                print("  command: \(cli.joined(separator: " "))")
            }
            if let http = plan.httpEndpoint {
                print("  endpoint: \(http)")
            }
        }
    }

    static func loadArtifact(_ path: String) throws -> BenchmarkRunArtifact {
        let expanded = (path as NSString).expandingTildeInPath
        let data = try Data(contentsOf: URL(fileURLWithPath: expanded))
        return try JSONDecoder().decode(BenchmarkRunArtifact.self, from: data)
    }
}
