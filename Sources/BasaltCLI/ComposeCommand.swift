import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

struct ComposeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "compose",
        abstract: "Bring up a model system described in one YAML file: local and cloud models, how each stays in memory, and the perceive → decide → answer pipeline (#412)",
        discussion: """
            A compose file names models (source: a repo, a local path, or cloud:openrouter/<model> with \
            api_key_env), how each is kept (resident, on-demand with idle_unload, or sequential) and the \
            pipeline: perceive.documents and perceive.audio (a model, apple-vision, speech-analyzer), \
            decide.cascade (a local model, or a cloud model followed by a local one) and answer. Secrets are \
            never written in the file; a cloud model names the environment variable that holds its key. \
            Bundled files: \(ComposeFile.bundled.keys.sorted().joined(separator: ", ")).
            """,
        subcommands: [Plan.self, Up.self])

    struct Source: ParsableArguments {
        @Option(name: [.short, .customLong("file")], help: "The compose file")
        var file: String?

        @Argument(help: "A bundled compose file instead of -f: \(ComposeFile.bundled.keys.sorted().joined(separator: ", "))")
        var bundled: String?

        func validate() throws {
            guard (file == nil) != (bundled == nil) else { throw ValidationError("give either -f <file> or the name of a bundled compose file") }
        }

        func spec() throws -> ComposeSpec {
            do {
                return try ComposeFile.load(file ?? bundled ?? "")
            } catch {
                ComposeCommand.log("\(red("[error]")) \(error.description)")
                throw ExitCode.failure
            }
        }
    }

    struct Plan: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Validate a compose file and show its compute plan on this Mac, refusing a system whose largest stage does not fit")

        @OptionGroup var source: Source

        func run() async throws {
            let spec = try source.spec()
            let (verdict, unknown) = spec.plan(ComposeCommand.sizes, budgetBytes: await SystemProfiler().detectResources().gpuMemory, power: PowerSource.current())
            for line in ComposeCommand.lines(spec, verdict: verdict, unknown: unknown) { print(line) }
            if verdict.refusal != nil { throw ExitCode.failure }
        }
    }

    struct Up: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Serve a compose file: the same REST and MCP surfaces as basalt serve, with the composed models; serve flags after -- are added")

        @OptionGroup var source: Source

        @Argument(parsing: .postTerminator, help: "More basalt serve flags, after --")
        var serve: [String] = []

        func run() async throws {
            let spec = try source.spec()
            ComposeCommand.log("Compose \(bold(spec.name)) → basalt serve \(spec.serveArguments.joined(separator: " "))")
            var command = try ServeCommand.parse(spec.serveArguments + serve)
            try await command.run()
        }
    }

    static func log(_ message: String) {
        fputs("[compose] \(message)\n", stderr)
    }

    static let sizes = ComposeSpec.Sizes(
        bytes: { source in
            guard let directory = ModelCache.resolve(repoId: source), FileManager.default.fileExists(atPath: directory.path) else { return nil }
            let bytes = ServeCommand.weightBytes(directory.path)
            return bytes > 0 ? bytes : nil
        },
        onNeuralEngine: { source in ModelCache.resolve(repoId: source).map(CoreAIDecisionEngine.isBundle) ?? false })

    static func lines(_ spec: ComposeSpec, verdict: PlanVerdict, unknown: [String]) -> [String] {
        ["compose \(spec.name): basalt serve \(spec.serveArguments.joined(separator: " "))"] + ComputePlan.describe(verdict) + spec.onDemandLines + spec.cloudLines
            + unknown.map { "  \($0): not downloaded, so its size is unknown and counted as 0; run basalt models download \(spec.local($0) ?? $0) for an exact plan" }
            + (verdict.refusal.map { ["refused: \($0)"] } ?? [])
    }
}
