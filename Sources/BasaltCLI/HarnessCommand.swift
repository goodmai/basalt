import Foundation
import ArgumentParser
import BasaltCore
import BasaltHarness
import BasaltInference

struct HarnessCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "harness",
        abstract: "Validate or optimize an agent harness (Claude Code, opencode, AGENTS.md)",
        subcommands: [HarnessValidateCommand.self, HarnessOptimizeCommand.self]
    )
}

struct HarnessSurfaceOptions: ParsableArguments {
    @Option(help: "Harness layout: claude | opencode | agents")
    var surface = "claude"

    @Option(name: .customLong("always-on-max-chars"), help: "Budget for the always-on files in characters")
    var alwaysOnMaxChars: Int?

    func spec() throws -> HarnessSurfaceSpec {
        guard let spec = HarnessSurfaceSpec.named(surface) else {
            throw ValidationError("Unknown surface '\(surface)'; expected one of: \(HarnessSurfaceSpec.all.map(\.name).joined(separator: ", "))")
        }
        return alwaysOnMaxChars.map(spec.withAlwaysOnMaxChars) ?? spec
    }
}

struct HarnessValidateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "validate",
        abstract: "Validate the harness under a repository root; exits 1 when it is invalid"
    )

    @Argument(help: "Repository root holding CLAUDE.md / AGENTS.md and the surface directory")
    var directory = "."

    @OptionGroup var surface: HarnessSurfaceOptions

    func run() async throws {
        let root = URL(fileURLWithPath: directory, isDirectory: true)
        let spec = try surface.spec()
        let loaded = try HarnessLoader.load(root: root, spec: spec, allowedRoots: [root])
        var report = HarnessValidator.validate(loaded.tree, spec: spec)
        report.warnings += loaded.skipped.map { "skipped \($0)" }
        try printJSON(report)
        if !report.isValid { throw ExitCode.failure }
    }
}

struct HarnessOptimizeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "optimize",
        abstract: "Run one Harness² propose–probe–compose pass on a local model and print the result as JSON; the harness on disk is not changed"
    )

    @Argument(help: "Repository root holding CLAUDE.md / AGENTS.md and the surface directory")
    var directory = "."

    @OptionGroup var surface: HarnessSurfaceOptions

    @Option(help: "Model repo id from the local cache, or a local model directory")
    var model: String

    @Option(help: "The task the harness should get better at")
    var task: String

    @Option(help: "Task id for the library entry and the prompt-cache namespace (default: derived from --task)")
    var slug: String?

    @Option(help: "Experience Library domain")
    var domain = "general"

    @Option(name: .customLong("require"), help: "A term the deliverable must contain (repeatable)")
    var requiredTerms: [String] = []

    @Flag(name: .customLong("expects-json"), help: "The deliverable must be a JSON object")
    var expectsJSON = false

    @Option(help: "Budget parameter k: 4k rollouts, 3k proposer calls")
    var k = 1

    @Option(name: .customLong("proposer-max-tokens"), help: "Token budget of each proposer reply (default: fits the largest editable file, 2048...12288)")
    var proposerMaxTokens: Int?

    @Option(help: "Experience Library directory (default ~/.basalt/harness-library)")
    var library: String?

    @Flag(name: .customLong("no-library"), help: "Do not read or write the Experience Library")
    var noLibrary = false

    func validate() throws {
        guard HarnessLedger.kRange.contains(k) else { throw ValidationError("--k must be in \(HarnessLedger.kRange)") }
        if let proposerMaxTokens, !HarnessOptimizer.TokenBudgets.proposerRange.contains(proposerMaxTokens) {
            throw ValidationError("--proposer-max-tokens must be in \(HarnessOptimizer.TokenBudgets.proposerRange)")
        }
    }

    func run() async throws {
        let spec = try surface.spec()
        let root = URL(fileURLWithPath: directory, isDirectory: true)
        let loaded = try HarnessLoader.load(root: root, spec: spec, allowedRoots: [root])
        let modelPath = FileManager.default.fileExists(atPath: model) ? model : ModelCache.resolve(repoId: model)?.path
        guard let modelPath else {
            throw ValidationError("Model '\(model)' is not in the local cache; run `basalt models download \(model)` first")
        }
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine())
        try await orchestrator.loadModel(path: modelPath)
        let libraryStore = noLibrary ? nil : HarnessExperienceLibrary(
            rootDirectory: library.map { URL(fileURLWithPath: $0, isDirectory: true) } ?? HarnessExperienceLibrary.defaultRoot
        )
        let harnessTask = HarnessTask(
            slug: HarnessExperienceLibrary.slugify(slug ?? String(task.prefix(48))),
            prompt: task, domain: domain, requiredTerms: requiredTerms, expectsJSON: expectsJSON
        )
        var tokens = HarnessOptimizer.TokenBudgets()
        tokens.proposer = proposerMaxTokens
        let result = try await HarnessOptimizer(orchestrator: orchestrator, spec: spec, library: libraryStore, tokens: tokens)
            .optimize(task: harnessTask, base: loaded.tree, k: k)
        try printJSON(result)
    }
}
