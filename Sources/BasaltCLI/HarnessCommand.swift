import Foundation
import ArgumentParser
import BasaltCore
import BasaltHarness

struct HarnessCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "harness",
        abstract: "Validate an agent harness (Claude Code, opencode, AGENTS.md)",
        subcommands: [HarnessValidateCommand.self]
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
        let loaded = try HarnessLoader.load(root: root, spec: try surface.spec(), allowedRoots: [root])
        var report = HarnessValidator.validate(loaded.tree, spec: try surface.spec())
        report.warnings += loaded.skipped.map { "skipped \($0)" }
        try printJSON(report)
        if !report.isValid { throw ExitCode.failure }
    }
}
