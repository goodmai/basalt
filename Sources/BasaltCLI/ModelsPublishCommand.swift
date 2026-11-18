import ArgumentParser
import BasaltCore
import Foundation

struct PublishSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "publish",
        abstract: "Publish a local or cached MLX checkpoint directory to Hugging Face Hub"
    )

    @Argument(help: "Cached repo id (org/name) or local checkpoint directory path")
    var source: String

    @Option(name: .long, help: "Target Hugging Face repository id (owner/name)")
    var repo: String

    @Flag(name: .long, help: "Create repository as private")
    var `private`: Bool = false

    @Option(name: .shortAndLong, help: "Commit message")
    var message: String = "Upload MLX checkpoint via Basalt"

    @Option(name: .long, help: "Optional Hugging Face write token (defaults to HF_TOKEN / ~/.cache/huggingface/token)")
    var token: String?

    @Option(name: .long, help: "Maximum concurrent file uploads")
    var concurrency: Int = 3

    mutating func run() async throws {
        guard HuggingFaceHub.isRepoId(repo), repo.contains("/") else {
            throw ValidationError("Target --repo must be a valid owner/name Hugging Face repository id.")
        }
        let directory = try Self.resolveSourceDirectory(source)
        let reporter = DownloadProgressReporter()
        let hub = HuggingFaceHub()
        let opts = HubPublishOptions(
            repoId: repo,
            isPrivate: `private`,
            commitMessage: message,
            token: token,
            maxConcurrent: concurrency
        )
        let result = try await hub.publishDirectory(
            at: directory,
            options: opts,
            onFile: reporter.callAsFunction
        )
        print()
        print("Published \(result.uploadedFiles) files (\(ByteCountFormatter.string(fromByteCount: result.totalBytes, countStyle: .file))) -> \(result.url)")
    }

    static func resolveSourceDirectory(_ raw: String) throws -> URL {
        let expanded = NSString(string: raw).expandingTildeInPath
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir), isDir.boolValue {
            return URL(fileURLWithPath: expanded)
        }
        if let resolved = ModelCache.resolve(repoId: raw) {
            return resolved
        }
        throw ValidationError("Could not find model directory or cached checkpoint for '\(raw)'.")
    }
}
