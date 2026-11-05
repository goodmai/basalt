import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

struct ReviewCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "review",
        abstract: "Review a diff or a GitLab MR / Gitea PR with a local model; prints a local report and posts only with --post",
        discussion: """
        git diff origin/main | basalt review --diff - --model <id>
        GITLAB_TOKEN=… basalt review --gitlab https://gitlab.example --repo group/project --number 42 --model <id>
        GITEA_TOKEN=… basalt review --gitea https://gitea.example --repo owner/name --number 7 --model <id> --post
        """
    )

    @Option(help: "Unified diff file to review, or - for stdin")
    var diff: String?

    @Option(help: "Self-hosted GitLab instance URL (https, or http on loopback); token from GITLAB_TOKEN")
    var gitlab: String?

    @Option(help: "Self-hosted Gitea instance URL (https, or http on loopback); token from GITEA_TOKEN")
    var gitea: String?

    @Option(help: "GitLab project path or id, or Gitea owner/name")
    var repo: String?

    @Option(help: "Merge request iid (GitLab) or pull request index (Gitea)")
    var number: Int?

    @Option(help: "Model repo id from the local cache, or a local model directory")
    var model: String?

    @Option(name: .customLong("max-tokens"), help: "Token budget of the review")
    var maxTokens = 2048

    @Option(name: .customLong("context-chars"), help: "Character budget of the prompt; touched symbol signatures come first, then diff hunks until it is spent")
    var contextChars = 48_000

    @Flag(name: .customLong("print-context"), help: "Print the review prompt and exit without loading a model")
    var printContext = false

    @Flag(help: "Post the review as a comment on the MR/PR; without it nothing is sent to the remote")
    var post = false

    func validate() throws {
        let sources = [diff, gitlab, gitea].compactMap { $0 }.count
        guard sources == 1 else { throw ValidationError("Give exactly one of --diff, --gitlab or --gitea") }
        if diff == nil, repo == nil || number == nil { throw ValidationError("--gitlab and --gitea need --repo and --number") }
        if post, diff != nil { throw ValidationError("--post needs --gitlab or --gitea") }
        if model == nil, !printContext { throw ValidationError("Give --model, or --print-context to see the prompt only") }
        guard contextChars > 0, maxTokens > 0 else { throw ValidationError("--context-chars and --max-tokens must be positive") }
    }

    func resolvedRemote(environment: [String: String]) throws -> ReviewRemote? {
        guard let target = gitlab.map({ (ReviewRemote.Forge.gitlab, $0) }) ?? gitea.map({ (ReviewRemote.Forge.gitea, $0) }) else { return nil }
        let (forge, instance) = target
        do {
            return try ReviewRemote(forge: forge, instance: instance, repo: repo ?? "", number: number ?? 0,
                                    token: environment[forge.tokenVariable])
        } catch {
            throw ValidationError(error.description)
        }
    }

    static func readCapped(_ handle: FileHandle) throws -> String {
        let data = try handle.read(upToCount: ReviewRemote.maxDiffBytes + 1) ?? Data()
        guard data.count <= ReviewRemote.maxDiffBytes else {
            throw ValidationError("Diff exceeds \(ReviewRemote.maxDiffBytes) bytes")
        }
        return String(decoding: data, as: UTF8.self)
    }

    func loadDiff(remote: ReviewRemote?, session: URLSession) async throws -> String {
        if let remote { return try await remote.fetchDiff(session: session) }
        guard let diff else { return "" }
        if diff == "-" { return try Self.readCapped(.standardInput) }
        return try Self.readCapped(FileHandle(forReadingFrom: URL(fileURLWithPath: diff)))
    }

    static func review(prompt: String, orchestrator: ModelOrchestratorActor, maxTokens: Int) async throws -> String {
        var request = GenerationRequest(prompt: prompt, maxTokens: maxTokens, temperature: 0,
                                        messages: [ChatMessageInput(role: .user, text: prompt)])
        request.enableThinking = false
        request = request.withOrigin(RequestOrigin(source: "cli", endpoint: "review", stream: false))
        return try await orchestrator.generate(request: request).generatedText
    }

    func run() async throws {
        try await perform(session: .shared, environment: ProcessInfo.processInfo.environment)
    }

    func perform(session: URLSession, environment: [String: String]) async throws {
        let remote = try resolvedRemote(environment: environment)
        let prompt = ReviewContext.prompt(diff: try await loadDiff(remote: remote, session: session), budgetChars: contextChars)
        if printContext {
            print(prompt)
            return
        }
        guard let model, let modelPath = FileManager.default.fileExists(atPath: model) ? model : ModelCache.resolve(repoId: model)?.path else {
            throw ValidationError("Model '\(model ?? "")' is not in the local cache; run `basalt models download \(model ?? "")` first")
        }
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine())
        defer { await orchestrator.shutdown() }
        try await orchestrator.loadModel(path: modelPath)
        let report = try await Self.review(prompt: prompt, orchestrator: orchestrator, maxTokens: maxTokens)
        print(report)
        if post, let remote {
            try await remote.post(report, session: session)
            fputs("[review] posted to \(remote.forge.rawValue) \(remote.repo) #\(remote.number)\n", stderr)
        }
    }
}
