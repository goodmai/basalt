import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltServer

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
        subcommands: [Config.self, Schema.self, Pull.self, Plan.self, Diff.self, Up.self, Ps.self, Down.self])

    struct Source: ParsableArguments {
        @Option(name: [.short, .customLong("file")], help: "The compose file")
        var file: String?

        @Argument(help: "A bundled compose file instead of -f: \(ComposeFile.bundled.keys.sorted().joined(separator: ", "))")
        var bundled: String?

        @Option(name: .customLong("set"), help: "Override a variable declared under variables: --set name=value (repeatable)")
        var assignments: [String] = []

        @Option(name: .customLong("profile"), help: "Turn on this profile instead of choosing profiles by their when (repeatable)")
        var profiles: [String] = []

        func validate() throws {
            guard (file == nil) != (bundled == nil) else { throw ValidationError("give either -f <file> or the name of a bundled compose file") }
            for assignment in assignments where !assignment.contains("=") { throw ValidationError("--set takes name=value, not \(assignment)") }
        }

        var reference: String { file ?? bundled ?? "" }

        var context: ComposeContext { ComposeContext.machine.overriding(assignments: assignments, profiles: profiles) }

        var serveFlags: [String] {
            ["--compose", reference] + assignments.flatMap { ["--compose-set", $0] } + profiles.flatMap { ["--compose-profile", $0] }
        }

        func source() throws -> (text: String, file: String, context: ComposeContext) {
            do {
                let source = try ComposeFile.read(reference)
                return (source.text, source.file, source.isFile ? context.with(base: reference) : context)
            } catch {
                ComposeCommand.log("\(red("[error]")) \(error.description)")
                throw ExitCode.failure
            }
        }

        func load() throws -> (spec: ComposeSpec, text: String, file: String, context: ComposeContext) {
            let source = try source()
            do {
                return (try ComposeFile.parse(source.text, file: source.file, context: source.context), source.text, source.file, source.context)
            } catch {
                ComposeCommand.log("\(red("[error]")) \(error.description)")
                throw ExitCode.failure
            }
        }

        func spec() throws -> ComposeSpec { try load().spec }
    }

    struct Config: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Print the compose file as it resolves on this Mac: the profiles that are on, with variables substituted")

        @OptionGroup var source: Source

        func run() async throws {
            let (_, text, name, context) = try source.load()
            let resolved = ComposeFile.resolve(text, root: try? ComposeFile.compose(text, file: name), context: context)
            let on = resolved.active.sorted().joined(separator: ", ")
            ComposeCommand.log("profiles on: \(on.isEmpty ? "none" : on) (declared: \(resolved.declared.sorted().joined(separator: ", ")))")
            print(resolved.text)
        }
    }

    struct Schema: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Print the JSON Schema of compose files, for editors and validators")

        func run() async throws { print(ComposeSchema.jsonSchema) }
    }

    struct Ps: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Show each running compose system: its pipelines and, per model, whether it is loaded, when an on-demand model was last used and unloads, and cloud spend today")

        @Argument(help: "The system name (default: every running system)")
        var name: String?

        func run() async throws {
            let records = name.map { ComposeRecord.running($0).map { [$0] } ?? [] } ?? ComposeRecord.all()
            guard !records.isEmpty else {
                print(name.map { "\($0) is not running" } ?? "no compose system is running")
                return
            }
            for record in records {
                let (status, body) = try await ComposeCommand.call(record, "/v1/compose")
                guard status == 200, let state = try? ComposeCommand.decoder.decode(ComposeStatus.self, from: body) else {
                    print("\(record.name): pid \(record.pid), port \(record.port), no status (\(ComposeCommand.message(body)))")
                    continue
                }
                for line in ComposeCommand.table(state, record: record) { print(line) }
            }
        }
    }

    struct Down: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Stop a running compose system: the server finishes in-flight requests and unloads its models")

        @Argument(help: "The system name")
        var name: String

        func run() async throws {
            guard let record = ComposeRecord.running(name) else {
                print("\(name) is not running")
                return
            }
            try await ComposeCommand.stop(record, grace: .seconds(30))
        }
    }

    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    static func call(_ record: ComposeRecord, _ path: String, body: Data? = nil) async throws -> (Int, Data) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(record.port)\(path)")!)
        request.timeoutInterval = 1800
        if let body {
            (request.httpMethod, request.httpBody) = ("POST", body)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        if let key = ProcessInfo.processInfo.environment["BASALT_API_KEY"] { request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
        let (data, response) = try await URLSession.shared.data(for: request)
        return ((response as? HTTPURLResponse)?.statusCode ?? 0, data)
    }

    static func message(_ body: Data) -> String {
        let json = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
        return ((json?["error"] as? [String: Any])?["message"] as? String) ?? String(decoding: body.prefix(300), as: UTF8.self)
    }

    static func table(_ state: ComposeStatus, record: ComposeRecord) -> [String] {
        let header = String(format: "%@  pid %d  port %d  pipelines %@  cloud today $%.4f", state.name, record.pid, record.port, state.pipelines.joined(separator: ", "), state.spentToday)
        return [header] + state.models.map { model in
            var notes: [String] = []
            if let used = model.lastUsed { notes.append("last used \(Int(Date().timeIntervalSince(used))) s ago") }
            if let idle = model.idleUnload { notes.append("unloads after \(idle) s idle") }
            if let spent = model.spentToday { notes.append(String(format: "$%.4f today", spent) + (model.limit.map { String(format: " of $%.2f", $0) } ?? "")) }
            if let refusal = model.refusal { notes.append(refusal) }
            let name = model.name.padding(toLength: 12, withPad: " ", startingAt: 0)
            let state = model.state.rawValue.padding(toLength: 13, withPad: " ", startingAt: 0)
            return "  \(name) \(state) \(model.keep.padding(toLength: 10, withPad: " ", startingAt: 0)) \(model.source)" + (notes.isEmpty ? "" : "  (\(notes.joined(separator: "; ")))")
        }
    }

    static func log(_ message: String) {
        fputs("[compose] \(message)\n", stderr)
    }

    static let sizes = ComposeSpec.Sizes(
        bytes: { source in
            guard let directory = ModelCache.resolve(repoId: source), FileManager.default.fileExists(atPath: directory.path) else { return nil }
            let bytes = ServeCommand.weightBytes(directory.path) + jevBaseBytes(directory)
            return bytes > 0 ? bytes : nil
        },
        onNeuralEngine: { source in ModelCache.resolve(repoId: source).map(CoreAIDecisionEngine.isBundle) ?? false })

    static func jevBaseBytes(_ directory: URL) -> Int64 {
        guard JevEvaluator.isBundle(directory), let base = try? JevEvaluator.baseModel(of: directory), let path = ModelCache.resolve(repoId: base) else { return 0 }
        return 2 * ServeCommand.weightBytes(path.path)
    }

    static func lines(_ spec: ComposeSpec, verdict: PlanVerdict, unknown: [String]) -> [String] {
        ["compose \(spec.name): basalt serve \(spec.serveArguments.joined(separator: " "))"] + ComputePlan.describe(verdict) + spec.onDemandLines + spec.cloudLines
            + unknown.map { "  \($0): size unknown — not downloaded, and the Hugging Face API did not say; run compose pull, or pass --assume-size <GB>" }
            + (verdict.refusal.map { ["refused: \($0)"] } ?? [])
    }
}
