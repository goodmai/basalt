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
        subcommands: [Plan.self, Up.self, Ps.self, Down.self])

    struct Source: ParsableArguments {
        @Option(name: [.short, .customLong("file")], help: "The compose file")
        var file: String?

        @Argument(help: "A bundled compose file instead of -f: \(ComposeFile.bundled.keys.sorted().joined(separator: ", "))")
        var bundled: String?

        func validate() throws {
            guard (file == nil) != (bundled == nil) else { throw ValidationError("give either -f <file> or the name of a bundled compose file") }
        }

        func yaml() throws -> String {
            if let bundled, let text = ComposeFile.bundled[bundled] { return text }
            guard let file, let data = FileManager.default.contents(atPath: file) else { throw ExitCode.failure }
            return String(decoding: data, as: UTF8.self)
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
            if let running = ComposeRecord.running(spec.name) {
                let (status, body) = try await ComposeCommand.call(running, "/v1/compose/apply", body: try source.yaml())
                guard status == 200, let diff = try? JSONDecoder().decode(ComposeDiff.self, from: body) else {
                    ComposeCommand.log("\(red("[error]")) \(spec.name) is running (pid \(running.pid)) and refused the change: \(ComposeCommand.message(body))")
                    throw ExitCode.failure
                }
                print("compose \(spec.name): applied to the running system (pid \(running.pid), port \(running.port))")
                for line in diff.lines { print(line) }
                return
            }
            ComposeCommand.log("Compose \(bold(spec.name)) → basalt serve \(spec.serveArguments.joined(separator: " "))")
            var command = try ServeCommand.parse(spec.serveArguments + ["--compose", source.file ?? source.bundled ?? ""] + serve)
            try await command.run()
        }
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
            kill(record.pid, SIGINT)
            for _ in 0 ..< 300 where kill(record.pid, 0) == 0 { try await Task.sleep(for: .milliseconds(100)) }
            print(kill(record.pid, 0) == 0 ? "\(name) (pid \(record.pid)) is still stopping" : "\(name) stopped")
        }
    }

    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    static func call(_ record: ComposeRecord, _ path: String, body: String? = nil) async throws -> (Int, Data) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(record.port)\(path)")!)
        request.timeoutInterval = 1800
        if let body {
            (request.httpMethod, request.httpBody) = ("POST", try JSONEncoder().encode(["yaml": body]))
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
            let state = model.state.padding(toLength: 10, withPad: " ", startingAt: 0)
            return "  \(name) \(state) \(model.keep.padding(toLength: 10, withPad: " ", startingAt: 0)) \(model.source)" + (notes.isEmpty ? "" : "  (\(notes.joined(separator: "; ")))")
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
