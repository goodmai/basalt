import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct SessionsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sessions",
        abstract: "Read the local session log without loading a model.",
        subcommands: [List.self, Show.self, Stats.self, Prune.self]
    )

    struct Location: ParsableArguments {
        @Option(name: .customLong("session-log-dir"), help: "Directory of the session log (default: ~/.basalt/sessions)")
        var directory: String?

        var url: URL {
            directory.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? SessionLogWriter.defaultDirectory
        }
    }

    static func resolve(_ id: String, in directory: URL) throws -> SessionLogReader.Session {
        let sessions = SessionLogReader.sessions(in: directory)
        let found = id == "latest" ? sessions.first : sessions.first { $0.id == id }
        guard let found else {
            throw ValidationError(id == "latest" ? "No sessions in \(directory.path)" : "No session \(id) in \(directory.path)")
        }
        return found
    }

    struct List: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "List sessions, newest first.")
        @OptionGroup var location: Location
        @Option(name: .long, help: "Show at most this many sessions") var limit: Int = 20

        mutating func run() async throws {
            let formatter = ISO8601DateFormatter()
            for session in SessionLogReader.sessions(in: location.url).prefix(limit) {
                let lines = SessionLogReader.lines(of: session.file)
                let requests = lines.filter { $0.type == "request" }.count
                print("\(session.id)\t\(formatter.string(from: session.modified))\t\(SessionLogReader.mode(of: lines) ?? "?")\t\(requests) requests")
            }
        }
    }

    struct Show: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Rebuild a session's latest conversation from the log.")
        @OptionGroup var location: Location
        @Argument(help: "Session id, or latest") var id: String = "latest"

        mutating func run() async throws {
            let session = try SessionsCommand.resolve(id, in: location.url)
            let lines = SessionLogReader.lines(of: session.file)
            print("session \(session.id) (\(SessionLogReader.mode(of: lines) ?? "?") mode) — \(session.file.path)")
            guard let exchange = SessionLogReader.lastExchange(in: lines) else {
                for line in lines where line.type == "response" {
                    let tokens = line.tokens.map { "\($0.prompt) in / \($0.completion) out" } ?? "-"
                    print("\(line.ts ?? "")\t\(line.status ?? "")\t\(tokens)\t\(line.output?.chars ?? 0) chars")
                }
                return
            }
            for message in exchange.messages {
                print("\n[\(message.role.rawValue)]\n\(message.text)")
            }
            if let reply = exchange.reply { print("\n[assistant]\n\(reply)") }
        }
    }

    struct Stats: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Totals for a session: requests, tokens, speed.")
        @OptionGroup var location: Location
        @Argument(help: "Session id, or latest") var id: String = "latest"

        mutating func run() async throws {
            let session = try SessionsCommand.resolve(id, in: location.url)
            let responses = SessionLogReader.lines(of: session.file).filter { $0.type == "response" }
            let ok = responses.filter { $0.status == "ok" }
            let prompt = responses.compactMap(\.tokens).reduce(0) { $0 + $1.prompt }
            let cached = responses.compactMap(\.tokens).reduce(0) { $0 + $1.cached }
            let completion = responses.compactMap(\.tokens).reduce(0) { $0 + $1.completion }
            let timings = ok.compactMap(\.timing)
            let tps = timings.isEmpty ? 0 : timings.map(\.tokensPerSecond).reduce(0, +) / Double(timings.count)
            let ttft = timings.isEmpty ? 0 : timings.map(\.ttftMs).reduce(0, +) / timings.count
            print("session \(session.id)")
            print("responses: \(responses.count) (\(ok.count) ok, \(responses.count - ok.count) other)")
            print("tokens: \(prompt) prompt (\(cached) cached), \(completion) completion")
            print(String(format: "mean: %.1f tok/s, %d ms to first token", tps, ttft))
        }
    }

    struct Prune: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Delete persisted prompt-cache files past their retention, beyond a size cap, and leftovers of interrupted writes.")
        @Option(name: .customLong("prompt-cache-dir"), help: "Directory of the persisted prompt cache (default: ~/.basalt/prompt-cache)")
        var directory: String?
        @Option(name: .customLong("prompt-cache-disk-days"), help: "Delete files unused for this many days (default: 7)")
        var retentionDays: Int = 7
        @Option(name: .customLong("prompt-cache-disk-mb"), help: "Then delete the least recently used files until at most this many MB remain (default: no cap; 0 deletes every file)")
        var megabytes: Int?

        func validate() throws {
            guard retentionDays >= 0, (megabytes ?? 0) >= 0 else {
                throw ValidationError("--prompt-cache-disk-days and --prompt-cache-disk-mb must not be negative")
            }
        }

        mutating func run() async throws {
            let url = PromptCacheDiskOptions.url(directory)
            guard FileManager.default.fileExists(atPath: url.path) else {
                print("No prompt cache in \(url.path)")
                return
            }
            let store = PromptCacheDiskStore(
                directory: url, byteQuota: megabytes.map { Int64(min($0, 1 << 30)) << 20 } ?? .max, retentionDays: retentionDays)
            let removed = try store.prune()
            print("Removed \(removed.files) files (\(removed.bytes >> 20) MB) from \(url.path)")
        }
    }
}
