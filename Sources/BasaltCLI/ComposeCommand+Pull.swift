import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltServer
import Synchronization

extension ComposeCommand {
    struct Plan: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Validate a compose file and show its compute plan on this Mac, refusing a system whose largest stage does not fit or whose sizes are unknown")

        @OptionGroup var source: Source

        @Option(name: .customLong("assume-size"), help: "Count a model whose size is neither in the cache nor known to the Hugging Face API as this many GB, instead of refusing")
        var assumeSize: Double?

        func run() async throws {
            let spec = try source.spec()
            let remote = await ComposeCommand.remoteSizes(spec) { try await HuggingFaceHub().remoteWeightBytes(repoId: $0) }
            let sizes = ComposeCommand.sizes(remote: remote, assume: assumeSize.map { Int64($0 * 1_000_000_000) })
            let (verdict, unknown) = spec.plan(sizes, budgetBytes: await SystemProfiler().detectResources().gpuMemory, power: PowerSource.current())
            for line in ComposeCommand.lines(spec, verdict: verdict, unknown: unknown) { print(line) }
            if verdict.refusal != nil || !unknown.isEmpty { throw ExitCode.failure }
        }
    }

    struct Pull: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Download every local model of a compose file into the Hugging Face cache, verifying each file, so that plan has exact sizes and up loads without waiting for downloads")

        @OptionGroup var source: Source

        func run() async throws {
            let spec = try source.spec()
            let hub = HuggingFaceHub()
            let failed = await ComposeCommand.pull(spec) { repo, progress in _ = try await hub.download(repoId: repo) { _, done, total in progress(done, total) } }
            if !failed.isEmpty { throw ExitCode.failure }
        }
    }

    struct Diff: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Show what compose up would change on the running system with the same name, without changing it: loads, unloads, keys, budgets, a needed restart, or a refusal")

        @OptionGroup var source: Source

        func run() async throws {
            let (spec, text, _, context) = try source.load()
            guard let running = ComposeRecord.running(spec.name) else {
                return print("compose \(spec.name) is not running; compose up starts it (compose plan shows what it needs)")
            }
            let request = ComposeApply(yaml: text, base: context.base?.path, variables: context.variables, profiles: context.profiles)
            let (status, body) = try await ComposeCommand.call(running, "/v1/compose/diff", body: try JSONEncoder().encode(request))
            guard status == 200, let diff = try? JSONDecoder().decode(ComposeDiff.self, from: body) else {
                ComposeCommand.log("\(red("[error]")) \(spec.name) could not compute the difference: \(ComposeCommand.message(body))")
                throw ExitCode.failure
            }
            print("compose \(spec.name): compose up would change the running system (pid \(running.pid)) like this")
            for line in diff.lines { print(line) }
            if diff.restart != nil || diff.refusal != nil { throw ExitCode.failure }
        }
    }

    static func repos(_ spec: ComposeSpec) -> [String] {
        Set(spec.models.keys.compactMap { spec.local($0) }.filter { !$0.hasPrefix("/") && !$0.hasPrefix(".") && !$0.hasPrefix("~") && $0.contains("/") }).sorted()
    }

    static func remoteSizes(_ spec: ComposeSpec, fetch: @escaping @Sendable (String) async throws -> Int64?) async -> [String: Int64] {
        var sizes: [String: Int64] = [:]
        for repo in repos(spec) where Self.sizes.bytes(repo) == nil {
            if let bytes = try? await fetch(repo) { sizes[repo] = bytes }
        }
        return sizes
    }

    static func sizes(remote: [String: Int64], assume: Int64?) -> ComposeSpec.Sizes {
        let local = Self.sizes
        return ComposeSpec.Sizes(bytes: { local.bytes($0) ?? remote[$0] ?? assume }, onNeuralEngine: local.onNeuralEngine)
    }

    static func pull(_ spec: ComposeSpec, download: (String, @escaping @Sendable (Int64, Int64) -> Void) async throws -> Void) async -> [String] {
        var failed: [String] = []
        for repo in repos(spec) {
            log("pulling \(repo)")
            let shown = PullProgress()
            do {
                try await download(repo) { done, total in if let line = shown.step(done, total) { log("  \(repo): \(line)") } }
                log("  \(repo): ready")
            } catch {
                log("\(red("[error]")) \(repo): \(error.localizedDescription)")
                failed.append(repo)
            }
        }
        log(failed.isEmpty ? "pulled \(repos(spec).count) model(s); compose plan now has exact sizes" : "\(failed.count) model(s) failed: \(failed.joined(separator: ", "))")
        return failed
    }
}

final class PullProgress: Sendable {
    private let last = Mutex(-1)

    func step(_ done: Int64, _ total: Int64) -> String? {
        guard total > 0 else { return nil }
        let percent = Int(done * 100 / total)
        let shown = last.withLock { last -> Bool in
            guard percent / 10 > last / 10 || (percent == 100 && last != 100) else { return false }
            last = percent
            return true
        }
        return shown ? "\(percent)% of \(ByteCountFormatter.string(fromByteCount: total, countStyle: .file))" : nil
    }
}
