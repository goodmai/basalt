import Foundation
import ArgumentParser
import BasaltCore
import BasaltServer

extension ComposeCommand {
    struct Up: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Serve a compose file: the same REST and MCP surfaces as basalt serve, with the composed models; serve flags after -- are added. On a running system with the same name, apply the difference instead")

        @OptionGroup var source: Source

        @Flag(name: .customLong("wait"), help: "On a running system, wait until /v1/health/ready answers 200 after the change, and exit non-zero if it does not within --wait-timeout")
        var wait = false

        @Option(name: .customLong("wait-timeout"), help: "Seconds to wait for a loading system to take the change, and with --wait to become ready (default: 1800)")
        var waitTimeout = 1800

        @Argument(parsing: .postTerminator, help: "More basalt serve flags, after --")
        var serve: [String] = []

        func run() async throws {
            let (spec, text, _, context) = try source.load()
            if let running = ComposeRecord.running(spec.name) {
                let request = ComposeApply(yaml: text, base: context.base?.path, variables: context.variables, profiles: context.profiles)
                let deadline = ContinuousClock.now + .seconds(waitTimeout)
                let diff = try await ComposeCommand.apply(request, to: running, until: deadline)
                print("compose \(spec.name): applied to the running system (pid \(running.pid), port \(running.port))")
                for line in diff.lines { print(line) }
                if wait { try await ComposeCommand.waitReady(running, until: deadline) }
                return
            }
            try ComposeCommand.announce(spec, environment: ProcessInfo.processInfo.environment)
            var command = try ServeCommand.parse(spec.serveArguments + source.serveFlags + serve)
            try await command.run()
        }
    }

    static func apply(_ request: ComposeApply, to running: ComposeRecord, until deadline: ContinuousClock.Instant) async throws -> ComposeDiff {
        let body = try JSONEncoder().encode(request)
        var announced = false
        while true {
            let (status, reply) = try await call(running, "/v1/compose/apply", body: body)
            if status == 200, let diff = try? JSONDecoder().decode(ComposeDiff.self, from: reply) { return diff }
            guard status == 503, ContinuousClock.now < deadline else {
                log("\(red("[error]")) \(running.name) is running (pid \(running.pid)) and refused the change: \(message(reply))")
                throw ExitCode.failure
            }
            if !announced { log("\(running.name) is still loading or applying another change; the change goes in when it is done") }
            announced = true
            try await Task.sleep(for: .seconds(2))
        }
    }

    static func waitReady(_ running: ComposeRecord, until deadline: ContinuousClock.Instant) async throws {
        var last: [String] = []
        while ContinuousClock.now < deadline {
            let (status, reply) = try await call(running, "/v1/health/ready")
            if status == 200 { return print("compose \(running.name): ready") }
            let waiting = ((try? JSONSerialization.jsonObject(with: reply)) as? [String: Any])?["waiting"] as? [String] ?? []
            if waiting != last { log("waiting for \(waiting.joined(separator: ", "))") }
            last = waiting
            try await Task.sleep(for: .seconds(2))
        }
        log("\(red("[error]")) \(running.name) was not ready in time")
        throw ExitCode.failure
    }
}
