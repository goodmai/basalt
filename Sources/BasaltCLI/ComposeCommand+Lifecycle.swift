import Foundation
import ArgumentParser
import BasaltCore

extension ComposeCommand {
    static func announce(_ spec: ComposeSpec, environment: [String: String]) throws {
        let missing = spec.missingKeys(environment)
        guard missing.isEmpty else {
            for line in missing { log("\(red("[error]")) \(line); export it or remove the model") }
            throw ExitCode.failure
        }
        log("Compose \(bold(spec.name)) → basalt serve \(spec.serveArguments.joined(separator: " "))")
    }

    static func stop(_ record: ComposeRecord, grace: Duration) async throws {
        kill(record.pid, SIGINT)
        let deadline = ContinuousClock.now + grace
        while ComposeRecord.alive(record), ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(100)) }
        guard ComposeRecord.alive(record) else { return print("\(record.name) stopped") }
        log("\(red("[error]")) \(record.name) (pid \(record.pid)) is still running after \(grace); it may be finishing a request or a load. Run compose down again, or kill \(record.pid)")
        throw ExitCode.failure
    }
}
