import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

struct SessionLogOptions: ParsableArguments {
    enum Toggle: String, ExpressibleByArgument, CaseIterable { case on, off }

    @Option(
        name: .customLong("session-log"),
        help: "JSONL session log of requests and replies: full (default; prompts and replies are stored on this machine) | meta (metrics, sizes and hashes only) | off"
    )
    var mode: SessionLogMode = .full

    @Option(
        name: .customLong("session-log-redact"),
        help: "Replace private keys and API tokens (sk-, ghp_, AKIA, xox, Bearer) in full-mode records with [REDACTED:*]: on (default) | off; best effort, not a guarantee"
    )
    var redact: Toggle = .on

    @Option(name: .customLong("session-idle-minutes"), help: "Write session_suspend and close a session file after this many idle minutes; the next request resumes it (default: 30)")
    var idleMinutes: Int = 30

    @Option(name: .customLong("session-log-dir"), help: "Directory of the session log (default: ~/.basalt/sessions)")
    var directory: String?

    @Option(name: .customLong("session-log-retention-days"), help: "Delete session files older than this many days at start (default: 14)")
    var retentionDays: Int = 14

    @Option(name: .customLong("session-log-max-total-mb"), help: "Delete the oldest session files beyond this total size at start (default: 1024)")
    var maxTotalMB: Int = 1024

    func validateRetention() throws {
        guard retentionDays >= 0, maxTotalMB >= 0 else {
            throw ValidationError("--session-log-retention-days and --session-log-max-total-mb must not be negative")
        }
    }

    func writer() -> SessionLogWriter? {
        ServeCommand.sessionLogWriter(
            mode: mode, directory: directory, retention: SessionLogRetention(days: retentionDays, maxTotalMB: maxTotalMB),
            redacts: redact == .on, idleMinutes: idleMinutes)
    }
}

struct SessionLogRetention: Sendable, Equatable {
    var days = 14
    var maxTotalMB = 1024
}

extension ServeCommand {
    static func sessionLogWriter(
        mode: SessionLogMode, directory: String?, retention: SessionLogRetention = SessionLogRetention(), redacts: Bool = true,
        idleMinutes: Int = 30, defaultSession: String = SessionLogWriter.processSession()
    ) -> SessionLogWriter? {
        guard mode != .off else { return nil }
        let url = directory.map(ConfigSection.fileURL) ?? SessionLogWriter.defaultDirectory
        SessionLogWriter.applyRetention(directory: url, days: retention.days, maxTotalBytes: Int64(retention.maxTotalMB) << 20)
        let writer = SessionLogWriter(
            mode: mode, directory: url, redacts: redacts, defaultSession: defaultSession, idle: .init(timeout: TimeInterval(max(1, idleMinutes) * 60)))
        Task { await writer.startFlushing() }
        return writer
    }
}
