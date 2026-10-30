import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

struct PromptCacheDiskOptions: ParsableArguments {
    @Option(
        name: .customLong("prompt-cache-disk-mb"),
        help: "Persist the prompt cache to disk on shutdown, model switch and chat exit, up to this many MB, so a resumed session skips the prefill (default: 0, off; needs --session-log full: the files are the conversation as KV tensors and cannot be redacted)"
    )
    var megabytes: Int = 0

    @Option(name: .customLong("prompt-cache-disk-days"), help: "Delete persisted prompt-cache files unused for this many days at start (default: 7)")
    var retentionDays: Int = 7

    @Option(name: .customLong("prompt-cache-dir"), help: "Directory of the persisted prompt cache (default: ~/.basalt/prompt-cache)")
    var directory: String?

    static func url(_ directory: String?) -> URL {
        directory.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? PromptCacheDiskStore.defaultDirectory
    }

    func validate(sessionLog: SessionLogMode) throws {
        guard megabytes >= 0, retentionDays >= 0 else {
            throw ValidationError("--prompt-cache-disk-mb and --prompt-cache-disk-days must not be negative")
        }
        guard megabytes == 0 || sessionLog == .full else {
            throw ValidationError("--prompt-cache-disk-mb writes the conversation to disk as KV tensors, which --session-log \(sessionLog.rawValue) promises not to keep and which cannot be redacted; use --session-log full or leave --prompt-cache-disk-mb at 0")
        }
    }

    func store() throws -> PromptCacheDiskStore? {
        guard megabytes > 0 else { return nil }
        let store = PromptCacheDiskStore(
            directory: Self.url(directory), byteQuota: Int64(min(megabytes, 1 << 30)) << 20, retentionDays: retentionDays)
        try store.prune()
        return store
    }
}
