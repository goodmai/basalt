import Foundation
import Synchronization

/// Serialises download progress output.
///
/// `HuggingFaceHub.download` runs one task per file in a task group, so the
/// progress callback fires concurrently from several of them. Each call site
/// used to keep its own `nonisolated(unsafe) var lastFile` — a bare Swift String
/// mutated from multiple threads, which is a data race on a copy-on-write
/// buffer, not merely garbled output. One of them even carried a comment
/// asserting the downloads were sequential; they never were.
///
/// Holding the lock only around the bookkeeping keeps the actual writing
/// outside the critical section.
final class DownloadProgressReporter: Sendable {
    private let lastFile = Mutex<String>("")

    func callAsFunction(_ filename: String, _ downloaded: Int64, _ total: Int64) {
        let startedNewFile = lastFile.withLock { last -> Bool in
            guard filename != last else { return false }
            let hadPrevious = !last.isEmpty
            last = filename
            return hadPrevious
        }
        if startedNewFile { print() }   // close off the previous file's line
        printFileProgress(filename: filename, downloaded: downloaded, total: total)
    }
}
