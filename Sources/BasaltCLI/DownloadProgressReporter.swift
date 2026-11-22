import Foundation
import Synchronization

final class DownloadProgressReporter: Sendable {
    private let lastFile = Mutex<String>("")
    private let useStderr: Bool

    init(useStderr: Bool = false) {
        self.useStderr = useStderr
    }

    func callAsFunction(_ filename: String, _ downloaded: Int64, _ total: Int64) {
        let startedNewFile = lastFile.withLock { last -> Bool in
            guard filename != last else { return false }
            let hadPrevious = !last.isEmpty
            last = filename
            return hadPrevious
        }
        if startedNewFile {
            if useStderr { fputs("\n", stderr) } else { print() }
        }
        printFileProgress(filename: filename, downloaded: downloaded, total: total, useStderr: useStderr)
    }
}
