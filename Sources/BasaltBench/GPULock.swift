import Foundation
import Darwin

public struct GPULockLease: ~Copyable, Sendable {
    public let path: String
    private let pid: Int32
    private var held: Bool

    init(path: String, pid: Int32) {
        self.path = path
        self.pid = pid
        self.held = true
    }

    public var isHeld: Bool { held }

    public mutating func close() {
        guard held else { return }
        held = false
        let (path, pid) = (path, pid)
        withTaskCancellationShield { _ = GPULock.release(path, pid: pid) }
    }

    public consuming func release() {
        var copy = self
        copy.close()
    }

    deinit {
        guard held else { return }
        let (path, pid) = (path, pid)
        withTaskCancellationShield { _ = GPULock.release(path, pid: pid) }
    }
}

public enum GPULockClaim: Equatable, Sendable {
    case taken
    case busy(owner: String?)
}

public struct GPULockError: Error, CustomStringConvertible, Sendable {
    public let path: String
    public let code: Int32

    public var description: String { "\(path): \(String(cString: strerror(code)))" }
}

public enum GPULock {
    public static let defaultPath = ProcessInfo.processInfo.environment["BASALT_GPU_LOCK"] ?? "/private/tmp/basalt-gpu.lock"

    public static func claim(
        _ path: String, pid: Int32, reclaiming: Bool = true,
        writeOwner: (String, Int32) throws -> Void = writeOwnerFile
    ) throws -> GPULockClaim {
        guard mkdir(path, 0o755) == 0 else {
            let code = errno
            guard code == EEXIST else { throw GPULockError(path: path, code: code) }
            guard reclaiming else { return .busy(owner: currentOwner(path)) }
            return try reclaimingStale(path) { try claim(path, pid: pid, reclaiming: false, writeOwner: writeOwner) }
        }
        do {
            try writeOwner(path + "/owner", pid)
        } catch {
            rmdir(path)
            throw error
        }
        return .taken
    }

    static func reclaimingStale(_ path: String, retry: () throws -> GPULockClaim) throws -> GPULockClaim {
        let guardDescriptor = Darwin.open(path + ".reclaim", O_RDONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o644)
        guard guardDescriptor >= 0 else { return .busy(owner: currentOwner(path)) }
        defer { _ = Darwin.close(guardDescriptor) }
        guard flock(guardDescriptor, LOCK_EX | LOCK_NB) == 0 else { return .busy(owner: currentOwner(path)) }
        if isRegularFile(path) {
            guard removeFreeFile(path) else { return .busy(owner: legacyOwner(path)) }
            return try retry()
        }
        let holder = owner(path)
        if let holder {
            guard let holderPID = Int32(holder), holderPID > 0, kill(holderPID, 0) != 0, errno == ESRCH else { return .busy(owner: holder) }
            unlink(path + "/owner")
        } else if !isRegularFile(path + "/.flock") {
            return .busy(owner: nil)
        }
        guard removeLeftoverFlock(path), rmdir(path) == 0 || errno == ENOENT else { return .busy(owner: holder) }
        return try retry()
    }

    public static func writeOwnerFile(_ path: String, _ pid: Int32) throws {
        try String(pid).write(toFile: path, atomically: true, encoding: .utf8)
    }

    @discardableResult
    public static func release(_ path: String, pid: Int32) -> Bool {
        guard owner(path) == String(pid) else { return false }
        unlink(path + "/owner")
        _ = removeLeftoverFlock(path)
        return rmdir(path) == 0 || errno == ENOENT
    }

    static func owner(_ path: String) -> String? {
        (try? String(contentsOfFile: path + "/owner", encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func currentOwner(_ path: String) -> String? {
        isRegularFile(path) ? legacyOwner(path) : owner(path)
    }

    static func isRegularFile(_ path: String) -> Bool {
        var info = stat()
        return lstat(path, &info) == 0 && (info.st_mode & S_IFMT) == S_IFREG
    }

    static func removeLeftoverFlock(_ path: String) -> Bool {
        !isRegularFile(path + "/.flock") || removeFreeFile(path + "/.flock")
    }

    static func removeFreeFile(_ path: String) -> Bool {
        let descriptor = Darwin.open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { return false }
        defer { _ = Darwin.close(descriptor) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else { return false }
        var opened = stat(), named = stat()
        guard fstat(descriptor, &opened) == 0, lstat(path, &named) == 0,
              opened.st_dev == named.st_dev, opened.st_ino == named.st_ino, (named.st_mode & S_IFMT) == S_IFREG
        else { return false }
        return unlink(path) == 0
    }

    static func legacyOwner(_ path: String) -> String? {
        (try? String(contentsOfFile: path, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func tryAcquire(path: String = defaultPath) throws(BenchmarkError) -> GPULockLease {
        let pid = getpid()
        let result: GPULockClaim
        do {
            result = try claim(path, pid: pid)
        } catch {
            throw .gpuLockIO("\(error)")
        }
        switch result {
        case .taken:
            return GPULockLease(path: path, pid: pid)
        case .busy(let owner):
            throw .gpuLockBusy("\(path) held by \(owner ?? "a process that wrote no owner")")
        }
    }

    public static func acquire(
        path: String = defaultPath,
        timeout: Duration = .seconds(30),
        pollInterval: Duration = .milliseconds(50),
        sleep: @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) async throws -> GPULockLease {
        let deadline = ContinuousClock.now + timeout
        while true {
            try Task.checkCancellation()
            do {
                return try tryAcquire(path: path)
            } catch BenchmarkError.gpuLockBusy(let busyPath) {
                guard ContinuousClock.now < deadline else {
                    throw BenchmarkError.gpuLockBusy(busyPath)
                }
                try await sleep(pollInterval)
            }
        }
    }

    public static func withLock<T: Sendable>(
        path: String = defaultPath,
        timeout: Duration = .seconds(30),
        pollInterval: Duration = .milliseconds(50),
        sleep: @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) },
        body: @Sendable () async throws -> T
    ) async throws -> T {
        var lease = try await acquire(path: path, timeout: timeout, pollInterval: pollInterval, sleep: sleep)
        defer { lease.close() }
        return try await body()
    }
}
