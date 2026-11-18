import Foundation
import Darwin

public struct GPULockLease: ~Copyable, Sendable {
    public let path: String
    private var fd: Int32

    init(path: String, fd: Int32) {
        self.path = path
        self.fd = fd
    }

    public var isHeld: Bool { fd >= 0 }

    public mutating func close() {
        guard fd >= 0 else { return }
        let descriptor = fd
        fd = -1
        withTaskCancellationShield {
            _ = flock(descriptor, LOCK_UN)
            _ = Darwin.close(descriptor)
        }
    }

    public consuming func release() {
        var copy = self
        copy.close()
    }

    deinit {
        guard fd >= 0 else { return }
        let descriptor = fd
        withTaskCancellationShield {
            _ = flock(descriptor, LOCK_UN)
            _ = Darwin.close(descriptor)
        }
    }
}

public enum GPULock {
    public static let defaultPath = "/private/tmp/basalt-gpu.lock"

    static func openDescriptor(at path: String) throws(BenchmarkError) -> (fd: Int32, resolvedPath: String) {
        var isDir: ObjCBool = false
        let actualPath: String
        if FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue {
            actualPath = (path as NSString).appendingPathComponent(".flock")
        } else {
            actualPath = path
        }
        let descriptor = Darwin.open(actualPath, O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0o666)
        guard descriptor >= 0 else {
            throw .gpuLockIO("cannot open \(actualPath): errno \(errno)")
        }
        return (descriptor, actualPath)
    }

    public static func tryAcquire(path: String = defaultPath) throws(BenchmarkError) -> GPULockLease {
        let opened = try openDescriptor(at: path)
        if flock(opened.fd, LOCK_EX | LOCK_NB) != 0 {
            let code = errno
            _ = Darwin.close(opened.fd)
            if code == EWOULDBLOCK || code == EAGAIN {
                throw .gpuLockBusy(opened.resolvedPath)
            }
            throw .gpuLockIO("flock failed on \(opened.resolvedPath): errno \(code)")
        }
        writeOwnerPID(fd: opened.fd)
        return GPULockLease(path: opened.resolvedPath, fd: opened.fd)
    }

    static func writeOwnerPID(fd: Int32) {
        let pidText = "\(ProcessInfo.processInfo.processIdentifier)\n"
        _ = Darwin.ftruncate(fd, 0)
        _ = Darwin.lseek(fd, 0, SEEK_SET)
        _ = pidText.withCString { ptr in
            Darwin.write(fd, ptr, strlen(ptr))
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
