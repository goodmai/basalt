import Foundation
import Testing
import Darwin

@Suite("The GPU lock real-model runs share (#137)", .serialized)
struct ModelRunLogTests {
    static func scratch() throws -> String {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("gpu-lock-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("basalt-gpu.lock").path
    }

    static func finishedProcess() throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/true")
        try process.run()
        process.waitUntilExit()
        return process.processIdentifier
    }

    @Test("A free lock is taken with the owner pid, a live owner's lock is busy, and only the owner releases it")
    func liveOwnerKeepsTheLock() throws {
        let path = try Self.scratch()
        #expect(try ModelRunLog.claim(path, pid: getpid()) == .taken)
        #expect(try String(contentsOfFile: path + "/owner", encoding: .utf8) == String(getpid()))
        #expect(try ModelRunLog.claim(path, pid: getpid() + 1) == .busy(owner: String(getpid())))
        #expect(!ModelRunLog.release(path, pid: getpid() + 1))
        #expect(ModelRunLog.release(path, pid: getpid()))
        #expect(!FileManager.default.fileExists(atPath: path))
        #expect(!ModelRunLog.release(path, pid: getpid()))
    }

    @Test("A lock left by a process that is gone is reclaimed; one without an owner file stays busy")
    func staleLockIsReclaimed() throws {
        let path = try Self.scratch()
        let dead = try Self.finishedProcess()
        #expect(mkdir(path, 0o755) == 0)
        try String(dead).write(toFile: path + "/owner", atomically: true, encoding: .utf8)
        #expect(try ModelRunLog.claim(path, pid: getpid()) == .taken)
        #expect(ModelRunLog.release(path, pid: getpid()))

        #expect(mkdir(path, 0o755) == 0)
        #expect(try ModelRunLog.claim(path, pid: getpid()) == .busy(owner: nil))
        rmdir(path)
    }

    @Test("A lock path whose directory is missing is an error, not a busy GPU")
    func missingDirectoryIsAnError() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("missing-\(UUID().uuidString)/lock").path
        #expect(throws: GPULockError.self) { try ModelRunLog.claim(path, pid: getpid()) }
    }

    @Test("When the owner file cannot be written the lock directory is removed again")
    func ownerWriteFailureLeavesNoLock() throws {
        let path = try Self.scratch()
        #expect(throws: CocoaError.self) {
            try ModelRunLog.claim(path, pid: getpid()) { _, _ in throw CocoaError(.fileWriteNoPermission) }
        }
        #expect(!FileManager.default.fileExists(atPath: path))
    }

    @Test("GPULockGuard releases the lock on scope exit without manual release")
    func guardAutoReleasesOnScopeExit() throws {
        let path = try Self.scratch()
        #expect(try ModelRunLog.claim(path, pid: getpid()) == .taken)
        #expect(FileManager.default.fileExists(atPath: path))
        do {
            let lockGuard = GPULockGuard(path: path, pid: getpid())
            _ = lockGuard
        }
        #expect(!FileManager.default.fileExists(atPath: path))
    }

    @Test("GPULockGuard explicit release unlinks once, and deinit does not repeat it")
    func guardExplicitReleaseIsIdempotentWithDeinit() throws {
        let path = try Self.scratch()
        #expect(try ModelRunLog.claim(path, pid: getpid()) == .taken)
        var lockGuard = GPULockGuard(path: path, pid: getpid())
        let first = lockGuard.release()
        #expect(first)
        #expect(!FileManager.default.fileExists(atPath: path))
        let second = lockGuard.release()
        #expect(!second)
        _ = consume lockGuard
        #expect(!FileManager.default.fileExists(atPath: path))
    }
}
