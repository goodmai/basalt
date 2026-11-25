import Darwin
import Foundation
import Testing
@testable import BasaltBench

@Suite("One GPU-lock protocol: GPULock and ModelRunLog both take the mkdir + owner directory and exclude each other (#795)")
struct GPULockProtocolTests {
    static func path() -> String {
        FileManager.default.temporaryDirectory.appending(path: "gpu-lock-\(UUID().uuidString)").path
    }

    static func remove(_ path: String) {
        try? FileManager.default.removeItem(atPath: path)
        try? FileManager.default.removeItem(atPath: path + ".reclaim")
    }

    static func directory(_ path: String, owner: String?, flock: Bool) throws {
        try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: false)
        if let owner { try owner.write(toFile: path + "/owner", atomically: true, encoding: .utf8) }
        if flock { try "".write(toFile: path + "/.flock", atomically: true, encoding: .utf8) }
    }

    static func deadPID() throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/true")
        try process.run()
        process.waitUntilExit()
        return process.processIdentifier
    }

    @Test("A test's directory lock held by a live pid makes GPULock refuse, instead of flocking a file inside it")
    func directoryHeldByLivePIDRefusesGPULock() throws {
        let path = Self.path()
        defer { Self.remove(path) }
        #expect(try ModelRunLog.claim(path, pid: getpid()) == .taken)
        #expect(throws: BenchmarkError.self) { _ = try GPULock.tryAcquire(path: path) }
        #expect(!FileManager.default.fileExists(atPath: path + "/.flock"))
        #expect(ModelRunLog.release(path, pid: getpid()))
    }

    @Test("A GPULock lease is the same directory a test claims, so a test is refused while the lease is held and taken after")
    func leaseRefusesTestUntilReleased() throws {
        let path = Self.path()
        defer { Self.remove(path) }
        let lease = try GPULock.tryAcquire(path: path)
        var isDirectory: ObjCBool = false
        #expect(FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) && isDirectory.boolValue)
        #expect(try ModelRunLog.claim(path, pid: getpid() + 1) == .busy(owner: String(getpid())))
        lease.release()
        #expect(!FileManager.default.fileExists(atPath: path))
        #expect(try ModelRunLog.claim(path, pid: getpid()) == .taken)
        #expect(ModelRunLog.release(path, pid: getpid()))
    }

    @Test("A leftover regular lock file that no process flocks is cleared and the lock is taken")
    func leftoverLegacyFileIsCleared() throws {
        let path = Self.path()
        defer { Self.remove(path) }
        try "12345\n".write(toFile: path, atomically: true, encoding: .utf8)
        #expect(try ModelRunLog.claim(path, pid: getpid()) == .taken)
        #expect(ModelRunLog.release(path, pid: getpid()))
        try "12345\n".write(toFile: path, atomically: true, encoding: .utf8)
        var lease = try GPULock.tryAcquire(path: path)
        let held = lease.isHeld
        #expect(held)
        lease.close()
    }

    @Test("A legacy lock file that a live process still flocks is reported busy, with the pid written in it")
    func flockedLegacyFileIsBusy() throws {
        let path = Self.path()
        defer { Self.remove(path) }
        try "4242\n".write(toFile: path, atomically: true, encoding: .utf8)
        let descriptor = Darwin.open(path, O_RDWR)
        #expect(descriptor >= 0)
        defer { _ = Darwin.close(descriptor) }
        #expect(flock(descriptor, LOCK_EX | LOCK_NB) == 0)
        #expect(try ModelRunLog.claim(path, pid: getpid()) == .busy(owner: "4242"))
        #expect(throws: BenchmarkError.self) { _ = try GPULock.tryAcquire(path: path) }
    }

    @Test("A directory whose owner pid is dead is reclaimed by GPULock")
    func deadOwnerIsReclaimed() throws {
        let path = Self.path()
        defer { Self.remove(path) }
        let dead = try Self.deadPID()
        #expect(try ModelRunLog.claim(path, pid: dead) == .taken)
        let lease = try GPULock.tryAcquire(path: path)
        #expect(GPULock.owner(path) == String(getpid()))
        lease.release()
    }

    @Test("A stale owner is reclaimed only by the process holding the reclaim flock, so two waiters never both delete and retake the lock")
    func reclaimIsSerialized() throws {
        let path = Self.path()
        defer { Self.remove(path) }
        try Self.directory(path, owner: String(try Self.deadPID()), flock: false)
        let other = Darwin.open(path + ".reclaim", O_RDONLY | O_CREAT, 0o644)
        #expect(other >= 0 && flock(other, LOCK_EX | LOCK_NB) == 0)
        #expect(try GPULock.claim(path, pid: getpid()) != .taken)
        _ = Darwin.close(other)
        #expect(try GPULock.claim(path, pid: getpid()) == .taken)
        #expect(GPULock.release(path, pid: getpid()))
    }

    @Test("An owner that is not a positive pid is never probed with kill or reclaimed", arguments: ["0", "-1", "-4242"])
    func nonPositiveOwnerIsBusy(owner: String) throws {
        let path = Self.path()
        defer { Self.remove(path) }
        try Self.directory(path, owner: owner, flock: false)
        #expect(try GPULock.claim(path, pid: getpid()) == .busy(owner: owner))
    }

    @Test("A #795 leftover: an ownerless directory holding a free .flock is reclaimed; one whose .flock is still flocked stays busy")
    func leftoverFlockDirectory() throws {
        let path = Self.path()
        defer { Self.remove(path) }
        try Self.directory(path, owner: nil, flock: true)
        let holder = Darwin.open(path + "/.flock", O_RDONLY)
        #expect(holder >= 0 && flock(holder, LOCK_EX | LOCK_NB) == 0)
        #expect(try GPULock.claim(path, pid: getpid()) == .busy(owner: nil))
        _ = Darwin.close(holder)
        #expect(try GPULock.claim(path, pid: getpid()) == .taken)
        #expect(GPULock.release(path, pid: getpid()))
        #expect(!FileManager.default.fileExists(atPath: path))
    }

    @Test("An ownerless directory without a .flock is a lock being created, and is never reclaimed")
    func ownerlessDirectoryIsBusy() throws {
        let path = Self.path()
        defer { Self.remove(path) }
        try Self.directory(path, owner: nil, flock: false)
        #expect(try GPULock.claim(path, pid: getpid()) == .busy(owner: nil))
    }

    @Test("Release removes a leftover .flock inside the lock, so the next claim is not stuck on an ownerless directory")
    func releaseRemovesLeftoverFlock() throws {
        let path = Self.path()
        defer { Self.remove(path) }
        #expect(try GPULock.claim(path, pid: getpid()) == .taken)
        try "".write(toFile: path + "/.flock", atomically: true, encoding: .utf8)
        #expect(GPULock.release(path, pid: getpid()))
        #expect(!FileManager.default.fileExists(atPath: path))
    }

    @Test("GPULock and the test harness lock the same path, BASALT_GPU_LOCK when it is set")
    func defaultPathHonoursEnvironment() {
        #expect(GPULock.defaultPath == (ProcessInfo.processInfo.environment["BASALT_GPU_LOCK"] ?? "/private/tmp/basalt-gpu.lock"))
        #expect(ModelRunLog.gpuLock == GPULock.defaultPath)
    }
}
