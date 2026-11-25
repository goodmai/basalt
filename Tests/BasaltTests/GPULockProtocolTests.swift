import Darwin
import Foundation
import Testing
@testable import BasaltBench

@Suite("One GPU-lock protocol: GPULock and ModelRunLog both take the mkdir + owner directory and exclude each other (#795)")
struct GPULockProtocolTests {
    static func path() -> String {
        FileManager.default.temporaryDirectory.appending(path: "gpu-lock-\(UUID().uuidString)").path
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
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(try ModelRunLog.claim(path, pid: getpid()) == .taken)
        #expect(throws: BenchmarkError.self) { _ = try GPULock.tryAcquire(path: path) }
        #expect(!FileManager.default.fileExists(atPath: path + "/.flock"))
        #expect(ModelRunLog.release(path, pid: getpid()))
    }

    @Test("A GPULock lease is the same directory a test claims, so a test is refused while the lease is held and taken after")
    func leaseRefusesTestUntilReleased() throws {
        let path = Self.path()
        defer { try? FileManager.default.removeItem(atPath: path) }
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
        defer { try? FileManager.default.removeItem(atPath: path) }
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
        defer { try? FileManager.default.removeItem(atPath: path) }
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
        defer { try? FileManager.default.removeItem(atPath: path) }
        let dead = try Self.deadPID()
        #expect(try ModelRunLog.claim(path, pid: dead) == .taken)
        let lease = try GPULock.tryAcquire(path: path)
        #expect(GPULock.owner(path) == String(getpid()))
        lease.release()
    }
}
