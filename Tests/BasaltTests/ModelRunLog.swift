import Foundation
import Darwin
import Synchronization
@testable import BasaltCore

public struct GPULockGuard: ~Copyable, Sendable {
    public let path: String
    public let pid: Int32
    private var isArmed: Bool

    public init(path: String, pid: Int32) {
        self.path = path
        self.pid = pid
        self.isArmed = true
    }

    public mutating func release() -> Bool {
        guard isArmed else { return false }
        isArmed = false
        return ModelRunLog.release(path, pid: pid)
    }

    deinit {
        if isArmed {
            _ = ModelRunLog.release(path, pid: pid)
        }
    }
}

enum GPULockClaim: Equatable {
    case taken
    case busy(owner: String?)
}

struct GPULockError: Error, CustomStringConvertible {
    let path: String
    let code: Int32

    var description: String { "\(path): \(String(cString: strerror(code)))" }
}

final class ModelRunLog: Sendable {
    static let gpuLock = ProcessInfo.processInfo.environment["BASALT_GPU_LOCK"] ?? "/private/tmp/basalt-gpu.lock"

    let url: URL
    let model: String
    private let logLock = Mutex<Void>(())
    private let heldGuard = Mutex<GPULockGuard?>(nil)

    init(_ runner: String, modelPath: String) async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let day = Date().ISO8601Format(.iso8601.year().month().day())
        let directory = root.appendingPathComponent("logs")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        url = directory.appendingPathComponent("\(runner)-\(day).log")
        let snapshot = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
        model = snapshot.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent
            .replacingOccurrences(of: "models--", with: "").replacingOccurrences(of: "--", with: "/")
        record("start", "commit \(Self.output("/usr/bin/git", ["-C", root.path, "rev-parse", "HEAD"])) revision \(snapshot.lastPathComponent) machine \(SystemProfiler.detectChipModel()) \(ProcessInfo.processInfo.physicalMemory >> 30) GB pid \(getpid())")
        let report = await ModelInspector().inspect(modelIdOrPath: snapshot.path)
        guard report.intact else {
            record("FAIL inspect", report.errors.joined(separator: "; "))
            throw BasaltError.modelNotFound(identifier: modelPath)
        }
        record("inspect", "intact, \(report.totalShards) shards, \(report.totalSizeFormatted), \(report.bitWidthDescription)")
        try acquireGPU()
    }

    deinit { releaseGPU() }

    func record(_ step: String, _ detail: String) {
        let line = "\(Date().ISO8601Format()) \(model) \(step) \(detail)\n"
        print(line, terminator: "")
        logLock.withLock { _ in
            let descriptor = open(url.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
            guard descriptor >= 0 else { return }
            _ = line.withCString { write(descriptor, $0, strlen($0)) }
            close(descriptor)
        }
    }

    func step<T>(_ name: String, _ body: () async throws -> T) async throws -> T {
        record(name, "begin, \(host())")
        do {
            let result = try await body()
            record(name, "done, \(host())")
            return result
        } catch {
            record("FAIL \(name)", "\(error)")
            throw error
        }
    }

    func host() -> String {
        let load = Self.load()
        let ci = !Self.output("/usr/bin/pgrep", ["-f", "Runner.Worker"]).isEmpty
        return String(format: "load %.1f ci %@", load, ci ? "busy" : "idle")
    }

    static func load() -> Double {
        var load = [Double](repeating: 0, count: 3)
        return getloadavg(&load, 3) == 3 ? load[0] : .infinity
    }

    func releaseGPU() {
        guard var g = heldGuard.withLock({ $0.take() }) else { return }
        let path = g.path
        if g.release() {
            record("gpu-release", path)
        } else {
            record("FAIL gpu-release", "\(path) is no longer owned by \(getpid()), left in place")
        }
    }

    private func acquireGPU() throws {
        do {
            switch try Self.claim(Self.gpuLock, pid: getpid()) {
            case .taken:
                heldGuard.withLock { $0 = GPULockGuard(path: Self.gpuLock, pid: getpid()) }
                record("gpu-acquire", Self.gpuLock)
            case .busy(let owner):
                record("FAIL gpu-busy", "\(Self.gpuLock) held by \(owner ?? "no owner file yet"); a test does not wait for the GPU, relaunch it when the lock is free")
                throw CancellationError()
            }
        } catch let error as GPULockError {
            record("FAIL gpu-lock", error.description)
            throw error
        }
    }

    static func claim(
        _ path: String, pid: Int32, reclaiming: Bool = true,
        writeOwner: (String, Int32) throws -> Void = { try String($1).write(toFile: $0, atomically: true, encoding: .utf8) }
    ) throws -> GPULockClaim {
        guard mkdir(path, 0o755) == 0 else {
            let code = errno
            guard code == EEXIST else { throw GPULockError(path: path, code: code) }
            let owner = (try? String(contentsOfFile: path + "/owner", encoding: .utf8))?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if reclaiming, let owner, let holder = Int32(owner), kill(holder, 0) != 0, errno == ESRCH {
                unlink(path + "/owner")
                rmdir(path)
                return try claim(path, pid: pid, reclaiming: false, writeOwner: writeOwner)
            }
            return .busy(owner: owner)
        }
        do {
            try writeOwner(path + "/owner", pid)
        } catch {
            rmdir(path)
            throw error
        }
        return .taken
    }

    static func release(_ path: String, pid: Int32) -> Bool {
        let owner = (try? String(contentsOfFile: path + "/owner", encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard owner == String(pid) else { return false }
        unlink(path + "/owner")
        rmdir(path)
        return true
    }

    static func output(_ tool: String, _ arguments: [String]) -> String {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: tool)
        process.arguments = arguments
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { return "" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
