import Foundation
import Darwin
@testable import BasaltCore

final class ModelRunLog: @unchecked Sendable {
    static let gpuLocks = ["/private/tmp/basalt-gpu.lock", "/private/tmp/claude-501/basalt-gpu.lock"]
    static let quietLoad = 8.0
    static let quietMarker = "/private/tmp/claude-501/basalt-quiet"

    let url: URL
    let model: String
    let issue: String
    private let state = NSLock()
    private var held: [String] = []

    init(_ runner: String, issue: String, modelPath: String) async throws {
        self.issue = issue
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
        try await acquireGPU()
    }

    deinit { releaseGPU() }

    func record(_ step: String, _ detail: String) {
        let line = "\(Date().ISO8601Format()) \(model) \(step) \(detail)\n"
        print(line, terminator: "")
        state.withLock {
            if let handle = try? FileHandle(forWritingTo: url) {
                handle.seekToEndOfFile()
                handle.write(Data(line.utf8))
                try? handle.close()
            } else {
                try? Data(line.utf8).write(to: url)
            }
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

    func timed<T>(_ name: String, _ body: () async throws -> T) async throws -> T {
        var waited = false
        while true {
            let ci = !Self.output("/usr/bin/pgrep", ["-f", "Runner.Worker"]).isEmpty
            let load = Self.load()
            if !ci, load <= Self.quietLoad {
                let descriptor = open(Self.quietMarker, O_CREAT | O_EXCL | O_WRONLY, 0o644)
                if descriptor >= 0 {
                    _ = "\(getpid()) \(issue)\n".withCString { write(descriptor, $0, strlen($0)) }
                    close(descriptor)
                    break
                }
            }
            if !waited {
                let holder = (try? String(contentsOfFile: Self.quietMarker, encoding: .utf8)) ?? ""
                record("quiet-wait", ci ? "a CI job is running" : load > Self.quietLoad
                    ? String(format: "load %.1f above %.0f", load, Self.quietLoad)
                    : "quiet marker held by \(holder.trimmingCharacters(in: .whitespacesAndNewlines))")
                waited = true
            }
            try await Task.sleep(for: .seconds(30))
        }
        record("quiet-acquire", Self.quietMarker)
        defer {
            let holder = try? String(contentsOfFile: Self.quietMarker, encoding: .utf8)
            if holder?.split(separator: " ").first == "\(getpid())" {
                unlink(Self.quietMarker)
                record("quiet-release", Self.quietMarker)
            } else {
                record("FAIL quiet-release", "marker is held by \(holder ?? "nobody"), left in place")
            }
        }
        return try await step(name, body)
    }

    func host() -> String {
        let load = Self.load()
        let ci = !Self.output("/usr/bin/pgrep", ["-f", "Runner.Worker"]).isEmpty
        let quiet = load <= Self.quietLoad && !ci
        return String(format: "load %.1f ci %@ speed %@", load, ci ? "busy" : "idle", quiet ? "valid" : "invalid")
    }

    static func load() -> Double {
        var load = [Double](repeating: 0, count: 3)
        return getloadavg(&load, 3) == 3 ? load[0] : .infinity
    }

    func releaseGPU() {
        let releasing = state.withLock { () -> [String] in
            defer { held = [] }
            return held
        }
        for path in releasing.reversed() {
            let owner = try? String(contentsOfFile: path + "/owner", encoding: .utf8)
            guard owner?.trimmingCharacters(in: .whitespacesAndNewlines) == String(getpid()) else {
                record("FAIL gpu-release", "\(path) is owned by \(owner ?? "nobody"), left in place")
                continue
            }
            try? FileManager.default.removeItem(atPath: path + "/owner")
            rmdir(path)
        }
        if !releasing.isEmpty { record("gpu-release", releasing.joined(separator: ", ")) }
    }

    private func acquireGPU() async throws {
        for path in Self.gpuLocks {
            var waited = false
            while mkdir(path, 0o755) != 0 {
                if !waited {
                    let owner = (try? String(contentsOfFile: path + "/owner", encoding: .utf8))?
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let alive = owner.flatMap { Int32($0) }.map { kill($0, 0) == 0 }
                    record("gpu-wait", "\(path) held by \(owner ?? "no owner file")\(alive == false ? " (process gone)" : "")")
                    waited = true
                }
                try await Task.sleep(for: .seconds(15))
            }
            try String(getpid()).write(toFile: path + "/owner", atomically: true, encoding: .utf8)
            state.withLock { held.append(path) }
        }
        record("gpu-acquire", Self.gpuLocks.joined(separator: ", "))
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
