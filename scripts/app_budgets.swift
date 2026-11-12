import AppKit
import CoreGraphics
import Foundation

struct Config {
    var app = ".build/Basalt.app"
    var model = "mlx-community/Qwen3-1.7B-4bit"
    var launches = 6
    var settle = 45.0
    var idle = 10

    init(_ arguments: [String]) {
        var values: [String: String] = [:]
        var iterator = arguments.makeIterator()
        while let key = iterator.next() { values[key] = iterator.next() }
        app = values["--app"] ?? app
        model = values["--model"] ?? model
        launches = values["--launches"].flatMap(Int.init) ?? launches
        settle = values["--settle-seconds"].flatMap(Double.init) ?? settle
        idle = values["--idle-seconds"].flatMap(Int.init) ?? idle
        guard launches >= 2, idle > 0, FileManager.default.isExecutableFile(atPath: app + "/Contents/MacOS/Basalt") else {
            FileHandle.standardError.write(Data("usage: app_budgets.swift [--app .build/Basalt.app] [--model repo] [--launches 6] [--settle-seconds 45] [--idle-seconds 10]\nbuild the bundle first: swift scripts/bundle_app.swift\n".utf8))
            exit(2)
        }
    }
}

final class Log {
    let handle: FileHandle
    var failures = 0

    init(_ path: String) throws {
        if !FileManager.default.fileExists(atPath: path) { FileManager.default.createFile(atPath: path, contents: nil) }
        handle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
        try handle.seekToEnd()
    }

    func line(_ text: String) {
        let stamped = "\(ISO8601DateFormatter().string(from: Date())) \(text)"
        print(stamped)
        try? handle.write(contentsOf: Data((stamped + "\n").utf8))
    }

    func fail(_ text: String) {
        failures += 1
        line("FAIL \(text)")
    }
}

struct GPULock {
    static let path = "/private/tmp/basalt-gpu.lock"
    let owner = "\(ProcessInfo.processInfo.processIdentifier)"

    func take(_ log: Log) {
        while (try? FileManager.default.createDirectory(atPath: Self.path, withIntermediateDirectories: false)) == nil {
            log.line("WAIT \(Self.path) is held")
            Thread.sleep(forTimeInterval: 30)
        }
        try? owner.write(toFile: Self.path + "/owner", atomically: true, encoding: .utf8)
        log.line("LOCKED \(Self.path) owner=\(owner)")
    }

    func release(_ log: Log) {
        guard (try? String(contentsOfFile: Self.path + "/owner", encoding: .utf8)) == owner else { return }
        try? FileManager.default.removeItem(atPath: Self.path)
        log.line("UNLOCKED \(Self.path) owner=\(owner)")
    }
}

func shell(_ arguments: [String]) -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = arguments
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = Pipe()
    try? process.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return String(decoding: data, as: UTF8.self)
}

func windowShown(by pid: pid_t) -> Bool {
    let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
    return windows.contains { window in
        (window[kCGWindowOwnerPID as String] as? Int32) == pid && (window[kCGWindowLayer as String] as? Int) == 0
            && ((window[kCGWindowBounds as String] as? [String: Double])?["Height"] ?? 0) > 100
    }
}

func launch(_ config: Config) -> (process: Process, window: Double?)? {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: config.app + "/Contents/MacOS/Basalt")
    process.arguments = ["-model", config.model, "-compose", ""]
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle.nullDevice
    let start = Date()
    guard (try? process.run()) != nil else { return nil }
    while Date().timeIntervalSince(start) < 10 {
        if windowShown(by: process.processIdentifier) { return (process, Date().timeIntervalSince(start) * 1000) }
        Thread.sleep(forTimeInterval: 0.002)
    }
    return (process, nil)
}

func quit(_ process: Process) {
    NSRunningApplication(processIdentifier: process.processIdentifier)?.terminate()
    let deadline = Date().addingTimeInterval(30)
    while process.isRunning, Date() < deadline { Thread.sleep(forTimeInterval: 0.1) }
    if process.isRunning { process.terminate() }
    process.waitUntilExit()
}

func idleWakeups(_ pid: pid_t, seconds: Int) -> Int? {
    let output = shell(["top", "-l", "2", "-s", "\(seconds)", "-pid", "\(pid)", "-stats", "pid,idlew"])
    let rows = output.split(separator: "\n").map { $0.split(separator: " ") }.filter { $0.count == 2 && $0[0] == Substring("\(pid)") }
    return rows.last.flatMap { Int($0[1]) }
}

func footprintMB(_ pid: pid_t) -> Double? {
    let output = shell(["footprint", "-p", "\(pid)"])
    guard let line = output.split(separator: "\n").first(where: { $0.contains("Footprint:") }) else { return nil }
    let parts = line.split(separator: " ")
    guard let index = parts.firstIndex(of: "Footprint:"), index + 2 < parts.count, let value = Double(parts[index + 1]) else { return nil }
    switch parts[index + 2] {
    case "KB": return value / 1024
    case "GB": return value * 1024
    default: return value
    }
}

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted.count.isMultiple(of: 2) ? (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2 : sorted[sorted.count / 2]
}

func measure(_ config: Config, _ log: Log) {
    var warm: [Double] = []
    for index in 0 ..< config.launches {
        guard let run = launch(config) else { return log.fail("could not start \(config.app)") }
        guard let window = run.window else {
            quit(run.process)
            return log.fail("launch \(index): no window within 10 s")
        }
        log.line("STEP launch=\(index) \(index == 0 ? "cold" : "warm") window_ms=\(String(format: "%.0f", window))")
        if index > 0 { warm.append(window) }
        guard index == config.launches - 1 else {
            quit(run.process)
            continue
        }
        Thread.sleep(forTimeInterval: config.settle)
        let wakeups = idleWakeups(run.process.processIdentifier, seconds: config.idle)
        let footprint = footprintMB(run.process.processIdentifier)
        log.line("STEP idle seconds=\(config.idle) idle_wakeups=\(wakeups.map(String.init) ?? "n/a") footprint_mb=\(footprint.map { String(format: "%.1f", $0) } ?? "n/a")")
        quit(run.process)
        let budgets = [
            ("warm_window_ms", median(warm), 400.0),
            ("idle_wakeups_per_s", wakeups.map { Double($0) / Double(config.idle) } ?? .infinity, 0.0),
            ("footprint_mb", footprint ?? .infinity, 150.0),
        ]
        for (name, value, limit) in budgets {
            log.line("RESULT \(name)=\(String(format: "%.1f", value)) budget=\(String(format: "%.0f", limit)) \(value <= limit ? "PASS" : "OVER")")
        }
    }
}

let config = Config(Array(CommandLine.arguments.dropFirst()))
let date = String(ISO8601DateFormatter().string(from: Date()).prefix(10))
try FileManager.default.createDirectory(atPath: "logs", withIntermediateDirectories: true)
let log = try Log("logs/app-budgets-\(date).log")
let head = shell(["git", "rev-parse", "HEAD"]).trimmingCharacters(in: .whitespacesAndNewlines)
let chip = shell(["sysctl", "-n", "machdep.cpu.brand_string"]).trimmingCharacters(in: .whitespacesAndNewlines)
log.line("RUN commit=\(head) chip=\(chip) memory=\(ProcessInfo.processInfo.physicalMemory) app=\(config.app) model=\(config.model) launches=\(config.launches)")
let lock = GPULock()
lock.take(log)
measure(config, log)
lock.release(log)
log.line("END failures=\(log.failures)")
exit(log.failures == 0 ? 0 : 1)
