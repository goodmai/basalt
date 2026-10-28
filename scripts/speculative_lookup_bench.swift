#!/usr/bin/env swift
import Foundation

let usage = """
    Usage:
      ./scripts/speculative_lookup_bench.swift [--models repo,repo] [--runs N] [--tokens N] [--variant label=binary]...

    Prompt-lookup speculative decoding against plain decoding, per model:
    workloads code edit and prose, temperatures 0 and 0.7, lookup off and on.
    Each model runs every cell once in order and once in reverse, and every
    ratio is taken between cells of the same pass, so host drift cancels.

    Options:
      --models     comma-separated HF repo ids (default: \(Options().models.joined(separator: ",")))
      --runs       timed iterations per cell (default: 3)
      --tokens     tokens generated per iteration (default: 300)
      --variant    another PerformanceBenchmark binary, run with lookup on next to the one built
                   from this checkout; labels are unique and not on or off
      -h, --help   show this help
    """

struct Options {
    var models = ["mlx-community/Qwen3-4B-4bit", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"]
    var runs = 3
    var tokens = 300
    var variants: [(label: String, binary: String)] = []
}

func parse(_ arguments: [String]) -> Options? {
    var options = Options()
    var rest = arguments[...]
    while let flag = rest.popFirst() {
        if flag == "-h" || flag == "--help" { return nil }
        guard let value = rest.popFirst() else { return nil }
        switch flag {
        case "--models":
            options.models = value.split(separator: ",").map { trimmed(String($0)) }.filter { !$0.isEmpty }
            guard !options.models.isEmpty else { return nil }
        case "--runs": guard let runs = Int(value), runs > 0 else { return nil }; options.runs = runs
        case "--tokens": guard let tokens = Int(value), tokens > 0 else { return nil }; options.tokens = tokens
        case "--variant":
            let parts = value.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2, !["on", "off"].contains(parts[0]),
                  !options.variants.contains(where: { $0.label == parts[0] })
            else { return nil }
            options.variants.append((parts[0], parts[1]))
        default: return nil
        }
    }
    return options
}

guard let options = parse(Array(CommandLine.arguments.dropFirst())) else {
    print(usage)
    exit(CommandLine.arguments.contains("-h") || CommandLine.arguments.contains("--help") ? 0 : 2)
}

let repo = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    .deletingLastPathComponent().deletingLastPathComponent().path
let releaseDirectory = repo + "/.build/release"
let lockDirectory = "/private/tmp/basalt-gpu.lock"
let pid = ProcessInfo.processInfo.processIdentifier
let ownerLine = String(pid)
let maxLoad = 8.0
let stamp = ISO8601DateFormatter().string(from: Date())
let logPath = repo + "/logs/speculative_lookup_bench-\(stamp.replacingOccurrences(of: ":", with: "")).log"
try? FileManager.default.createDirectory(atPath: repo + "/logs", withIntermediateDirectories: true)
FileManager.default.createFile(atPath: logPath, contents: nil)
guard let logHandle = FileHandle(forWritingAtPath: logPath) else {
    print("FAIL cannot open the log file \(logPath)")
    exit(1)
}
let state = NSLock()
nonisolated(unsafe) var lockHeld = false
nonisolated(unsafe) var child: Process?
var failures = 0

func log(_ line: String) {
    let stamped = "\(ISO8601DateFormatter().string(from: Date())) \(line)"
    print(stamped)
    fflush(stdout)
    do {
        try logHandle.write(contentsOf: Data((stamped + "\n").utf8))
    } catch {
        failures += 1
        FileHandle.standardError.write(Data("FAIL log write \(error)\n".utf8))
    }
}

func fail(_ step: String, _ detail: String) {
    failures += 1
    log("FAIL \(step) \(detail)")
}

func run(_ executable: String, _ arguments: [String], in directory: String = repo) -> (status: Int32, output: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    process.currentDirectoryURL = URL(fileURLWithPath: directory)
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    state.withLock { child = process }
    defer { state.withLock { child = nil } }
    do { try process.run() } catch { return (-1, "\(error)") }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (process.terminationStatus, String(decoding: data, as: UTF8.self))
}

func trimmed(_ text: String) -> String { text.trimmingCharacters(in: .whitespacesAndNewlines) }

func releaseLock() {
    state.withLock {
        guard lockHeld else { return }
        lockHeld = false
        let owner = lockOwner()
        guard owner == ownerLine else {
            fail("release lock", "held by someone else, left untouched: owner=\(owner)")
            return
        }
        unlink(lockDirectory + "/owner")
        rmdir(lockDirectory)
        log("lock released")
    }
}

func lockOwner() -> String {
    trimmed((try? String(contentsOfFile: lockDirectory + "/owner", encoding: .utf8)) ?? "")
}

func reclaimIfStale(owner: String) -> Bool {
    guard let holder = Int32(owner), kill(holder, 0) != 0, errno == ESRCH else { return false }
    unlink(lockDirectory + "/owner")
    rmdir(lockDirectory)
    log("reclaimed the GPU lock from pid \(holder), which no longer runs")
    return true
}

func acquireLock() {
    var waiting = false
    while true {
        let taken: Bool = state.withLock {
            guard mkdir(lockDirectory, 0o755) == 0 else { return false }
            do {
                try ownerLine.write(toFile: lockDirectory + "/owner", atomically: true, encoding: .utf8)
            } catch {
                rmdir(lockDirectory)
                fail("lock", "cannot write the owner file: \(error)")
                exit(1)
            }
            lockHeld = true
            return true
        }
        if taken {
            log("lock acquired by pid \(ownerLine)")
            return
        }
        let owner = lockOwner()
        if reclaimIfStale(owner: owner) { continue }
        if !waiting {
            log("waiting for the GPU lock held by pid \(owner.isEmpty ? "unknown" : owner)")
            waiting = true
        }
        Thread.sleep(forTimeInterval: 15)
    }
}

signal(SIGPIPE, SIG_IGN)
let signalSources = [SIGINT, SIGTERM, SIGHUP].map { signalNumber in
    signal(signalNumber, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: .global())
    source.setEventHandler {
        if let running = state.withLock({ child }) {
            running.terminate()
            running.waitUntilExit()
        }
        log("FAIL interrupted by signal \(signalNumber)")
        releaseLock()
        exit(128 + signalNumber)
    }
    source.resume()
    return source
}

struct Host {
    let load: Double
    let ciRunning: Bool
    var quiet: Bool { load <= maxLoad && !ciRunning }
    var text: String { String(format: "load=%.2f ci=%@", load, ciRunning ? "yes" : "no") }
}

func host() -> Host {
    let raw = run("/usr/sbin/sysctl", ["-n", "vm.loadavg"]).output
    let load = raw.split(separator: " ").compactMap { Double($0) }.first ?? .infinity
    let ci = run("/usr/bin/pgrep", ["-x", "Runner.Worker"]).status == 0
    return Host(load: load, ciRunning: ci)
}

func waitForQuietHost() {
    var reported = false
    var now = host()
    while !now.quiet {
        if !reported { log("waiting for a quiet host: \(now.text)") }
        reported = true
        Thread.sleep(forTimeInterval: 30)
        now = host()
    }
}

func revision(of model: String, inspector: String) -> String? {
    let result = run(inspector, ["models", "inspect", model, "--json"])
    guard result.status == 0,
          let start = result.output.firstIndex(of: "{"),
          let report = try? JSONSerialization.jsonObject(with: Data(result.output[start...].utf8)) as? [String: Any]
    else {
        fail("snapshot \(model)", "inspect exit=\(result.status) \(result.output.suffix(300))")
        return nil
    }
    guard report["intact"] as? Bool == true, let path = report["path"] as? String else {
        fail("snapshot \(model)", "not intact: \(report["errors"] ?? "no errors reported")")
        return nil
    }
    log("snapshot ok model=\(model) path=\(path)")
    return URL(fileURLWithPath: path).lastPathComponent
}

struct Cell: Hashable {
    let workload: String
    let temperature: String
    let mode: String
    var label: String { "\(workload) T=\(temperature) \(mode)" }
}

struct Measurement {
    let tps: Double
    let sigma: Double
    let accepted: String
    let output: String
    let valid: Bool
}

func measure(_ cell: Cell, model: String, binary: String, prompt: String, pass: Int) -> Measurement? {
    var arguments = ["-m", model, "-i", String(options.runs), "-t", String(options.tokens), "-p", prompt,
                     "--temperature", cell.temperature, "--seed", "1", "--reasoning-effort", "none", "--print-output"]
    if cell.mode != "off" { arguments.append("--speculative-lookup") }
    let step = "model=\(model) pass=\(pass) \(cell.label)"
    let before = host()
    let result = run(binary, arguments, in: releaseDirectory)
    let after = host()
    let lines = result.output.split(separator: "\n").map(String.init)
    guard result.status == 0, let tpsLine = lines.first(where: { $0.contains("TPS") && $0.contains("avg=") }) else {
        fail(step, "exit=\(result.status) tail=\(lines.suffix(6).joined(separator: " | "))")
        return nil
    }
    func number(after key: String) -> Double? {
        tpsLine.range(of: key).flatMap { Double(tpsLine[$0.upperBound...].prefix { $0 != " " }) }
    }
    guard let tps = number(after: "avg="), let sigma = number(after: "σ="), tps > 0 else {
        fail(step, "unparsable TPS line: \(tpsLine)")
        return nil
    }
    let accepted = lines.first { $0.contains("Drafts") }
        .map { $0.split(separator: " ").dropFirst().joined(separator: " ") } ?? "no drafts"
    let output = result.output.components(separatedBy: "--- OUTPUT ---").last?
        .components(separatedBy: "--- END OUTPUT ---").first ?? ""
    let valid = before.quiet && after.quiet && sigma / tps <= 0.03
    log("result \(step) \(valid ? "valid" : "INVALID") tps=\(tps) sigma=\(sigma) \(accepted) before:\(before.text) after:\(after.text)")
    return Measurement(tps: tps, sigma: sigma, accepted: accepted, output: output, valid: valid)
}

func ratio(_ numerator: Measurement?, _ denominator: Measurement?) -> String {
    guard let numerator, let denominator else { return "failed" }
    let mark = numerator.valid && denominator.valid ? "" : "*"
    return String(format: "%.2f×%@", numerator.tps / denominator.tps, mark)
}

let commit = trimmed(run("/usr/bin/git", ["rev-parse", "HEAD"]).output)
let dirty = !trimmed(run("/usr/bin/git", ["status", "--porcelain", "--untracked-files=no"]).output).isEmpty
let chip = trimmed(run("/usr/sbin/sysctl", ["-n", "machdep.cpu.brand_string"]).output)
let memory = (Int64(trimmed(run("/usr/sbin/sysctl", ["-n", "hw.memsize"]).output)) ?? 0) >> 30
log("start commit=\(commit)\(dirty ? " (dirty)" : "") machine=\(chip) \(memory)GB os=\(ProcessInfo.processInfo.operatingSystemVersionString) runs=\(options.runs) tokens=\(options.tokens) variants=\(options.variants.map { "\($0.label)=\($0.binary)" }) log=\(logPath)")

for product in ["PerformanceBenchmark", "Basalt"] {
    log("build release \(product)")
    let build = run("/usr/bin/swift", ["build", "-c", "release", "--product", product])
    guard build.status == 0 else {
        fail("build \(product)", build.output.split(separator: "\n").suffix(6).joined(separator: " | "))
        exit(1)
    }
}
let benchmark = releaseDirectory + "/PerformanceBenchmark"
for variant in options.variants where !FileManager.default.isExecutableFile(atPath: variant.binary) {
    fail("variant \(variant.label)", "not an executable: \(variant.binary)")
    exit(1)
}

var prompts: [String: String] = [:]
for workload in ["code", "prose"] {
    let path = repo + "/benchmarks/speculative-lookup/\(workload).txt"
    guard let text = try? String(contentsOfFile: path, encoding: .utf8) else {
        fail("workload \(workload)", "missing \(path)")
        exit(1)
    }
    prompts[workload] = text
}

var cells: [Cell] = []
for workload in ["code", "prose"] {
    for temperature in ["0", "0.7"] {
        for mode in ["off", "on"] + options.variants.map(\.label) {
            cells.append(Cell(workload: workload, temperature: temperature, mode: mode))
        }
    }
}

var summary: [String] = []
for model in options.models {
    guard let revision = revision(of: model, inspector: releaseDirectory + "/Basalt") else { continue }
    acquireLock()
    var results: [Int: [Cell: Measurement]] = [:]
    for (pass, order) in [cells, cells.reversed()].enumerated() {
        for cell in order {
            waitForQuietHost()
            let binary = options.variants.first { $0.label == cell.mode }?.binary ?? benchmark
            if let measurement = measure(cell, model: model, binary: binary, prompt: prompts[cell.workload]!, pass: pass + 1) {
                results[pass + 1, default: [:]][cell] = measurement
            }
        }
    }
    releaseLock()

    summary.append("")
    summary.append("\(model) @ \(revision), commit \(commit.prefix(7)), \(chip) \(memory)GB, \(options.runs) runs × \(options.tokens) tokens per cell")
    summary.append("| workload | T | on / off (pass 1, pass 2) | drafts, pass 1 | same text as off |"
        + options.variants.map { " \($0.label) / on (pass 1, pass 2) |" }.joined())
    summary.append("|---|---|---|---|---|" + String(repeating: "---|", count: options.variants.count))
    for workload in ["code", "prose"] {
        for temperature in ["0", "0.7"] {
            func cell(_ mode: String) -> Cell { Cell(workload: workload, temperature: temperature, mode: mode) }
            func pair(_ mode: String, over base: String) -> String {
                [1, 2].map { ratio(results[$0]?[cell(mode)], results[$0]?[cell(base)]) }.joined(separator: ", ")
            }
            let lookupModes = ["on"] + options.variants.map(\.label)
            let sameText = [1, 2].allSatisfy { pass in
                guard let off = results[pass]?[cell("off")] else { return false }
                return lookupModes.allSatisfy { results[pass]?[cell($0)]?.output == off.output }
            }
            let alpha = results[1]?[cell("on")]?.accepted ?? "n/a"
            summary.append("| \(workload) | \(temperature) | \(pair("on", over: "off")) | \(alpha) | \(sameText) |"
                + options.variants.map { " \(pair($0.label, over: "on")) |" }.joined())
        }
    }
}

summary.append("")
summary.append("* a cell in the ratio was INVALID: σ above 3% of the mean, or load above \(maxLoad) or a CI job before or after it")
for line in summary { log("summary \(line)") }
log("done failures=\(failures)")
exit(failures == 0 ? 0 : 1)
