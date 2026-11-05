#!/usr/bin/env swift
import Foundation

let usage = """
    Usage:
      swift scripts/acceptance.swift --model REPO --output REPORT [--repo CHECKOUT] [--baseline REPORT] [--port 8090]
      swift scripts/acceptance.swift --baseline REPORT --candidate REPORT

    Acceptance gate between a baseline commit and a candidate on one model. Measuring a checkout
    (default: this one; it needs a release build) runs the fixed task pool of
    scripts/task_pool_eval.swift against `basalt serve` for accuracy and tok/s, then the golden-logit
    test for its max KL against Tests/BasaltTests/Fixtures/golden, and writes REPORT. With --baseline
    the new report is compared with the baseline's; with --candidate two saved reports are compared.
    Thresholds come from scripts/acceptance.json. Exits 1 on any regression or failed step, and
    writes one log in logs/ with both commits.
    """

struct Report: Codable, Equatable {
    var commit: String
    var model: String
    var chip: String
    var golden: String
    var tasks: Int
    var concurrency: Int
    var tokens: Int
    var accuracy: Double
    var tokPerSec: Double
    var maxKL: Double
}

struct Config: Codable {
    var tasks: Int
    var concurrency: Int
    var tokens: Int
    var maxAccuracyDrop: Double
    var maxKLIncrease: Double
    var minTokPerSecRatio: Double
}

struct Gate {
    let name: String
    let baseline: Double
    let candidate: Double
    let limit: Double
    let pass: Bool
}

func gates(baseline: Report, candidate: Report, config: Config) -> [Gate] {
    let accuracyFloor = baseline.accuracy - config.maxAccuracyDrop
    let klCeiling = baseline.maxKL + config.maxKLIncrease
    let speedFloor = baseline.tokPerSec * config.minTokPerSecRatio
    return [
        Gate(name: "accuracy", baseline: baseline.accuracy, candidate: candidate.accuracy, limit: accuracyFloor,
             pass: candidate.accuracy >= accuracyFloor - 1e-9),
        Gate(name: "golden_kl", baseline: baseline.maxKL, candidate: candidate.maxKL, limit: klCeiling,
             pass: candidate.maxKL <= klCeiling + 1e-12),
        Gate(name: "tok_s", baseline: baseline.tokPerSec, candidate: candidate.tokPerSec, limit: speedFloor,
             pass: candidate.tokPerSec >= speedFloor - 1e-9),
    ]
}

struct Options {
    var model: String?
    var output: String?
    var repo: String?
    var baseline: String?
    var candidate: String?
    var port = 8090
}

func parse(_ arguments: [String]) -> Options? {
    var options = Options()
    var rest = arguments[...]
    while let flag = rest.popFirst() {
        guard let value = rest.popFirst() else { return nil }
        switch flag {
        case "--model": options.model = value
        case "--output": options.output = value
        case "--repo": options.repo = value
        case "--baseline": options.baseline = value
        case "--candidate": options.candidate = value
        case "--port": guard let port = Int(value), port > 0 else { return nil }; options.port = port
        default: return nil
        }
    }
    let measuring = options.model != nil && options.output != nil && options.candidate == nil
    let comparing = options.model == nil && options.output == nil && options.baseline != nil && options.candidate != nil
    return measuring || comparing ? options : nil
}

guard let options = parse(Array(CommandLine.arguments.dropFirst())) else {
    print(usage)
    exit(CommandLine.arguments.contains("-h") || CommandLine.arguments.contains("--help") ? 0 : 2)
}

let scripts = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.deletingLastPathComponent().path
let ownRepo = URL(fileURLWithPath: scripts).deletingLastPathComponent().path
let lockDirectory = ProcessInfo.processInfo.environment["BASALT_GPU_LOCK"] ?? "/private/tmp/basalt-gpu.lock"
let pid = String(ProcessInfo.processInfo.processIdentifier)
let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
let logPath = ownRepo + "/logs/acceptance-\(stamp).log"
try? FileManager.default.createDirectory(atPath: ownRepo + "/logs", withIntermediateDirectories: true)
FileManager.default.createFile(atPath: logPath, contents: nil)
guard let logHandle = FileHandle(forWritingAtPath: logPath) else {
    print("FAIL cannot open the log file \(logPath)")
    exit(1)
}

func log(_ line: String) {
    let stamped = "\(ISO8601DateFormatter().string(from: Date())) \(line)"
    print(stamped)
    fflush(stdout)
    try? logHandle.write(contentsOf: Data((stamped + "\n").utf8))
}

func fail(_ step: String, _ detail: String) -> Never {
    log("FAIL \(step) \(detail)")
    log("done regressions=unknown log=\(logPath)")
    exit(1)
}

func trimmed(_ text: String) -> String { text.trimmingCharacters(in: .whitespacesAndNewlines) }

func run(_ executable: String, _ arguments: [String], in directory: String, environment: [String: String] = [:]) -> (status: Int32, output: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    process.currentDirectoryURL = URL(fileURLWithPath: directory)
    process.environment = ProcessInfo.processInfo.environment.merging(environment) { $1 }
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    do { try process.run() } catch { return (-1, "\(error)") }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (process.terminationStatus, String(decoding: data, as: UTF8.self))
}

func decode<T: Decodable>(_ type: T.Type, _ path: String) -> T {
    guard let data = FileManager.default.contents(atPath: path) else { fail("read", "cannot read \(path)") }
    do { return try JSONDecoder().decode(type, from: data) } catch { fail("decode", "\(path): \(error)") }
}

func lockOwner() -> String {
    trimmed((try? String(contentsOfFile: lockDirectory + "/owner", encoding: .utf8)) ?? "")
}

func acquireLock() {
    var waiting = false
    while mkdir(lockDirectory, 0o755) != 0 {
        guard errno == EEXIST else { fail("lock", "cannot create \(lockDirectory): \(String(cString: strerror(errno)))") }
        let owner = lockOwner()
        if let holder = Int32(owner), kill(holder, 0) != 0, errno == ESRCH {
            unlink(lockDirectory + "/owner")
            rmdir(lockDirectory)
            log("reclaimed the GPU lock from pid \(holder), which no longer runs")
            continue
        }
        if !waiting { log("waiting for the GPU lock held by pid \(owner.isEmpty ? "unknown" : owner)") }
        waiting = true
        Thread.sleep(forTimeInterval: 15)
    }
    guard (try? pid.write(toFile: lockDirectory + "/owner", atomically: true, encoding: .utf8)) != nil else {
        rmdir(lockDirectory)
        fail("lock", "cannot write the owner file")
    }
    log("lock acquired by pid \(pid)")
}

func releaseLock() {
    guard lockOwner() == pid else {
        log("FAIL release lock held by pid \(lockOwner()), left untouched")
        return
    }
    unlink(lockDirectory + "/owner")
    rmdir(lockDirectory)
    log("lock released")
}

func waitForHealth(_ url: String, server: Process) -> Bool {
    let deadline = Date().addingTimeInterval(900)
    while Date() < deadline, server.isRunning {
        if trimmed(run("/usr/bin/curl", ["-s", "-o", "/dev/null", "-w", "%{http_code}", url + "/api/v1/health"], in: ownRepo).output) == "200" {
            return true
        }
        Thread.sleep(forTimeInterval: 2)
    }
    return false
}

func taskPool(repo: String, model: String, config: Config, port: Int) -> (accuracy: Double, tokPerSec: Double) {
    let binary = repo + "/.build/release/Basalt"
    guard FileManager.default.isExecutableFile(atPath: binary) else { fail("binary", "no release build at \(binary)") }
    let server = Process()
    server.executableURL = URL(fileURLWithPath: binary)
    server.arguments = ["serve", "--model", model, "--port", String(port), "--max-concurrent", String(config.concurrency)]
    server.currentDirectoryURL = URL(fileURLWithPath: repo)
    server.standardOutput = FileHandle(forWritingAtPath: "/dev/null")
    server.standardError = FileHandle(forWritingAtPath: "/dev/null")
    func stop() {
        if server.isRunning { server.terminate(); server.waitUntilExit() }
        releaseLock()
    }
    func abort(_ step: String, _ detail: String) -> Never {
        stop()
        fail(step, detail)
    }
    acquireLock()
    do { try server.run() } catch { abort("serve", "cannot start \(binary): \(error)") }
    let url = "http://127.0.0.1:\(port)"
    guard waitForHealth(url, server: server) else { abort("serve", "\(model) never answered \(url)/api/v1/health") }
    let result = run("/usr/bin/env", ["swift", scripts + "/task_pool_eval.swift", "--label", "acceptance", "--url", url,
                                      "--tasks", String(config.tasks), "--concurrency", String(config.concurrency),
                                      "--tokens", String(config.tokens)], in: ownRepo)
    guard result.status == 0, let summary = result.output.matches(of: #/accuracy=([\d.]+) .* tok_s=([\d.]+)/#).last,
          let accuracy = Double(summary.output.1), let tokPerSec = Double(summary.output.2)
    else { abort("task pool", "exit=\(result.status) \(result.output.suffix(400))") }
    stop()
    log("task pool model=\(model) accuracy=\(accuracy) tok_s=\(tokPerSec)")
    return (accuracy, tokPerSec)
}

func goldenKL(repo: String, model: String) -> Double {
    let result = run("/usr/bin/env", ["swift", "test", "-c", "release", "--filter", "GoldenLogitsRealModelTests/goldenLogits"], in: repo,
                     environment: ["BASALT_E2E": "1", "BASALT_E2E_GOLDEN_MODELS": model])
    guard let line = result.output.matches(of: #/result GOLDEN max KL ([0-9.eE+-]+)(.*)/#).last,
          let kl = Double(line.output.1)
    else { fail("golden", "exit=\(result.status) no GOLDEN result line: \(result.output.suffix(400))") }
    log("golden model=\(model) max_kl=\(kl)\(line.output.2) exit=\(result.status)")
    return kl
}

func measure(repo: String, model: String, config: Config, port: Int) -> Report {
    let commit = trimmed(run("/usr/bin/git", ["rev-parse", "HEAD"], in: repo).output)
    let dirty = trimmed(run("/usr/bin/git", ["status", "--porcelain", "--untracked-files=no"], in: repo).output)
    guard dirty.isEmpty else { fail("checkout", "\(repo) has uncommitted changes, so commit \(commit) would not name what ran") }
    let golden = trimmed(run("/usr/bin/git", ["rev-parse", "HEAD:Tests/BasaltTests/Fixtures/golden"], in: repo).output)
    log("measure commit=\(commit) repo=\(repo) model=\(model) tasks=\(config.tasks) concurrency=\(config.concurrency) tokens=\(config.tokens)")
    let pool = taskPool(repo: repo, model: model, config: config, port: port)
    let kl = goldenKL(repo: repo, model: model)
    return Report(commit: commit, model: model, chip: chip, golden: golden, tasks: config.tasks, concurrency: config.concurrency,
                  tokens: config.tokens, accuracy: pool.accuracy, tokPerSec: pool.tokPerSec, maxKL: kl)
}

func compare(_ baseline: Report, _ candidate: Report, config: Config) -> Int {
    log("compare model=\(candidate.model) baseline=\(baseline.commit) candidate=\(candidate.commit)")
    let setup = { (report: Report) in
        "\(report.model) on \(report.chip), goldens \(report.golden), \(report.tasks) tasks x\(report.concurrency) at \(report.tokens) tokens"
    }
    guard setup(baseline) == setup(candidate) else {
        log("FAIL comparable baseline \(setup(baseline)), candidate \(setup(candidate))")
        return 1
    }
    let results = gates(baseline: baseline, candidate: candidate, config: config)
    for gate in results {
        log("gate \(gate.name) baseline=\(gate.baseline) candidate=\(gate.candidate) limit=\(gate.limit) \(gate.pass ? "PASS" : "FAIL")")
    }
    return results.filter { !$0.pass }.count
}

let config = decode(Config.self, scripts + "/acceptance.json")
let chip = trimmed(run("/usr/sbin/sysctl", ["-n", "machdep.cpu.brand_string"], in: ownRepo).output)
log("acceptance chip=\(chip) config=\(scripts)/acceptance.json log=\(logPath)")
let baseline = options.baseline.map { decode(Report.self, $0) }
var candidate: Report
if let model = options.model, let output = options.output {
    candidate = measure(repo: options.repo ?? ownRepo, model: model, config: config, port: options.port)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    do { try encoder.encode(candidate).write(to: URL(fileURLWithPath: output)) } catch { fail("report", "cannot write \(output): \(error)") }
    log("report \(output)")
} else {
    candidate = decode(Report.self, options.candidate ?? "")
}
let regressions = baseline.map { compare($0, candidate, config: config) } ?? 0
log("done regressions=\(regressions) log=\(logPath)")
exit(regressions == 0 ? 0 : 1)
