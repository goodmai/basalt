import Foundation

struct Load {
    var name: String
    var path: String
    var function: String
    var weightBytes: String
    var unit: String
}

struct Config {
    var gpuModels: [String] = []
    var loads: [Load] = []
    var duties = ["0.25", "0.5", "1.0"]
    var benchmark = ".build/release/PerformanceBenchmark"
    var aneLoad = ""
    var tokens = "512"
    var promptRepeat = "450"
    var iterations = "3"
    var aneSolo = 20.0

    init(_ arguments: [String]) {
        var values: [String: String] = [:]
        var iterator = arguments.makeIterator()
        while let key = iterator.next() { values[key] = iterator.next() }
        gpuModels = (values["--gpu-models"] ?? "").split(separator: ",").map(String.init)
        loads = (values["--ane"] ?? "").split(separator: ",").compactMap { entry in
            let parts = entry.split(separator: "=", maxSplits: 1).map(String.init)
            let fields = parts.count == 2 ? parts[1].split(separator: ":").map(String.init) : []
            return (3...4).contains(fields.count) ? Load(name: parts[0], path: fields[0], function: fields[1], weightBytes: fields[2], unit: fields.count == 4 ? fields[3] : "ane") : nil
        }
        duties = values["--duties"].map { $0.split(separator: ",").map(String.init) } ?? duties
        benchmark = values["--benchmark"] ?? benchmark
        aneLoad = values["--ane-load"] ?? aneLoad
        tokens = values["--tokens"] ?? tokens
        promptRepeat = values["--prompt-repeat"] ?? promptRepeat
        iterations = values["--iterations"] ?? iterations
        aneSolo = Double(values["--ane-solo-seconds"] ?? "") ?? aneSolo
        guard !gpuModels.isEmpty, !loads.isEmpty, !aneLoad.isEmpty else {
            FileHandle.standardError.write(Data("usage: ane_contention.swift --gpu-models a,b --ane name=path:function:bytes[:ane|gpu],... --ane-load <compiled ane_load> [--duties 0.25,0.5,1.0]\n".utf8))
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

func field(_ name: String, in text: String) -> Double? {
    guard let range = text.range(of: "\(name)=") else { return nil }
    return Double(text[range.upperBound...].prefix { "0123456789.".contains($0) })
}

struct GPULock {
    static let path = "/private/tmp/basalt-gpu.lock"
    let owner = "\(ProcessInfo.processInfo.processIdentifier)"

    func take(_ log: Log) async {
        while (try? FileManager.default.createDirectory(atPath: Self.path, withIntermediateDirectories: false)) == nil {
            log.line("WAIT \(Self.path) is held")
            try? await Task.sleep(for: .seconds(30))
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

func loadAverage() -> Double {
    var loads = [0.0, 0.0, 0.0]
    return getloadavg(&loads, 3) > 0 ? loads[0] : .nan
}

func waitQuiet(_ log: Log) {
    var waited = false
    while !shell(["pgrep", "-x", "swift-build|clang|swift-test"]).isEmpty || loadAverage() > 3 {
        if !waited { log.line(String(format: "WAIT machine busy: load %.2f or a compiler is running", loadAverage())) }
        waited = true
        Thread.sleep(forTimeInterval: 30)
    }
}

func benchmark(_ model: String, _ config: Config) -> (tps: Double?, start: Double, end: Double) {
    let start = Date().timeIntervalSince1970
    let output = shell([config.benchmark, "-m", model, "-t", config.tokens, "-i", config.iterations, "--prompt-repeat", config.promptRepeat, "--seed", "7"])
    let end = Date().timeIntervalSince1970
    let line = output.split(separator: "\n").first { $0.contains("TPS") && $0.contains("avg=") }.map(String.init) ?? ""
    return (field("avg", in: line), start, end)
}

func aneArguments(_ load: Load, duty: String, seconds: Double, window: Bool) -> [String] {
    [
        "--model", load.path, "--function", load.function, "--duty", duty, "--seconds", String(seconds),
        "--weight-bytes", load.weightBytes, "--window", window ? "5" : "0", "--unit", load.unit,
    ]
}

func aneSolo(_ load: Load, duty: String, _ config: Config) -> Double? {
    field("per_s", in: shell([config.aneLoad] + aneArguments(load, duty: duty, seconds: config.aneSolo, window: false)))
}

func overlap(_ load: Load, duty: String, model: String, _ config: Config) -> (tps: Double?, ane: Double?) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: config.aneLoad)
    process.arguments = aneArguments(load, duty: duty, seconds: 86_400, window: true)
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = Pipe()
    guard (try? process.run()) != nil else { return (nil, nil) }
    Thread.sleep(forTimeInterval: 15)
    let run = benchmark(model, config)
    process.terminate()
    let text = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    process.waitUntilExit()
    let rates = text.split(separator: "\n").filter { $0.hasPrefix("window") }.compactMap { line -> Double? in
        guard let time = field("t", in: String(line)), time > run.start + 5, time <= run.end else { return nil }
        return field("per_s", in: String(line))
    }
    return (run.tps, rates.isEmpty ? nil : rates.reduce(0, +) / Double(rates.count))
}

func percent(_ value: Double?, of base: Double?) -> String {
    guard let value, let base, base > 0 else { return "n/a" }
    return String(format: "%+.1f%%", (value / base - 1) * 100)
}

func measure(_ model: String, _ config: Config, _ log: Log) {
    waitQuiet(log)
    let solo = benchmark(model, config)
    guard let soloTPS = solo.tps else { return log.fail("\(model) solo benchmark printed no TPS") }
    log.line("RESULT \(model) gpu-solo tps=\(String(format: "%.2f", soloTPS)) load=\(String(format: "%.2f", loadAverage()))")
    for load in config.loads {
        for duty in config.duties {
            waitQuiet(log)
            let alone = aneSolo(load, duty: duty, config)
            let both = overlap(load, duty: duty, model: model, config)
            let gb = (Double(load.weightBytes) ?? 0) / 1e9
            log.line("RESULT \(model) ane=\(load.name) duty=\(duty) gpu_tps=\(both.tps.map { String(format: "%.2f", $0) } ?? "n/a") gpu_drop=\(percent(both.tps, of: soloTPS)) ane_solo_per_s=\(alone.map { String(format: "%.2f", $0) } ?? "n/a") ane_overlap_per_s=\(both.ane.map { String(format: "%.2f", $0) } ?? "n/a") ane_change=\(percent(both.ane, of: alone)) ane_solo_gb_s=\(alone.map { String(format: "%.1f", $0 * gb) } ?? "n/a") ane_overlap_gb_s=\(both.ane.map { String(format: "%.1f", $0 * gb) } ?? "n/a") load=\(String(format: "%.2f", loadAverage()))")
            if both.tps == nil || both.ane == nil { log.fail("\(model) \(load.name) duty=\(duty) missing a measurement") }
        }
    }
}

let config = Config(Array(CommandLine.arguments.dropFirst()))
let date = String(ISO8601DateFormatter().string(from: Date()).prefix(10))
try FileManager.default.createDirectory(atPath: "logs", withIntermediateDirectories: true)
let log = try Log("logs/ane-contention-\(date).log")
let head = shell(["git", "rev-parse", "HEAD"]).trimmingCharacters(in: .whitespacesAndNewlines)
let chip = shell(["sysctl", "-n", "machdep.cpu.brand_string"]).trimmingCharacters(in: .whitespacesAndNewlines)
log.line("RUN commit=\(head) chip=\(chip) memory=\(ProcessInfo.processInfo.physicalMemory) tokens=\(config.tokens) promptRepeat=\(config.promptRepeat) iterations=\(config.iterations)")
let lock = GPULock()
for model in config.gpuModels {
    await lock.take(log)
    measure(model, config, log)
    lock.release(log)
}
log.line("END failures=\(log.failures)")
exit(log.failures == 0 ? 0 : 1)
