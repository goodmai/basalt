#!/usr/bin/env swift
import Foundation

let usage = """
    Usage: swift -O scripts/ocr_bench.swift <basalt-binary> <document.pdf> <truth.txt> <model-repo>[@<resolution>]...

    Reads the document once per step with `basalt ocr --format json` (release build), holding the GPU
    lock for that step only, and logs per step: pages/s and tok/s over the summed page times, the
    peak memory footprint from /usr/bin/time -l, the generated tokens, and the
    character error rate of the joined page Markdown against <truth.txt> (Levenshtein distance over
    the truth's characters). Each model is checked with `basalt models inspect --json` first.
    Lines go to stdout and logs/ocr_bench-<date>.log; a failed step is logged and the run exits 1.
    """

let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.count >= 4 else {
    print(usage)
    exit(1)
}
let (basalt, document, truthPath, steps) = (arguments[0], arguments[1], arguments[2], Array(arguments.dropFirst(3)))
let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let day = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate])
let logURL = repo.appending(path: "logs/ocr_bench-\(day).log")
try? FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)

func log(_ line: String) {
    let stamped = "\(ISO8601DateFormatter().string(from: Date())) \(line)"
    print(stamped)
    guard let handle = try? FileHandle(forWritingTo: logURL) else {
        try? (stamped + "\n").write(to: logURL, atomically: true, encoding: .utf8)
        return
    }
    handle.seekToEndOfFile()
    handle.write(Data((stamped + "\n").utf8))
    try? handle.close()
}

func run(_ executable: String, _ arguments: [String]) throws -> (status: Int32, out: Data, err: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    let errURL = FileManager.default.temporaryDirectory.appending(path: "ocr-bench-\(UUID().uuidString).err")
    FileManager.default.createFile(atPath: errURL.path, contents: nil)
    defer { try? FileManager.default.removeItem(at: errURL) }
    let out = Pipe()
    process.standardOutput = out
    process.standardError = try FileHandle(forWritingTo: errURL)
    try process.run()
    let data = out.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (process.terminationStatus, data, (try? String(contentsOf: errURL, encoding: .utf8)) ?? "")
}

func editDistance(_ a: [UInt8], _ b: [UInt8]) -> Int {
    var previous = Array(0 ... b.count), current = [Int](repeating: 0, count: b.count + 1)
    for i in 1 ... max(a.count, 1) where !a.isEmpty {
        current[0] = i
        for j in 1 ... max(b.count, 1) where !b.isEmpty {
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1))
        }
        swap(&previous, &current)
    }
    return a.isEmpty ? b.count : previous[b.count]
}

enum GPULock {
    static let path = "/private/tmp/basalt-gpu.lock"
    static let owner = String(ProcessInfo.processInfo.processIdentifier)

    static func holding<T>(_ body: () throws -> T) rethrows -> T {
        while (try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: false)) == nil { sleep(10) }
        try? owner.write(toFile: path + "/owner", atomically: true, encoding: .utf8)
        defer {
            if (try? String(contentsOfFile: path + "/owner", encoding: .utf8)) == owner { try? FileManager.default.removeItem(atPath: path) }
        }
        return try body()
    }
}

func measure(_ model: String, resolution: String?, truth: [UInt8]) throws -> String {
    let inspect = try run(basalt, ["models", "inspect", model, "--json"])
    let report = (try? JSONSerialization.jsonObject(with: inspect.out)) as? [String: Any]
    guard inspect.status == 0, report?["intact"] as? Bool == true else { throw NSError(domain: "inspect: snapshot not intact", code: 1) }
    let ocr = ["ocr", document, "--model", model, "--format", "json"] + (resolution.map { ["--resolution", $0] } ?? [])
    let result = try GPULock.holding { try run("/usr/bin/time", ["-l", basalt] + ocr) }
    guard result.status == 0, let json = (try? JSONSerialization.jsonObject(with: result.out)) as? [String: Any],
          let pages = json["pages"] as? [[String: Any]] else {
        throw NSError(domain: "ocr exited \(result.status): \(result.err.suffix(400))", code: 2)
    }
    let ms = pages.compactMap { $0["ms"] as? Int }.reduce(0, +)
    let tokens = pages.compactMap { $0["tokens"] as? Int }.reduce(0, +)
    let text = Array(pages.compactMap { $0["markdown"] as? String }.joined(separator: "\n").utf8)
    let peak = result.err.split(separator: "\n").first { $0.contains("peak memory footprint") }
        .flatMap { Double($0.trimmingCharacters(in: .whitespaces).split(separator: " ").first ?? "") } ?? 0
    let seconds = Double(ms) / 1000
    return String(
        format: "pages=%d pages/s=%.3f tok/s=%.1f tokens=%d peak=%.2fGB cer=%.4f",
        pages.count, Double(pages.count) / seconds, Double(tokens) / seconds, tokens, peak / 1e9,
        Double(editDistance(text, truth)) / Double(max(truth.count, 1)))
}

let head = (try? run("/usr/bin/git", ["-C", repo.path, "rev-parse", "HEAD"]).out).map { String(decoding: $0, as: UTF8.self) } ?? "?"
let chip = (try? run("/usr/sbin/sysctl", ["-n", "machdep.cpu.brand_string", "hw.memsize"]).out).map { String(decoding: $0, as: UTF8.self) } ?? "?"
log("HEADER commit=\(head.trimmingCharacters(in: .whitespacesAndNewlines)) machine=\(chip.split(separator: "\n").joined(separator: " ")) document=\(document)")
let truth = Array(try String(contentsOfFile: truthPath, encoding: .utf8).utf8)
var failed = false
for step in steps {
    let parts = step.split(separator: "@").map(String.init)
    let (model, resolution) = (parts[0], parts.count > 1 ? parts[1] : nil)
    do {
        log("OK \(model) resolution=\(resolution ?? "default") \(try measure(model, resolution: resolution, truth: truth))")
    } catch {
        failed = true
        log("FAIL \(model) resolution=\(resolution ?? "default") \((error as NSError).domain)")
    }
}
exit(failed ? 1 : 0)
