#!/usr/bin/env swift
import Foundation

let usage = """
    Usage:
      swift scripts/compose_mom_bench.swift <basalt binary> <workload.json> <compose file>...

    Serves each compose file with `basalt compose up` (one at a time, holding the GPU lock) and sends
    the same workload to /v1/pipeline (#412 phase 2). The workload is a JSON list of
    {kind, body, expect}: body is a /v1/pipeline request, expect maps question IDs to the right label.
    Per compose file it logs decision accuracy, cloud cost per 1000 requests, p50/p95 latency and
    the milliseconds spent in local stages (stages without a cost), to logs/compose-mom-<date>.log.
    Cloud keys come from the environment the compose file names; they are never logged.
    """

let arguments = CommandLine.arguments
guard arguments.count >= 4 else {
    print(usage)
    exit(2)
}
let (basalt, workloadPath, files) = (arguments[1], arguments[2], Array(arguments.dropFirst(3)))
let port = 18_490
let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let day = ISO8601DateFormatter().string(from: Date()).prefix(10)
let logURL = repo.appending(path: "logs/compose-mom-\(day).log")
try? FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)

func log(_ line: String) {
    let stamped = "\(ISO8601DateFormatter().string(from: Date())) \(line)\n"
    print(stamped, terminator: "")
    if let handle = try? FileHandle(forWritingTo: logURL) {
        handle.seekToEndOfFile()
        handle.write(Data(stamped.utf8))
        try? handle.close()
    } else {
        try? Data(stamped.utf8).write(to: logURL)
    }
}

enum GPULock {
    static let path = "/private/tmp/basalt-gpu.lock"
    static let owner = String(ProcessInfo.processInfo.processIdentifier)

    static func take() {
        while (try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: false)) == nil { sleep(2) }
        try? owner.write(toFile: path + "/owner", atomically: true, encoding: .utf8)
    }

    static func release() {
        if (try? String(contentsOfFile: path + "/owner", encoding: .utf8)) == owner { try? FileManager.default.removeItem(atPath: path) }
    }
}

struct Outcome {
    var ms: Int
    var correct = 0
    var asked = 0
    var cost = 0.0
    var localMs = 0
    var routes: [String] = []
}

func post(_ body: Any) async -> (json: [String: Any], ms: Int)? {
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/pipeline")!)
    (request.httpMethod, request.timeoutInterval) = ("POST", 600)
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try? JSONSerialization.data(withJSONObject: body)
    let started = Date()
    guard let (data, _) = try? await URLSession.shared.data(for: request),
          let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
    return (json, Int(Date().timeIntervalSince(started) * 1000))
}

func outcome(_ json: [String: Any], ms: Int, expect: [String: String]) -> Outcome {
    var result = Outcome(ms: ms)
    let answers = json["answers"] as? [String: [String: Any]] ?? [:]
    for (question, label) in expect {
        result.asked += 1
        if answers[question]?["label"] as? String == label { result.correct += 1 }
    }
    for stage in ((json["receipt"] as? [String: Any])?["stages"] as? [[String: Any]]) ?? [] {
        if let cost = stage["cost"] as? Double { result.cost += cost } else { result.localMs += stage["ms"] as? Int ?? 0 }
        if stage["stage"] as? String == "answer" { result.routes.append("\(stage["model"] as? String ?? "?")") }
    }
    return result
}

func percentile(_ values: [Int], _ p: Double) -> Int {
    let sorted = values.sorted()
    return sorted.isEmpty ? 0 : sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * p + 0.5))]
}

func serve(_ file: String) throws -> Process {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: basalt)
    process.arguments = ["compose", "up", "-f", file, "--", "--port", String(port), "--session-log", "off"]
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle(forWritingAtPath: "/dev/null")
    try process.run()
    return process
}

func ready() async -> Bool {
    for _ in 0 ..< 300 {
        if let (_, response) = try? await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/api/v1/health")!),
           (response as? HTTPURLResponse)?.statusCode == 200 { return true }
        try? await Task.sleep(for: .seconds(2))
    }
    return false
}

let workload = (try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: workloadPath)))) as? [[String: Any]] ?? []
let head = (try? String(contentsOf: repo.appending(path: ".git/HEAD"), encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "?"
log("HEADER workload=\(workloadPath) requests=\(workload.count) basalt=\(basalt) head=\(head)")
var failed = false
for file in files {
    GPULock.take()
    defer { GPULock.release() }
    let process = try serve(file)
    defer { process.interrupt(); process.waitUntilExit() }
    guard await ready() else {
        log("FAIL \(file): serve did not become ready")
        failed = true
        continue
    }
    var outcomes: [String: [Outcome]] = [:]
    for item in workload {
        let kind = item["kind"] as? String ?? "?"
        guard let body = item["body"], let (json, ms) = await post(body), json["error"] == nil else {
            log("FAIL \(file) \(kind): no answer")
            failed = true
            continue
        }
        outcomes[kind, default: []].append(outcome(json, ms: ms, expect: item["expect"] as? [String: String] ?? [:]))
    }
    let all = outcomes.values.flatMap { $0 }
    let (correct, asked) = (all.map(\.correct).reduce(0, +), all.map(\.asked).reduce(0, +))
    let cost = all.map(\.cost).reduce(0, +)
    log(String(format: "RESULT %@: requests %d, labels right %d/%d (%.1f%%), cloud $%.5f ($%.4f per 1000), p50 %d ms, p95 %d ms, local stage time %.1f s",
               (file as NSString).lastPathComponent, all.count, correct, asked, 100 * Double(correct) / Double(max(asked, 1)), cost, 1000 * cost / Double(max(all.count, 1)),
               percentile(all.map(\.ms), 0.5), percentile(all.map(\.ms), 0.95), Double(all.map(\.localMs).reduce(0, +)) / 1000))
    for (kind, items) in outcomes.sorted(by: { $0.key < $1.key }) {
        let models = Dictionary(items.flatMap(\.routes).map { ($0, 1) }, uniquingKeysWith: +).map { "\($0.key)×\($0.value)" }.sorted().joined(separator: " ")
        log(String(format: "  %@ %@: %d requests, labels %d/%d, p50 %d ms, cloud $%.5f, local %.1f s, answered by %@", (file as NSString).lastPathComponent, kind, items.count,
                   items.map(\.correct).reduce(0, +), items.map(\.asked).reduce(0, +), percentile(items.map(\.ms), 0.5), items.map(\.cost).reduce(0, +),
                   Double(items.map(\.localMs).reduce(0, +)) / 1000, models.isEmpty ? "-" : models))
    }
}
exit(failed ? 1 : 0)
