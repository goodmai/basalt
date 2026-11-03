#!/usr/bin/env swift
import Foundation

let usage = """
    Usage:
      ./scripts/batch_throughput.swift --label NAME [--url http://127.0.0.1:8080] [--concurrency 1,2,4,8] [--tokens N] [--runs N]

    Aggregate decode throughput of a running `basalt serve` at several concurrency levels.
    Each run sends N different greedy chat requests at once and divides the completion
    tokens the server reports by the wall time until the last one answered. Start the
    server yourself, once with --batch-slots 1 and once with --batch-slots N, both with
    --max-concurrent at least the highest level, and give each pass its own --label.
    """

struct Options {
    var label = ""
    var url = "http://127.0.0.1:8080"
    var concurrency = [1, 2, 4, 8]
    var tokens = 128
    var runs = 3
}

func parse(_ arguments: [String]) -> Options? {
    var options = Options()
    var rest = arguments[...]
    while let flag = rest.popFirst() {
        guard flag != "-h", flag != "--help", let value = rest.popFirst() else { return nil }
        switch flag {
        case "--label": options.label = value
        case "--url": options.url = value
        case "--concurrency":
            options.concurrency = value.split(separator: ",").compactMap { Int($0) }
            guard !options.concurrency.isEmpty, options.concurrency.allSatisfy({ $0 > 0 }) else { return nil }
        case "--tokens": guard let tokens = Int(value), tokens > 0 else { return nil }; options.tokens = tokens
        case "--runs": guard let runs = Int(value), runs > 0 else { return nil }; options.runs = runs
        default: return nil
        }
    }
    return options.label.isEmpty ? nil : options
}

guard let options = parse(Array(CommandLine.arguments.dropFirst())) else {
    print(usage)
    exit(CommandLine.arguments.contains("-h") || CommandLine.arguments.contains("--help") ? 0 : 2)
}

let repo = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    .deletingLastPathComponent().deletingLastPathComponent().path
let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
let logPath = repo + "/logs/batch_throughput-\(options.label)-\(stamp).log"
try? FileManager.default.createDirectory(atPath: repo + "/logs", withIntermediateDirectories: true)
FileManager.default.createFile(atPath: logPath, contents: nil)
guard let logHandle = FileHandle(forWritingAtPath: logPath) else {
    print("FAIL cannot open the log file \(logPath)")
    exit(1)
}

func log(_ line: String) {
    let stamped = "\(ISO8601DateFormatter().string(from: Date())) \(line)"
    print(stamped)
    try? logHandle.write(contentsOf: Data((stamped + "\n").utf8))
}

func shell(_ arguments: [String]) -> String {
    let process = Process()
    let pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = arguments
    process.standardOutput = pipe
    try? process.run()
    process.waitUntilExit()
    return String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

struct Reply: Decodable {
    struct Usage: Decodable { let completion_tokens: Int }
    let usage: Usage
}

func complete(_ index: Int) async throws -> Int {
    var request = URLRequest(url: URL(string: options.url + "/v1/chat/completions")!)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let key = ProcessInfo.processInfo.environment["BASALT_API_KEY"] {
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
    }
    let prompt = "Request \(index): write a long, detailed story about a lighthouse keeper number \(index * 7 + 3)."
    request.httpBody = try JSONSerialization.data(withJSONObject: [
        "model": "basalt", "temperature": 0, "max_tokens": options.tokens,
        "messages": [["role": "user", "content": prompt]],
    ])
    request.timeoutInterval = 600
    let (data, response) = try await URLSession.shared.data(for: request)
    guard (response as? HTTPURLResponse)?.statusCode == 200 else {
        throw NSError(domain: "batch_throughput", code: (response as? HTTPURLResponse)?.statusCode ?? -1,
                      userInfo: [NSLocalizedDescriptionKey: String(decoding: data.prefix(300), as: UTF8.self)])
    }
    return try JSONDecoder().decode(Reply.self, from: data).usage.completion_tokens
}

log("label=\(options.label) url=\(options.url) commit=\(shell(["git", "-C", repo, "rev-parse", "--short", "HEAD"])) chip=\(shell(["sysctl", "-n", "machdep.cpu.brand_string"])) memory=\(shell(["sysctl", "-n", "hw.memsize"])) tokens=\(options.tokens)")
var failures = 0
for concurrency in options.concurrency {
    for run in 1 ... options.runs {
        let started = ContinuousClock.now
        do {
            let tokens = try await withThrowingTaskGroup(of: Int.self) { group in
                for index in 0 ..< concurrency { group.addTask { try await complete(index) } }
                return try await group.reduce(0, +)
            }
            let seconds = (ContinuousClock.now - started) / .seconds(1)
            log(String(format: "label=%@ concurrency=%d run=%d tokens=%d seconds=%.2f aggregate_tok_s=%.1f",
                       options.label, concurrency, run, tokens, seconds, Double(tokens) / seconds))
        } catch {
            failures += 1
            log("FAIL label=\(options.label) concurrency=\(concurrency) run=\(run) error=\(error.localizedDescription)")
        }
    }
}
log("done label=\(options.label) failures=\(failures) log=\(logPath)")
exit(failures == 0 ? 0 : 1)
