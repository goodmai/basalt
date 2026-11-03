#!/usr/bin/env swift
import Foundation

let usage = """
    Usage:
      ./scripts/task_pool_eval.swift --label NAME [--url http://127.0.0.1:8080] [--tasks 120] [--concurrency 4]
                                     [--tokens 384] [--temperature 0] [--reasoning-effort none] [--compare LOG]

    Accuracy of a running `basalt serve` on a fixed pool of multi-step arithmetic word problems
    with exact integer answers, sent N at a time. Every task's expected and returned answer is
    logged, so two passes (say --batch-slots 1 and 4) can be compared with --compare: same
    answers, tasks only one pass solved, and the accuracy of each. The pool is generated from a
    fixed seed, so every pass sees the same tasks.
    """

struct Options {
    var label = ""
    var url = "http://127.0.0.1:8080"
    var tasks = 120
    var concurrency = 4
    var tokens = 384
    var temperature = 0.0
    var reasoningEffort = "none"
    var compare: String?
}

func parse(_ arguments: [String]) -> Options? {
    var options = Options()
    var rest = arguments[...]
    while let flag = rest.popFirst() {
        guard flag != "-h", flag != "--help", let value = rest.popFirst() else { return nil }
        switch flag {
        case "--label": options.label = value
        case "--url": options.url = value
        case "--tasks": guard let n = Int(value), n > 0 else { return nil }; options.tasks = n
        case "--concurrency": guard let n = Int(value), n > 0 else { return nil }; options.concurrency = n
        case "--tokens": guard let n = Int(value), n > 0 else { return nil }; options.tokens = n
        case "--temperature": guard let t = Double(value), t >= 0 else { return nil }; options.temperature = t
        case "--reasoning-effort": options.reasoningEffort = value
        case "--compare": options.compare = value
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
let logPath = repo + "/logs/task_pool-\(options.label)-\(stamp).log"
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

struct Generator {
    var state: UInt64
    mutating func next(_ range: ClosedRange<Int>) -> Int {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return range.lowerBound + Int((state >> 33) % UInt64(range.count))
    }
}

struct Task {
    let id: Int
    let text: String
    let answer: Int
}

func pool(_ count: Int) -> [Task] {
    var g = Generator(state: 2026)
    return (0 ..< count).map { id in
        let scale = 1 + id % 3
        switch id % 4 {
        case 0:
            let stores = g.next(2 ... 9), each = g.next(3 ... 40 * scale), shipped = g.next(10 ... 90 * scale)
            let pallets = g.next(2 ... 9), perPallet = g.next(5 ... 30)
            let start = stores * each + shipped - pallets * perPallet
            guard start > 0 else { return Task(id: id, text: "What is \(shipped) plus \(stores * each)?", answer: shipped + stores * each) }
            return Task(id: id, text: "A warehouse holds \(start) crates. It ships \(shipped) crates, then receives \(pallets) pallets of \(perPallet) crates each, and finally splits all its crates equally among \(stores) stores. How many crates does each store get?", answer: each)
        case 1:
            let v1 = g.next(30 ... 90), t1 = g.next(2 ... 5 * scale), v2 = g.next(20 ... 70), t2 = g.next(1 ... 4 * scale), back = g.next(5 ... 60)
            return Task(id: id, text: "A truck drives at \(v1) km/h for \(t1) hours, then at \(v2) km/h for \(t2) hours, then turns around and drives \(back) km back. How far from its starting point is it now, in km?", answer: v1 * t1 + v2 * t2 - back)
        case 2:
            let bob = g.next(4 ... 30 * scale), gap = g.next(1 ... 15), years = g.next(2 ... 12)
            let sum = 2 * bob + gap + 2 * years
            return Task(id: id, text: "Ann is \(gap) years older than Bob. In \(years) years the sum of their ages will be \(sum). How old is Bob now?", answer: bob)
        default:
            let price = g.next(3 ... 25 * scale), count = g.next(4 ... 30), discount = g.next(1 ... min(9, price - 1)), paid = (price * count / 10 + 1) * 10 + g.next(0 ... 5) * 10
            return Task(id: id, text: "A notebook costs \(price) dollars. A school buys \(count) notebooks and gets \(discount) dollars off each one. It pays with \(paid) dollars. How much change does it get, in dollars?", answer: paid - (price - discount) * count)
        }
    }
}

struct Reply: Decodable {
    struct Usage: Decodable { let completion_tokens: Int }
    struct Choice: Decodable {
        struct Message: Decodable { let content: String?; let reasoning_content: String? }
        let message: Message
    }
    let usage: Usage
    let choices: [Choice]
}

func extract(_ text: String) -> Int? {
    guard let match = text.matches(of: #/Answer:\s*\$?(-?[\d,]+)/#).last else { return nil }
    return Int(match.output.1.replacingOccurrences(of: ",", with: ""))
}

@MainActor
func solve(_ task: Task) async throws -> (answer: Int?, tokens: Int) {
    var request = URLRequest(url: URL(string: options.url + "/v1/chat/completions")!)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let key = ProcessInfo.processInfo.environment["BASALT_API_KEY"] {
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
    }
    request.httpBody = try JSONSerialization.data(withJSONObject: [
        "model": "basalt", "temperature": options.temperature, "max_tokens": options.tokens, "seed": task.id,
        "reasoning_effort": options.reasoningEffort,
        "messages": [["role": "user", "content": task.text + "\nWork it out step by step, then end with a line of the form 'Answer: <integer>'."]],
    ])
    request.timeoutInterval = 900
    let (data, response) = try await URLSession.shared.data(for: request)
    guard (response as? HTTPURLResponse)?.statusCode == 200 else {
        throw NSError(domain: "task_pool", code: (response as? HTTPURLResponse)?.statusCode ?? -1,
                      userInfo: [NSLocalizedDescriptionKey: String(decoding: data.prefix(300), as: UTF8.self)])
    }
    let reply = try JSONDecoder().decode(Reply.self, from: data)
    let message = reply.choices.first?.message
    return (extract(message?.content ?? "") ?? extract(message?.reasoning_content ?? ""), reply.usage.completion_tokens)
}

let tasks = pool(options.tasks)
log("label=\(options.label) url=\(options.url) tasks=\(tasks.count) concurrency=\(options.concurrency) tokens=\(options.tokens) temperature=\(options.temperature) reasoning_effort=\(options.reasoningEffort)")
let started = ContinuousClock.now

@MainActor
func run(_ tasks: [Task]) async -> (results: [Int: Int?], tokens: Int, failures: Int) {
    var results: [Int: Int?] = [:]
    var tokens = 0
    var failures = 0
    var queue = tasks[...]
    await withTaskGroup(of: (Task, Result<(answer: Int?, tokens: Int), Error>).self) { group in
        func submit() {
            guard let task = queue.popFirst() else { return }
            group.addTask { @MainActor in
                do { return (task, .success(try await solve(task))) } catch { return (task, .failure(error)) }
            }
        }
        for _ in 0 ..< options.concurrency { submit() }
        for await (task, result) in group {
            switch result {
            case .success(let reply):
                results[task.id] = reply.answer
                tokens += reply.tokens
                log("task id=\(task.id) expected=\(task.answer) got=\(reply.answer.map(String.init) ?? "none") correct=\(reply.answer == task.answer ? 1 : 0)")
            case .failure(let error):
                failures += 1
                log("FAIL task id=\(task.id) error=\(error.localizedDescription)")
            }
            submit()
        }
    }
    return (results, tokens, failures)
}

let (results, totalTokens, failures) = await run(tasks)
let seconds = (ContinuousClock.now - started) / .seconds(1)
let correct = tasks.filter { results[$0.id] == .some($0.answer) }.count
log(String(format: "summary label=%@ correct=%d of %d accuracy=%.3f tokens=%d seconds=%.1f tok_s=%.1f failures=%d",
           options.label, correct, tasks.count, Double(correct) / Double(tasks.count), totalTokens, seconds, Double(totalTokens) / seconds, failures))

if let other = options.compare, let text = try? String(contentsOfFile: other, encoding: .utf8) {
    var theirs: [Int: String] = [:]
    for match in text.matches(of: #/task id=(\d+) expected=-?\d+ got=(\S+)/#) { theirs[Int(match.output.1)!] = String(match.output.2) }
    let mine = Dictionary(uniqueKeysWithValues: tasks.map { ($0.id, results[$0.id].flatMap { $0 }.map(String.init) ?? "none") })
    let both = tasks.filter { mine[$0.id] == String($0.answer) && theirs[$0.id] == String($0.answer) }.count
    let onlyMine = tasks.filter { mine[$0.id] == String($0.answer) && theirs[$0.id] != String($0.answer) }.count
    let onlyTheirs = tasks.filter { mine[$0.id] != String($0.answer) && theirs[$0.id] == String($0.answer) }.count
    let same = tasks.filter { mine[$0.id] == theirs[$0.id] }.count
    log("compare label=\(options.label) against=\(other) same_answer=\(same) of \(tasks.count) both_correct=\(both) only_this=\(onlyMine) only_other=\(onlyTheirs)")
}
log("done label=\(options.label) failures=\(failures) log=\(logPath)")
exit(failures == 0 ? 0 : 1)
