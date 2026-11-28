#!/usr/bin/env swift
import Darwin
import Foundation
import notify

let usage = """
    Event bus for the coding agents on this Mac. One append-only log, instant wake-up over Darwin notifications.

      bus.swift send <to> <type> <text...> [--ref <id>]   append an event (sender: $AGENT_NAME); <to> is an agent or "all"
      bus.swift wait <me> [--timeout <seconds>]           block until an event for <me> arrives, print it, exit 0; exit 3 on timeout
      bus.swift watch <me>                                stream events for <me> as they arrive (for a monitor)
      bus.swift peek <me> [n]                             print the last n events for <me> without moving the cursor
      bus.swift status                                    who holds the GPU, open review requests, last activity per agent
      bus.swift gpu claim <what...> [--minutes <n>] [--ref <id>]  take the GPU; exit 2 if another agent holds an unexpired claim
      bus.swift gpu release                               give the GPU back
      bus.swift agents                                    known agents
      bus.swift --self-test                               check the log, cursors, the GPU rule and wake-up without touching the real bus

    Types: msg, gpu.claim, gpu.release, ci.busy, ci.idle, review.request, review.verdict, pr.opened, pr.merged, heartbeat.
    The log lives in $BASALT_BUS_DIR, default /private/tmp/basalt-agents/bus.
    """

struct Event: Codable, Equatable {
    var seq: Int
    var ts: String
    var from: String
    var to: String
    var type: String
    var text: String
    var ref: String?
    var minutes: Int?

    var line: String {
        "[\(ts)] #\(seq) \(from) → \(to) \(type)\(ref.map { " (\($0))" } ?? ""): \(text)"
    }
}

struct Failure: Error {
    let message: String
    let code: Int32
}

enum Names {
    static let known = ["claude", "agy", "glm", "qwen"]
    static let types: Set<String> = [
        "msg", "gpu.claim", "gpu.release", "ci.busy", "ci.idle", "review.request", "review.verdict", "pr.opened", "pr.merged", "heartbeat",
    ]

    static func agent(_ name: String, allowAll: Bool = false) throws -> String {
        if allowAll, name == "all" { return name }
        guard name.range(of: #"^[a-z][a-z0-9-]{0,31}$"#, options: .regularExpression) != nil, name != "all" else {
            throw Failure(message: "bad agent name '\(name)' (lowercase, like claude, agy, glm, qwen)", code: 2)
        }
        return name
    }
}

struct GPUState: Equatable {
    var holder: String?
    var what = ""
    var until: Date?

    func heldByOther(than agent: String, at now: Date) -> Bool {
        guard let holder, holder != agent, let until else { return false }
        return until > now
    }
}

enum Reducer {
    static let stamp: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    static func gpu(_ events: [Event]) -> GPUState {
        events.reduce(into: GPUState()) { state, event in
            switch event.type {
            case "gpu.claim":
                let start = stamp.date(from: event.ts) ?? .distantPast
                if state.heldByOther(than: event.from, at: start) { break }
                state = GPUState(holder: event.from, what: event.text, until: start.addingTimeInterval(Double(event.minutes ?? 60) * 60))
            case "gpu.release" where state.holder == event.from:
                state = GPUState()
            default:
                break
            }
        }
    }

    static func openReviews(_ events: [Event]) -> [Event] {
        var open: [String: Event] = [:]
        for event in events {
            guard let ref = event.ref else { continue }
            if event.type == "review.request" { open[ref] = event }
            if event.type == "review.verdict" || event.type == "pr.merged" { open[ref] = nil }
        }
        return open.values.sorted { $0.seq < $1.seq }
    }

    static func lastSeen(_ events: [Event]) -> [String: Event] {
        events.reduce(into: [:]) { $0[$1.from] = $1 }
    }

    static func inbox(_ events: [Event], for agent: String, after cursor: Int) -> [Event] {
        events.filter { $0.seq > cursor && $0.from != agent && ($0.to == agent || $0.to == "all") }
    }
}

struct Bus {
    static let channel = "ai.basalt.agents.bus"
    let root: URL

    init(root: URL? = nil) throws {
        let path = ProcessInfo.processInfo.environment["BASALT_BUS_DIR"] ?? "/private/tmp/basalt-agents/bus"
        self.root = root ?? URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: self.root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    }

    var log: URL { root.appendingPathComponent("events.jsonl") }
    func cursorFile(_ agent: String) -> URL { root.appendingPathComponent("cursor-\(agent)") }

    func events() -> [Event] {
        guard let text = try? String(contentsOf: log, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        return text.split(separator: "\n").compactMap { try? decoder.decode(Event.self, from: Data($0.utf8)) }
    }

    func cursor(_ agent: String) -> Int {
        (try? String(contentsOf: cursorFile(agent), encoding: .utf8)).flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) } ?? 0
    }

    func setCursor(_ agent: String, _ seq: Int) throws {
        try String(seq).write(to: cursorFile(agent), atomically: true, encoding: .utf8)
    }

    @discardableResult
    func append(from: String, to: String, type: String, text: String, ref: String? = nil, minutes: Int? = nil, now: Date = Date()) throws -> Event {
        guard Names.types.contains(type) else {
            throw Failure(message: "unknown event type '\(type)' (one of: \(Names.types.sorted().joined(separator: ", ")))", code: 2)
        }
        let descriptor = open(log.path, O_WRONLY | O_APPEND | O_CREAT, 0o600)
        guard descriptor >= 0 else { throw Failure(message: "cannot open \(log.path): \(String(cString: strerror(errno)))", code: 1) }
        defer { close(descriptor) }
        guard flock(descriptor, LOCK_EX) == 0 else { throw Failure(message: "cannot lock \(log.path)", code: 1) }
        defer { flock(descriptor, LOCK_UN) }
        let event = Event(
            seq: (events().last?.seq ?? 0) + 1, ts: Reducer.stamp.string(from: now), from: from, to: to, type: type, text: text, ref: ref, minutes: minutes)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var line = try encoder.encode(event)
        line.append(0x0A)
        let written = line.withUnsafeBytes { write(descriptor, $0.baseAddress, $0.count) }
        guard written == line.count else { throw Failure(message: "short write to \(log.path)", code: 1) }
        notify_post(Self.channel)
        return event
    }

    func take(_ agent: String) throws -> [Event] {
        let fresh = Reducer.inbox(events(), for: agent, after: cursor(agent))
        if let last = events().last, last.seq > cursor(agent) { try setCursor(agent, last.seq) }
        return fresh
    }

    func wait(_ agent: String, timeout: Double?) throws -> [Event] {
        var descriptor: Int32 = -1
        var token: Int32 = 0
        guard notify_register_file_descriptor(Self.channel, &descriptor, 0, &token) == NOTIFY_STATUS_OK else {
            throw Failure(message: "cannot register on the notification bus", code: 1)
        }
        defer { notify_cancel(token) }
        let deadline = timeout.map { Date().addingTimeInterval($0) }
        while true {
            let fresh = try take(agent)
            if !fresh.isEmpty { return fresh }
            let remaining = deadline.map { $0.timeIntervalSinceNow }
            if let remaining, remaining <= 0 { return [] }
            var poller = pollfd(fd: descriptor, events: Int16(POLLIN), revents: 0)
            let slice = Int32(min(remaining ?? 30, 30) * 1000)
            if poll(&poller, 1, slice) > 0 {
                var drained: Int32 = 0
                _ = read(descriptor, &drained, MemoryLayout<Int32>.size)
            }
        }
    }
}

func option(_ name: String, in arguments: inout [String]) -> String? {
    guard let index = arguments.firstIndex(of: name), index + 1 < arguments.count else { return nil }
    let value = arguments[index + 1]
    arguments.removeSubrange(index ... index + 1)
    return value
}

func sender() throws -> String {
    try Names.agent(ProcessInfo.processInfo.environment["AGENT_NAME"] ?? "user")
}

func status(_ bus: Bus, now: Date = Date()) -> String {
    let events = bus.events()
    let gpu = Reducer.gpu(events)
    var lines: [String] = []
    if let holder = gpu.holder, let until = gpu.until, until > now {
        lines.append("GPU: \(holder) — \(gpu.what) (until \(Reducer.stamp.string(from: until)))")
    } else {
        lines.append("GPU: free")
    }
    let reviews = Reducer.openReviews(events)
    lines.append(reviews.isEmpty ? "Reviews waiting: none" : "Reviews waiting:")
    lines += reviews.map { "  \($0.ref ?? "?") from \($0.from) to \($0.to), since \($0.ts)" }
    let seen = Reducer.lastSeen(events)
    lines.append("Agents:")
    for agent in Array(Set(Names.known + seen.keys)).sorted() {
        lines.append("  \(agent): " + (seen[agent].map { "last event #\($0.seq) \($0.type) at \($0.ts)" } ?? "never seen"))
    }
    return lines.joined(separator: "\n")
}

func selfTest() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-bus-test-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let bus = try Bus(root: root)
    func check(_ condition: Bool, _ what: String) throws {
        guard condition else { throw Failure(message: "self-test failed: \(what)", code: 1) }
    }
    let start = Date(timeIntervalSince1970: 1_800_000_000)
    let first = try bus.append(from: "claude", to: "agy", type: "msg", text: "hello", now: start)
    let second = try bus.append(from: "agy", to: "all", type: "heartbeat", text: "alive", now: start)
    try check(first.seq == 1 && second.seq == 2, "sequence numbers rise by one")
    try check(try bus.take("agy").map(\.seq) == [1], "an agent gets what is addressed to it, not its own broadcast")
    try check(try bus.take("agy").isEmpty, "a second read returns nothing")
    try check(try bus.take("glm").map(\.seq) == [2], "a third agent gets the broadcast and not another agent's mail")
    try check(try bus.take("claude").map(\.seq) == [2], "the sender does not get its own message back")
    try bus.append(from: "claude", to: "all", type: "gpu.claim", text: "bench", minutes: 30, now: start)
    let held = Reducer.gpu(bus.events())
    try check(held.holder == "claude" && held.heldByOther(than: "agy", at: start.addingTimeInterval(60)), "a claim holds the GPU against others")
    try check(!held.heldByOther(than: "claude", at: start), "the holder may claim again")
    try check(!held.heldByOther(than: "agy", at: start.addingTimeInterval(31 * 60)), "a claim expires")
    try bus.append(from: "agy", to: "all", type: "gpu.claim", text: "late", minutes: 30, now: start.addingTimeInterval(1))
    try check(Reducer.gpu(bus.events()).holder == "claude", "a claim that lands while another holds is ignored")
    try bus.append(from: "agy", to: "all", type: "gpu.release", text: "", now: start)
    try check(Reducer.gpu(bus.events()).holder == "claude", "only the holder can release")
    try bus.append(from: "claude", to: "all", type: "gpu.release", text: "", now: start)
    try check(Reducer.gpu(bus.events()).holder == nil, "the holder's release frees the GPU")
    try bus.append(from: "claude", to: "agy", type: "review.request", text: "please", ref: "#1", now: start)
    try check(Reducer.openReviews(bus.events()).map(\.ref) == ["#1"], "a review request stays open")
    try bus.append(from: "agy", to: "claude", type: "review.verdict", text: "LGTM", ref: "#1", now: start)
    try check(Reducer.openReviews(bus.events()).isEmpty, "a verdict closes the request")
    try check((try? bus.append(from: "claude", to: "agy", type: "nonsense", text: "")) == nil, "an unknown type is refused")
    try check((try? Names.agent("Bad Name")) == nil && (try? Names.agent("all")) == nil, "bad agent names are refused")
    _ = try bus.take("agy")
    try check(try bus.wait("agy", timeout: 0).isEmpty, "wait with nothing new times out")
    try bus.append(from: "glm", to: "agy", type: "msg", text: "wake", now: start)
    try check(try bus.wait("agy", timeout: 5).map(\.text) == ["wake"], "wait returns an event that is already there")
    print("self-test passed: 17 checks")
}

func run(_ arguments: [String]) throws -> Int32 {
    var arguments = arguments
    guard let command = arguments.first else { print(usage); return 2 }
    arguments.removeFirst()
    if command == "--self-test" { try selfTest(); return 0 }
    let bus = try Bus()
    switch command {
    case "send":
        let ref = option("--ref", in: &arguments)
        guard arguments.count >= 3 else { throw Failure(message: "usage: bus.swift send <to> <type> <text...> [--ref <id>]", code: 2) }
        let event = try bus.append(
            from: try sender(), to: try Names.agent(arguments[0], allowAll: true), type: arguments[1],
            text: arguments.dropFirst(2).joined(separator: " "), ref: ref)
        print("sent #\(event.seq) to \(event.to)")
    case "wait":
        let timeout = option("--timeout", in: &arguments).flatMap(Double.init)
        guard let me = arguments.first else { throw Failure(message: "usage: bus.swift wait <me> [--timeout <seconds>]", code: 2) }
        let fresh = try bus.wait(try Names.agent(me), timeout: timeout)
        fresh.forEach { print($0.line) }
        return fresh.isEmpty ? 3 : 0
    case "watch":
        guard let me = arguments.first else { throw Failure(message: "usage: bus.swift watch <me>", code: 2) }
        setvbuf(stdout, nil, _IOLBF, 0)
        while true { try bus.wait(try Names.agent(me), timeout: nil).forEach { print($0.line) } }
    case "peek":
        guard let me = arguments.first else { throw Failure(message: "usage: bus.swift peek <me> [n]", code: 2) }
        let count = arguments.dropFirst().first.flatMap(Int.init) ?? 10
        Reducer.inbox(bus.events(), for: try Names.agent(me), after: 0).suffix(count).forEach { print($0.line) }
    case "status":
        print(status(bus))
    case "gpu":
        let me = try sender()
        let minutes = option("--minutes", in: &arguments).flatMap(Int.init) ?? 60
        let ref = option("--ref", in: &arguments)
        guard let action = arguments.first, ["claim", "release"].contains(action) else {
            throw Failure(message: "usage: bus.swift gpu claim <what...> [--minutes <n>] | gpu release", code: 2)
        }
        if action == "release" {
            try bus.append(from: me, to: "all", type: "gpu.release", text: "")
            print("GPU released by \(me)")
            return 0
        }
        let state = Reducer.gpu(bus.events())
        if state.heldByOther(than: me, at: Date()) {
            throw Failure(message: "GPU is held by \(state.holder ?? "?") (\(state.what)) until \(state.until.map(Reducer.stamp.string) ?? "?")", code: 2)
        }
        try bus.append(from: me, to: "all", type: "gpu.claim", text: arguments.dropFirst().joined(separator: " "), ref: ref, minutes: minutes)
        let after = Reducer.gpu(bus.events())
        guard after.holder == me else {
            throw Failure(message: "GPU was taken first by \(after.holder ?? "?") (\(after.what))", code: 2)
        }
        print("GPU claimed by \(me) for \(minutes) min")
    case "agents":
        print(Array(Set(Names.known + Reducer.lastSeen(bus.events()).keys)).sorted().joined(separator: "\n"))
    default:
        print(usage)
        return 2
    }
    return 0
}

do {
    exit(try run(Array(CommandLine.arguments.dropFirst())))
} catch let failure as Failure {
    FileHandle.standardError.write(Data((failure.message + "\n").utf8))
    exit(failure.code)
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
