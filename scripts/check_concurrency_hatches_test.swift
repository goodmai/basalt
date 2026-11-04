#!/usr/bin/env swift
import Foundation

let checker = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appending(path: "check_concurrency_hatches.swift").path

let clean = #"""
// @unchecked Sendable, NSLock, Task.detached { }
/* nonisolated(unsafe) var x = DispatchQueue.main /* nested Thread { } */ */
let s = "Task.detached { } NSCondition() os_unfair_lock"
let m = """
    DispatchSemaphore(value: 0)
    @unchecked Sendable
    """
let raw = #"Thread.detachNewThread { } "# + "x"
let r = #/NSRecursiveLock/#
let myNSLockish = 1
let taskDetached = Thread.current
Thread.sleep(forTimeInterval: 0)
let isMain = Thread.isMainThread
let detached = options.detached
nonisolated func f() {}

"""#

let dirty = [
    "final class A: @unchecked Sendable {}",
    "nonisolated(unsafe) var b = 0",
    "let c = NSLock()",
    "let d = NSRecursiveLock()",
    "let e = NSCondition()",
    "let f = DispatchQueue(label: \"x\")",
    "let g = DispatchSemaphore(value: 0)",
    "var h = os_unfair_lock()",
    "let i = Task.detached { }",
    "Thread.detachNewThread { }",
    "let k = Thread { }",
    "let l = Task<Void, Never>.detached { }",
    "let m = \"\\(Task.detached { 1 })\"",
    "let n = NSConditionLock(condition: 0)",
    "let o = Task.immediateDetached { }",
]

let allowEngine = "Sources/E.swift:Task.detached: outlives the call; carries only Sendable values\n"

func shell(_ arguments: [String], in directory: URL) -> (status: Int32, output: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = arguments
    process.currentDirectoryURL = directory
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    try! process.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (process.terminationStatus, String(decoding: data, as: UTF8.self))
}

func check(_ files: [String: String], allow: String? = nil) -> (status: Int32, output: String) {
    let directory = FileManager.default.temporaryDirectory.appending(path: "hatches-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: directory) }
    for (name, body) in files.merging(["scripts/concurrency-hatches.allow": allow].compactMapValues { $0 }) { $1 } {
        let file = directory.appending(path: name)
        try! FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try! body.write(to: file, atomically: true, encoding: .utf8)
    }
    return shell(["swift", checker], in: directory)
}

var failures = 0
func expect(_ condition: Bool, _ name: String, _ output: String) {
    if condition { print("ok   \(name)") } else {
        failures += 1
        print("FAIL \(name)\n\(output)")
    }
}

var result = check(["Sources/Clean.swift": clean])
expect(result.status == 0 && result.output.contains("No concurrency hatch outside scripts/concurrency-hatches.allow."),
       "comments, strings, regexes and look-alike names pass", result.output)

result = check(["Sources/Deep/Bad.swift": dirty.joined(separator: "\n") + "\n"])
let kinds = ["@unchecked Sendable", "nonisolated(unsafe)", "NSLock", "NSRecursiveLock", "NSCondition", "DispatchQueue",
             "DispatchSemaphore", "os_unfair_lock", "Task.detached", "Thread", "Thread", "Task.detached", "Task.detached",
             "NSCondition", "Task.detached"]
expect(result.status == 1 && zip(1..., kinds).allSatisfy { result.output.contains("Sources/Deep/Bad.swift:\($0): \($1)\n") }
    && result.output.contains("\n\n\(dirty.count) problem(s)."),
    "every hatch kind is reported with its line", result.output)

result = check(["Sources/E.swift": "let t = Task\n    .detached { }\n"], allow: allowEngine)
expect(result.status == 0, "an allowed site passes, even split across lines", result.output)

result = check(["Sources/E.swift": "let t = Task.detached { }\nlet u = Task.detached { }\n"], allow: allowEngine)
expect(result.status == 1 && result.output.contains("Sources/E.swift:1: Task.detached") && result.output.contains("Sources/E.swift:2: Task.detached"),
       "a second site in an allowed file needs its own line", result.output)

result = check(["Sources/E.swift": "let t = 1\n"], allow: allowEngine)
expect(result.status == 1 && result.output.contains("Sources/E.swift:Task.detached is listed 1 time(s) but found 0: remove the stale line"),
       "a stale allowlist line fails, so the list only shrinks", result.output)

result = check(["Sources/E.swift": "let t = Task.detached { }\n"], allow: "Sources/E.swift:Task.detached\n")
expect(result.status == 1 && result.output.contains("scripts/concurrency-hatches.allow:1: expected `path:symbol: reason`"),
       "an allowlist line without a reason is rejected", result.output)

result = check(["Tests/T.swift": "let c = NSLock()\n", "Sources/Ok.swift": "let a = 1\r\n// NSLock\r\nlet b = NSLock()\r\n"])
expect(result.status == 1 && result.output.contains("Sources/Ok.swift:3: NSLock") && !result.output.contains("Tests/"),
       "only Sources is scanned, crlf keeps the line count", result.output)

exit(failures == 0 ? 0 : 1)
