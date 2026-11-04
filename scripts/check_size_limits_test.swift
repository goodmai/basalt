#!/usr/bin/env swift
import Foundation

let checker = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appending(path: "check_size_limits.swift").path

func branchy(_ name: String, _ branches: Int, params: Int = 1) -> String {
    let arguments = (0 ..< params).map { "p\($0): Int" }.joined(separator: ", ")
    let body = (0 ..< branches).map { "    if p0 == \($0) { return \($0) }\n" }.joined()
    return "func \(name)(\(arguments)) -> Int {\n\(body)    return -1\n}\n"
}

func long(_ name: String, _ lines: Int) -> String {
    "func \(name)() -> Int {\n    var total = 0\n" + (0 ..< lines).map { "    total += \($0)\n" }.joined() + "    return total\n}\n"
}

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

func check(before: [String: String], after: [String: String]) -> (status: Int32, output: String) {
    let directory = FileManager.default.temporaryDirectory.appending(path: "size-limits-\(UUID().uuidString)")
    try! FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    _ = shell(["git", "init", "-q", "-b", "main"], in: directory)
    _ = shell(["git", "config", "user.name", "t"], in: directory)
    _ = shell(["git", "config", "user.email", "t@t"], in: directory)
    for (files, message) in [(before, "base"), (after, "pr")] {
        for (name, body) in files {
            let file = directory.appending(path: name)
            try! FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try! body.write(to: file, atomically: true, encoding: .utf8)
        }
        _ = shell(["git", "add", "-A"], in: directory)
        _ = shell(["git", "commit", "-qm", message, "--allow-empty"], in: directory)
        if message == "base" { _ = shell(["git", "checkout", "-qb", "pr"], in: directory) }
    }
    return shell(["swift", checker, "main"], in: directory)
}

var failures = 0
func expect(_ condition: Bool, _ name: String, _ output: String) {
    if condition { print("ok   \(name)") } else {
        failures += 1
        print("FAIL \(name)\n\(output)")
    }
}

var result = check(before: [:], after: ["Sources/A.swift": branchy("small", 5) + long("short", 20)])
expect(result.status == 0, "new code within the limits passes", result.output)

result = check(before: [:], after: ["Sources/A.swift": branchy("complex", 20) + long("long", 80) + branchy("wide", 1, params: 8)])
expect(result.status == 1 && result.output.contains("complex p0 : Int: CCN 21 > 15") && result.output.contains(": long: lines of code")
    && result.output.contains("parameters 8 > 6"), "a new function over each limit fails", result.output)

let legacy = ["Sources/A.swift": branchy("legacy", 20)]
result = check(before: legacy, after: ["Sources/A.swift": branchy("legacy", 18)])
expect(result.status == 0, "an existing function over a limit may shrink", result.output)
result = check(before: legacy, after: ["Sources/A.swift": branchy("legacy", 22)])
expect(result.status == 1 && result.output.contains("CCN 23 > 15, was 21"), "an existing function over a limit may not grow", result.output)

result = check(before: ["Sources/A.swift": branchy("legacy", 20) + branchy("other", 2)],
               after: ["Sources/A.swift": branchy("legacy", 20) + branchy("other", 3)])
expect(result.status == 0, "an untouched function over a limit does not fail an edit elsewhere", result.output)

let big = (0 ..< 700).map { "let v\($0) = \($0)\n" }.joined()
result = check(before: ["Sources/Big.swift": big, "Tests/BigTests.swift": big],
               after: ["Sources/Big.swift": big + "let extra = 1\n", "Tests/BigTests.swift": big + "let extra = 1\n"])
expect(result.status == 1 && result.output.contains("Sources/Big.swift: 701 lines > 600, was 700") && !result.output.contains("Tests/BigTests.swift"),
       "a long source file may not grow, a test file may", result.output)

let setters = "final class Engine {\n    private(set) var a = 0\n    public private(set) var b = 0\n"
    + (0 ..< 10).map { "    var v\($0) = \($0)\n" }.joined() + long("near", 55) + "}\n"
result = check(before: [:], after: ["Sources/S.swift": setters])
expect(result.status == 0 && !result.output.contains(": set:"), "private(set) is not read as a setter that swallows the declarations after it", result.output)

let sparse = "func serve() {\n    group.run { value in\n" + (0 ..< 40).map { "        print(\($0))\n\n" }.joined() + "    }\n}\n"
result = check(before: [:], after: ["Sources/C.swift": sparse])
expect(result.status == 0, "a trailing closure over 60 lines long but under 60 lines of code passes: NLOC is checked, not length", result.output)

func type(_ kind: String, _ name: String, _ lines: Int) -> String {
    "\(kind) \(name) {\n" + (0 ..< lines).map { "    var f\($0) = \"{ not a brace }\" // { nor this\n" }.joined() + "}\n"
}

result = check(before: [:], after: ["Sources/T.swift": type("struct", "Core", 200) + type("extension", "Core", 150)])
expect(result.status == 0, "a type of 200 lines and an extension of 150 pass, braces in strings and comments ignored", result.output)
result = check(before: [:], after: ["Sources/T.swift": type("actor", "Shell", 201) + type("extension", "Shell", 151)])
expect(result.status == 1 && result.output.contains("actor Shell: 201 lines of code > 200")
    && result.output.contains("extension Shell: 151 lines of code > 150"), "a new type over 200 or extension over 150 fails", result.output)
result = check(before: ["Sources/T.swift": type("final class", "Legacy", 300)], after: ["Sources/T.swift": type("final class", "Legacy", 290)])
expect(result.status == 0, "a type already over the limit may shrink", result.output)
result = check(before: ["Sources/T.swift": type("final class", "Legacy", 300)], after: ["Sources/T.swift": type("final class", "Legacy", 301)])
expect(result.status == 1 && result.output.contains("class Legacy: 301 lines of code > 200, was 300"), "a type already over the limit may not grow", result.output)

exit(failures == 0 ? 0 : 1)
