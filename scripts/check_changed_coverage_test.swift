#!/usr/bin/env swift
import Foundation

let checker = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appending(path: "check_changed_coverage.swift").path

let lib = """
public func f(_ x: Int) -> Int {
    if x > 0 {
        return 1
    }
    return 2
}

public struct S {
    public var a: Int
    public init(a: Int) { self.a = a }
    public func g() -> Int {
        let y = a
            + 1
        return y
    }
}

"""

let libSegments = """
[[1,32,1,true,true,false],[2,8,1,true,true,false],[2,13,1,true,false,false],[2,14,1,true,true,false],\
[4,6,0,true,true,false],[5,13,1,true,false,false],[6,2,0,false,false,false],[10,25,0,true,true,false],\
[10,39,0,false,false,false],[11,28,0,true,true,false],[15,6,0,false,false,false]]
"""

let edgeSegments = """
[[1,1,5,true,true,false],[2,1,0,false,true,false],[3,1,0,false,false,false],[4,1,7,true,true,true],[5,1,0,false,false,false]]
"""

func export(_ files: [(String, String)]) -> String {
    let entries = files.map { #"{"filename":"\#($0.0)","segments":\#($0.1),"summary":{}}"# }
    return #"{"type":"llvm.coverage.json.export","version":"3.0.1","data":[{"files":[\#(entries.joined(separator: ","))],"functions":[],"totals":{}}]}"#
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

func check(base: [String: String], head: [String: String], report: String, flags: [String] = []) -> (status: Int32, output: String) {
    let directory = FileManager.default.temporaryDirectory.appending(path: "changed-coverage-\(UUID().uuidString)")
    try! FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    func write(_ files: [String: String]) {
        for (name, body) in files {
            let file = directory.appending(path: name)
            try! FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try! body.write(to: file, atomically: true, encoding: .utf8)
        }
        _ = shell(["git", "add", "-A"], in: directory)
    }
    _ = shell(["git", "init", "-q", "-b", "main"], in: directory)
    for (key, value) in [("user.name", "t"), ("user.email", "t@t")] { _ = shell(["git", "config", key, value], in: directory) }
    write(base.merging(["README": "x"]) { $1 })
    _ = shell(["git", "commit", "-qm", "base"], in: directory)
    _ = shell(["git", "checkout", "-qb", "pr"], in: directory)
    write(head)
    _ = shell(["git", "commit", "-qm", "pr", "--allow-empty"], in: directory)
    let json = directory.appending(path: "codecov.json")
    try! report.write(to: json, atomically: true, encoding: .utf8)
    return shell(["swift", checker, json.path, "main"] + flags, in: directory)
}

var failures = 0
func expect(_ condition: Bool, _ name: String, _ output: String) {
    if condition { print("ok   \(name)") } else {
        failures += 1
        print("FAIL \(name)\n\(output)")
    }
}

let libReport = export([("/runner/work/basalt/Sources/Lib/Lib.swift", libSegments)])

var result = check(base: ["Sources/Lib/Lib.swift": ""], head: ["Sources/Lib/Lib.swift": lib], report: libReport)
expect(result.status == 1
    && result.output.contains("Sources/Lib/Lib.swift: 5/12 changed executable lines covered; uncovered: 5, 10-15")
    && result.output.contains("Changed-line coverage: 41.7% (5/12), minimum 60%.")
    && result.output.contains("Below the minimum"),
    "lines follow llvm-cov (wrapped counts, closing braces, blank lines unmapped) and fail under 60%", result.output)

result = check(base: ["Sources/Lib/Lib.swift": ""], head: ["Sources/Lib/Lib.swift": lib], report: libReport, flags: ["--min", "40"])
expect(result.status == 0 && result.output.contains("minimum 40%."), "--min sets the threshold", result.output)

let firstSix = lib.split(separator: "\n", omittingEmptySubsequences: false).prefix(6).joined(separator: "\n") + "\n"
result = check(base: ["Sources/Lib/Lib.swift": firstSix], head: ["Sources/Lib/Lib.swift": lib], report: libReport, flags: ["--min", "40"])
expect(result.status == 1 && result.output.contains("0/6 changed executable lines covered; uncovered: 10-15"),
       "only added lines count", result.output)

result = check(base: [:], head: ["Sources/E.swift": "a\nb\nc\nd\ne\n"], report: export([("/w/Sources/E.swift", edgeSegments)]))
expect(result.output.contains("Sources/E.swift: 2/2 changed executable lines covered"),
       "a skipped region and a gap region do not make a line executable", result.output)

result = check(
    base: [:], head: ["Sources/Lib/Lib.swift": lib, "Sources/Gone.swift": "let x = 1\n", "Tests/T.swift": "let t = 1\n"],
    report: export([("/w/.build/checkouts/dep/Sources/Lib/Lib.swift", "[[1,1,9,true,true,false]]"), ("/w/Sources/Lib/Lib.swift", libSegments)]),
    flags: ["--min", "40"])
expect(result.status == 0 && result.output.contains("Sources/Gone.swift: not in the coverage report")
    && result.output.contains("5/12") && !result.output.contains("Tests/"),
    "dependency copies are ignored, a file missing from the report is named, tests are not counted", result.output)

result = check(base: ["Sources/Lib/Lib.swift": lib], head: ["Sources/Lib/Lib.swift": firstSix, "Tests/T.swift": "x\n"], report: libReport)
expect(result.status == 0 && result.output.contains("No changed executable lines in Sources."), "a deletion-only change passes", result.output)

result = check(base: [:], head: [:], report: libReport, flags: ["--min"])
expect(result.status == 1 && result.output.contains("usage:"), "a bad flag prints usage", result.output)

exit(failures == 0 ? 0 : 1)
