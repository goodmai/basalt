#!/usr/bin/env swift
import Foundation

let scripts = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.deletingLastPathComponent().path
let fixtures = scripts + "/fixtures/acceptance/"

func compare(_ candidate: String) -> (status: Int32, output: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["swift", scripts + "/acceptance.swift", "--baseline", fixtures + "baseline.json", "--candidate", fixtures + candidate + ".json"]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    try! process.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (process.terminationStatus, String(decoding: data, as: UTF8.self))
}

let cases: [(candidate: String, status: Int32, failing: [String])] = [
    ("baseline", 0, []),
    ("noise", 0, []),
    ("slower", 1, ["tok_s"]),
    ("less-accurate", 1, ["accuracy"]),
    ("drifted", 1, ["golden_kl"]),
    ("other-pool", 1, ["comparable"]),
]

var failures = 0
for test in cases {
    let result = compare(test.candidate)
    let failing = result.output.matches(of: #/(?:gate (\w+) .* FAIL|FAIL (comparable))/#).map { String($0.output.1 ?? $0.output.2 ?? "") }
    let bothCommits = result.output.contains("baseline=8c1f17d") && result.output.contains("candidate=")
    guard result.status == test.status, failing == test.failing, bothCommits else {
        failures += 1
        print("FAIL \(test.candidate): exit \(result.status) failing \(failing), expected exit \(test.status) failing \(test.failing)\n\(result.output)")
        continue
    }
    print("ok \(test.candidate): exit \(result.status) failing \(failing)")
}
print(failures == 0 ? "acceptance gate self-test passed" : "acceptance gate self-test: \(failures) failure(s)")
exit(failures == 0 ? 0 : 1)
