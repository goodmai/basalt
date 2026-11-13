#!/usr/bin/env swift
import Foundation

let checker = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appending(path: "check_readme_models.swift").path

func run(readme: String, fixtures: [String]) throws -> (status: Int32, output: String) {
    let root = FileManager.default.temporaryDirectory.appending(path: "readme-models-\(UUID().uuidString)")
    let golden = root.appending(path: "Tests/BasaltTests/Fixtures/golden")
    try FileManager.default.createDirectory(at: golden, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    for fixture in fixtures { try Data("{}".utf8).write(to: golden.appending(path: fixture + ".json")) }
    try Data(readme.utf8).write(to: root.appending(path: "README.md"))
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["swift", checker, root.path]
    let pipe = Pipe()
    process.standardOutput = pipe
    try process.run()
    process.waitUntilExit()
    return (process.terminationStatus, String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self))
}

var failed = 0
func check(_ label: String, _ condition: Bool, _ output: String) {
    if !condition {
        failed += 1
        print("FAIL \(label)\n\(output)")
    }
}

let row = "| [`org/model`](https://huggingface.co/org/model) | [fixture](Tests/BasaltTests/Fixtures/golden/org--model.json) |"
var result = try run(readme: "# x\n" + row + "\n", fixtures: ["org--model"])
check("a listed model with its fixture passes", result.status == 0, result.output)
result = try run(readme: "# x\n" + row + "\n", fixtures: [])
check("a fixture link to a missing file fails", result.status != 0 && result.output.contains("does not exist"), result.output)
result = try run(readme: "# x\n| [`org/other`](x) | [fixture](Tests/BasaltTests/Fixtures/golden/org--model.json) |\n", fixtures: ["org--model"])
check("a repo id without its own fixture fails", result.status != 0 && result.output.contains("org/other"), result.output)
result = try run(readme: "# x\nno models here\n", fixtures: ["org--model"])
check("a README listing no golden-tested model fails", result.status != 0 && result.output.contains("lists no golden-tested model"), result.output)

print(failed == 0 ? "check_readme_models: 4 cases pass" : "check_readme_models: \(failed) case(s) failed")
exit(failed == 0 ? 0 : 1)
