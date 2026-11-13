#!/usr/bin/env swift
import Foundation

let root = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : FileManager.default.currentDirectoryPath)
let golden = root.appending(path: "Tests/BasaltTests/Fixtures/golden")

func captures(_ pattern: String, in line: String) -> [String] {
    let regex = try! NSRegularExpression(pattern: pattern)
    let range = NSRange(line.startIndex..., in: line)
    return regex.matches(in: line, range: range).compactMap { Range($0.range(at: 1), in: line).map { String(line[$0]) } }
}

var failures: [String] = []
for name in ["README.md", "README.ru.md"] {
    guard let text = try? String(contentsOf: root.appending(path: name), encoding: .utf8) else { continue }
    var listed = 0
    for (index, line) in text.components(separatedBy: "\n").enumerated() where line.contains("Fixtures/golden/") {
        let fixtures = captures(#"Fixtures/golden/([A-Za-z0-9._-]+)\.json"#, in: line)
        let repos = captures(#"`([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)`"#, in: line)
        for fixture in fixtures where !FileManager.default.fileExists(atPath: golden.appending(path: fixture + ".json").path) {
            failures.append("\(name):\(index + 1): links Fixtures/golden/\(fixture).json, which does not exist")
        }
        for repo in repos where !fixtures.contains(repo.replacingOccurrences(of: "/", with: "--")) {
            failures.append("\(name):\(index + 1): lists \(repo) as golden-tested without linking its fixture Fixtures/golden/\(repo.replacingOccurrences(of: "/", with: "--")).json")
        }
        listed += repos.count
    }
    if listed == 0 { failures.append("\(name): lists no golden-tested model (a line linking Fixtures/golden/<owner>--<repo>.json with the repo id in backticks)") }
}

if failures.isEmpty {
    print("README models: every golden-tested model links an existing fixture.")
} else {
    failures.forEach { print($0) }
    exit(1)
}
