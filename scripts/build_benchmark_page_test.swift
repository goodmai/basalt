#!/usr/bin/env swift
import Foundation

let builder = URL(fileURLWithPath: CommandLine.arguments[0])
    .deletingLastPathComponent()
    .appendingPathComponent("build_benchmark_page.swift").path

struct Sandbox {
    let root: URL

    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("bench-page-\(UUID().uuidString)")
        for dir in [
            "Tests/BasaltTests/Fixtures/golden",
            "Tests/BasaltTests/Fixtures/ReferenceLogits/qwen3",
            "docs",
            "benchmarks/ladder",
            "benchmarks/cacheblend",
            "logs"
        ] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(dir), withIntermediateDirectories: true)
        }
    }

    func cleanup() {
        try? FileManager.default.removeItem(at: root)
    }

    func write(_ rel: String, _ text: String) throws {
        try Data(text.utf8).write(to: root.appendingPathComponent(rel))
    }

    func run(_ extraArgs: [String] = []) throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["swift", builder, "--root", root.path] + extraArgs
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(decoding: data, as: UTF8.self))
    }
}

func seedValid(_ box: Sandbox) throws {
    try box.write(
        "Tests/BasaltTests/Fixtures/golden/org--alpha.json",
        #"{"revision":"abc123","tolerance":0.0001,"prompts":{"chat-capital":{"tokens":[1,2,3]},"code-fib":{"tokens":[4,5]}}}"#
    )
    try box.write(
        "README.md",
        """
        # basalt
        | Golden-tested model | Class | Fixture |
        |---|---|---|
        | [`org/alpha`](https://huggingface.co/org/alpha) | dense | [golden](Tests/BasaltTests/Fixtures/golden/org--alpha.json) |

        ## Inference: fast
        - Fast decode at 80 tok/s ([#239](https://github.com/goodmai/basalt/pull/239)).
        """
    )
    try box.write(
        "docs/benchmarks.md",
        """
        # Features and benchmarks
        ## Features at a glance
        | Feature | Flag | What it does | Measured |
        |---|---|---|---|
        | KV bits | `serve --kv-bits 4\\|8` | compresses KV | 1.0 GB |

        ## Where the README's numbers come from
        | Claim | Source |
        |---|---|
        | Fast decode 80 tok/s | [#239](https://github.com/goodmai/basalt/pull/239) |

        ### Results
        | Model | Ladder |
        |---|---|
        | `org/alpha` | 4/5 |
        """
    )
    try box.write(
        "docs/models.md",
        """
        # Models
        ### Measured for 0.8
        | Model | Params | Active memory | Decode, tok/s | Notes |
        |---|---|---|---|---|
        | `org/alpha` | 4B | 2.2 GB | 80 | dense |
        """
    )
    try box.write("benchmarks/ladder/run.py", "print('ok')\n")
    try box.write("benchmarks/cacheblend/file0.txt", "chunk0\n")
}

var failed = 0
func expect(_ label: String, _ condition: Bool, _ detail: String = "") {
    if !condition {
        failed += 1
        print("FAIL \(label)\n\(detail)")
    }
}

do {
    let box = try Sandbox()
    defer { box.cleanup() }
    try seedValid(box)

    let gen = try box.run()
    expect("generates page and collection manifest", gen.status == 0, gen.output)

    let json = try String(contentsOf: box.root.appendingPathComponent("docs/benchmark-page/hf-collection.json"), encoding: .utf8)
    let html = try String(contentsOf: box.root.appendingPathComponent("docs/benchmark-page/index.html"), encoding: .utf8)
    expect("manifest stays unpublished until human review", json.contains("\"publish\" : false") && json.contains("\"requiresHumanReview\" : true"), json)
    expect("manifest includes logged 0.8 tok/s and step count", json.contains("\"decodeTokPerSec\" : \"80\"") && json.contains("\"verifiedGreedySteps\" : 5"), json)
    expect("html preserves escaped pipe in flag cell", html.contains("<code>serve --kv-bits 4|8</code>"), html)

    let checkOK = try box.run(["--check"])
    expect("--check passes when generated files match", checkOK.status == 0, checkOK.output)

    try box.write("docs/benchmark-page/index.html", html + " ")
    let stale = try box.run(["--check"])
    expect("--check fails when page is stale", stale.status != 0 && stale.output.contains("out of date"), stale.output)

    try box.write(
        "logs/runner.log",
        """
        2026-11-13T12:00:00Z start commit=deadbeef machine=Mac16,7
        2026-11-13T12:01:00Z summary | prompt | tok/s |
        2026-11-13T12:01:00Z summary | --- | --- |
        2026-11-13T12:01:00Z summary | code | 81.2 |
        """
    )
    let withLogs = try box.run(["--logs", box.root.appendingPathComponent("logs").path])
    let logsHTML = try String(contentsOf: box.root.appendingPathComponent("docs/benchmark-page/index.html"), encoding: .utf8)
    expect("--logs renders runner summary table", withLogs.status == 0 && logsHTML.contains("Logged Runner Tables") && logsHTML.contains("81.2"), logsHTML)
}

do {
    let box = try Sandbox()
    defer { box.cleanup() }
    try seedValid(box)
    try box.write(
        "README.md",
        """
        # basalt
        | [`org/alpha`](https://huggingface.co/org/alpha) | dense | [golden](Tests/BasaltTests/Fixtures/golden/org--alpha.json) |
        ## Inference: fast
        - Uncited claim at 999 tok/s with no link.
        """
    )
    let badClaim = try box.run()
    expect("uncited quantitative README bullet fails", badClaim.status != 0 && badClaim.output.contains("lacks a PR, test or fixture link"), badClaim.output)

    try seedValid(box)
    try FileManager.default.removeItem(at: box.root.appendingPathComponent("Tests/BasaltTests/Fixtures/golden/org--alpha.json"))
    let missingFixture = try box.run()
    expect("missing golden fixture fails", missingFixture.status != 0 && missingFixture.output.contains("unreadable or empty golden fixture"), missingFixture.output)
}

print(failed == 0 ? "build_benchmark_page: 8 cases pass" : "build_benchmark_page: \(failed) case(s) failed")
exit(failed == 0 ? 0 : 1)
