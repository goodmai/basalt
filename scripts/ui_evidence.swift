#!/usr/bin/env swift
import Foundation

struct CLIOptions {
    var recordBeforeDir: String?
    var baselineDir: String?
    var outputDir: String?
    var commitBase: String?
    var commitHead: String?
    var skipVideos = false
    var dryRun = false
}

func printUsage() {
    print("""
    Usage:
      ./scripts/ui_evidence.swift [options]

    Options:
      --record-before=DIR   Export deterministic golden PNGs of current UI into DIR
      --baseline=DIR        Directory of 'before' PNGs to compare against (default: Tests/BasaltTests/Fixtures/VisualGoldens)
      --output=DIR          Output evidence directory (default: logs/ui-evidence-YYYY-MM-DD)
      --commit-base=REF     Base commit label for manifest.json / report.md
      --commit-head=REF     Head commit label for manifest.json / report.md
      --skip-videos         Skip 60 fps MP4 video rendering
      --dry-run             Print planned command without executing swift test
      -h, --help            Show this help
    """)
}

func parseOptions(_ arguments: [String]) -> CLIOptions {
    var opts = CLIOptions()
    for arg in arguments {
        switch arg {
        case "-h", "--help":
            printUsage()
            exit(0)
        case "--skip-videos":
            opts.skipVideos = true
        case "--dry-run":
            opts.dryRun = true
        case _ where arg.hasPrefix("--record-before="):
            opts.recordBeforeDir = String(arg.dropFirst("--record-before=".count))
        case _ where arg.hasPrefix("--baseline="):
            opts.baselineDir = String(arg.dropFirst("--baseline=".count))
        case _ where arg.hasPrefix("--output="):
            opts.outputDir = String(arg.dropFirst("--output=".count))
        case _ where arg.hasPrefix("--commit-base="):
            opts.commitBase = String(arg.dropFirst("--commit-base=".count))
        case _ where arg.hasPrefix("--commit-head="):
            opts.commitHead = String(arg.dropFirst("--commit-head=".count))
        default:
            fputs("Unknown option: \(arg)\n", stderr)
            printUsage()
            exit(1)
        }
    }
    return opts
}

func gitShortRev(_ ref: String) -> String {
    let task = Process()
    let pipe = Pipe()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    task.arguments = ["rev-parse", "--short", ref]
    task.standardOutput = pipe
    task.standardError = FileHandle.nullDevice
    guard (try? task.run()) != nil else { return ref }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    task.waitUntilExit()
    guard task.terminationStatus == 0 else { return ref }
    let trimmed = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? ref : trimmed
}

func runSwiftTest(filter: String, extraEnv: [String: String], dryRun: Bool) -> Int32 {
    if dryRun {
        let pairs = extraEnv.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " ")
        print("[dry-run] \(pairs) swift test --filter \(filter)")
        return 0
    }
    let task = Process()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    task.arguments = ["swift", "test", "--filter", filter]
    var env = ProcessInfo.processInfo.environment
    for (key, value) in extraEnv { env[key] = value }
    task.environment = env
    do {
        try task.run()
        task.waitUntilExit()
        return task.terminationStatus
    } catch {
        fputs("Failed to launch swift test: \(error.localizedDescription)\n", stderr)
        return 1
    }
}

let opts = parseOptions(Array(ProcessInfo.processInfo.arguments.dropFirst()))
let cwd = FileManager.default.currentDirectoryPath

if let recordDir = opts.recordBeforeDir {
    let absDir = recordDir.hasPrefix("/") ? recordDir : "\(cwd)/\(recordDir)"
    let status = runSwiftTest(
        filter: "AppVisualSnapshotTests",
        extraEnv: ["BASALT_UPDATE_VISUAL_GOLDENS": "1", "BASALT_VISUAL_EXPORT_DIR": absDir],
        dryRun: opts.dryRun
    )
    exit(status)
}

let formatter = DateFormatter()
formatter.dateFormat = "yyyy-MM-dd"
formatter.timeZone = TimeZone(secondsFromGMT: 0)
let defaultOut = "logs/ui-evidence-\(formatter.string(from: Date()))"
let rawOut = opts.outputDir ?? defaultOut
let absOut = rawOut.hasPrefix("/") ? rawOut : "\(cwd)/\(rawOut)"
let rawBase = opts.baselineDir ?? "Tests/BasaltTests/Fixtures/VisualGoldens"
let absBase = rawBase.hasPrefix("/") ? rawBase : "\(cwd)/\(rawBase)"
let baseLabel = opts.commitBase ?? gitShortRev("origin/main")
let headLabel = opts.commitHead ?? gitShortRev("HEAD")

let status = runSwiftTest(
    filter: "AppEvidencePackageRunner",
    extraEnv: [
        "BASALT_GENERATE_UI_EVIDENCE": "1",
        "BASALT_UI_BASELINE_DIR": absBase,
        "BASALT_UI_EVIDENCE_DIR": absOut,
        "BASALT_UI_COMMIT_BASE": baseLabel,
        "BASALT_UI_COMMIT_HEAD": headLabel,
        "BASALT_UI_SKIP_VIDEOS": opts.skipVideos ? "1" : "0",
    ],
    dryRun: opts.dryRun
)
exit(status)
