#!/usr/bin/env swift
import Foundation

// One cleanup script for the repository, run from its root.
//   daily   — rotate logs/, images/ and reports/: remove files older than --max-age-hours
//   release — remove build/runtime artifacts before publishing (the old clean_for_github.swift)
//   all     — both
// Every mode is a dry run unless --apply is given, and nothing tracked by git is ever removed
// (images/ holds the README screenshots next to generated ones).

enum LogLevel: Int { case debug = 0, info, warning, error }

var currentLogLevel: LogLevel = .info
func log(_ message: String, level: LogLevel = .info) {
    if level.rawValue >= currentLogLevel.rawValue { print(message) }
}

func printUsage() {
    print("""
    Usage:
      ./scripts/cleanup.swift [--mode=daily|release|all] [--apply] [--include-local] [--max-age-hours=N] [--verbose]

    Options:
      --mode=MODE         daily: rotate logs/, images/, reports/; release: build/runtime artifacts;
                          all: both (default: release)
      --dry-run           Show what would be removed (default)
      --apply             Actually remove the listed artifacts
      --include-local     release: also remove local working artifacts (.archive, .claude, .gemini.md,
                          test_*.swift in the repo root)
      --max-age-hours=N   daily: remove files older than N hours (default 24; 0 removes all)
      --verbose           Show debug output
      -h, --help          Show this help
    """)
}

var mode = "release"
var apply = false
var includeLocal = false
var maxAgeHours = 24.0

for arg in ProcessInfo.processInfo.arguments.dropFirst() {
    switch arg {
    case "--dry-run": apply = false
    case "--apply": apply = true
    case "--include-local": includeLocal = true
    case "--verbose": currentLogLevel = .debug
    case "-h", "--help":
        printUsage()
        exit(0)
    case _ where arg.hasPrefix("--mode="):
        mode = String(arg.dropFirst("--mode=".count))
        guard ["daily", "release", "all"].contains(mode) else {
            log("Unknown mode: \(mode)", level: .error)
            printUsage()
            exit(1)
        }
    case _ where arg.hasPrefix("--max-age-hours="):
        guard let hours = Double(arg.dropFirst("--max-age-hours=".count)), hours >= 0 else {
            log("Invalid \(arg)", level: .error)
            exit(1)
        }
        maxAgeHours = hours
    default:
        log("Unknown option: \(arg)", level: .error)
        printUsage()
        exit(1)
    }
}

let fm = FileManager.default
let currentDir = fm.currentDirectoryPath

func relative(_ path: String) -> String { path.replacingOccurrences(of: currentDir + "/", with: "") }

func regexMatch(_ string: String, pattern: String) -> Bool {
    let regex = try? NSRegularExpression(pattern: pattern)
    return regex?.firstMatch(in: string, range: NSRange(location: 0, length: string.utf16.count)) != nil
}

/// True when git tracks the path (for a directory: anything inside it).
func isTracked(_ path: String) -> Bool {
    let task = Process()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    task.arguments = ["ls-files", "--", relative(path)]
    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = FileHandle.nullDevice
    guard (try? task.run()) != nil else { return false }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    task.waitUntilExit()
    return !data.isEmpty
}

// MARK: — release

func releaseCandidates() -> [String] {
    let alwaysDirs = [".build", ".swiftpm", "logs", "docs-output", "docs-static", ".docc"]
    // default.metallib is deliberately absent: the MLX runtime loads it from the repo root.
    let alwaysFiles = [".coverage", "context_latency.json", "auth.sqlite3"]
    let recursiveFilePatterns = [
        "\\.DS_Store", ".*\\.log$", ".*\\.tmp$", ".*\\.swp$", ".*~", ".*\\.profdata$",
        "^server_.*\\.log$", "^benchmark_.*\\.json$", "^res_.*\\.json$",
    ]
    let rootFilePatterns = [".*\\.db$", ".*\\.sqlite$", ".*\\.sqlite3$"]
    let optionalLocalPaths = [".archive", ".claude", ".gemini.md"]

    var collected = (alwaysDirs + alwaysFiles + (includeLocal ? optionalLocalPaths : []))
        .map { currentDir + "/" + $0 }
        .filter { fm.fileExists(atPath: $0) }

    if let enumerator = fm.enumerator(atPath: currentDir) {
        for case let relPath as String in enumerator {
            if relPath == ".git" || relPath.hasPrefix(".git/") || relPath.hasPrefix(".build/") { continue }
            let url = URL(fileURLWithPath: currentDir + "/" + relPath)
            if (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true { continue }
            let filename = url.lastPathComponent
            if recursiveFilePatterns.contains(where: { regexMatch(filename, pattern: $0) }) {
                collected.append(url.path)
            } else if !relPath.contains("/") {
                if rootFilePatterns.contains(where: { regexMatch(filename, pattern: $0) })
                    || (includeLocal && regexMatch(filename, pattern: "^test_.*\\.swift$")) {
                    collected.append(url.path)
                }
            }
        }
    }
    return collected
}

// MARK: — daily

func dailyCandidates() -> [String] {
    let cutoff = Date().addingTimeInterval(-maxAgeHours * 3600)
    var collected: [String] = []
    for dirName in ["logs", "images", "reports"] {
        let dir = URL(fileURLWithPath: currentDir).appendingPathComponent(dirName)
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else {
            log("Directory \(dirName) does not exist, skipping.", level: .debug)
            continue
        }
        let keys: [URLResourceKey] = [.contentModificationDateKey, .creationDateKey]
        let contents = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: keys)) ?? []
        for file in contents where file.lastPathComponent != ".gitkeep" {
            let values = try? file.resourceValues(forKeys: Set(keys))
            let date = values?.contentModificationDate ?? values?.creationDate ?? Date()
            if maxAgeHours == 0 || date < cutoff {
                collected.append(file.path)
            } else {
                log("Kept recent: \(relative(file.path))", level: .debug)
            }
        }
    }
    return collected
}

// MARK: — run

var candidates: [String] = []
if mode == "daily" || mode == "all" { candidates += dailyCandidates() }
if mode == "release" || mode == "all" { candidates += releaseCandidates() }

var toRemove: [String] = []
var skippedTracked: [String] = []
for path in Set(candidates) where fm.fileExists(atPath: path) {
    if isTracked(path) { skippedTracked.append(path) } else { toRemove.append(path) }
}
// A file inside a directory that is removed anyway needs no line of its own.
toRemove = toRemove.filter { path in !toRemove.contains { path.hasPrefix($0 + "/") } }

log("Repository: \(currentDir)")
log("Mode: \(mode), \(apply ? "apply" : "dry-run")")
if mode != "daily" { log("Include local artifacts: \(includeLocal)") }
if mode != "release" { log("Daily max age: \(maxAgeHours == 0 ? "none (all files)" : "\(maxAgeHours) h")") }
log("")

if toRemove.isEmpty {
    log("No removable artifacts found.")
} else {
    log("Artifacts selected for cleanup (\(toRemove.count)):")
    for path in toRemove.sorted() { log("  - \(relative(path))") }
}
if !skippedTracked.isEmpty {
    log("\nSkipped because tracked by git (\(skippedTracked.count)):")
    for path in skippedTracked.sorted() { log("  - \(relative(path))") }
}

guard apply else {
    log("\nDry-run complete. Re-run with --apply to delete listed artifacts.")
    exit(0)
}

var failed = false
for path in toRemove {
    do {
        try fm.removeItem(atPath: path)
        log("Removed \(relative(path))", level: .debug)
    } catch {
        log("Failed to remove \(relative(path)): \(error.localizedDescription)", level: .error)
        failed = true
    }
}
log(failed ? "\nCleanup finished with errors." : "\nCleanup applied.")
exit(failed ? 1 : 0)
