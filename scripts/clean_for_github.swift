#!/usr/bin/env swift
import Foundation

enum LogLevel: Int {
    case debug = 0, info, warning, error
}

var currentLogLevel: LogLevel = .info

func log(_ message: String, level: LogLevel = .info) {
    if level.rawValue >= currentLogLevel.rawValue {
        print(message)
    }
}

// Clean temporary/local artifacts before publishing repository to GitHub.
// Default mode is dry-run (no deletions).

let args = ProcessInfo.processInfo.arguments.dropFirst()
var mode = "dry-run"
var includeLocal = false

func printUsage() {
    print("""
    Usage:
      ./scripts/clean_for_github.swift [--dry-run] [--apply] [--include-local] [--verbose]

    Options:
      --dry-run        Show what would be removed (default)
      --apply          Actually remove matched artifacts
      --include-local  Also remove local working artifacts (.archive, .claude, .gemini.md, test_*.swift in repo root)
      --verbose        Show debug output
      -h, --help       Show this help
    """)
}

for arg in args {
    switch arg {
    case "--dry-run": mode = "dry-run"
    case "--apply": mode = "apply"
    case "--include-local": includeLocal = true
    case "--verbose": currentLogLevel = .debug
    case "-h", "--help":
        printUsage()
        exit(0)
    default:
        log("Unknown option: \(arg)", level: .error)
        printUsage()
        exit(1)
    }
}

let fm = FileManager.default
let currentDir = fm.currentDirectoryPath

let alwaysDirs = [
    ".build", ".swiftpm", "logs", "docs-output", "docs-static", ".docc"
]

// Exclude default.metallib explicitly
let alwaysFiles = [
    ".coverage", "context_latency.json", "auth.sqlite3" // Notice: default.metallib removed from here
]

let recursiveFilePatterns = [
    "\\.DS_Store", ".*\\.log$", ".*\\.tmp$", ".*\\.swp$", ".*~", ".*\\.profdata$",
    "^server_.*\\.log$", "^benchmark_.*\\.json$", "^res_.*\\.json$"
]

let rootFilePatterns = [
    ".*\\.db$", ".*\\.sqlite$", ".*\\.sqlite3$"
]

let optionalLocalPaths = [
    ".archive", ".claude", ".gemini.md"
]

var collected: [String] = []

func regexMatch(_ string: String, pattern: String) -> Bool {
    let regex = try? NSRegularExpression(pattern: pattern)
    let range = NSRange(location: 0, length: string.utf16.count)
    return regex?.firstMatch(in: string, options: [], range: range) != nil
}

log("Scanning for artifacts to clean...", level: .debug)

// Add always dirs
for d in alwaysDirs {
    let path = currentDir + "/" + d
    if fm.fileExists(atPath: path) {
        collected.append(path)
        log("Found directory: \(path)", level: .debug)
    }
}

// Add always files
for f in alwaysFiles {
    let path = currentDir + "/" + f
    if fm.fileExists(atPath: path) {
        collected.append(path)
        log("Found file: \(path)", level: .debug)
    }
}

// Recursively find files
if let enumerator = fm.enumerator(atPath: currentDir) {
    for case let relPath as String in enumerator {
        if relPath.hasPrefix(".git/") { continue }
        
        let url = URL(fileURLWithPath: currentDir + "/" + relPath)
        let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
        
        if isDir { continue }
        
        let filename = url.lastPathComponent
        
        for pattern in recursiveFilePatterns {
            if regexMatch(filename, pattern: pattern) {
                collected.append(url.path)
                log("Found file matching recursive pattern \(pattern): \(url.path)", level: .debug)
                break
            }
        }
        
        if !relPath.contains("/") {
            for pattern in rootFilePatterns {
                if regexMatch(filename, pattern: pattern) {
                    collected.append(url.path)
                    log("Found file matching root pattern \(pattern): \(url.path)", level: .debug)
                    break
                }
            }
            if includeLocal && regexMatch(filename, pattern: "^test_.*\\.swift$") {
                collected.append(url.path)
                log("Found local test file: \(url.path)", level: .debug)
            }
        }
    }
}

if includeLocal {
    for p in optionalLocalPaths {
        let path = currentDir + "/" + p
        if fm.fileExists(atPath: path) {
            collected.append(path)
            log("Found optional local path: \(path)", level: .debug)
        }
    }
}

let uniqueCollected = Array(Set(collected))

func isTracked(path: String) -> Bool {
    let relPath = path.replacingOccurrences(of: currentDir + "/", with: "")
    let task = Process()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    task.arguments = ["ls-files", "--error-unmatch", relPath]
    try? task.run()
    task.waitUntilExit()
    return task.terminationStatus == 0
}

func hasTrackedInDir(path: String) -> Bool {
    let relPath = path.replacingOccurrences(of: currentDir + "/", with: "")
    let task = Process()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    task.arguments = ["ls-files", relPath]
    let pipe = Pipe()
    task.standardOutput = pipe
    try? task.run()
    task.waitUntilExit()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    return !data.isEmpty
}

var toRemove: [String] = []
var skippedTracked: [String] = []

for p in uniqueCollected {
    var isDir: ObjCBool = false
    if fm.fileExists(atPath: p, isDirectory: &isDir) {
        if isDir.boolValue {
            if hasTrackedInDir(path: p) {
                skippedTracked.append(p)
            } else {
                toRemove.append(p)
            }
        } else {
            if isTracked(path: p) {
                skippedTracked.append(p)
            } else {
                toRemove.append(p)
            }
        }
    }
}

log("Repository: \(currentDir)", level: .info)
log("Mode: \(mode)", level: .info)
log("Include local artifacts: \(includeLocal)\n", level: .info)

if toRemove.isEmpty {
    log("No removable temporary artifacts found.", level: .info)
} else {
    log("Artifacts selected for cleanup (\(toRemove.count)):", level: .info)
    for p in toRemove.sorted() {
        log("  - \(p.replacingOccurrences(of: currentDir + "/", with: ""))", level: .info)
    }
}

if !skippedTracked.isEmpty {
    log("\nSkipped because tracked by git (\(skippedTracked.count)):", level: .info)
    for p in skippedTracked.sorted() {
        log("  - \(p.replacingOccurrences(of: currentDir + "/", with: ""))", level: .info)
    }
}

if mode == "dry-run" {
    log("\nDry-run complete. Re-run with --apply to delete listed artifacts.", level: .info)
    exit(0)
}

if toRemove.isEmpty {
    log("\nNothing to delete.", level: .info)
    exit(0)
}

for p in toRemove {
    try? fm.removeItem(atPath: p)
    log("Removed \(p)", level: .debug)
}

log("\nCleanup applied.", level: .info)
