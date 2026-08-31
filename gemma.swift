#!/usr/bin/swift

import Foundation
#if os(macOS)
import Darwin
#elseif os(Linux)
import Glibc
#endif

// MARK: - Constants & Config
let defaultModel = "mlx-community/Qwen3.5-4B-4bit"
let defaultPort = 8080

// MARK: - ANSI Logging
func log(_ message: String) { print("\u{001B}[2m[\(currentTime())]\u{001B}[0m \(message)") }
func info(_ message: String) { print("\u{001B}[1;36m[\(currentTime())]\u{001B}[0m \(message)") }
func ok(_ message: String) { print("\u{001B}[1;32m[\(currentTime())] \(message)\u{001B}[0m") }
func err(_ message: String) { print("\u{001B}[1;31m[\(currentTime())] ERROR: \(message)\u{001B}[0m") }
func currentTime() -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "HH:mm:ss"
    return formatter.string(from: Date())
}

// MARK: - App State
var model = defaultModel
var port = defaultPort
var build = true
var claudeArgs: [String] = []

// MARK: - Parsing Args
var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let arg = args.removeFirst()
    switch arg {
    case "--model":
        if !args.isEmpty { model = args.removeFirst() }
    case "--port":
        if !args.isEmpty, let p = Int(args.removeFirst()) { port = p }
    case "--no-build":
        build = false
    case "--":
        claudeArgs = args
        args.removeAll()
    default:
        fputs("Unknown argument: \(arg)\n", stderr)
        exit(1)
    }
}

// MARK: - Paths
let fileManager = FileManager.default
let scriptPath = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
let scriptDir = scriptPath.deletingLastPathComponent().path
let binaryPath = "\(scriptDir)/.build/release/gemm"
let homeDir = fileManager.homeDirectoryForCurrentUser.path
let gemDir = "\(homeDir)/.gemm"
let pidFile = "\(gemDir)/server.pid"
let logFile = "\(gemDir)/server.log"

try? fileManager.createDirectory(atPath: gemDir, withIntermediateDirectories: true)

// MARK: - Cleanup functions
func cleanup() {
    if let pidStr = try? String(contentsOfFile: pidFile, encoding: .utf8), 
       let pid = Int32(pidStr.trimmingCharacters(in: .whitespacesAndNewlines)) {
        log("Stopping Gemm server (PID \(pid))…")
        kill(pid, SIGTERM)
        for _ in 0..<50 {
            if kill(pid, 0) != 0 { break }
            Thread.sleep(forTimeInterval: 0.1)
        }
        kill(pid, SIGKILL)
        try? fileManager.removeItem(atPath: pidFile)
        log("Server stopped.")
    }
}

// Ensure cleanup on normal exit
atexit {
    cleanup()
}

// Handle signals
signal(SIGINT) { _ in cleanup(); exit(1) }
signal(SIGTERM) { _ in cleanup(); exit(1) }
signal(SIGHUP) { _ in cleanup(); exit(1) }

// Stop stale server
cleanup()

// MARK: - Build
if build {
    info("Building gemm (incremental)…")
    let buildProc = Process()
    buildProc.executableURL = URL(fileURLWithPath: "/bin/bash")
    buildProc.arguments = ["-c", "swift build -c release 2>&1 | tail -5"]
    buildProc.currentDirectoryURL = URL(fileURLWithPath: scriptDir)
    try? buildProc.run()
    buildProc.waitUntilExit()
    if buildProc.terminationStatus != 0 {
        err("Build failed.")
        exit(1)
    }
    ok("Build done ✓")
} else {
    guard fileManager.fileExists(atPath: binaryPath) else {
        err("Binary not found and --no-build set. Run without --no-build first.")
        exit(1)
    }
    log("Skipping build (--no-build).")
}

print("\n")
info("Model : \(model)")
info("Port  : \(port)")
info("Log   : tail -f \(logFile)")
print("\n")

let sessionLog = "\n── Gemma session \(Date()) ──\n"
if let handle = FileHandle(forWritingAtPath: logFile) {
    handle.seekToEndOfFile()
    handle.write(sessionLog.data(using: .utf8)!)
    try? handle.close()
} else {
    try? sessionLog.write(toFile: logFile, atomically: false, encoding: .utf8)
}

// MARK: - Start Server
let serverProc = Process()
serverProc.executableURL = URL(fileURLWithPath: binaryPath)
serverProc.arguments = ["serve", "--model", model, "--port", String(port), "--rest"]

if let logOut = FileHandle(forWritingAtPath: logFile) {
    logOut.seekToEndOfFile()
    serverProc.standardOutput = logOut
    serverProc.standardError = logOut
}

do {
    try serverProc.run()
} catch {
    err("Failed to start Gemm server: \(error)")
    exit(1)
}

let serverPID = serverProc.processIdentifier
try? String(serverPID).write(toFile: pidFile, atomically: false, encoding: .utf8)
log("Server started (PID \(serverPID))")

log("Waiting for server (model loading may take 10–60 s)…")

// Start tailing
let tailProc = Process()
tailProc.executableURL = URL(fileURLWithPath: "/usr/bin/tail")
tailProc.arguments = ["-F", logFile]
try? tailProc.run()

// Wait for ready
let healthUrl = URL(string: "http://127.0.0.1:\(port)/v1/models")!
var ready = false
for _ in 0..<120 {
    if kill(serverPID, 0) != 0 {
        err("Server process died during startup. Check: tail -f \(logFile)")
        tailProc.terminate()
        exit(1)
    }
    
    let group = DispatchGroup()
    group.enter()
    var success = false
    let task = URLSession.shared.dataTask(with: healthUrl) { data, response, error in
        if let httpRes = response as? HTTPURLResponse, httpRes.statusCode == 200 {
            success = true
        }
        group.leave()
    }
    task.resume()
    _ = group.wait(timeout: .now() + 1)
    if success {
        ready = true
        break
    }
    Thread.sleep(forTimeInterval: 1.0)
}

tailProc.terminate()

if !ready {
    err("Server not ready after 120 s. Check: tail -f \(logFile)")
    exit(1)
}

print("\n")
ok("Server ready ✓  (PID \(serverPID))")

// Preflight test
log("Pre-flight: testing POST /v1/messages…")
var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/messages")!)
request.httpMethod = "POST"
request.addValue("application/json", forHTTPHeaderField: "Content-Type")
request.addValue("local", forHTTPHeaderField: "x-api-key")
request.httpBody = "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"stream\":false}".data(using: .utf8)

var preflightStatus = 0
let pfGroup = DispatchGroup()
pfGroup.enter()
URLSession.shared.dataTask(with: request) { _, response, _ in
    if let res = response as? HTTPURLResponse {
        preflightStatus = res.statusCode
    }
    pfGroup.leave()
}.resume()
pfGroup.wait()

if preflightStatus == 0 {
    err("Pre-flight FAILED — server not reachable on 127.0.0.1:\(port)")
    exit(1)
} else if preflightStatus >= 200 && preflightStatus < 300 {
    ok("Pre-flight passed ✓  (HTTP \(preflightStatus))")
} else {
    ok("Pre-flight passed ✓  (HTTP \(preflightStatus) — endpoint reachable)")
}

let claudeLocalId = "claude-local/" + model.replacingOccurrences(of: "/", with: "--")
let anthropicBaseUrl = "http://127.0.0.1:\(port)"
let anthropicApiKey = ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"] ?? "local"

let claudeSettings = "{\"model\": \"\(claudeLocalId)\", \"env\": { \"ANTHROPIC_BASE_URL\": \"\(anthropicBaseUrl)\", \"ANTHROPIC_API_KEY\": \"\(anthropicApiKey)\" }}"

print("\n")
info("┌─────────────────────────────────────────────────────┐")
info("│  Gemm → Claude Code proxy active                    │")
info("├─────────────────────────────────────────────────────┤")
info("│  ANTHROPIC_BASE_URL = \(anthropicBaseUrl)")
info("│  ANTHROPIC_API_KEY  = \(String(anthropicApiKey.prefix(6)))…")
info("│  Model              = \(model)")
info("└─────────────────────────────────────────────────────┘")
print("\n")

info("Launching: claude \(claudeArgs.joined(separator: " "))")
print("\n")

let claudeProc = Process()
claudeProc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
claudeProc.arguments = ["claude"] + claudeArgs + ["--settings", claudeSettings]

var env = ProcessInfo.processInfo.environment
env["ANTHROPIC_BASE_URL"] = anthropicBaseUrl
env["ANTHROPIC_API_KEY"] = anthropicApiKey
claudeProc.environment = env

do {
    try claudeProc.run()
    claudeProc.waitUntilExit()
} catch {
    err("Failed to run claude: \(error)")
    exit(1)
}
