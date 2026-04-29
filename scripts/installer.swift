#!/usr/bin/env swift
import Foundation

enum LogLevel: Int {
    case debug = 0, info, warning, error
}

var currentLogLevel: LogLevel = .info
var nonInteractive = false
var isDryRun = false

func log(_ message: String, level: LogLevel = .info) {
    if level.rawValue >= currentLogLevel.rawValue {
        print(message)
    }
}

let arguments = ProcessInfo.processInfo.arguments.dropFirst()

func printUsage() {
    print("""
    Usage:
      ./scripts/installer.swift [--verbose] [--non-interactive] [--dry-run]

    Options:
      --verbose          Show debug output
      --non-interactive  Skip user prompts (will perform setup without asking, or exit if input needed)
      --dry-run          Don't actually install or build
      -h, --help         Show this help
    """)
}

for arg in arguments {
    switch arg {
    case "--verbose": currentLogLevel = .debug
    case "--non-interactive": nonInteractive = true
    case "--dry-run": isDryRun = true
    case "-h", "--help":
        printUsage()
        exit(0)
    case "setup":
        break // backward compatibility
    default:
        log("Unknown option: \(arg)", level: .warning)
    }
}

log("==============================================", level: .info)
log("🚀 Installing Gemm for Apple Silicon...", level: .info)
log("==============================================", level: .info)

@discardableResult
func runCommand(_ command: String, args: [String]) -> (status: Int32, output: String) {
    log("Running: \(command) \(args.joined(separator: " "))", level: .debug)
    
    if isDryRun && command != "/usr/bin/uname" && command != "/usr/bin/which" {
        return (0, "")
    }

    let task = Process()
    let pipe = Pipe()
    task.executableURL = URL(fileURLWithPath: command)
    task.arguments = args
    task.standardOutput = pipe
    try? task.run()
    task.waitUntilExit()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    let output = String(data: data, encoding: .utf8) ?? ""
    return (task.terminationStatus, output.trimmingCharacters(in: .whitespacesAndNewlines))
}

let unameResult = runCommand("/usr/bin/uname", args: [])
if unameResult.output != "Darwin" && !isDryRun {
    log("❌ Error: Gemm requires macOS.", level: .error)
    exit(1)
}

let archResult = runCommand("/usr/bin/uname", args: ["-m"])
if archResult.output != "arm64" && !isDryRun {
    log("❌ Error: Gemm requires Apple Silicon (M1/M2/M3/M4).", level: .error)
    exit(1)
}

let swiftResult = runCommand("/usr/bin/which", args: ["swift"])
if swiftResult.status != 0 && !isDryRun {
    log("❌ Error: Swift is not installed. Please install Xcode or Xcode Command Line Tools.", level: .error)
    exit(1)
}

log("📦 Building release version...", level: .info)
let buildResult = runCommand("/usr/bin/swift", args: ["build", "-c", "release"])
if buildResult.status != 0 && !isDryRun {
    log("❌ Error: Build failed.", level: .error)
    exit(1)
}

let binPath = ".build/release/Gemm"
let fm = FileManager.default

if !fm.fileExists(atPath: binPath) && !isDryRun {
    log("❌ Error: Build failed, binary not found.", level: .error)
    exit(1)
}

if isDryRun {
    log("Dry run: Skipping installation prompts.", level: .info)
    exit(0)
}

var choice = "3"
if nonInteractive {
    log("Running in non-interactive mode. Skipping installation.", level: .info)
} else {
    log("\nChoose installation method:", level: .info)
    log("1) Install to /usr/local/bin (requires sudo)", level: .info)
    log("2) Add alias to shell config", level: .info)
    log("3) Skip installation", level: .info)
    print("Enter choice (1-3): ", terminator: "")
    
    if let userChoice = readLine() {
        choice = userChoice
    }
}

switch choice {
case "1":
    log("🔑 Requesting sudo to install to /usr/local/bin/gemm...", level: .info)
    let _ = runCommand("/usr/bin/sudo", args: ["mkdir", "-p", "/usr/local/bin"])
    let sudoResult2 = runCommand("/usr/bin/sudo", args: ["cp", binPath, "/usr/local/bin/gemm"])
    if sudoResult2.status == 0 {
        log("✅ Installation complete! You can now run: gemm --help", level: .info)
    } else {
        log("❌ Error during installation.", level: .error)
    }
case "2":
    let home = fm.homeDirectoryForCurrentUser.path
    var shellRC = ""
    let zshrc = home + "/.zshrc"
    let bashrc = home + "/.bashrc"
    
    if fm.fileExists(atPath: zshrc) {
        shellRC = zshrc
    } else if fm.fileExists(atPath: bashrc) {
        shellRC = bashrc
    }
    
    let pwd = fm.currentDirectoryPath
    let aliasStr = "\n# Gemm alias\nalias gemm='swift run --package-path \(pwd) Gemm'\n"
    
    if !shellRC.isEmpty {
        if let data = aliasStr.data(using: .utf8),
           let fileHandle = FileHandle(forWritingAtPath: shellRC) {
            fileHandle.seekToEndOfFile()
            fileHandle.write(data)
            fileHandle.closeFile()
            log("✅ Alias added to \(shellRC)", level: .info)
            log("Run: source \(shellRC)", level: .info)
        } else {
            log("❌ Failed to write to \(shellRC)", level: .error)
        }
    } else {
        log("❌ Could not detect shell config. Add this to your shell config:", level: .error)
        log("alias gemm='swift run --package-path \(pwd) Gemm'", level: .info)
    }
case "3":
    log("Skipped. Run with: swift run Gemm --help", level: .info)
default:
    log("Invalid choice. Exiting.", level: .warning)
}

log("\n🎉 Setup complete!\n", level: .info)
log("Next steps:", level: .info)
log("1. Download a model: huggingface-cli download mlx-community/Qwen3.5-4B-4bit", level: .info)
log("2. Start server: gemm serve --model mlx-community/Qwen3.5-4B-4bit --rest", level: .info)
log("3. Test it: curl http://localhost:8080/api/v1/health", level: .info)
