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

let args = ProcessInfo.processInfo.arguments.dropFirst()
if args.contains("--verbose") {
    currentLogLevel = .debug
}
if args.contains("--help") || args.contains("-h") {
    print("""
    Usage:
      ./scripts/build_metal.swift [--verbose] [--dry-run]

    Options:
      --verbose        Show debug output
      --dry-run        Don't actually build anything
      -h, --help       Show this help
    """)
    exit(0)
}
// CI runs an older SDK than the one MLX targets, and a few kernels include
// headers that only exist in the newer one (MetalPerformancePrimitives). The
// resulting library is still loadable, which is all the test suite needs — but
// a developer build must stay strict, so this is opt-in.
let skipUnsupported = args.contains("--skip-unsupported")

if args.contains("--dry-run") {
    log("Dry run: build_metal", level: .info)
    exit(0)
}

let kernelDir = ".build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels"
let genKernelDir = ".build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"
let outputDir = ".build/debug"
let tempDir = ".build/metal_temp"
let metalLibPath = "\(outputDir)/default.metallib"

let fm = FileManager.default

// MARK: — Check if metallib is up to date

func getModificationDate(of path: String) -> Date? {
    guard let attrs = try? fm.attributesOfItem(atPath: path) else { return nil }
    return attrs[.modificationDate] as? Date
}

func findMetalFiles(in directory: String) -> [String] {
    var files: [String] = []
    guard let enumerator = fm.enumerator(atPath: directory) else { return files }
    for case let file as String in enumerator {
        if file.hasSuffix(".metal") {
            files.append(directory + "/" + file)
        }
    }
    return files
}

var metalFiles = findMetalFiles(in: kernelDir)
if fm.fileExists(atPath: genKernelDir) {
    metalFiles.append(contentsOf: findMetalFiles(in: genKernelDir))
}

// Check if metallib exists and is newer than all .metal files
if fm.fileExists(atPath: metalLibPath),
   let metalLibDate = getModificationDate(of: metalLibPath) {
    let allSourcesNewer = metalFiles.allSatisfy { metalFile in
        if let metalDate = getModificationDate(of: metalFile) {
            return metalDate < metalLibDate
        }
        return true
    }
    if allSourcesNewer {
        log("Metal shaders up to date, skipping rebuild", level: .info)
        exit(0)
    }
}

try? fm.createDirectory(atPath: tempDir, withIntermediateDirectories: true)
try? fm.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

// Xcode 26 ships the Metal compiler as a downloadable component. Without it
// every kernel fails with the same opaque error, so say it once, up front.
let probe = Process()
probe.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
probe.arguments = ["metal", "--version"]
probe.standardOutput = FileHandle.nullDevice
probe.standardError = FileHandle.nullDevice
try? probe.run()
probe.waitUntilExit()
if probe.terminationStatus != 0 {
    log("""
        Metal toolchain not installed — MLX's kernels cannot be compiled.
        Install it once, then re-run this script:

          xcodebuild -downloadComponent MetalToolchain
        """, level: .error)
    exit(1)
}

log("Compiling Metal kernels...", level: .info)

@discardableResult
func run(_ args: [String]) -> Int32 {
    let task = Process()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
    task.arguments = args
    try? task.run()
    task.waitUntilExit()
    return task.terminationStatus
}

for file in metalFiles {
    let name = (file as NSString).lastPathComponent.replacingOccurrences(of: ".metal", with: "")
    let args = [
        "-sdk", "macosx", "metal",
        "-I", ".build/checkouts/mlx-swift/Source/Cmlx/mlx/",
        "-I", ".build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/",
        "-c", file,
        "-o", "\(tempDir)/\(name).air"
    ]
    log("Compiling \(file)...", level: .debug)
    let status = run(args)
    if status != 0 {
        if skipUnsupported {
            log("Skipping \(name): this SDK cannot compile it", level: .warning)
            continue
        }
        log("Error compiling \(file)", level: .error)
        exit(status)
    }
}

var airFiles: [String] = []
if let contents = try? fm.contentsOfDirectory(atPath: tempDir) {
    airFiles = contents.filter { $0.hasSuffix(".air") }.map { "\(tempDir)/\($0)" }
}

var metallibArgs = ["-sdk", "macosx", "metallib"]
metallibArgs.append(contentsOf: airFiles)
metallibArgs.append(contentsOf: ["-o", "\(outputDir)/default.metallib"])

log("Creating metallib...", level: .debug)
let status = run(metallibArgs)
if status != 0 {
    log("Error creating metallib", level: .error)
    exit(status)
}

// Two placements, because MLX has two ways of finding this file. The copy at
// the repo root is the last resort in its search order and only works when the
// process CWD happens to be the repo. The `mlx.metallib` copies sit next to the
// built binaries, which is the FIRST thing MLX looks for and the only one that
// survives `brew install` or running the binary from anywhere else.
try? fm.removeItem(atPath: "default.metallib")
try? fm.copyItem(atPath: "\(outputDir)/default.metallib", toPath: "default.metallib")

for buildDir in [".build/debug", ".build/release"] where fm.fileExists(atPath: buildDir) {
    let colocated = "\(buildDir)/mlx.metallib"
    try? fm.removeItem(atPath: colocated)
    try? fm.copyItem(atPath: "\(outputDir)/default.metallib", toPath: colocated)
    log("Placed \(colocated)", level: .debug)
}

log("Successfully built \(outputDir)/default.metallib, copied to root and next to the binaries", level: .info)
try? fm.removeItem(atPath: tempDir)
