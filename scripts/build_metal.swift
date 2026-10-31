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

func queryBinPath(configuration: String, buildSystem: String? = nil) -> (path: String, stderr: String, status: Int32) {
    let task = Process()
    let outPipe = Pipe()
    let errPipe = Pipe()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    var command = ["swift", "build", "-c", configuration, "--disable-sandbox", "--show-bin-path"]
    if let buildSystem {
        command += ["--build-system", buildSystem]
    }
    task.arguments = command
    task.standardOutput = outPipe
    task.standardError = errPipe
    try? task.run()
    let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
    let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
    task.waitUntilExit()
    let path = String(data: outData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    let stderrText = String(data: errData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    return (path, stderrText, task.terminationStatus)
}

func binPath(configuration: String) -> String {
    let result = queryBinPath(configuration: configuration)
    guard result.status == 0, !result.path.isEmpty else {
        log("Could not determine \(configuration) bin path via swift build --show-bin-path: \(result.stderr)", level: .error)
        exit(1)
    }
    return result.path
}

let kernelDir = ".build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels"
let genKernelDir = ".build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"
let debugBinDir = binPath(configuration: "debug")
let releaseBinDir = binPath(configuration: "release")
let nativeBinDirs = ["debug", "release"].compactMap { cfg -> String? in
    let res = queryBinPath(configuration: cfg, buildSystem: "native")
    return res.status == 0 && !res.path.isEmpty ? res.path : nil
}
let colocatedDirs = Array(Set([debugBinDir, releaseBinDir] + nativeBinDirs)).sorted()
let outputDir = debugBinDir
let tempDir = ".build/metal_temp"
let metalLibPath = "\(outputDir)/default.metallib"

let fm = FileManager.default

func atomicCopy(_ src: String, to dst: String) {
    let staging = dst + ".partial"
    try? fm.removeItem(atPath: staging)
    guard (try? fm.copyItem(atPath: src, toPath: staging)) != nil, rename(staging, dst) == 0 else {
        try? fm.removeItem(atPath: staging)
        log("Could not place \(dst): \(String(cString: strerror(errno)))", level: .error)
        exit(1)
    }
}

func placeColocated(from source: String) {
    let sourceDate = getModificationDate(of: source)
    for buildDir in colocatedDirs where fm.fileExists(atPath: buildDir) {
        let colocated = "\(buildDir)/mlx.metallib"
        if let have = getModificationDate(of: colocated), let want = sourceDate, have >= want { continue }
        atomicCopy(source, to: colocated)
        log("Placed \(colocated)", level: .debug)
    }
}

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

func findHeaderFiles(in directory: String) -> [String] {
    var files: [String] = []
    guard let enumerator = fm.enumerator(atPath: directory) else { return files }
    for case let file as String in enumerator {
        if file.hasSuffix(".h") {
            files.append(directory + "/" + file)
        }
    }
    return files
}

func readMlxRevision() -> String? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: "Package.resolved")),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let pins = json["pins"] as? [[String: Any]] else {
        return nil
    }
    for pin in pins {
        if let identity = pin["identity"] as? String, identity == "mlx-swift",
           let state = pin["state"] as? [String: Any],
           let revision = state["revision"] as? String {
            return revision
        }
    }
    return nil
}

let revisionStampPath = "\(outputDir)/.mlx_revision"
var metalFiles = findMetalFiles(in: kernelDir)
var headerFiles = findHeaderFiles(in: kernelDir)
if fm.fileExists(atPath: genKernelDir) {
    metalFiles.append(contentsOf: findMetalFiles(in: genKernelDir))
    headerFiles.append(contentsOf: findHeaderFiles(in: genKernelDir))
}
let currentRevision = readMlxRevision()

if !fm.fileExists(atPath: metalLibPath) && fm.fileExists(atPath: "default.metallib") {
    try? fm.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
    atomicCopy("default.metallib", to: metalLibPath)
}

// Check if metallib exists and is newer than all .metal files
if fm.fileExists(atPath: metalLibPath),
   let metalLibDate = getModificationDate(of: metalLibPath) {
    let cachedRevision = try? String(contentsOfFile: revisionStampPath, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
    let revisionMatches = currentRevision == nil || cachedRevision == currentRevision
    let allSourcesNewer = (metalFiles + headerFiles).allSatisfy { file in
        if let fileDate = getModificationDate(of: file) {
            return fileDate < metalLibDate
        }
        return true
    }
    if allSourcesNewer && revisionMatches && !metalFiles.isEmpty {
        // The library is current, but the colocated copies may not be (a run cancelled
        // mid-placement, a wiped build dir): restore them before declaring success.
        placeColocated(from: metalLibPath)
        log("Metal shaders up to date, skipping rebuild", level: .info)
        exit(0)
    }
}

if metalFiles.isEmpty {
    // Nothing to compile means mlx-swift was never checked out (run `swift build` first), not that
    // the work is done: a green step here only moves the failure to MLX aborting the test run.
    log("No .metal files found to compile — run `swift build` first", level: .error)
    exit(1)
}

// Fresh every run: a stale .air from an earlier SDK or mlx revision (kept when its kernel is now
// skipped by --skip-unsupported, or left by a failed run) must not be linked into the library.
try? fm.removeItem(atPath: tempDir)
try? fm.createDirectory(atPath: tempDir, withIntermediateDirectories: true)
try? fm.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

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

if airFiles.isEmpty {
    log("No .air files produced — every kernel failed to compile", level: .error)
    exit(1)
}

var metallibArgs = ["-sdk", "macosx", "metallib"]
metallibArgs.append(contentsOf: airFiles)
// Linked into the temp dir and moved into place at the end: an interrupted run (a cancelled CI
// job) must never leave a truncated metallib with a fresh mtime that the check above trusts.
let stagedLib = "\(tempDir)/default.metallib"
metallibArgs.append(contentsOf: ["-o", stagedLib])

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
// The copies go first and metalLibPath — whose mtime is the up-to-date marker — last: a run
// cancelled in between leaves the marker stale, so the next run redoes every placement.
atomicCopy(stagedLib, to: "default.metallib")

placeColocated(from: stagedLib)

// Same filesystem as stagedLib, so a plain rename(2).
guard rename(stagedLib, metalLibPath) == 0 else {
    log("Could not place \(metalLibPath): \(String(cString: strerror(errno)))", level: .error)
    exit(1)
}

if let currentRevision {
    try? currentRevision.write(toFile: revisionStampPath, atomically: true, encoding: .utf8)
}

log("Successfully built \(outputDir)/default.metallib, copied to root and next to the binaries", level: .info)
try? fm.removeItem(atPath: tempDir)
