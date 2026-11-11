#!/usr/bin/env swift
import Foundation

let usage = """
    Usage:
      ./scripts/bundle_app.swift [--out PATH] [--bin PATH] [--identity ID | --no-sign] [--skip-metal] [--dry-run]

    Builds the release products, then assembles Basalt.app:
      Contents/MacOS/Basalt          the SwiftUI app (BasaltMac)
      Contents/Helpers/basalt        the engine the app supervises (the basalt CLI)
      Contents/Resources/mlx.metallib  MLX's kernels; the app passes the path to the engine in BASALT_METALLIB
      Contents/Info.plist

    Options:
      --out PATH       Where to write the app (default: .build/Basalt.app)
      --bin PATH       Use binaries already built in PATH instead of building
      --identity ID    Codesign identity with the hardened runtime (default: - for ad hoc)
      --no-sign        Leave the bundle unsigned
      --skip-metal     Do not run scripts/build_metal.swift (needs mlx.metallib in the bin folder)
      --dry-run        Print the plan and change nothing
      -h, --help       Show this help
    """

struct Options {
    var out = ".build/Basalt.app"
    var bin: String?
    var identity: String? = "-"
    var skipMetal = false
    var dryRun = false
}

func parse(_ arguments: [String]) -> Options {
    var options = Options()
    var iterator = arguments.makeIterator()
    while let argument = iterator.next() {
        switch argument {
        case "--out": options.out = iterator.next() ?? options.out
        case "--bin": options.bin = iterator.next()
        case "--identity": options.identity = iterator.next()
        case "--no-sign": options.identity = nil
        case "--skip-metal": options.skipMetal = true
        case "--dry-run": options.dryRun = true
        case "-h", "--help":
            print(usage)
            exit(0)
        default:
            FileHandle.standardError.write(Data("unknown option \(argument)\n\(usage)\n".utf8))
            exit(2)
        }
    }
    return options
}

@discardableResult
func run(_ arguments: [String], dryRun: Bool, capture: Bool = false) -> String {
    print("+ \(arguments.joined(separator: " "))")
    guard !dryRun else { return "" }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = arguments
    let pipe = Pipe()
    if capture { process.standardOutput = pipe }
    do { try process.run() } catch {
        FileHandle.standardError.write(Data("could not run \(arguments[0]): \(error.localizedDescription)\n".utf8))
        exit(1)
    }
    let output = capture ? String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self) : ""
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        FileHandle.standardError.write(Data("\(arguments.joined(separator: " ")) failed with status \(process.terminationStatus)\n".utf8))
        exit(1)
    }
    return output.trimmingCharacters(in: .whitespacesAndNewlines)
}

func version() -> String {
    let source = (try? String(contentsOfFile: "Sources/BasaltCore/DTOs.swift", encoding: .utf8)) ?? ""
    return source.firstMatch(of: #/static let version = "([^"]+)"/#).map { String($0.1) } ?? "0.0.0"
}

func infoPlist(version: String) -> [String: Any] {
    [
        "CFBundleIdentifier": "app.basalt.mac", "CFBundleName": "Basalt", "CFBundleDisplayName": "Basalt",
        "CFBundleExecutable": "Basalt", "CFBundlePackageType": "APPL", "CFBundleShortVersionString": version,
        "CFBundleVersion": version, "LSMinimumSystemVersion": "27.0", "NSHighResolutionCapable": true,
        "LSApplicationCategoryType": "public.app-category.developer-tools", "NSPrincipalClass": "NSApplication",
    ]
}

func assemble(_ options: Options, bin: String) {
    let fm = FileManager.default
    let contents = URL(fileURLWithPath: options.out).appending(path: "Contents")
    let layout = [
        ("\(bin)/BasaltMac", contents.appending(path: "MacOS/Basalt")),
        ("\(bin)/Basalt", contents.appending(path: "Helpers/basalt")),
        ("\(bin)/mlx.metallib", contents.appending(path: "Resources/mlx.metallib")),
    ]
    for (source, target) in layout { print("  \(source) → \(target.path)") }
    guard !options.dryRun else { return }
    try? fm.removeItem(atPath: options.out)
    do {
        for (source, target) in layout {
            try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.copyItem(atPath: source, toPath: target.path)
        }
        let plist = try PropertyListSerialization.data(fromPropertyList: infoPlist(version: version()), format: .xml, options: 0)
        try plist.write(to: contents.appending(path: "Info.plist"))
    } catch {
        FileHandle.standardError.write(Data("could not assemble \(options.out): \(error.localizedDescription)\n".utf8))
        exit(1)
    }
}

func sign(_ options: Options) {
    guard let identity = options.identity else { return }
    let contents = "\(options.out)/Contents"
    for path in ["\(contents)/Helpers/basalt", options.out] {
        run(["codesign", "--force", "--options", "runtime", "--timestamp=none", "--sign", identity, path], dryRun: options.dryRun)
    }
}

let options = parse(Array(CommandLine.arguments.dropFirst()))
let bin: String
if let given = options.bin {
    bin = given
} else {
    run(["swift", "build", "-c", "release", "--product", "Basalt"], dryRun: options.dryRun)
    run(["swift", "build", "-c", "release", "--product", "BasaltMac"], dryRun: options.dryRun)
    if !options.skipMetal { run(["./scripts/build_metal.swift"], dryRun: options.dryRun) }
    bin = options.dryRun ? ".build/release" : run(["swift", "build", "-c", "release", "--show-bin-path"], dryRun: false, capture: true)
}
assemble(options, bin: bin)
sign(options)
print(options.dryRun ? "dry run: nothing written" : "wrote \(options.out)")
