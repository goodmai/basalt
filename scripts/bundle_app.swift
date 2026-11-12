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
        "NSServices": [[
            "NSMenuItem": ["default": "Ask Basalt"], "NSMessage": "askBasalt", "NSPortName": "Basalt",
            "NSSendTypes": ["NSStringPboardType", "public.utf8-plain-text"], "NSRequiredContext": [String: String](),
        ]],
        "CFBundleDocumentTypes": [[
            "CFBundleTypeName": "Attachment", "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate",
            "LSItemContentTypes": ["com.adobe.pdf", "public.image", "public.audio"],
        ]],
    ]
}

func intentsMetadata(bin: String, resources: URL, dryRun: Bool) {
    let objects = URL(fileURLWithPath: bin).deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "Intermediates.noindex/Basalt.build/\(URL(fileURLWithPath: bin).lastPathComponent)/BasaltMac-p.build/Objects-normal/arm64")
    let fm = FileManager.default
    let constants = ((try? fm.contentsOfDirectory(atPath: objects.path)) ?? []).filter { $0.hasSuffix(".swiftconstvalues") }.map { objects.appending(path: $0).path }
    let sources = ((try? fm.contentsOfDirectory(atPath: "Sources/BasaltMac")) ?? []).filter { $0.hasSuffix(".swift") }.map { fm.currentDirectoryPath + "/Sources/BasaltMac/" + $0 }
    print("  App Intents metadata → \(resources.path)/Metadata.appintents")
    guard !dryRun else { return }
    guard !constants.isEmpty else { return print("  no Swift constant values in \(objects.path); Shortcuts will not list the intents") }
    let lists = fm.temporaryDirectory.appending(path: "basalt-intents-\(UUID().uuidString)")
    try? fm.createDirectory(at: lists, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: lists) }
    try? sources.joined(separator: "\n").write(to: lists.appending(path: "sources"), atomically: true, encoding: .utf8)
    try? constants.joined(separator: "\n").write(to: lists.appending(path: "constants"), atomically: true, encoding: .utf8)
    let swift = run(["xcrun", "--find", "swift"], dryRun: false, capture: true)
    let toolchain = URL(fileURLWithPath: swift).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().path
    let xcode = run(["xcodebuild", "-version"], dryRun: false, capture: true).split(separator: " ").last.map(String.init) ?? "27A1"
    run(["xcrun", "appintentsmetadataprocessor", "--output", resources.path, "--toolchain-dir", toolchain, "--module-name", "BasaltMac",
         "--sdk-root", run(["xcrun", "--show-sdk-path"], dryRun: false, capture: true), "--xcode-version", xcode, "--platform-family", "macOS",
         "--deployment-target", "27.0", "--target-triple", "arm64-apple-macos27.0", "--source-file-list", lists.appending(path: "sources").path,
         "--swift-const-vals-list", lists.appending(path: "constants").path, "--binary-file", "\(bin)/BasaltMac"], dryRun: false)
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
    if options.dryRun { intentsMetadata(bin: bin, resources: contents.appending(path: "Resources"), dryRun: true) }
    guard !options.dryRun else { return }
    try? fm.removeItem(atPath: options.out)
    do {
        for (source, target) in layout {
            try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.copyItem(atPath: source, toPath: target.path)
        }
        let plist = try PropertyListSerialization.data(fromPropertyList: infoPlist(version: version()), format: .xml, options: 0)
        try plist.write(to: contents.appending(path: "Info.plist"))
        intentsMetadata(bin: bin, resources: contents.appending(path: "Resources"), dryRun: false)
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
