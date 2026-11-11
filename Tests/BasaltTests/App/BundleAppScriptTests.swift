import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("scripts/bundle_app.swift: the plan, and the Basalt.app layout from a given bin folder (#434 phase 1)")
struct BundleAppScriptTests {
    static func run(_ path: String, args: [String]) -> Int32 {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = ["swift", URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appending(path: path).path] + args
        task.standardOutput = FileHandle.nullDevice
        try? task.run()
        task.waitUntilExit()
        return task.terminationStatus
    }

    @Test("bundle_app.swift prints its plan with --dry-run, and assembles the app layout and Info.plist from a given bin folder")
    func bundleApp() throws {
        #expect(Self.run("scripts/bundle_app.swift", args: ["--dry-run"]) == 0)
        #expect(Self.run("scripts/bundle_app.swift", args: ["--bogus"]) != 0)
        let fm = FileManager.default
        let folder = fm.temporaryDirectory.appending(path: "bundle-\(UUID().uuidString)")
        let bin = folder.appending(path: "bin")
        try fm.createDirectory(at: bin, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: folder) }
        for name in ["BasaltMac", "Basalt", "mlx.metallib"] { try Data(name.utf8).write(to: bin.appending(path: name)) }
        let app = folder.appending(path: "Basalt.app")
        #expect(Self.run("scripts/bundle_app.swift", args: ["--bin", bin.path, "--out", app.path, "--no-sign"]) == 0)
        #expect(try String(contentsOf: app.appending(path: "Contents/MacOS/Basalt"), encoding: .utf8) == "BasaltMac")
        #expect(try String(contentsOf: app.appending(path: "Contents/Helpers/basalt"), encoding: .utf8) == "Basalt")
        #expect(fm.fileExists(atPath: app.appending(path: "Contents/Resources/mlx.metallib").path))
        let plist = try #require(NSDictionary(contentsOf: app.appending(path: "Contents/Info.plist")) as? [String: Any])
        #expect(plist["CFBundleExecutable"] as? String == "Basalt" && plist["CFBundleShortVersionString"] as? String == HealthResponse.version)
    }
}

@Suite("BASALT_METALLIB: the engine takes MLX's kernel library path from the app (#434 phase 1)")
struct MetallibOverrideTests {
    @Test("A missing or empty path changes nothing")
    func ignoresMissing() {
        #expect(MetallibOverride.apply([:]) == nil)
        #expect(MetallibOverride.apply(["BASALT_METALLIB": ""]) == nil)
        #expect(MetallibOverride.apply(["BASALT_METALLIB": "/nonexistent/mlx.metallib"]) == nil)
    }
}
