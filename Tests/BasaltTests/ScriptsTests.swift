import Testing
import Foundation

@Suite("Scripts Tests")
struct ScriptsTests {
    
    private func runScript(_ path: String, args: [String], in directory: URL? = nil) -> Int32 {
        let task = Process()
        let fm = FileManager.default
        let currentDir = fm.currentDirectoryPath
        let scriptPath = URL(fileURLWithPath: "\(currentDir)/\(path)")
        
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = ["swift", scriptPath.path] + args
        if let directory {
            task.currentDirectoryURL = directory
            task.standardOutput = FileHandle.nullDevice
        }
        try? task.run()
        task.waitUntilExit()
        return task.terminationStatus
    }
    
    @Test("build_metal.swift runs with --help")
    func testBuildMetalHelp() {
        let status = runScript("scripts/build_metal.swift", args: ["--help"])
        #expect(status == 0)
    }
    
    @Test("build_metal.swift runs with --dry-run")
    func testBuildMetalDryRun() {
        let status = runScript("scripts/build_metal.swift", args: ["--dry-run", "--verbose"])
        #expect(status == 0)
    }
    
    @Test("cleanup.swift runs with --help")
    func testCleanupHelp() {
        let status = runScript("scripts/cleanup.swift", args: ["--help"])
        #expect(status == 0)
    }
    
    @Test("cleanup.swift runs every mode as a dry run; an unknown mode is an error")
    func testCleanupDryRun() {
        for mode in ["daily", "release", "all"] {
            #expect(runScript("scripts/cleanup.swift", args: ["--mode=\(mode)", "--dry-run", "--verbose"]) == 0, "\(mode)")
        }
        #expect(runScript("scripts/cleanup.swift", args: ["--mode=weekly"]) != 0)
    }

    @Test("cleanup.swift --mode=all --apply removes old untracked files and artifacts, never tracked ones")
    func testCleanupApply() throws {
        let fm = FileManager.default
        let repo = fm.temporaryDirectory.appendingPathComponent("cleanup-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: repo) }
        for dir in ["images", "logs", "reports", "Sources"] {
            try fm.createDirectory(at: repo.appendingPathComponent(dir), withIntermediateDirectories: true)
        }
        func write(_ path: String, age: TimeInterval = 0) throws {
            let url = repo.appendingPathComponent(path)
            try Data("x".utf8).write(to: url)
            try fm.setAttributes([.modificationDate: Date().addingTimeInterval(-age)], ofItemAtPath: url.path)
        }
        let twoDays: TimeInterval = 2 * 86400
        try write("images/readme_shot.png", age: twoDays)   // tracked: a README asset
        try write("images/old_capture.png", age: twoDays)   // untracked, old: rotated
        try write("reports/today.md")                       // untracked, recent: kept
        try write("reports/.gitkeep", age: twoDays)          // kept by name
        try write("Sources/notes.tmp")                      // release artifact
        try write("local.sqlite")                           // root runtime DB
        try write("Sources/main.swift")                     // tracked source
        // Someone else's work: a nested work tree (a worktree has a .git file) and Claude
        // Code's worktree dir. Their untracked files are not this repository's artifacts.
        for dir in ["nested", ".claude/worktrees/w1"] {
            try fm.createDirectory(at: repo.appendingPathComponent(dir), withIntermediateDirectories: true)
        }
        try write("nested/.git")
        try write("nested/work.log")
        try write(".claude/worktrees/w1/draft.tmp")
        try write("notes~draft.md")                         // `~` mid-name is not a backup file
        try fm.createDirectory(at: repo.appendingPathComponent("reports/run1"), withIntermediateDirectories: true)
        try write("reports/run1/today.md")                    // fresh file in an old-looking dir
        try fm.setAttributes([.modificationDate: Date().addingTimeInterval(-twoDays)],
                             ofItemAtPath: repo.appendingPathComponent("reports/run1").path)
        try fm.createDirectory(at: repo.appendingPathComponent(".archive/clone"), withIntermediateDirectories: true)
        try write(".archive/clone/.git")                    // a clone parked in .archive
        try write(".archive/clone/wip.swift")

        let git = Process()
        git.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        git.currentDirectoryURL = repo
        git.arguments = ["-c", "init.defaultBranch=main", "init", "-q"]
        try git.run(); git.waitUntilExit()
        let add = Process()
        add.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        add.currentDirectoryURL = repo
        add.arguments = ["add", "images/readme_shot.png", "Sources/main.swift"]
        try add.run(); add.waitUntilExit()
        #expect(add.terminationStatus == 0)

        // A dry run touches nothing.
        #expect(runScript("scripts/cleanup.swift", args: ["--mode=all"], in: repo) == 0)
        #expect(fm.fileExists(atPath: repo.appendingPathComponent("images/old_capture.png").path))

        #expect(runScript("scripts/cleanup.swift", args: ["--mode=all", "--apply"], in: repo) == 0)
        let exists = { (path: String) in fm.fileExists(atPath: repo.appendingPathComponent(path).path) }
        #expect(exists("images/readme_shot.png"))
        #expect(!exists("images/old_capture.png"))
        #expect(exists("reports/today.md"))
        #expect(exists("reports/.gitkeep"))
        #expect(!exists("Sources/notes.tmp"))
        #expect(!exists("local.sqlite"))
        #expect(exists("Sources/main.swift"))
        #expect(exists("nested/work.log"))
        #expect(exists(".claude/worktrees/w1/draft.tmp"))
        #expect(exists("notes~draft.md"))
        #expect(exists("reports/run1/today.md"), "daily rotation never removes a directory by its own mtime")
        #expect(runScript("scripts/cleanup.swift", args: ["--mode=release", "--include-local", "--apply"], in: repo) == 0)
        #expect(exists(".archive/clone/wip.swift"), ".archive holding a work tree is kept whole")
    }

    @Test("cleanup.swift refuses to run outside the root of a git work tree")
    func testCleanupOutsideRepo() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("cleanup-nogit-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        let file = dir.appendingPathComponent("keep.log")
        try Data("x".utf8).write(to: file)
        // Without git nothing could count as tracked, so --apply would delete freely.
        #expect(runScript("scripts/cleanup.swift", args: ["--mode=all", "--apply"], in: dir) != 0)
        #expect(fm.fileExists(atPath: file.path))

        // A subdirectory of a work tree is refused too: relative paths would miss the root.
        let git = Process()
        git.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        git.currentDirectoryURL = dir
        git.arguments = ["init", "-q"]
        try git.run(); git.waitUntilExit()
        let sub = dir.appendingPathComponent("sub")
        try fm.createDirectory(at: sub, withIntermediateDirectories: true)
        #expect(runScript("scripts/cleanup.swift", args: ["--mode=all", "--apply"], in: sub) != 0)
        #expect(fm.fileExists(atPath: file.path))
    }
    
    @Test("installer.swift runs with --help")
    func testInstallerHelp() {
        let status = runScript("scripts/installer.swift", args: ["--help"])
        #expect(status == 0)
    }
    
    @Test("installer.swift runs with --dry-run and --non-interactive")
    func testInstallerDryRunNonInteractive() {
        let status = runScript("scripts/installer.swift", args: ["--dry-run", "--non-interactive", "--verbose"])
        #expect(status == 0)
    }

    @Test("build_metal.swift places mlx.metallib into swift build --show-bin-path directories instead of .build/debug")
    func testBuildMetalUsesShowBinPath() throws {
        let fm = FileManager.default
        let repo = fm.temporaryDirectory.appendingPathComponent("build-metal-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: repo) }
        let kernelDir = repo.appendingPathComponent(".build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels")
        let customDebug = repo.appendingPathComponent(".build/out/Products/Debug")
        let customRelease = repo.appendingPathComponent(".build/out/Products/Release")
        let nativeDebug = repo.appendingPathComponent(".build/arm64-apple-macosx/debug")
        let nativeRelease = repo.appendingPathComponent(".build/arm64-apple-macosx/release")
        let binDir = repo.appendingPathComponent("bin")
        for dir in [kernelDir, customRelease, nativeDebug, binDir] {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        let kernel = kernelDir.appendingPathComponent("noop.metal")
        try Data("#include <metal_stdlib>\nusing namespace metal;\nkernel void noop() {}\n".utf8).write(to: kernel)
        try fm.setAttributes([.modificationDate: Date().addingTimeInterval(-120)], ofItemAtPath: kernel.path)
        try Data("metallib".utf8).write(to: repo.appendingPathComponent("default.metallib"))

        let swiftShim = binDir.appendingPathComponent("swift")
        let shimScript = """
        #!/bin/bash
        if [ "$1" = "build" ] && [[ " $* " == *" --show-bin-path "* ]]; then
          if [[ " $* " == *" --build-system native "* ]]; then
            if [[ " $* " == *" -c debug "* ]]; then
              echo "\(nativeDebug.path)"
            else
              echo "\(nativeRelease.path)"
            fi
          elif [[ " $* " == *" -c debug "* ]]; then
            echo "\(customDebug.path)"
          else
            echo "\(customRelease.path)"
          fi
          exit 0
        fi
        exec /usr/bin/swift "$@"
        """
        try Data(shimScript.utf8).write(to: swiftShim)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: swiftShim.path)

        let scriptURL = URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("scripts/build_metal.swift")
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/swift")
        task.arguments = [scriptURL.path, "--skip-unsupported"]
        task.currentDirectoryURL = repo
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "\(binDir.path):\(env["PATH"] ?? "/usr/bin:/bin")"
        task.environment = env
        task.standardOutput = FileHandle.nullDevice
        try task.run()
        task.waitUntilExit()

        #expect(task.terminationStatus == 0)
        #expect(fm.fileExists(atPath: customDebug.appendingPathComponent("mlx.metallib").path))
        #expect(fm.fileExists(atPath: customRelease.appendingPathComponent("mlx.metallib").path))
        #expect(fm.fileExists(atPath: nativeDebug.appendingPathComponent("mlx.metallib").path))
        #expect(!fm.fileExists(atPath: nativeRelease.appendingPathComponent("mlx.metallib").path))
        #expect(!fm.fileExists(atPath: repo.appendingPathComponent(".build/debug/mlx.metallib").path))
    }

    @Test("xcode-toolchain action resolves symlinks before filtering release candidates and enforces Swift 6.4+")
    func testXcodeToolchainSelectionSkipsSymlinkedRC() throws {
        let fm = FileManager.default
        let actionPath = URL(fileURLWithPath: fm.currentDirectoryPath)
            .appendingPathComponent(".github/actions/xcode-toolchain/action.yml")
        let yaml = try String(contentsOf: actionPath, encoding: .utf8)
        let lines = yaml.components(separatedBy: "\n")
        let runIdx = try #require(lines.firstIndex(where: { $0.trimmingCharacters(in: .whitespaces) == "run: |" }))
        let script = lines[(runIdx + 1)...]
            .map { $0.hasPrefix("        ") ? String($0.dropFirst(8)) : $0 }
            .joined(separator: "\n")

        let temp = fm.temporaryDirectory.resolvingSymlinksInPath()
            .appendingPathComponent("xcode-action-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: temp) }
        let apps = temp.appendingPathComponent("Applications")
        let rcOnlyApps = temp.appendingPathComponent("RCOnlyApplications")
        let olderDev = apps.appendingPathComponent("Xcode_26.6.app/Contents/Developer")
        let gaApp = apps.appendingPathComponent("Xcode.app")
        let gaDev = gaApp.appendingPathComponent("Contents/Developer")
        let rcApp = apps.appendingPathComponent("Xcode_27.2_Release_Candidate.app")
        let rcDev = rcApp.appendingPathComponent("Contents/Developer")
        let rcOnlyDev = rcOnlyApps.appendingPathComponent("Xcode_27.2_Release_Candidate.app/Contents/Developer")
        let binDir = temp.appendingPathComponent("bin")
        for dir in [olderDev, gaDev, rcDev, rcOnlyDev, binDir] {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        func writePlist(app: URL, version: String) throws {
            let plist: [String: Any] = ["CFBundleShortVersionString": version]
            let data = try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
            try data.write(to: app.appendingPathComponent("Contents/Info.plist"))
        }
        try writePlist(app: apps.appendingPathComponent("Xcode_26.6.app"), version: "26.6")
        try writePlist(app: gaApp, version: "27.1")
        try writePlist(app: rcApp, version: "27.2")
        try writePlist(app: rcOnlyApps.appendingPathComponent("Xcode_27.2_Release_Candidate.app"), version: "27.2")
        try fm.createSymbolicLink(
            at: apps.appendingPathComponent("Xcode_27.2.app"),
            withDestinationURL: rcApp
        )
        let envFile = temp.appendingPathComponent("github_env")
        try Data().write(to: envFile)

        let swiftShim = binDir.appendingPathComponent("swift")
        try Data("#!/bin/bash\necho \"${FAKE_SWIFT_OUT:-Apple Swift version ${FAKE_SWIFT_VER:-6.4.2} (swiftlang-6.4.2)}\"\n".utf8).write(to: swiftShim)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: swiftShim.path)

        func runAction(appsDir: URL = apps, swiftVersion: String, swiftOut: String? = nil) throws -> Int32 {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/bin/bash")
            task.arguments = ["-euo", "pipefail", "-c", script]
            var env = ProcessInfo.processInfo.environment
            env["XCODE_APPS_DIR"] = appsDir.path
            env["GITHUB_ENV"] = envFile.path
            env["FAKE_SWIFT_VER"] = swiftVersion
            if let swiftOut { env["FAKE_SWIFT_OUT"] = swiftOut }
            env["PATH"] = "\(binDir.path):/usr/bin:/bin"
            task.environment = env
            task.standardOutput = FileHandle.nullDevice
            task.standardError = FileHandle.nullDevice
            try task.run()
            task.waitUntilExit()
            return task.terminationStatus
        }

        #expect(try runAction(swiftVersion: "6.4.2") == 0)
        let writtenEnv = try String(contentsOf: envFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(writtenEnv.hasPrefix("DEVELOPER_DIR="))
        let selectedDev = URL(fileURLWithPath: String(writtenEnv.dropFirst("DEVELOPER_DIR=".count)))
            .resolvingSymlinksInPath().path
        #expect(selectedDev == gaDev.resolvingSymlinksInPath().path)
        try Data().write(to: envFile)
        #expect(try runAction(appsDir: rcOnlyApps, swiftVersion: "6.4.2") == 0)
        #expect(try String(contentsOf: envFile, encoding: .utf8).isEmpty)
        #expect(try runAction(swiftVersion: "6.3.3") != 0)
        #expect(try runAction(swiftVersion: "6.4.2", swiftOut: "xcrun: error: unable to find utility") != 0)
    }

    @Test("build_metal.swift rebuilds when a header is newer than metallib or when Package.resolved revision drifts")
    func testBuildMetalFreshnessWithHeadersAndRevision() throws {
        let fm = FileManager.default
        let repo = fm.temporaryDirectory.appendingPathComponent("build-metal-fresh-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: repo) }
        let kernelDir = repo.appendingPathComponent(".build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels")
        let debugDir = repo.appendingPathComponent(".build/out/Products/Debug")
        let binDir = repo.appendingPathComponent("bin")
        for dir in [kernelDir, debugDir, binDir] {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        let metalFile = kernelDir.appendingPathComponent("noop.metal")
        let headerFile = kernelDir.appendingPathComponent("types.h")
        try Data("#include <metal_stdlib>\nusing namespace metal;\nkernel void noop() {}\n".utf8).write(to: metalFile)
        try Data("#pragma once\n".utf8).write(to: headerFile)
        try fm.setAttributes([.modificationDate: Date().addingTimeInterval(-200)], ofItemAtPath: metalFile.path)
        try fm.setAttributes([.modificationDate: Date().addingTimeInterval(-200)], ofItemAtPath: headerFile.path)

        func writePackageResolved(revision: String) throws {
            let json = """
            {
              "pins": [
                {
                  "identity": "mlx-swift",
                  "state": {
                    "revision": "\(revision)",
                    "version": "0.32.3"
                  }
                }
              ]
            }
            """
            try Data(json.utf8).write(to: repo.appendingPathComponent("Package.resolved"))
        }

        try writePackageResolved(revision: "rev-alpha")
        try Data("metallib-content".utf8).write(to: repo.appendingPathComponent("default.metallib"))

        let swiftShim = binDir.appendingPathComponent("swift")
        let shimScript = """
        #!/bin/bash
        if [ "$1" = "build" ] && [[ " $* " == *" --show-bin-path "* ]]; then
          echo "\(debugDir.path)"
          exit 0
        fi
        exec /usr/bin/swift "$@"
        """
        try Data(shimScript.utf8).write(to: swiftShim)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: swiftShim.path)

        let scriptURL = URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("scripts/build_metal.swift")
        func runBuildMetal() -> (status: Int32, output: String) {
            let task = Process()
            let pipe = Pipe()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/swift")
            task.arguments = [scriptURL.path, "--skip-unsupported", "--verbose"]
            task.currentDirectoryURL = repo
            var env = ProcessInfo.processInfo.environment
            env["PATH"] = "\(binDir.path):\(env["PATH"] ?? "/usr/bin:/bin")"
            task.environment = env
            task.standardOutput = pipe
            task.standardError = pipe
            try? task.run()
            task.waitUntilExit()
            let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            return (task.terminationStatus, out)
        }

        let firstRun = runBuildMetal()
        #expect(firstRun.status == 0)
        let stampFile = debugDir.appendingPathComponent(".mlx_revision")
        #expect(fm.fileExists(atPath: stampFile.path))
        #expect((try? String(contentsOf: stampFile, encoding: .utf8)) == "rev-alpha")

        let secondRun = runBuildMetal()
        #expect(secondRun.status == 0)
        #expect(secondRun.output.contains("Metal shaders up to date, skipping rebuild"))

        try fm.setAttributes([.modificationDate: Date().addingTimeInterval(50)], ofItemAtPath: headerFile.path)
        let headerTouchedRun = runBuildMetal()
        #expect(headerTouchedRun.status == 0)
        #expect(!headerTouchedRun.output.contains("Metal shaders up to date, skipping rebuild"))

        try writePackageResolved(revision: "rev-beta")
        let revisionChangedRun = runBuildMetal()
        #expect(revisionChangedRun.status == 0)
        #expect(!revisionChangedRun.output.contains("Metal shaders up to date, skipping rebuild"))
        #expect((try? String(contentsOf: stampFile, encoding: .utf8)) == "rev-beta")
    }

    @Test("All third-party GitHub Actions across workflows are pinned to 40-character SHAs")
    func testAllGitHubActionsPinnedToSHAs() throws {
        let fm = FileManager.default
        let root = URL(fileURLWithPath: fm.currentDirectoryPath)
        let workflows = try fm.contentsOfDirectory(at: root.appendingPathComponent(".github/workflows"), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "yml" || $0.pathExtension == "yaml" }
        let compositeActions = [
            root.appendingPathComponent(".github/actions/mlx-metallib/action.yml"),
            root.appendingPathComponent(".github/actions/xcode-toolchain/action.yml")
        ]

        let hex40Pattern = try Regex("^[0-9a-f]{40}$")
        for file in workflows + compositeActions {
            guard fm.fileExists(atPath: file.path) else { continue }
            let content = try String(contentsOf: file, encoding: .utf8)
            let lines = content.components(separatedBy: "\n")
            for (idx, line) in lines.enumerated() {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard trimmed.hasPrefix("uses:") || trimmed.hasPrefix("- uses:") else { continue }
                let afterUses = trimmed.components(separatedBy: "uses:").last?.trimmingCharacters(in: .whitespaces) ?? ""
                let actionPart = afterUses.components(separatedBy: "#").first?.trimmingCharacters(in: .whitespaces) ?? ""
                if actionPart.hasPrefix("./") { continue }
                let parts = actionPart.components(separatedBy: "@")
                #expect(parts.count == 2, "Unversioned action on line \(idx + 1) in \(file.lastPathComponent): \(trimmed)")
                let sha = parts[1]
                #expect(sha.contains(hex40Pattern), "Action not pinned to 40-char SHA on line \(idx + 1) in \(file.lastPathComponent): \(sha)")
            }
        }
    }
}
