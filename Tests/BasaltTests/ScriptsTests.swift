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
}
