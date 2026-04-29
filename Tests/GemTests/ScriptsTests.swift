import Testing
import Foundation

@Suite("Scripts Tests")
struct ScriptsTests {
    
    private func runScript(_ path: String, args: [String]) -> Int32 {
        let task = Process()
        let fm = FileManager.default
        let currentDir = fm.currentDirectoryPath
        let scriptPath = URL(fileURLWithPath: "\(currentDir)/\(path)")
        
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = ["swift", scriptPath.path] + args
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
    
    @Test("clean_for_github.swift runs with --help")
    func testCleanForGithubHelp() {
        let status = runScript("scripts/clean_for_github.swift", args: ["--help"])
        #expect(status == 0)
    }
    
    @Test("clean_for_github.swift runs with --dry-run")
    func testCleanForGithubDryRun() {
        let status = runScript("scripts/clean_for_github.swift", args: ["--dry-run", "--verbose"])
        #expect(status == 0)
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
