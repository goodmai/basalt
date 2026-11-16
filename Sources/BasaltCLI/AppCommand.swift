import ArgumentParser
import BasaltCore
import Foundation

struct AppCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "app",
        abstract: "Launch Basalt.app or a sibling BasaltMac build"
    )

    enum LaunchTarget: Sendable, Equatable {
        case appBundle(URL)
        case binary(executable: URL, helper: URL?)
    }

    @Option(name: .long, help: "Hugging Face repo id to select on launch.")
    var model: String?

    @Option(name: .long, help: "Compose file path to select on launch.")
    var compose: String?

    @Flag(name: .long, help: "Wait until the app process exits.")
    var wait = false

    mutating func run() async throws {
        let fm = FileManager.default
        let executablePath = Bundle.main.executableURL?.path ?? CommandLine.arguments.first
        guard let target = Self.resolveTarget(
            executablePath: executablePath,
            currentDirectory: fm.currentDirectoryPath,
            environment: ProcessInfo.processInfo.environment,
            isDirectory: { path in
                var dir: ObjCBool = false
                return fm.fileExists(atPath: path, isDirectory: &dir) && dir.boolValue
            },
            isExecutable: { fm.isExecutableFile(atPath: $0) }
        ) else {
            throw ValidationError("Basalt.app was not found. Build it with ./scripts/bundle_app.swift or swift build --product BasaltMac.")
        }
        let process = Self.makeProcess(target: target, model: model, compose: compose, wait: wait)
        try process.run()
        if wait || Self.isDirectBinary(target) {
            process.waitUntilExit()
            if process.terminationStatus != 0 {
                throw ExitCode(process.terminationStatus)
            }
        }
    }

    private static func isDirectBinary(_ target: LaunchTarget) -> Bool {
        if case .binary = target { return true }
        return false
    }

    static func resolveTarget(
        executablePath: String?,
        currentDirectory: String,
        environment: [String: String],
        isDirectory: (String) -> Bool,
        isExecutable: (String) -> Bool
    ) -> LaunchTarget? {
        if let override = environment["BASALT_APP"], !override.isEmpty {
            let url = URL(fileURLWithPath: override)
            if url.pathExtension == "app", isDirectory(url.path) { return .appBundle(url) }
            if isExecutable(url.path) { return .binary(executable: url, helper: executablePath.map(URL.init(fileURLWithPath:))) }
        }
        if let executablePath {
            let execURL = URL(fileURLWithPath: executablePath).standardizedFileURL
            let helperParent = execURL.deletingLastPathComponent()
            let enclosedApp = helperParent.deletingLastPathComponent().deletingLastPathComponent()
            if enclosedApp.pathExtension == "app", isDirectory(enclosedApp.path) {
                return .appBundle(URL(fileURLWithPath: enclosedApp.path))
            }
            let siblingBundle = helperParent.deletingLastPathComponent().appending(path: "Basalt.app")
            if isDirectory(siblingBundle.path) { return .appBundle(URL(fileURLWithPath: siblingBundle.path)) }
            let siblingMac = helperParent.appending(path: "BasaltMac")
            if isExecutable(siblingMac.path) {
                return .binary(
                    executable: URL(fileURLWithPath: siblingMac.path),
                    helper: isExecutable(execURL.path) ? URL(fileURLWithPath: execURL.path) : nil
                )
            }
        }
        let cwdBundle = URL(fileURLWithPath: currentDirectory).appending(path: ".build/Basalt.app")
        if isDirectory(cwdBundle.path) { return .appBundle(cwdBundle) }
        let homeApps = FileManager.default.homeDirectoryForCurrentUser.appending(path: "Applications/Basalt.app")
        for installed in [URL(fileURLWithPath: "/Applications/Basalt.app"), homeApps] where isDirectory(installed.path) {
            return .appBundle(installed)
        }
        return nil
    }

    static func makeProcess(target: LaunchTarget, model: String?, compose: String?, wait: Bool) -> Process {
        let process = Process()
        var extraArgs: [String] = []
        if let model, !model.isEmpty { extraArgs += ["-model", model] }
        if let compose { extraArgs += ["-compose", compose] }
        switch target {
        case .appBundle(let bundle):
            process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
            var openArgs = ["-a", bundle.path]
            if wait { openArgs.append("-W") }
            if !extraArgs.isEmpty { openArgs += ["--args"] + extraArgs }
            process.arguments = openArgs
        case .binary(let executable, let helper):
            process.executableURL = executable
            process.arguments = extraArgs
            if let helper {
                var env = ProcessInfo.processInfo.environment
                if env["BASALT_HELPER"] == nil { env["BASALT_HELPER"] = helper.path }
                process.environment = env
            }
        }
        return process
    }
}
