import ArgumentParser
import BasaltCore
import BasaltInference
import Darwin
import Foundation

extension Benchmark {
    func runMetalKernelBench() throws {
        let (profile, buildConfig, baseline) = try validateAndLoadBaseline()
        let lockPath = ProcessInfo.processInfo.environment["BASALT_GPU_LOCK"] ?? "/private/tmp/basalt-gpu.lock"
        let pid = getpid()
        guard MetalBenchEnv.claimGPULock(lockPath, pid: pid) else {
            throw ValidationError("GPU lock \(lockPath) is held by another process")
        }
        defer { MetalBenchEnv.releaseGPULock(lockPath, pid: pid) }

        let repeats = profile == .production ? max(7, iterations) : max(3, iterations)
        let metadata = MetalBenchEnv.collectMetadata(profile: profile, iterations: repeats, buildConfig: buildConfig)
        let report = MetalKernelBenchRunner.runAll(profile: profile, iterations: repeats, metadata: metadata)
        print(MetalKernelBenchMath.renderMarkdown(report))

        if let path = metalJson {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: URL(fileURLWithPath: path), options: .atomic)
            print("\nWrote Metal kernel benchmark JSON to \(path)")
        }
        if let baseline {
            let cmp = MetalKernelBenchMath.compare(
                baseline: baseline,
                candidate: report,
                maxRegressionPct: maxRegressionPct,
                allowDebug: allowDebug,
                allowDirty: allowDirty
            )
            print("\n" + MetalKernelBenchMath.renderComparisonMarkdown(cmp))
            if cmp.hasFailure { throw ExitCode(1) }
        }
        if !report.allParityPassed { throw ExitCode(2) }
    }

    private func validateAndLoadBaseline() throws -> (MetalKernelBenchProfile, String, MetalKernelBenchReport?) {
        guard let profile = MetalKernelBenchProfile(rawValue: metalProfile) else {
            throw ValidationError("--metal-profile must be 'production' or 'smoke', got '\(metalProfile)'")
        }
        guard maxRegressionPct.isFinite, maxRegressionPct >= 0 else {
            throw ValidationError("--max-regression-pct must be finite and >= 0, got \(maxRegressionPct)")
        }
        if let jsonPath = metalJson, let basePath = compare {
            let lhs = URL(fileURLWithPath: jsonPath).standardizedFileURL.path
            let rhs = URL(fileURLWithPath: basePath).standardizedFileURL.path
            guard lhs != rhs else {
                throw ValidationError("--metal-json and --compare cannot point to the same file (\(lhs))")
            }
        }
        let buildConfig = MetalBenchEnv.currentBuildConfig
        if compare != nil, !allowDebug, buildConfig != "release" {
            throw ValidationError("--compare requires a release build (pass --allow-debug to override)")
        }
        let baseline: MetalKernelBenchReport? = try compare.map { path in
            try MetalKernelBenchMath.decodeReport(from: Data(contentsOf: URL(fileURLWithPath: path)))
        }
        if !allowBusyCI, MetalBenchEnv.isCIRunnerBusy() {
            throw ValidationError("CI Runner.Worker is active; wait until CI is idle or pass --allow-busy-ci")
        }
        return (profile, buildConfig, baseline)
    }
}

private enum MetalBenchEnv {
    static var currentBuildConfig: String {
        #if DEBUG
        return "debug"
        #else
        return "release"
        #endif
    }

    static func collectMetadata(
        profile: MetalKernelBenchProfile,
        iterations: Int,
        buildConfig: String
    ) -> MetalKernelBenchMetadata {
        var load: Double = 0
        _ = getloadavg(&load, 1)
        let commit = commandOutput("/usr/bin/git", ["rev-parse", "--short=12", "HEAD"]) ?? "unknown"
        let dirty = isWorkingTreeDirty()
        let memoryGB = Int(ProcessInfo.processInfo.physicalMemory / (1 << 30))
        return MetalKernelBenchMetadata(
            commit: commit,
            dirty: dirty,
            buildConfig: buildConfig,
            host: MetalKernelBenchHost(
                chip: SystemProfiler.detectChipModel(),
                osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
                memoryGB: memoryGB,
                loadAvg1m: load
            ),
            runConfig: MetalKernelBenchRunConfig(
                profile: profile.rawValue,
                iterations: iterations,
                callsPerSample: profile == .production ? 16 : 2
            )
        )
    }

    static func commandOutput(_ executable: String, _ args: [String]) -> String? {
        let proc = Process()
        let pipe = Pipe()
        proc.executableURL = URL(fileURLWithPath: executable)
        proc.arguments = args
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            proc.waitUntilExit()
            guard proc.terminationStatus == 0 else { return nil }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
            return text?.isEmpty == false ? text : nil
        } catch {
            return nil
        }
    }

    static func isWorkingTreeDirty() -> Bool {
        let proc = Process()
        let pipe = Pipe()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        proc.arguments = ["status", "--porcelain"]
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            proc.waitUntilExit()
            guard proc.terminationStatus == 0 else { return true }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return !text.isEmpty
        } catch {
            return true
        }
    }

    static func isCIRunnerBusy() -> Bool {
        commandOutput("/usr/bin/pgrep", ["-f", "[R]unner\\.Worker"]) != nil
    }

    static func claimGPULock(_ path: String, pid: Int32, reclaiming: Bool = true) -> Bool {
        if mkdir(path, 0o755) == 0 {
            do {
                try String(pid).write(toFile: path + "/owner", atomically: true, encoding: .utf8)
                return true
            } catch {
                rmdir(path)
                return false
            }
        }
        guard errno == EEXIST, reclaiming else { return false }
        let guardFD = Darwin.open(path + ".reclaim", O_RDONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o644)
        guard guardFD >= 0 else { return false }
        defer { _ = Darwin.close(guardFD) }
        guard flock(guardFD, LOCK_EX | LOCK_NB) == 0 else { return false }
        if let ownerStr = try? String(contentsOfFile: path + "/owner", encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           let ownerPID = Int32(ownerStr), ownerPID > 0, kill(ownerPID, 0) != 0, errno == ESRCH {
            unlink(path + "/owner")
            if rmdir(path) == 0 || errno == ENOENT {
                return claimGPULock(path, pid: pid, reclaiming: false)
            }
        }
        return false
    }

    static func releaseGPULock(_ path: String, pid: Int32) {
        let owner = (try? String(contentsOfFile: path + "/owner", encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard owner == String(pid) else { return }
        unlink(path + "/owner")
        rmdir(path)
    }
}
