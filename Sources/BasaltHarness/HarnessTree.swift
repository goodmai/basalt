import Foundation
import BasaltCore

public enum HarnessComponentKind: String, Codable, CaseIterable, Sendable {
    case systemPrompt = "systemprompt"
    case guardrails = "guardrails"
    case memory = "memory"
    case toolGuidance = "tools"
    case subagents = "subagents"
    case skills = "skills"
    case scripts = "scripts"
    case hooks = "hooks"
}

public struct HarnessSurfaceSpec: Sendable, Equatable {
    public enum AgentFormat: Sendable, Equatable {
        case claude
        case opencode
    }

    public let name: String
    public let instructionFiles: [String]
    public let directory: String
    public let skillDirectories: [String]
    public let commandDirectories: [String]
    public let agentDirectories: [String]
    public let agentFormat: AgentFormat
    public let hookSettingsFile: String?
    public let pluginDirectories: [String]
    public var alwaysOnMaxChars: Int
    public var minDiversity: Double

    public static let claude = HarnessSurfaceSpec(
        name: "claude",
        instructionFiles: ["CLAUDE.md", ".claude/CLAUDE.md", "CLAUDE.local.md"],
        directory: ".claude",
        skillDirectories: [".claude/skills"],
        commandDirectories: [".claude/commands"],
        agentDirectories: [".claude/agents"],
        agentFormat: .claude,
        hookSettingsFile: ".claude/settings.json",
        pluginDirectories: [".claude/hooks"],
        alwaysOnMaxChars: 8000,
        minDiversity: 0.05
    )

    public static let opencode = HarnessSurfaceSpec(
        name: "opencode",
        instructionFiles: ["AGENTS.md"],
        directory: ".opencode",
        skillDirectories: [".opencode/skill", ".opencode/skills"],
        commandDirectories: [".opencode/command", ".opencode/commands"],
        agentDirectories: [".opencode/agent", ".opencode/agents"],
        agentFormat: .opencode,
        hookSettingsFile: nil,
        pluginDirectories: [".opencode/plugin", ".opencode/plugins"],
        alwaysOnMaxChars: 8000,
        minDiversity: 0.05
    )

    public static let agents = HarnessSurfaceSpec(
        name: "agents",
        instructionFiles: ["AGENTS.md"],
        directory: ".agents",
        skillDirectories: [".agents/skills"],
        commandDirectories: [],
        agentDirectories: [],
        agentFormat: .claude,
        hookSettingsFile: nil,
        pluginDirectories: [],
        alwaysOnMaxChars: 8000,
        minDiversity: 0.05
    )

    public static let all: [HarnessSurfaceSpec] = [.claude, .opencode, .agents]

    public static func named(_ name: String) -> HarnessSurfaceSpec? {
        all.first { $0.name == name.lowercased() }
    }

    public func withAlwaysOnMaxChars(_ limit: Int) -> HarnessSurfaceSpec {
        var copy = self
        copy.alwaysOnMaxChars = limit
        return copy
    }

    public func owns(_ path: String) -> Bool {
        instructionFiles.contains(path) || path.hasPrefix(directory + "/")
    }

    func isUnder(_ directories: [String], _ path: String) -> Bool {
        directories.contains { path.hasPrefix($0 + "/") }
    }

    public func classify(path: String) -> HarnessComponentKind {
        if isUnder(skillDirectories + commandDirectories, path) { return .skills }
        if isUnder(agentDirectories, path) { return .subagents }
        if path == hookSettingsFile || isUnder(pluginDirectories, path) { return .hooks }
        let lower = path.lowercased()
        if lower.split(separator: "/").contains("scripts") { return .scripts }
        if lower.contains("guardrail") { return .guardrails }
        if lower.contains("memory") { return .memory }
        if lower.contains("tools") { return .toolGuidance }
        return .systemPrompt
    }

    func skillName(at path: String) -> String? {
        for directory in skillDirectories where path.hasPrefix(directory + "/") {
            let parts = path.dropFirst(directory.count + 1).split(separator: "/")
            if parts.count == 2, parts[1] == "SKILL.md" { return String(parts[0]) }
        }
        return nil
    }

    func isAgentFile(_ path: String) -> Bool {
        isUnder(agentDirectories, path) && path.hasSuffix(".md")
    }
}

public struct HarnessTree: Codable, Sendable, Equatable {
    public var files: [String: String]

    public init(files: [String: String] = [:]) {
        self.files = Dictionary(
            files.map { (HarnessPath.normalized($0.key) ?? $0.key, $0.value) },
            uniquingKeysWith: { _, last in last }
        )
    }

    public func applying(delta: [String: String]) -> HarnessTree {
        HarnessTree(files: files.merging(delta) { _, new in new })
    }

    public func alwaysOnPaths(spec: HarnessSurfaceSpec) -> [String] {
        var ordered: [String] = []
        var pending = spec.instructionFiles.filter { files[$0] != nil }
        var depth = 0
        while !pending.isEmpty, depth <= 5 {
            var next: [String] = []
            for path in pending where !ordered.contains(path) {
                ordered.append(path)
                next += HarnessPath.imports(in: files[path] ?? "", from: path).filter { files[$0] != nil }
            }
            pending = next
            depth += 1
        }
        return ordered
    }

    public func alwaysOnText(spec: HarnessSurfaceSpec) -> String {
        alwaysOnPaths(spec: spec).compactMap { files[$0] }.joined(separator: "\n\n")
    }

    public func alwaysOnCharCount(spec: HarnessSurfaceSpec) -> Int {
        alwaysOnPaths(spec: spec).reduce(0) { $0 + (files[$1]?.count ?? 0) }
    }

    func serializedLines() -> Set<String> {
        var result = Set<String>()
        for (path, content) in files {
            for line in content.components(separatedBy: .newlines) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if !trimmed.isEmpty { result.insert("\(path):\(trimmed)") }
            }
        }
        return result
    }
}

enum HarnessPath {
    static func normalized(_ raw: String) -> String? {
        var parts: [Substring] = []
        for part in raw.split(separator: "/", omittingEmptySubsequences: true) {
            switch part {
            case ".": continue
            case "..":
                guard !parts.isEmpty else { return nil }
                parts.removeLast()
            default: parts.append(part)
            }
        }
        return parts.isEmpty ? nil : parts.joined(separator: "/")
    }

    static func resolve(_ reference: String, from file: String) -> [String] {
        let directory = file.split(separator: "/").dropLast().joined(separator: "/")
        let local = directory.isEmpty ? nil : normalized(directory + "/" + reference)
        return [normalized(reference), local].compactMap { $0 }
    }

    static func imports(in content: String, from file: String) -> [String] {
        matches(of: #"(?<![^\s(\[])@((?:\.{1,2}/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.md)"#, in: content)
            .flatMap { resolve($0, from: file) }
    }

    static func scriptReferences(in content: String) -> [String] {
        matches(of: #"(?<![^\s`(\["'])((?:\./)?(?:[A-Za-z0-9_.-]+/)*scripts/[A-Za-z0-9_./-]+\.(?:py|sh|bash|swift|ts|js|mjs))"#, in: content)
            .filter { reference in
                let head = reference.split(separator: "/").first ?? ""
                return head.hasPrefix(".") || !head.contains(".")
            }
    }

    private static func matches(of pattern: String, in content: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        return regex.matches(in: content, range: NSRange(content.startIndex..., in: content)).compactMap {
            Range($0.range(at: 1), in: content).map { String(content[$0]) }
        }
    }
}

public struct HarnessLoadLimits: Sendable, Equatable {
    public var maxFiles: Int
    public var maxFileBytes: Int
    public var maxTotalBytes: Int
    public var maxDepth: Int

    public static let standard = HarnessLoadLimits(
        maxFiles: 2000, maxFileBytes: 256 * 1024, maxTotalBytes: 16 * 1024 * 1024, maxDepth: 8
    )

    public init(maxFiles: Int, maxFileBytes: Int, maxTotalBytes: Int, maxDepth: Int) {
        self.maxFiles = maxFiles
        self.maxFileBytes = maxFileBytes
        self.maxTotalBytes = maxTotalBytes
        self.maxDepth = maxDepth
    }
}

public enum HarnessLoadError: Error, Equatable, CustomStringConvertible {
    case notADirectory(String)
    case outsideAllowedRoots(String)
    case tooManyFiles(limit: Int)
    case tooManyBytes(limit: Int)

    public var description: String {
        switch self {
        case .notADirectory(let path): "\(path) is not a directory"
        case .outsideAllowedRoots(let path): "\(path) is outside the allowed harness roots"
        case .tooManyFiles(let limit): "harness has more than \(limit) files"
        case .tooManyBytes(let limit): "harness is larger than \(limit) bytes"
        }
    }
}

public struct LoadedHarness: Sendable {
    public let tree: HarnessTree
    public let skipped: [String]
}

public enum HarnessLoader {
    public static func allowedRoots(environment: [String: String] = ProcessInfo.processInfo.environment) -> [URL] {
        let configured = (environment["BASALT_HARNESS_ROOTS"] ?? "")
            .split(separator: ":").map { URL(fileURLWithPath: String($0), isDirectory: true) }
        return [URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)] + configured
    }

    public static func load(
        root: URL,
        spec: HarnessSurfaceSpec,
        allowedRoots: [URL],
        limits: HarnessLoadLimits = .standard
    ) throws -> LoadedHarness {
        let base = root.resolvingSymlinksInPath().standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: base.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw HarnessLoadError.notADirectory(root.path)
        }
        guard allowedRoots.contains(where: { contains($0.resolvingSymlinksInPath().standardizedFileURL, base) }) else {
            throw HarnessLoadError.outsideAllowedRoots(root.path)
        }
        var reader = Reader(base: base, limits: limits)
        var pending = spec.instructionFiles
        for _ in 0...5 where !pending.isEmpty {
            var next: [String] = []
            for path in pending where reader.files[path] == nil {
                try reader.read(path)
                next += HarnessPath.imports(in: reader.files[path] ?? "", from: path)
            }
            pending = next
        }
        try reader.walk(spec.directory)
        for (path, content) in reader.files {
            for reference in HarnessPath.scriptReferences(in: content) {
                for candidate in HarnessPath.resolve(reference, from: path) where reader.files[candidate] == nil {
                    try reader.read(candidate)
                }
            }
        }
        return LoadedHarness(tree: HarnessTree(files: reader.files), skipped: reader.skipped.sorted())
    }

    static func contains(_ root: URL, _ candidate: URL) -> Bool {
        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        return candidate.path == root.path || candidate.path.hasPrefix(rootPath)
    }

    private struct Reader {
        let base: URL
        let limits: HarnessLoadLimits
        var files: [String: String] = [:]
        var skipped: [String] = []
        var totalBytes = 0

        init(base: URL, limits: HarnessLoadLimits) {
            self.base = base
            self.limits = limits
        }

        mutating func read(_ path: String) throws {
            let url = base.appendingPathComponent(path)
            let resolved = url.resolvingSymlinksInPath().standardizedFileURL
            guard let values = try? resolved.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  values.isRegularFile == true else { return }
            guard HarnessLoader.contains(base, resolved) else {
                skipped.append("\(path): resolves outside the harness root")
                return
            }
            let size = values.fileSize ?? 0
            guard size <= limits.maxFileBytes else {
                skipped.append("\(path): \(size) bytes exceeds the \(limits.maxFileBytes)-byte file limit")
                return
            }
            guard files.count < limits.maxFiles else { throw HarnessLoadError.tooManyFiles(limit: limits.maxFiles) }
            guard totalBytes + size <= limits.maxTotalBytes else {
                throw HarnessLoadError.tooManyBytes(limit: limits.maxTotalBytes)
            }
            guard let text = try? String(contentsOf: resolved, encoding: .utf8) else {
                skipped.append("\(path): not UTF-8 text")
                return
            }
            totalBytes += size
            files[path] = text
        }

        mutating func walk(_ directory: String) throws {
            let start = base.appendingPathComponent(directory, isDirectory: true)
            guard let enumerator = FileManager.default.enumerator(
                at: start, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]
            ) else { return }
            for case let url as URL in enumerator {
                let isDirectory = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                if isDirectory {
                    if url.lastPathComponent == "node_modules" || enumerator.level >= limits.maxDepth {
                        enumerator.skipDescendants()
                    }
                    continue
                }
                let relative = url.standardizedFileURL.path.dropFirst(start.standardizedFileURL.path.count + 1)
                try read(directory + "/" + relative)
            }
        }
    }
}
