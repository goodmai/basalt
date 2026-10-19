import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltHarness

@Suite("Harness validator on real layouts")
struct HarnessValidatorTests {

    static let claudeLayout = HarnessTree(files: [
        "CLAUDE.md": "Project rules. Run scripts/check.sh before finishing.\nSee @docs/guardrails.md\nUse the ast-scan skill for static analysis.",
        "docs/guardrails.md": "Never push to main.",
        "scripts/check.sh": "#!/bin/sh\necho ok\n",
        ".claude/agents/backend-dev.md": """
        ---
        name: backend-dev
        description: >
          Backend developer. Implements tasks
          in backend/.
        tools: Bash, Read
        model: sonnet
        ---
        You implement backend tasks.
        """,
        ".claude/skills/ast-scan/SKILL.md": """
        ---
        name: ast-scan
        description: AST-level static analysis routine.
        allowed-tools: Bash, Read
        metadata:
          name: not-the-name
        ---
        Run scripts/scan.py from this skill.
        """,
        ".claude/skills/ast-scan/scripts/scan.py": "print('scan')\n",
        ".claude/commands/G0.md": "---\ndescription: \"Start G0\"\n---\nRun G0.",
        ".claude/settings.json": #"{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo pre"}]}], "SubagentStart": [{"hooks": [{"type": "command", "command": "echo s"}]}], "Stop": [{"hooks": [{"type": "prompt", "prompt": "Check the work is done"}]}]}, "env": {"A": "1"}}"#
    ])

    @Test("A Claude Code layout validates: root CLAUDE.md with @imports, agents with name, settings.json hooks")
    func claudeLayoutIsValid() {
        let report = HarnessValidator.validate(Self.claudeLayout, spec: .claude)
        #expect(report.errors == [])
        #expect(report.isValid)
        #expect(report.alwaysOnFiles == ["CLAUDE.md", "docs/guardrails.md"])
        #expect(report.alwaysOnChars == Self.claudeLayout.files["CLAUDE.md"]!.count + "Never push to main.".count)
        #expect(report.skillCount == 1)
        #expect(report.subagentCount == 1)
        #expect(report.untriggeredSkills.isEmpty)
    }

    @Test("An opencode layout validates: AGENTS.md, mode: subagent with nested tools, skill/ directory")
    func opencodeLayout() {
        let tree = HarnessTree(files: [
            "AGENTS.md": "Use the review skill.",
            ".opencode/agent/backend-dev.md": "---\ndescription: \"Backend developer\"\nmode: subagent\ntools:\n  bash: true\n  mode: none\n---\nBody",
            ".opencode/skill/review/SKILL.md": "---\nname: review\ndescription: Review diffs\n---\n",
            ".opencode/plugin/guards.ts": "export const Guards = async () => ({})\n"
        ])
        let report = HarnessValidator.validate(tree, spec: .opencode)
        #expect(report.errors == [])
        #expect(report.skillCount == 1 && report.subagentCount == 1 && report.scriptCount == 1)

        let bad = tree.applying(delta: [".opencode/agent/backend-dev.md": "---\ndescription: x\nmode: sidekick\n---\n"])
        #expect(HarnessValidator.validate(bad, spec: .opencode).errors.contains { $0.contains("mode 'sidekick'") })
    }

    @Test("Structural violations are reported")
    func violations() {
        let tree = HarnessTree(files: [
            "CLAUDE.md": String(repeating: "x", count: 8100) + " scripts/missing.py",
            ".claude/skills/dead/SKILL.md": "---\nname: other\ndescription: x\n---\n",
            ".claude/skills/nodesc/SKILL.md": "---\nname: nodesc\n---\n",
            ".claude/agents/noname.md": "---\ndescription: d\n---\n",
            ".claude/settings.json": #"{"hooks": {"InvalidEvent": [], "PreToolUse": [{"hooks": [{"type": "command", "command": " "}]}], "Stop": [{"hooks": [{"type": "bogus"}]}], "PostToolUse": [{"matcher": "x"}]}}"#
        ])
        let errors = HarnessValidator.validate(tree, spec: .claude).errors
        for expected in [
            "always-on files exceed budget (8119 > 8000 chars)",
            "CLAUDE.md: references missing script 'scripts/missing.py'",
            ".claude/skills/dead/SKILL.md: skill name 'other' does not match directory 'dead'",
            ".claude/skills/nodesc/SKILL.md: frontmatter missing non-empty 'description'",
            ".claude/agents/noname.md: agent frontmatter missing 'name'",
            ".claude/settings.json: hook event 'PreToolUse' has a command handler with an empty 'command'",
            ".claude/settings.json: hook event 'PostToolUse' has a matcher group without a non-empty 'hooks' array"
        ] {
            #expect(errors.contains(expected), "\(expected)")
        }
        #expect(HarnessValidator.validate(tree, spec: .claude.withAlwaysOnMaxChars(9000)).errors.allSatisfy { !$0.contains("budget") })
    }

    @Test("An unknown hook event or handler type is a warning, so a newer Claude Code hook does not invalidate the harness")
    func unknownHooksAreWarnings() {
        let tree = Self.claudeLayout.applying(delta: [
            ".claude/settings.json": #"{"hooks": {"FutureEvent": [{"hooks": [{"type": "command", "command": "echo"}]}], "Stop": [{"hooks": [{"type": "agent", "agent": "x"}]}]}}"#
        ])
        let report = HarnessValidator.validate(tree, spec: .claude)
        #expect(report.isValid)
        #expect(report.errors == [])
        #expect(report.warnings == [
            ".claude/settings.json: hook event 'Stop' has a handler of unknown type 'agent'",
            ".claude/settings.json: unknown hook event 'FutureEvent'"
        ])
    }

    @Test("An unreferenced skill is a warning, not an error; mode: subagent is not required of Claude agents")
    func untriggeredIsWarning() {
        let tree = Self.claudeLayout.applying(delta: [
            ".claude/skills/lonely/SKILL.md": "---\nname: lonely\ndescription: Nobody mentions me\n---\n"
        ])
        let report = HarnessValidator.validate(tree, spec: .claude)
        #expect(report.isValid)
        #expect(report.untriggeredSkills == [".claude/skills/lonely/SKILL.md"])
        #expect(report.warnings.count == 1)
    }

    @Test("Script references resolve against the repository root or the referencing file; URLs are not scripts")
    func scriptReferences() {
        #expect(HarnessPath.scriptReferences(in: "see https://github.com/x/scripts/a.sh and `scripts/b.ts`, /abs/scripts/c.sh") == ["scripts/b.ts"])
        #expect(HarnessPath.resolve("scripts/scan.py", from: ".claude/skills/ast-scan/SKILL.md")
            == ["scripts/scan.py", ".claude/skills/ast-scan/scripts/scan.py"])
        #expect(HarnessPath.normalized("../etc/passwd") == nil)
    }

    @Test("A byte-identical copy of one admitted sibling fails the floor even when the other siblings differ a lot")
    func duplicateOfOneSiblingAmongMany() {
        let base = HarnessTree(files: ["CLAUDE.md": "Base."])
        let first = base.applying(delta: ["CLAUDE.md": "Base.\nRun the tests."])
        let second = base.applying(delta: [".claude/guardrails.md": "Never push.\nNever force.\nNever skip hooks."])
        let third = base.applying(delta: [".claude/memory.md": "Remember the cache key.\nRemember the slug."])
        let report = HarnessValidator.validate(first, spec: .claude, requireDiverseFrom: [first, second, third], relativeTo: base)
        #expect(report.diversityScore == 0)
        #expect(report.errors.contains { $0.hasPrefix("candidate diversity") })
    }

    @Test("@imports resolve relative to the importing file only, as Claude Code does")
    func importsAreFileRelative() {
        #expect(HarnessPath.imports(in: "See @docs/rules.md", from: ".claude/CLAUDE.md") == [".claude/docs/rules.md"])
        #expect(HarnessPath.imports(in: "See @../docs/rules.md", from: ".claude/CLAUDE.md") == ["docs/rules.md"])
        let tree = HarnessTree(files: [".claude/CLAUDE.md": "See @docs/rules.md", "docs/rules.md": "root rules"])
        #expect(tree.alwaysOnPaths(spec: .claude) == [".claude/CLAUDE.md"])
    }

    @Test("Identical sibling candidates fail the diversity floor")
    func diversity() {
        let report = HarnessValidator.validate(Self.claudeLayout, spec: .claude, requireDiverseFrom: [Self.claudeLayout])
        #expect(report.diversityScore == 0)
        #expect(report.errors.contains { $0.hasPrefix("candidate diversity") })
    }
}

@Suite("Harness loader bounds")
struct HarnessLoaderTests {
    static func makeRepo() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        for (path, content) in HarnessValidatorTests.claudeLayout.files {
            let url = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try content.write(to: url, atomically: true, encoding: .utf8)
        }
        return root
    }

    @Test("Loading a repository walks .claude/, reads root instructions, their @imports and referenced scripts, skips the rest")
    func loadsRealLayout() throws {
        let root = try Self.makeRepo()
        defer { try? FileManager.default.removeItem(at: root) }
        let modules = root.appendingPathComponent(".claude/hooks/node_modules/pkg")
        try FileManager.default.createDirectory(at: modules, withIntermediateDirectories: true)
        try "x".write(to: modules.appendingPathComponent("index.js"), atomically: true, encoding: .utf8)
        try "unrelated".write(to: root.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        try "big".padding(toLength: 300_000, withPad: "x", startingAt: 0)
            .write(to: root.appendingPathComponent(".claude/huge.md"), atomically: true, encoding: .utf8)

        let loaded = try HarnessLoader.load(root: root, spec: .claude, allowedRoots: [root])
        #expect(Set(loaded.tree.files.keys) == Set(HarnessValidatorTests.claudeLayout.files.keys))
        #expect(loaded.skipped.map(\.description) == [".claude/huge.md: 300000 bytes exceeds the 262144-byte file limit"])
    }

    @Test("The working directory is an allowed root unless it is / or $HOME; BASALT_HARNESS_ROOTS can still list them")
    func broadWorkingDirectoryIsNotAllowed() {
        let home = ["HOME": "/Users/dev"]
        #expect(HarnessLoader.allowedRoots(environment: home, currentDirectory: "/").isEmpty)
        #expect(HarnessLoader.allowedRoots(environment: home, currentDirectory: "/Users/dev/").isEmpty)
        #expect(HarnessLoader.allowedRoots(environment: home, currentDirectory: "/Users/dev/repo").map(\.path) == ["/Users/dev/repo"])
        let explicit = ["HOME": "/Users/dev", "BASALT_HARNESS_ROOTS": "/Users/dev"]
        #expect(HarnessLoader.allowedRoots(environment: explicit, currentDirectory: "/Users/dev").map(\.path) == ["/Users/dev"])
    }

    @Test("A root outside every allowed root is refused, also through a symlink")
    func outsideAllowedRoots() throws {
        let root = try Self.makeRepo()
        let allowed = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: allowed, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: allowed) }
        let link = allowed.appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: root)

        #expect(throws: HarnessLoadError.outsideAllowedRoots(root.path)) {
            try HarnessLoader.load(root: root, spec: .claude, allowedRoots: [allowed])
        }
        #expect(throws: HarnessLoadError.outsideAllowedRoots(link.path)) {
            try HarnessLoader.load(root: link, spec: .claude, allowedRoots: [allowed])
        }
        #expect(throws: HarnessLoadError.notADirectory(root.appendingPathComponent("CLAUDE.md").path)) {
            try HarnessLoader.load(root: root.appendingPathComponent("CLAUDE.md"), spec: .claude, allowedRoots: [root])
        }
    }

    @Test("Every visited directory entry counts toward maxEntries, also entries that are skipped")
    func visitedEntriesAreBounded() throws {
        let root = try Self.makeRepo()
        defer { try? FileManager.default.removeItem(at: root) }
        for index in 0..<30 {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(".claude/empty/\(index)"), withIntermediateDirectories: true)
        }
        let limits = HarnessLoadLimits(maxFiles: 100, maxFileBytes: 4096, maxTotalBytes: 1 << 20, maxDepth: 8, maxEntries: 20)
        #expect(throws: HarnessLoadError.tooManyEntries(limit: 20)) {
            try HarnessLoader.load(root: root, spec: .claude, allowedRoots: [root], limits: limits)
        }
    }

    @Test("Bytes of a skipped non-UTF-8 file count toward the total limit")
    func skippedBytesCount() throws {
        let root = try Self.makeRepo()
        defer { try? FileManager.default.removeItem(at: root) }
        let layoutBytes = HarnessValidatorTests.claudeLayout.files.values.reduce(0) { $0 + $1.utf8.count }
        try Data(repeating: 0xFF, count: 500).write(to: root.appendingPathComponent(".claude/blob.bin"))
        let limits = HarnessLoadLimits(maxFiles: 100, maxFileBytes: 4096, maxTotalBytes: layoutBytes + 100, maxDepth: 8)
        #expect(throws: HarnessLoadError.tooManyBytes(limit: layoutBytes + 100)) {
            try HarnessLoader.load(root: root, spec: .claude, allowedRoots: [root], limits: limits)
        }
        let roomy = HarnessLoadLimits(maxFiles: 100, maxFileBytes: 4096, maxTotalBytes: layoutBytes + 600, maxDepth: 8)
        let loaded = try HarnessLoader.load(root: root, spec: .claude, allowedRoots: [root], limits: roomy)
        #expect(loaded.skipped.map(\.description) == [".claude/blob.bin: not UTF-8 text"])
    }

    @Test("Claude Code worktrees under .claude/worktrees are not read as part of the harness; a FIFO is ignored")
    func worktreesAndSpecialFilesAreSkipped() throws {
        let root = try Self.makeRepo()
        defer { try? FileManager.default.removeItem(at: root) }
        let worktree = root.appendingPathComponent(".claude/worktrees/feature/.claude/agents")
        try FileManager.default.createDirectory(at: worktree, withIntermediateDirectories: true)
        try "---\nname: x\n---\n".write(to: worktree.appendingPathComponent("copy.md"), atomically: true, encoding: .utf8)
        try "copy".write(to: root.appendingPathComponent(".claude/worktrees/feature/CLAUDE.md"), atomically: true, encoding: .utf8)
        #expect(mkfifo(root.appendingPathComponent(".claude/pipe.md").path, 0o600) == 0)

        let loaded = try HarnessLoader.load(root: root, spec: .claude, allowedRoots: [root])
        #expect(Set(loaded.tree.files.keys) == Set(HarnessValidatorTests.claudeLayout.files.keys))
    }

    @Test("Inline files obey the same count, per-file and total limits as a directory")
    func inlineFilesAreBounded() throws {
        let limits = HarnessLoadLimits(maxFiles: 2, maxFileBytes: 10, maxTotalBytes: 25, maxDepth: 8)
        #expect(throws: HarnessLoadError.tooManyFiles(limit: 2)) {
            try HarnessLoader.inline(["a.md": "a", "b.md": "b", "c.md": "c"], limits: limits)
        }
        #expect(throws: HarnessLoadError.tooManyBytes(limit: 25)) {
            try HarnessLoader.inline(["a.md": String(repeating: "a", count: 20), "b.md": String(repeating: "b", count: 10)], limits: limits)
        }
        let loaded = try HarnessLoader.inline(["CLAUDE.md": "short", "big.md": String(repeating: "b", count: 11)], limits: limits)
        #expect(loaded.tree.files == ["CLAUDE.md": "short"])
        #expect(loaded.skipped.map(\.description) == ["big.md: 11 bytes exceeds the 10-byte file limit"])
    }

    @Test("File count, total size and depth are bounded; a symlink out of the root is skipped")
    func limits() throws {
        let root = try Self.makeRepo()
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".md")
        try "secret".write(to: outside, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: outside) }
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent(".claude/leak.md"), withDestinationURL: outside)
        let deep = root.appendingPathComponent(".claude/a/b/c")
        try FileManager.default.createDirectory(at: deep, withIntermediateDirectories: true)
        try "deep".write(to: deep.appendingPathComponent("deep.md"), atomically: true, encoding: .utf8)

        let tight = HarnessLoadLimits(maxFiles: 3, maxFileBytes: 1024, maxTotalBytes: 1 << 20, maxDepth: 8)
        #expect(throws: HarnessLoadError.tooManyFiles(limit: 3)) {
            try HarnessLoader.load(root: root, spec: .claude, allowedRoots: [root], limits: tight)
        }
        let small = HarnessLoadLimits(maxFiles: 100, maxFileBytes: 1024, maxTotalBytes: 100, maxDepth: 8)
        #expect(throws: HarnessLoadError.tooManyBytes(limit: 100)) {
            try HarnessLoader.load(root: root, spec: .claude, allowedRoots: [root], limits: small)
        }
        let shallow = HarnessLoadLimits(maxFiles: 100, maxFileBytes: 1024, maxTotalBytes: 1 << 20, maxDepth: 2)
        let loaded = try HarnessLoader.load(root: root, spec: .claude, allowedRoots: [root], limits: shallow)
        #expect(loaded.tree.files[".claude/a/b/c/deep.md"] == nil)
        #expect(loaded.tree.files[".claude/leak.md"] == nil)
        #expect(loaded.skipped.map(\.description).contains(".claude/leak.md: resolves outside the harness root"))
    }
}
