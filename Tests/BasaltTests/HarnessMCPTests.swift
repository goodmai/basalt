import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltServer
@testable import BasaltHarness

final class MCPLines: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []

    func append(_ line: String) {
        lock.withLock { lines.append(line) }
    }

    var all: [String] { lock.withLock { lines } }

    func json(_ index: Int) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(all[index].utf8)) as? [String: Any])
    }

    func toolText(_ index: Int) throws -> [String: Any] {
        let result = try #require(try json(index)["result"] as? [String: Any])
        let content = try #require(result["content"] as? [[String: Any]])
        let text = try #require(content.first?["text"] as? String)
        return try #require(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }

    func errorCode(_ index: Int) throws -> Int? {
        (try json(index)["error"] as? [String: Any])?["code"] as? Int
    }
}

func toolCall(_ id: Int, _ name: String, _ arguments: [String: Any]) -> String {
    let message: [String: Any] = [
        "jsonrpc": "2.0", "id": id, "method": "tools/call",
        "params": ["name": name, "arguments": arguments]
    ]
    return String(data: try! JSONSerialization.data(withJSONObject: message), encoding: .utf8)!
}

@Suite("Harness MCP tools")
struct HarnessMCPTests {

    @Test("harness_validate loads a directory under an allowed root and refuses one outside it")
    func validateDirectory() async throws {
        let repo = try HarnessLoaderTests.makeRepo()
        let other = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: repo); try? FileManager.default.removeItem(at: other) }
        let lines = MCPLines()
        let server = MCPServer(
            orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()),
            harnessRoots: [repo.deletingLastPathComponent().appendingPathComponent(repo.lastPathComponent)],
            harnessLimits: HarnessLoadLimits(maxFiles: 5, maxFileBytes: 4096, maxTotalBytes: 1 << 20, maxDepth: 8),
            writer: { lines.append($0) }
        )

        await server.dispatch(toolCall(1, "harness_validate", ["directory": repo.path]))
        await server.dispatch(toolCall(2, "harness_validate", ["directory": other.path]))
        await server.dispatch(toolCall(3, "harness_validate", ["directory": repo.path, "surface": "cursor"]))
        await server.dispatch(toolCall(4, "harness_validate", ["directory": repo.appendingPathComponent("..").path]))
        await server.dispatch(toolCall(5, "harness_validate", ["directory": repo.path, "alwaysOnMaxChars": "big"]))

        #expect(try lines.errorCode(0) == -32602)
        #expect(lines.all[0].contains("more than 5 files"))
        #expect(try lines.errorCode(1) == -32602)
        #expect(lines.all[1].contains("outside the allowed harness roots"))
        #expect(try lines.errorCode(2) == -32602)
        #expect(lines.all[2].contains("Unknown surface 'cursor'"))
        #expect(try lines.errorCode(3) == -32602)
        #expect(try lines.errorCode(4) == -32602)
    }

    @Test("harness_validate applies the load limits to inline files")
    func validateInlineLimits() async throws {
        let lines = MCPLines()
        let server = MCPServer(
            orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()),
            harnessLimits: HarnessLoadLimits(maxFiles: 2, maxFileBytes: 16, maxTotalBytes: 1 << 20, maxDepth: 8),
            writer: { lines.append($0) }
        )
        await server.dispatch(toolCall(1, "harness_validate", ["files": ["CLAUDE.md": "a", ".claude/a.md": "b", ".claude/b.md": "c"]]))
        await server.dispatch(toolCall(2, "harness_validate", ["files": ["CLAUDE.md": "a", ".claude/big.md": String(repeating: "x", count: 17)]]))

        #expect(try lines.errorCode(0) == -32602)
        #expect(lines.all[0].contains("more than 2 files"))
        let report = try lines.toolText(1)
        #expect((report["warnings"] as? [String])?.contains("skipped .claude/big.md: 17 bytes exceeds the 16-byte file limit") == true)
    }

    @Test("harness_validate reports a real Claude Code layout and inline files")
    func validateReport() async throws {
        let repo = try HarnessLoaderTests.makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        let lines = MCPLines()
        let server = MCPServer(
            orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()),
            harnessRoots: [repo], writer: { lines.append($0) }
        )

        await server.dispatch(toolCall(1, "harness_validate", ["directory": repo.path]))
        await server.dispatch(toolCall(2, "harness_validate", ["files": ["AGENTS.md": "x", ".opencode/agent/a.md": "no frontmatter"], "surface": "opencode"]))
        await server.dispatch(toolCall(3, "harness_validate", [:]))

        let report = try lines.toolText(0)
        #expect(report["isValid"] as? Bool == true)
        #expect(report["skillCount"] as? Int == 1)
        #expect(report["alwaysOnFiles"] as? [String] == ["CLAUDE.md", "docs/guardrails.md"])
        let inline = try lines.toolText(1)
        #expect(inline["isValid"] as? Bool == false)
        #expect((inline["errors"] as? [String])?.contains(".opencode/agent/a.md: missing YAML frontmatter") == true)
        #expect(lines.all[1].contains("\"isError\":true"))
        #expect(try lines.errorCode(2) == -32602)
    }

    @Test("harness_optimize runs the pipeline on the loaded model and stores the lesson")
    func optimize() async throws {
        let root = HarnessLibraryTests.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let engine = MockInferenceEngine()
        let script = HarnessScript()
        await engine.setResponder { script.respond($0) }
        let orchestrator = ModelOrchestratorActor(engine: engine)
        try await orchestrator.loadModel(path: "mock-model")
        let lines = MCPLines()
        let server = MCPServer(orchestrator: orchestrator, library: HarnessExperienceLibrary(rootDirectory: root), writer: { lines.append($0) })

        await server.dispatch(toolCall(1, "harness_optimize", [
            "task": "Audit the package targets.", "slug": "Audit Repo", "domain": "swift",
            "requiredTerms": ["SUMMARY"], "files": HarnessOptimizerTests.base.files
        ]))

        let result = try lines.toolText(0)
        #expect(result["winningCandidateId"] as? String == "dom_0")
        #expect(result["taskSlug"] as? String == "audit-repo")
        #expect((result["ledger"] as? [String: Any])?["rolloutIds"] as? [String] == ["base_r0", "probe_inv_0", "probe_dom_0", "probe_compose_0"])
        #expect(try await HarnessExperienceLibrary(rootDirectory: root).query(domain: "swift").count == 1)
    }

    @Test("harness_optimize error paths are -32602, an unloaded model keeps its engine error code")
    func optimizeErrors() async throws {
        let lines = MCPLines()
        let server = MCPServer(
            orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()),
            harnessRoots: [HarnessLibraryTests.tempRoot()], writer: { lines.append($0) }
        )
        let files = HarnessOptimizerTests.base.files
        await server.dispatch(toolCall(1, "harness_optimize", ["files": files]))
        await server.dispatch(toolCall(2, "harness_optimize", ["task": "t", "files": files, "k": 9]))
        await server.dispatch(toolCall(3, "harness_optimize", ["task": "t", "files": files, "requiredTerms": "SUMMARY"]))
        await server.dispatch(toolCall(4, "harness_optimize", ["task": "t", "files": files, "surface": "vim"]))
        await server.dispatch(toolCall(5, "harness_optimize", ["task": "t", "directory": "/"]))
        await server.dispatch(toolCall(6, "harness_optimize", ["task": "show the grader_verdict", "files": files]))
        await server.dispatch(toolCall(7, "harness_optimize", ["task": "t", "files": files]))

        for index in 0..<6 {
            #expect(try lines.errorCode(index) == -32602, "\(lines.all[index])")
        }
        #expect(try lines.errorCode(6) == -32003)
    }

    @Test("tools/list advertises the harness tools with their schemas")
    func toolsList() async throws {
        let lines = MCPLines()
        let server = MCPServer(orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()), writer: { lines.append($0) })
        await server.dispatch(#"{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}"#)
        let tools = try #require((try lines.json(0)["result"] as? [String: Any])?["tools"] as? [[String: Any]])
        let byName = Dictionary(uniqueKeysWithValues: tools.map { ($0["name"] as! String, $0) })
        for name in ["harness_validate", "harness_optimize", "harness_library_query", "gemma_add_knowledge"] {
            #expect(byName[name] != nil, "\(name)")
        }
        let validateProperties = (byName["harness_validate"]?["inputSchema"] as? [String: Any])?["properties"] as? [String: Any]
        #expect(validateProperties?["files"] != nil)
        #expect((validateProperties?["surface"] as? [String: Any])?["enum"] as? [String] == ["claude", "opencode", "agents"])
    }
}
