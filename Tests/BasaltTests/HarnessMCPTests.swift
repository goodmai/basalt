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
}
