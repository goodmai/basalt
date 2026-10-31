import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("Agent capability analyzer parsing")
struct AgentCapabilityAnalyzerTests {
    private static let agentsMdContent = """
    # Agent Capabilities

    ## Tool: calculate_math
    Description: Calculates arithmetic expressions safely.
    Parameters:
      - expression (String, required): The math expression (e.g. "2 + 2").
    Return Type: Number

    ## Tool: file_reader
    Description: Reads a file from disk.
    Parameters:
      - path (String, required): Absolute or relative path.
      - start_line (Int, optional): Starting line.
    Return Type: String
    """

    private static let geminiMdContent = """
    <available_skills>
      <skill>
        <name>skill-creator</name>
        <description>Guide for creating effective skills.</description>
        <location>/opt/homebrew/SKILL.md</location>
      </skill>
      <skill>
        <name>code-analyzer</name>
        <description>Analyzes swift code.</description>
        <location>/opt/homebrew/CODE.md</location>
      </skill>
    </available_skills>
    """

    private static func scratchDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    @Test("Parse agents.md capabilities")
    func parseAgentsMd() async throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("agents.md")
        try Self.agentsMdContent.write(to: url, atomically: true, encoding: .utf8)

        let analyzer = AgentCapabilityAnalyzer()
        let caps = try await analyzer.parse(file: url)

        #expect(caps.count == 2)
        #expect(caps[0].source == .agentsMd)
        #expect(caps[0].name == "calculate_math")
        #expect(caps[0].description == "Calculates arithmetic expressions safely.")
        #expect(caps[0].returnType == "Number")
        #expect(caps[0].parameters.count == 1)
        #expect(caps[0].parameters[0].name == "expression")
        #expect(caps[0].parameters[0].type == "String")
        #expect(caps[0].parameters[0].isRequired)

        #expect(caps[1].name == "file_reader")
        #expect(caps[1].parameters.count == 2)
        #expect(caps[1].parameters[1].name == "start_line")
        #expect(caps[1].parameters[1].type == "Int")
        #expect(!caps[1].parameters[1].isRequired)
    }

    @Test("Parse gemini.md capabilities")
    func parseGeminiMd() async throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("gemini.md")
        try Self.geminiMdContent.write(to: url, atomically: true, encoding: .utf8)

        let analyzer = AgentCapabilityAnalyzer()
        let caps = try await analyzer.parse(file: url)

        #expect(caps.count == 2)
        #expect(caps[0].source == .geminiMd)
        #expect(caps[0].name == "skill-creator")
        #expect(caps[0].description == "Guide for creating effective skills.")
        #expect(caps[0].returnType == nil)
    }

    @Test("Parse claude skill placeholder")
    func parseClaudeSkill() async throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("claude-skill.md")
        try "placeholder".write(to: url, atomically: true, encoding: .utf8)

        let analyzer = AgentCapabilityAnalyzer()
        let caps = try await analyzer.parse(file: url)

        #expect(caps.isEmpty)
    }

    @Test("Parse unknown file format")
    func parseUnknown() async throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("unknown.md")
        try "placeholder".write(to: url, atomically: true, encoding: .utf8)

        let analyzer = AgentCapabilityAnalyzer()
        let caps = try await analyzer.parse(file: url)

        #expect(caps.isEmpty)
    }
}
