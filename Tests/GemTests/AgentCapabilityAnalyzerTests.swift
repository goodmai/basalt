import XCTest
@testable import GemCore

final class AgentCapabilityAnalyzerTests: XCTestCase {
    
    // MARK: - Test Sample Files
    
    let agentsMdContent = """
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
    
    let geminiMdContent = """
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
    
    // MARK: - Setup
    
    var agentsMdURL: URL!
    var geminiMdURL: URL!
    
    override func setUp() {
        super.setUp()
        let tempDir = FileManager.default.temporaryDirectory
        
        agentsMdURL = tempDir.appendingPathComponent("agents.md")
        geminiMdURL = tempDir.appendingPathComponent("gemini.md")
        
        try! agentsMdContent.write(to: agentsMdURL, atomically: true, encoding: .utf8)
        try! geminiMdContent.write(to: geminiMdURL, atomically: true, encoding: .utf8)
    }
    
    override func tearDown() {
        try? FileManager.default.removeItem(at: agentsMdURL)
        try? FileManager.default.removeItem(at: geminiMdURL)
        super.tearDown()
    }
    
    // MARK: - Tests
    
    func testParseAgentsMd() async throws {
        let analyzer = AgentCapabilityAnalyzer()
        let caps = try await analyzer.parse(file: agentsMdURL)
        
        XCTAssertEqual(caps.count, 2)
        XCTAssertEqual(caps[0].source, .agentsMd)
        XCTAssertEqual(caps[0].name, "calculate_math")
        XCTAssertEqual(caps[0].description, "Calculates arithmetic expressions safely.")
        XCTAssertEqual(caps[0].returnType, "Number")
        XCTAssertEqual(caps[0].parameters.count, 1)
        XCTAssertEqual(caps[0].parameters[0].name, "expression")
        XCTAssertEqual(caps[0].parameters[0].type, "String")
        XCTAssertTrue(caps[0].parameters[0].isRequired)
        
        XCTAssertEqual(caps[1].name, "file_reader")
        XCTAssertEqual(caps[1].parameters.count, 2)
        XCTAssertEqual(caps[1].parameters[1].name, "start_line")
        XCTAssertEqual(caps[1].parameters[1].type, "Int")
        XCTAssertFalse(caps[1].parameters[1].isRequired)
    }
    
    func testParseGeminiMd() async throws {
        let analyzer = AgentCapabilityAnalyzer()
        let caps = try await analyzer.parse(file: geminiMdURL)
        
        XCTAssertEqual(caps.count, 2)
        XCTAssertEqual(caps[0].source, .geminiMd)
        XCTAssertEqual(caps[0].name, "skill-creator")
        XCTAssertEqual(caps[0].description, "Guide for creating effective skills.")
        XCTAssertNil(caps[0].returnType)
    }

    func testParseClaudeSkill() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let claudeURL = tempDir.appendingPathComponent("claude-skill.md")
        try! "placeholder".write(to: claudeURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: claudeURL) }
        
        let analyzer = AgentCapabilityAnalyzer()
        let caps = try await analyzer.parse(file: claudeURL)
        
        XCTAssertTrue(caps.isEmpty)
    }

    func testParseUnknown() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let unknownURL = tempDir.appendingPathComponent("unknown.md")
        try! "placeholder".write(to: unknownURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: unknownURL) }
        
        let analyzer = AgentCapabilityAnalyzer()
        let caps = try await analyzer.parse(file: unknownURL)
        
        XCTAssertTrue(caps.isEmpty)
    }
}