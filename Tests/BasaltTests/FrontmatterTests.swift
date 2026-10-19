import Testing
import Foundation
@testable import BasaltCore

@Suite("Frontmatter and skill detection")
struct FrontmatterTests {

    @Test("Folded and literal block scalars yield their text")
    func blockScalars() throws {
        let parsed = try #require(Frontmatter.parse("""
        ---
        name: chaindesk
        description: >
          Interact with the platform
          via its REST API.

          Second paragraph.
        notes: |-
          line one
          line two
        ---
        Body
        """))
        #expect(parsed.fields["description"] == "Interact with the platform via its REST API.\nSecond paragraph.")
        #expect(parsed.fields["notes"] == "line one\nline two")
        #expect(parsed.fields["name"] == "chaindesk")
        #expect(parsed.body == "Body")
    }

    @Test("Nested keys never override top-level keys")
    func nestedKeys() throws {
        let parsed = try #require(Frontmatter.parse("""
        ---
        name: top
        metadata:
          name: nested
          description: nested description
        tools:
          bash: true
        description: "Quoted: value with \\"escapes\\""
        ---
        """))
        #expect(parsed.fields["name"] == "top")
        #expect(parsed.fields["description"] == "Quoted: value with \"escapes\"")
        #expect(parsed.fields["metadata"] == nil)
        #expect(parsed.fields["tools"] == nil)
        #expect(parsed.fields["bash"] == nil)
    }

    @Test("Plain scalars continue on indented lines; single quotes unescape")
    func plainContinuation() throws {
        let parsed = try #require(Frontmatter.parse("---\ndescription: first part\n  second part\nmode: 'it''s'\n---\n"))
        #expect(parsed.fields["description"] == "first part second part")
        #expect(parsed.fields["mode"] == "it's")
    }

    @Test("Content without a closing fence has no frontmatter")
    func unterminated() {
        #expect(Frontmatter.parse("---\nname: x\n") == nil)
        #expect(Frontmatter.parse("name: x\n---\n") == nil)
    }

    @Test("Only skills/<name>/SKILL.md is a Claude skill; any other frontmatter file is not")
    func skillDetectionIsPathBased() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let skillDir = root.appendingPathComponent(".claude/skills/daml-reviewer")
        try FileManager.default.createDirectory(at: skillDir, withIntermediateDirectories: true)
        let skill = skillDir.appendingPathComponent("SKILL.md")
        try """
        ---
        name: daml-reviewer
        description: >
          Review DAML code and DAR packages.
        allowed-tools: Bash, Read
        ---

        - dar_path (String, required): not a parameter list, just prose.
        """.write(to: skill, atomically: true, encoding: .utf8)
        let agent = root.appendingPathComponent("backend-dev.md")
        try "---\nname: backend-dev\ndescription: Backend developer\n---\n".write(to: agent, atomically: true, encoding: .utf8)

        let analyzer = AgentCapabilityAnalyzer()
        let capabilities = try await analyzer.parse(file: skill)
        #expect(capabilities.count == 1)
        #expect(capabilities.first?.source == .claudeSkill)
        #expect(capabilities.first?.name == "daml-reviewer")
        #expect(capabilities.first?.description == "Review DAML code and DAR packages.")
        #expect(capabilities.first?.parameters.isEmpty == true)
        #expect(try await analyzer.parse(file: agent).isEmpty)
    }
}
