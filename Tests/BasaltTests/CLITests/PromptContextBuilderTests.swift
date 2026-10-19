import Testing
import Foundation
@testable import BasaltKit

@Suite("PromptContextBuilderTests")
struct PromptContextBuilderTests {
    
    @Test("Build parses single @file and injects content")
    func testSingleFileInjection() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("test_context_1.txt")
        try "Hello from test file".write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }
        
        let prompt = "Explain @\(fileURL.path)"
        let built = try await PromptContextBuilder.build(prompt: prompt)
        
        #expect(built.contains("Explain @\(fileURL.path)"))
        #expect(built.contains("--- Context Files ---"))
        #expect(built.contains("Hello from test file"))
    }
    
    @Test("Build parses multiple files")
    func testMultipleFilesInjection() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL1 = tempDir.appendingPathComponent("test_context_multi_1.txt")
        let fileURL2 = tempDir.appendingPathComponent("test_context_multi_2.txt")
        try "Content 1".write(to: fileURL1, atomically: true, encoding: .utf8)
        try "Content 2".write(to: fileURL2, atomically: true, encoding: .utf8)
        defer { 
            try? FileManager.default.removeItem(at: fileURL1)
            try? FileManager.default.removeItem(at: fileURL2)
        }
        
        let prompt = "Combine @\(fileURL1.path) and @\(fileURL2.path)"
        let built = try await PromptContextBuilder.build(prompt: prompt)
        
        #expect(built.contains("Content 1"))
        #expect(built.contains("Content 2"))
    }
    
    @Test("Escaped @ is ignored")
    func testEscapedAtIsIgnored() async throws {
        let prompt = "Email me at test\\@example.com"
        let built = try await PromptContextBuilder.build(prompt: prompt)
        
        #expect(built == "Email me at test@example.com")
        #expect(!built.contains("Context Files"))
    }
    
    @Test("Missing file throws error")
    func testMissingFileThrows() async throws {
        let prompt = "Fix @/path/to/missing/file.txt"
        await #expect(throws: BasaltError.self) {
            _ = try await PromptContextBuilder.build(prompt: prompt)
        }
    }
    
    @Test("Large file throws error")
    func testLargeFileThrows() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("test_context_large.txt")
        
        // Write >1MB file
        let data = Data(repeating: 65, count: 1024 * 1024 + 10)
        try data.write(to: fileURL)
        defer { try? FileManager.default.removeItem(at: fileURL) }
        
        let prompt = "Analyze @\(fileURL.path)"
        await #expect(throws: BasaltError.self) {
            _ = try await PromptContextBuilder.build(prompt: prompt)
        }
    }

    // MARK: — Tool Prefix Protection & Compaction (Issue #26)

    @Test("extractToolPrefix separates tool definitions from user turns")
    func testExtractToolPrefix() {
        let promptWithTools = """
        [Tools]
        - run_command: Run a shell command
        - view_file: View file content
        
        [User]: Can you build the project?
        [Assistant]: Running swift build...
        """

        let (prefix, tail) = PromptContextBuilder.extractToolPrefix(from: promptWithTools)
        #expect(prefix.contains("[Tools]"))
        #expect(prefix.contains("run_command"))
        #expect(tail.contains("[User]: Can you build the project?"))
        #expect(!tail.contains("[Tools]"))
    }

    @Test("compactWithToolPrefixProtection preserves tool prefix when budget exceeded")
    func testCompactWithToolPrefixProtection() {
        let toolPrefix = "System Prompt: You have access to bash, python, and git tools."
        var turns: [String] = []
        for i in 1...20 {
            turns.append("[User turn \(i)]: Please do step \(i)")
            turns.append("[Assistant turn \(i)]: Done step \(i)")
        }

        let compacted = PromptContextBuilder.compactWithToolPrefixProtection(
            toolPrefix: toolPrefix,
            historyTurns: turns,
            maxBudgetChars: 500
        )

        // 1. Tool prefix must be 100% intact at the head!
        #expect(compacted.hasPrefix(toolPrefix))
        // 2. Total compacted length must respect budget (with reasonable margin for notice)
        #expect(compacted.count <= 600)
        // 3. Compaction marker is present
        #expect(compacted.contains("... [earlier turns compacted to preserve tool prefix] ..."))
        // 4. Latest turn is preserved at the tail
        #expect(compacted.contains("[Assistant turn 20]"))
    }

    @Test("compactWithToolPrefixProtection keeps entire short history intact")
    func testCompactLeavesShortHistoryUntouched() {
        let toolPrefix = "Tools: [ping]"
        let turns = ["[User]: ping", "[Assistant]: pong"]

        let result = PromptContextBuilder.compactWithToolPrefixProtection(
            toolPrefix: toolPrefix,
            historyTurns: turns,
            maxBudgetChars: 2000
        )

        #expect(result.contains("Tools: [ping]"))
        #expect(result.contains("[User]: ping"))
        #expect(result.contains("[Assistant]: pong"))
        #expect(!result.contains("compacted"))
    }
}

