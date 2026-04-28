import Testing
import Foundation
@testable import GemmaServerCore

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
        await #expect(throws: GemmaServerError.self) {
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
        await #expect(throws: GemmaServerError.self) {
            _ = try await PromptContextBuilder.build(prompt: prompt)
        }
    }
}
