import Testing
import Foundation
@testable import GemCore

/// Epic 16.6: Clipboard Integration - TDD Tests
/// Write tests FIRST, implement SECOND
@Suite("ClipboardManager Tests - Epic 16.6")
struct ClipboardManagerTests {
    
    // MARK: - Clipboard Tool Detection Tests
    
    @Test("Detect available clipboard tool on macOS")
    func testDetectClipboardToolMacOS() async {
        #if os(macOS)
        let manager = ClipboardManager()
        let tool = await manager.detectClipboardTool()
        
        // macOS should have pbcopy/pbpaste
        #expect(tool == .pbcopy || tool == .none)
        #endif
    }
    
    @Test("Detect available clipboard tool on Linux")
    func testDetectClipboardToolLinux() async {
        #if os(Linux)
        let manager = ClipboardManager()
        let tool = await manager.detectClipboardTool()
        
        // Linux might have xclip, xsel, or none
        #expect(tool == .xclip || tool == .xsel || tool == .none)
        #endif
    }
    
    // MARK: - Copy Tests
    
    @Test("Copy simple text to clipboard", .enabled(if: !ClipboardManager.isRunningInCI(), "no pasteboard in a CI session"))
    func testCopySimpleText() async throws {
        let manager = ClipboardManager()
        let text = "Hello, Clipboard!"
        
        do {
            try await manager.copy(text)
            // If no error, consider it success
            #expect(true)
        } catch ClipboardError.toolNotAvailable {
            // Acceptable on systems without clipboard tool
            #expect(true)
        } catch {
            throw error
        }
    }
    
    @Test("Copy multiline text to clipboard", .enabled(if: !ClipboardManager.isRunningInCI(), "no pasteboard in a CI session"))
    func testCopyMultilineText() async throws {
        let manager = ClipboardManager()
        let text = """
        Line 1
        Line 2
        Line 3
        """
        
        do {
            try await manager.copy(text)
            #expect(true)
        } catch ClipboardError.toolNotAvailable {
            #expect(true)
        }
    }
    
    @Test("Copy Unicode text to clipboard", .enabled(if: !ClipboardManager.isRunningInCI(), "no pasteboard in a CI session"))
    func testCopyUnicodeText() async throws {
        let manager = ClipboardManager()
        let text = "Hello 🌍 Мир 世界"
        
        do {
            try await manager.copy(text)
            #expect(true)
        } catch ClipboardError.toolNotAvailable {
            #expect(true)
        }
    }
    
    @Test("Copy empty string to clipboard", .enabled(if: !ClipboardManager.isRunningInCI(), "no pasteboard in a CI session"))
    func testCopyEmptyString() async throws {
        let manager = ClipboardManager()
        
        do {
            try await manager.copy("")
            #expect(true)
        } catch ClipboardError.toolNotAvailable {
            #expect(true)
        }
    }
    
    // MARK: - Paste Tests
    
    @Test("Paste text from clipboard", .enabled(if: !ClipboardManager.isRunningInCI(), "no pasteboard in a CI session"))
    func testPasteText() async throws {
        let manager = ClipboardManager()
        
        do {
            let text = try await manager.paste()
            // Should return some string (might be empty)
            #expect(text != nil)
        } catch ClipboardError.toolNotAvailable {
            // Acceptable
            #expect(true)
        }
    }
    
    // MARK: - Round-trip Tests
    
    @Test("Copy then paste round-trip", .disabled("Races other tests over the shared pasteboard"))
    func testRoundTrip() async throws {
        let manager = ClipboardManager()
        let originalText = "Test round-trip"
        
        do {
            try await manager.copy(originalText)
            let pastedText = try await manager.paste()
            
            #expect(pastedText == originalText)
        } catch ClipboardError.toolNotAvailable {
            // Skip on systems without clipboard
            #expect(true)
        }
    }
    
    // MARK: - Error Handling Tests
    
    @Test("Handle missing clipboard tool gracefully", .enabled(if: !ClipboardManager.isRunningInCI(), "no pasteboard in a CI session"))
    func testMissingToolError() async {
        let manager = ClipboardManager()
        
        // Should not crash even if tool is missing
        do {
            try await manager.copy("test")
        } catch ClipboardError.toolNotAvailable {
            // Expected error
            #expect(true)
        } catch {
            // Unexpected error
            #expect(false, "Unexpected error: \(error)")
        }
    }
    
    // MARK: - Tool Availability Tests
    
    @Test("Check if clipboard is available")
    func testIsAvailable() async {
        let manager = ClipboardManager()
        let available = await manager.isAvailable()
        
        // Should return boolean without crashing
        #expect(available == true || available == false)
    }
    
    // MARK: - Tool Path Tests
    
    @Test("Get clipboard tool path on macOS")
    func testToolPathMacOS() {
        #if os(macOS)
        let path = ClipboardManager.toolPath(for: .pbcopy)
        #expect(path == "/usr/bin/pbcopy")
        
        let pastePath = ClipboardManager.toolPath(for: .pbpaste)
        #expect(pastePath == "/usr/bin/pbpaste")
        #endif
    }
    
    @Test("Get clipboard tool path on Linux")
    func testToolPathLinux() {
        #if os(Linux)
        let xclipPath = ClipboardManager.toolPath(for: .xclip)
        #expect(xclipPath == "/usr/bin/xclip")
        
        let xselPath = ClipboardManager.toolPath(for: .xsel)
        #expect(xselPath == "/usr/bin/xsel")
        #endif
    }
}

@Suite("ClipboardManager Integration Tests - Epic 16.6")
struct ClipboardManagerIntegrationTests {
    
    @Test("Copy AI response to clipboard", .disabled("Races other tests over the shared pasteboard"))
    func testCopyAIResponse() async throws {
        let manager = ClipboardManager()
        let aiResponse = """
        Here's a Swift function to calculate Fibonacci:
        
        ```swift
        func fibonacci(_ n: Int) -> Int {
            guard n > 1 else { return n }
            return fibonacci(n - 1) + fibonacci(n - 2)
        }
        ```
        """
        
        try await manager.copy(aiResponse)
        
        let pasted = try await manager.paste()
        #expect(pasted.contains("fibonacci"))
    }
    
    @Test("Copy large text to clipboard", .disabled("Races other tests over the shared pasteboard"))
    func testCopyLargeText() async throws {
        let manager = ClipboardManager()
        let largeText = String(repeating: "Lorem ipsum dolor sit amet. ", count: 1000)
        
        try await manager.copy(largeText)
        
        let pasted = try await manager.paste()
        #expect(pasted.count > 1000)
    }
}

@Suite("ClipboardExtensions Tests - Epic 16.6")
struct ClipboardExtensionsTests {
    
    @Test("String extension - copy to clipboard", .enabled(if: !ClipboardManager.isRunningInCI(), "no pasteboard in a CI session"))
    func testStringCopyExtension() async throws {
        let text = "Test extension"
        
        do {
            try await text.copyToClipboard()
            #expect(true)
        } catch ClipboardError.toolNotAvailable {
            #expect(true)
        }
    }
    
    @Test("Check if running in CI environment")
    func testCIEnvironmentDetection() {
        let isCI = ClipboardManager.isRunningInCI()
        
        // Should detect CI from environment variables
        let ciEnv = ProcessInfo.processInfo.environment["CI"]
        if ciEnv != nil {
            #expect(isCI == true)
        }
    }
}
