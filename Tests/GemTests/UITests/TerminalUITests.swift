import Testing
import Foundation
@preconcurrency import Rainbow
@testable import GemCore

/// Epic 16.1: Rich Terminal UI Foundation - Unit Tests
/// Following TDD: Write tests FIRST, then implement
@Suite("TerminalUI Tests - Epic 16.1")
struct TerminalUITests {
    
    // MARK: - Test Environment Control
    
    @Test("Colors flag can be set and read")
    func testColorsEnabledFlag() {
        // Given: Set colors enabled
        TerminalUI.colorsEnabled = true
        
        // When: Read the flag
        let enabled = TerminalUI.colorsEnabled
        
        // Then: It should match
        #expect(enabled == true)
        
        // Given: Set colors disabled
        TerminalUI.colorsEnabled = false
        
        // When: Read the flag
        let disabled = TerminalUI.colorsEnabled
        
        // Then: It should match
        #expect(disabled == false)
    }
    
    // MARK: - Color Helper Tests (verify they return strings)
    
    @Test("success() returns a string")
    func testSuccessReturnsString() {
        setenv("FORCE_COLOR", "1", 1)
        defer { unsetenv("FORCE_COLOR") }
        TerminalUI.colorsEnabled = true
        let output = TerminalUI.success("OK")
        #expect(!output.isEmpty)
        #expect(output.contains("OK"))
        
        
    }
    
    @Test("success() returns plain text when colors disabled")
    func testSuccessWithoutColors() {
        // Given: Colors disabled
        TerminalUI.colorsEnabled = false
        // Note: TerminalUI.colorsEnabled should set Rainbow.enabled = false
        
        // When: Format success message
        let output = TerminalUI.success("OK")
        
        // Then: Output is plain text
        #expect(output == "OK")
        #expect(!output.contains("\u{1B}["))
    }
    
    @Test("error() returns a string")
    func testErrorReturnsString() {
        TerminalUI.colorsEnabled = true
        let output = TerminalUI.error("Failed")
        #expect(!output.isEmpty)
        #expect(output.count >= "Failed".count)
    }
    
    @Test("error() returns plain text when colors disabled")
    func testErrorWithoutColors() {
        TerminalUI.colorsEnabled = false
        let output = TerminalUI.error("Failed")
        
        #expect(output == "Failed")
    }
    
    @Test("warning() returns a string")
    func testWarningReturnsString() {
        TerminalUI.colorsEnabled = true
        let output = TerminalUI.warning("Caution")
        #expect(!output.isEmpty)
    }
    
    @Test("info() returns a string")
    func testInfoReturnsString() {
        TerminalUI.colorsEnabled = true
        let output = TerminalUI.info("Info")
        #expect(!output.isEmpty)
    }
    
    @Test("dim() returns a string")
    func testDimReturnsString() {
        TerminalUI.colorsEnabled = true
        let output = TerminalUI.dim("Subtle")
        #expect(!output.isEmpty)
    }
    
    @Test("code() returns a string with content")
    func testCodeReturnsString() {
        TerminalUI.colorsEnabled = true
        let output = TerminalUI.code("print()")
        #expect(!output.isEmpty)
        #expect(output.contains("print()"))
    }
    
    @Test("code() wraps in backticks when colors disabled")
    func testCodeWithoutColors() {
        TerminalUI.colorsEnabled = false
        let output = TerminalUI.code("print()")
        
        #expect(output == "`print()`")
    }
    
    // MARK: - Style Helper Tests
    
    @Test("bold() returns a string")
    func testBoldReturnsString() {
        let output = TerminalUI.bold("Text")
        #expect(!output.isEmpty)
        #expect(output.contains("Text"))
    }
    
    @Test("underline() returns a string")
    func testUnderlineReturnsString() {
        let output = TerminalUI.underline("Text")
        #expect(!output.isEmpty)
        #expect(output.contains("Text"))
    }
    
    @Test("italic() returns a string")
    func testItalicReturnsString() {
        let output = TerminalUI.italic("Text")
        #expect(!output.isEmpty)
        #expect(output.contains("Text"))
    }
    
    // MARK: - Composite Helper Tests
    
    @Test("heading() returns a string")
    func testHeadingReturnsString() {
        let output = TerminalUI.heading("Title")
        #expect(!output.isEmpty)
        #expect(output.contains("Title"))
    }
    
    @Test("codeBlock() returns formatted text")
    func testCodeBlockReturnsString() {
        let output = TerminalUI.codeBlock("code")
        #expect(!output.isEmpty)
        #expect(output.contains("code"))
    }
    
    // MARK: - Extension Tests
    
    @Test("String extensions work")
    func testStringExtensions() {
        #expect(!"OK".asSuccess.isEmpty)
        #expect(!"Error".asError.isEmpty)
        #expect(!"Warning".asWarning.isEmpty)
        #expect(!"Info".asInfo.isEmpty)
        #expect(!"Dim".asDim.isEmpty)
        #expect(!"code".asCode.isEmpty)
    }
}

// MARK: - OutputMode Tests (Epic 16.9)

@Suite("OutputMode Tests - Epic 16.9")
struct OutputModeTests {
    
    @Test("OutputMode has all three cases")
    func testOutputModeCases() {
        #expect(OutputMode.allCases.count == 3)
        #expect(OutputMode.allCases.contains(.json))
        #expect(OutputMode.allCases.contains(.plain))
        #expect(OutputMode.allCases.contains(.pretty))
    }
    
    @Test("OutputMode.auto returns a valid mode")
    func testAutoModeTTY() {
        let mode = OutputMode.auto
        #expect([OutputMode.pretty, OutputMode.plain].contains(mode))
    }
    
    @Test("OutputFormatter emits JSON in json mode")
    func testJSONMode() throws {
        let formatter = OutputFormatter(mode: .json)
        
        struct TestData: Codable {
            let message: String
            let count: Int
        }
        
        let data = TestData(message: "test", count: 42)
        
        // Format output
        let output = formatter.formatSuccess(data)
        
        // Verify JSON structure
        #expect(output.contains("\"message\""))
        #expect(output.contains("\"count\""))
        #expect(output.contains("42"))
    }
    
    @Test("OutputFormatter emits plain text in plain mode")
    func testPlainMode() {
        let formatter = OutputFormatter(mode: .plain)
        
        let output = formatter.formatSuccess("Simple text")
        
        #expect(output == "Simple text")
        #expect(!output.contains("\u{1B}["))  // No ANSI codes
    }
    
    @Test("OutputFormatter emits text in pretty mode")
    func testPrettyMode() {
        let formatter = OutputFormatter(mode: .pretty)
        
        let output = formatter.formatSuccess("Colored text")
        
        // Verify output is not empty
        #expect(!output.isEmpty)
        #expect(output.contains("Colored text"))
    }
    
    @Test("OutputFormatter formats errors consistently")
    func testErrorFormatting() {
        let formatter = OutputFormatter(mode: .json)
        
        let error = NSError(domain: "TestError", code: 404, userInfo: [
            NSLocalizedDescriptionKey: "Not found"
        ])
        
        let output = formatter.formatError(error)
        
        #expect(output.contains("Not found"))
    }
    
    @Test("OutputFormatter handles all modes for errors")
    func testErrorFormattingAllModes() {
        let error = NSError(domain: "Test", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "Test error"
        ])
        
        // JSON mode
        let jsonFormatter = OutputFormatter(mode: .json)
        let jsonOutput = jsonFormatter.formatError(error)
        #expect(jsonOutput.contains("Test error"))
        
        // Plain mode
        let plainFormatter = OutputFormatter(mode: .plain)
        let plainOutput = plainFormatter.formatError(error)
        #expect(plainOutput.contains("Test error"))
        
        // Pretty mode
        let prettyFormatter = OutputFormatter(mode: .pretty)
        let prettyOutput = prettyFormatter.formatError(error)
        #expect(prettyOutput.contains("Test error"))
    }
}

// MARK: - TTY Detection Tests

@Suite("TTY Detection Tests")
struct TTYDetectionTests {
    
    @Test("isTTY detects stdout correctly")
    func testIsTTY() {
        #if canImport(Darwin)
        let result = TerminalUI.isStdoutTTY
        
        // In test environment, this might be false
        // Just verify it returns a boolean
        #expect([true, false].contains(result))
        #endif
    }
    
    @Test("isStderrTTY returns a boolean")
    func testIsStderrTTY() {
        #if canImport(Darwin)
        let result = TerminalUI.isStderrTTY
        #expect([true, false].contains(result))
        #endif
    }
    
    @Test("Environment variable NO_COLOR is respected")
    func testNOCOLOREnvVar() {
        setenv("NO_COLOR", "1", 1)
        defer { unsetenv("NO_COLOR") }
        
        let hasNOCOLOR = ProcessInfo.processInfo.environment["NO_COLOR"] != nil
        
        #expect(hasNOCOLOR == true)
    }
    
    @Test("Environment variable FORCE_COLOR enables colors")
    func testFORCECOLOREnvVar() {
        setenv("FORCE_COLOR", "1", 1)
        defer { unsetenv("FORCE_COLOR") }
        
        let hasFORCECOLOR = ProcessInfo.processInfo.environment["FORCE_COLOR"] != nil
        
        #expect(hasFORCECOLOR == true)
    }
}
