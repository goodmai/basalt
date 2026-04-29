import Testing
import Foundation
@testable import GemCore

/// Tests for Spinner and TerminalStatus utilities
@Suite("Spinner Tests")
struct SpinnerTests {
    
    @Test("Spinner can be created with default style")
    func testSpinnerCreation() async {
        let spinner = Spinner()
        // Just verify it initializes without crashing
        #expect(true)
    }
    
    @Test("Spinner can be created with custom style and message")
    func testSpinnerWithCustomStyle() async {
        let spinner = Spinner(style: .dots, message: "Loading...")
        // Verify initialization
        #expect(true)
    }
    
    @Test("Spinner styles have correct frame counts")
    func testSpinnerStyleFrames() {
        #expect(Spinner.Style.dots.frames.count == 10)
        #expect(Spinner.Style.line.frames.count == 4)
        #expect(Spinner.Style.pulse.frames.count == 6)
        #expect(Spinner.Style.bounce.frames.count == 6)
    }
    
    @Test("Spinner can start and stop without crashing")
    func testSpinnerStartStop() async {
        let spinner = Spinner(style: .line, message: "Test")
        
        // Start spinner
        await spinner.start()
        
        // Wait a bit
        try? await Task.sleep(for: .milliseconds(100))
        
        // Stop spinner
        await spinner.stop()
        
        #expect(true) // If we got here, no crash
    }
}

@Suite("LoadingState Tests")
struct LoadingStateTests {
    
    @Test("LoadingState has correct descriptions")
    func testLoadingStateDescriptions() {
        #expect(LoadingState.initializing.description == "Initializing engine...")
        #expect(LoadingState.thinking.description == "Thinking...")
        #expect(LoadingState.generating.description == "Generating tokens...")
    }
    
    @Test("LoadingState coloredDescription returns non-empty string")
    func testLoadingStateColored() {
        let colored = LoadingState.thinking.coloredDescription
        #expect(!colored.isEmpty)
        #expect(colored.contains("Thinking"))
    }
    
    @Test("All LoadingState cases are testable")
    func testAllLoadingStates() {
        let allStates: [LoadingState] = [
            .initializing,
            .readingFiles,
            .loadingMCP,
            .analyzingSkills,
            .thinking,
            .generating,
            .finalizing
        ]
        
        for state in allStates {
            #expect(!state.description.isEmpty)
            #expect(!state.coloredDescription.isEmpty)
        }
    }
}

@Suite("ContextStats Tests")
struct ContextStatsTests {
    
    @Test("ContextStats with no data has no context")
    func testEmptyContextStats() {
        let stats = ContextStats()
        
        #expect(stats.files == 0)
        #expect(stats.systemPrompts == 0)
        #expect(stats.mcpServers == 0)
        #expect(stats.skills == 0)
        #expect(!stats.hasContext)
    }
    
    @Test("ContextStats with files has context")
    func testContextStatsWithFiles() {
        let stats = ContextStats(files: 3)
        
        #expect(stats.files == 3)
        #expect(stats.hasContext)
        #expect(stats.description.contains("3 file(s)"))
    }
    
    @Test("ContextStats with multiple values")
    func testContextStatsMultiple() {
        let stats = ContextStats(
            files: 2,
            systemPrompts: 1,
            mcpServers: 3,
            skills: 5
        )
        
        #expect(stats.hasContext)
        #expect(stats.files == 2)
        #expect(stats.systemPrompts == 1)
        #expect(stats.mcpServers == 3)
        #expect(stats.skills == 5)
        
        let desc = stats.description
        #expect(desc.contains("2 file(s)"))
        #expect(desc.contains("1 prompt(s)"))
        #expect(desc.contains("3 MCP server(s)"))
        #expect(desc.contains("5 skill(s)"))
    }
    
    @Test("ContextStats colored description is formatted")
    func testContextStatsColoredDescription() {
        let stats = ContextStats(files: 1, mcpServers: 2)
        
        let colored = stats.coloredDescription
        #expect(!colored.isEmpty)
    }
    
    @Test("ContextStats empty description")
    func testContextStatsEmptyDescription() {
        let stats = ContextStats()
        
        #expect(stats.description == "No context")
    }
}

@Suite("TerminalUI Actor Tests")
struct TerminalUIActorTests {
    
    @Test("Configuration actor can be accessed")
    func testConfigurationActor() async {
        let config = TerminalUI.Configuration.shared
        let enabled = await config.colorsEnabled
        
        // Just verify it returns a boolean
        #expect([true, false].contains(enabled))
    }
    
    @Test("Configuration actor can set colors")
    func testConfigurationSetColors() async {
        let config = TerminalUI.Configuration.shared
        
        await config.setColorsEnabled(false)
        let disabled = await config.colorsEnabled
        #expect(disabled == false)
        
        await config.setColorsEnabled(true)
        let enabled = await config.colorsEnabled
        #expect(enabled == true)
    }
}
