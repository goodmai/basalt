import Testing
import Foundation
@testable import GemmaServerCore

@Suite("MCPServer logic")
struct MCPServerTests {

    @Test("initialize returns server info")
    func testInitialize() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        
        actor OutputCapturer {
            var lines: [String] = []
            func capture(_ line: String) { lines.append(line) }
        }
        let capturer = OutputCapturer()
        
        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }
        
        let request = """
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        """
        
        await server.dispatch(request)
        
        // Wait a bit for the Task in the writer closure to finish
        try? await Task.sleep(for: .milliseconds(50))
        
        let output = await capturer.lines
        #expect(output.count == 1)
        #expect(output[0].contains("\"jsonrpc\":\"2.0\""))
        #expect(output[0].contains("\"result\":"))
        #expect(output[0].contains("GemmaServer"))
    }

    @Test("tools/list returns available tools")
    func testToolsList() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        
        actor OutputCapturer {
            var lines: [String] = []
            func capture(_ line: String) { lines.append(line) }
        }
        let capturer = OutputCapturer()
        
        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }
        
        let request = """
        {"jsonrpc": "2.0", "id": "req-2", "method": "tools/list", "params": {}}
        """
        
        await server.dispatch(request)
        try? await Task.sleep(for: .milliseconds(50))
        
        let output = await capturer.lines
        #expect(output.count == 1)
        #expect(output[0].contains("gemma_generate"))
        #expect(output[0].contains("gemma_status"))
    }

    @Test("tools/call (gemma_status) returns model info")
    func testToolsCallStatus() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        
        actor OutputCapturer {
            var lines: [String] = []
            func capture(_ line: String) { lines.append(line) }
        }
        let capturer = OutputCapturer()
        
        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }
        
        let request = """
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "gemma_status", "arguments": {}}}
        """
        
        await server.dispatch(request)
        try? await Task.sleep(for: .milliseconds(50))
        
        let output = await capturer.lines
        #expect(output.count == 1)
        #expect(output[0].contains("status=initializing")) // because it's not loaded in mock yet
    }

    // MARK: — Fuzzing / Error cases (Task 4.1)

    @Test("Invalid JSON returns parse error")
    func testInvalidJSON() async throws {
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine())) { line in
            Task { await capturer.capture(line) }
        }
        
        await server.dispatch("not a json")
        try? await Task.sleep(for: .milliseconds(50))
        
        let output = await capturer.lines
        #expect(output.count == 1)
        #expect(output[0].contains("-32700")) // Parse error code
    }

    @Test("Unknown method returns method not found")
    func testUnknownMethod() async throws {
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine())) { line in
            Task { await capturer.capture(line) }
        }
        
        await server.dispatch("""
        {"jsonrpc": "2.0", "id": 99, "method": "unknown/method", "params": {}}
        """)
        try? await Task.sleep(for: .milliseconds(50))
        
        let output = await capturer.lines
        #expect(output.count == 1)
        #expect(output[0].contains("-32601")) // Method not found
    }

    @Test("Missing tool name returns error")
    func testMissingToolName() async throws {
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine())) { line in
            Task { await capturer.capture(line) }
        }
        
        await server.dispatch("""
        {"jsonrpc": "2.0", "id": 100, "method": "tools/call", "params": {"arguments": {}}}
        """)
        try? await Task.sleep(for: .milliseconds(50))
        
        let output = await capturer.lines
        #expect(output[0].contains("-32602")) // Invalid params
    }

    private actor OutputCapturer {
        var lines: [String] = []
        func capture(_ line: String) { lines.append(line) }
    }
}
