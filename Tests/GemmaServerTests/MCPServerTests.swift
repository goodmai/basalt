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
}
