import Testing
import Foundation
@testable import GemCore

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
        #expect(output[0].contains("Gem"))
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
        #expect(output[0].contains("\"name\":\"status\""))
        #expect(output[0].contains("check_hardware"))
        #expect(output[0].contains("inspect_model"))
        #expect(output[0].contains("switch_model"))
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

    @Test("tools/call (status) returns structured json health")
    func testToolsCallStatusJSON() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }
        
        let request = """
        {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "status", "arguments": {}}}
        """
        
        await server.dispatch(request)
        try? await Task.sleep(for: .milliseconds(50))
        
        let output = await capturer.lines
        #expect(output.count == 1)
        #expect(output[0].contains("status"))
        #expect(output[0].contains("ready"))
        #expect(output[0].contains("diagnostics"))
    }

    @Test("tools/call (check_hardware) returns system profiler report")
    func testToolsCallCheckHardware() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }
        
        let request = """
        {"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {"name": "check_hardware", "arguments": {}}}
        """
        
        await server.dispatch(request)
        try? await Task.sleep(for: .milliseconds(50))
        
        let output = await capturer.lines
        #expect(output.count == 1)
        #expect(output[0].contains("totalRAMGB"))
        #expect(output[0].contains("availableRAMBytes"))
        #expect(output[0].contains("chip"))
    }

    @Test("tools/call (inspect_model) verifies shard integrity")
    func testToolsCallInspectModel() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }

        // Create temporary valid dummy model dir
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let configData = """
        {"quantization": {"bits": 4, "group_size": 64, "mode": "optiq"}, "model_type": "qwen2"}
        """.data(using: .utf8)!
        try configData.write(to: tempDir.appendingPathComponent("config.json"))

        // Create dummy safetensors file with 8-byte header len
        let headerJSON = "{\"weight\": {\"dtype\": \"F16\", \"shape\": [2, 2], \"data_offsets\": [0, 8]}}"
        let headerData = headerJSON.data(using: .utf8)!
        var dummyData = Data()
        var headerLen = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLen) { dummyData.append(contentsOf: $0) }
        dummyData.append(headerData)
        dummyData.append(Data(repeating: 0, count: 8))
        try dummyData.write(to: tempDir.appendingPathComponent("model.safetensors"))

        let request = """
        {"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": {"name": "inspect_model", "arguments": {"modelId": "\(tempDir.path)"}}}
        """
        
        await server.dispatch(request)
        try? await Task.sleep(for: .milliseconds(100))
        
        let output = await capturer.lines
        #expect(output.count == 1)
        #expect(output[0].contains("intact"))
        #expect(output[0].contains("detectedBitWidth"))
        #expect(output[0].contains("optiq"))
    }

    @Test("tools/call (switch_model) fails with error code when model is not cached")
    func testToolsCallSwitchModelNotCached() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }
        
        let request = """
        {"jsonrpc": "2.0", "id": 13, "method": "tools/call", "params": {"name": "switch_model", "arguments": {"modelId": "non-existent-org/uncached-model"}}}
        """
        
        await server.dispatch(request)
        try? await Task.sleep(for: .milliseconds(50))
        
        let output = await capturer.lines
        #expect(output.count == 1)
        #expect(output[0].contains("-32010")) // modelNotCached code
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

    // MARK: — TC-1.2.2.5: Streaming response handling

    @Test("tools/call (gemma_generate) handles streaming responses")
    func testStreamingResponse() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "test-model")

        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }

        let request = """
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "gemma_generate", "arguments": {"prompt": "Hello", "maxTokens": 10}}}
        """

        await server.dispatch(request)
        try? await Task.sleep(for: .milliseconds(100))

        let output = await capturer.lines
        #expect(output.count >= 1)
        #expect(output[0].contains("\"result\""))
        #expect(output[0].contains("content"))
    }

    private actor OutputCapturer {
        var lines: [String] = []
        func capture(_ line: String) { lines.append(line) }
    }
}
