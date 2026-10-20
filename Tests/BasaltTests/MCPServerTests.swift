import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

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
        #expect(!output[0].contains("playwright_screenshot"))
    }

    @Test("playwright_screenshot is gone: a url carrying Python never reaches an interpreter")
    func screenshotToolIsRemoved() async throws {
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine())
        actor OutputCapturer {
            var lines: [String] = []
            func capture(_ line: String) { lines.append(line) }
        }
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orch) { line in Task { await capturer.capture(line) } }
        let marker = NSTemporaryDirectory() + "basalt-pwned-\(UUID().uuidString)"
        let url = "'); import os; os.system('touch \(marker)')#"
        let call: [String: Any] = ["jsonrpc": "2.0", "id": 9, "method": "tools/call",
                                   "params": ["name": "playwright_screenshot", "arguments": ["url": url]]]
        await server.dispatch(String(decoding: try JSONSerialization.data(withJSONObject: call), as: UTF8.self))
        try? await Task.sleep(for: .milliseconds(200))
        let output = await capturer.lines
        #expect(output.count == 1)
        #expect(output[0].contains("-32601"))
        #expect(!FileManager.default.fileExists(atPath: marker))
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

    @Test("tools/call (status) with a modelId that is not loaded reports the mismatch")
    func testToolsCallStatusRequestedModelMismatch() async throws {
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine())
        try await orch.loadModel(path: "loaded-path")
        await orch.setModelId("org/loaded-model")

        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }
        await server.dispatch("""
        {"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {"name": "status", "arguments": {"modelId": "org/other-model"}}}
        """)
        try? await Task.sleep(for: .milliseconds(50))

        let output = await capturer.lines
        #expect(output.count == 1)
        let outer = try #require(try JSONSerialization.jsonObject(with: Data(output[0].utf8)) as? [String: Any])
        let content = try #require((outer["result"] as? [String: Any])?["content"] as? [[String: Any]])
        let text = try #require(content.first?["text"] as? String)
        let status = try #require(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        #expect(status["loadedModel"] as? String == "org/loaded-model")
        #expect(status["requestedModel"] as? String == "org/other-model")
        #expect(status["requestedModelLoaded"] as? Bool == false)
    }

    @Test("tools/call with arguments that are not an object, even after repair, is -32602")
    func testToolsCallBadArguments() async throws {
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine())
        for args in [#""not json at all""#, "[1, 2]", "42"] {
            let capturer = OutputCapturer()
            let server = MCPServer(orchestrator: orch) { line in
                Task { await capturer.capture(line) }
            }
            await server.dispatch(#"{"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "inspect_model", "arguments": "# + args + "}}")
            try? await Task.sleep(for: .milliseconds(50))
            let output = await capturer.lines
            #expect(output.count == 1)
            #expect(output.first?.contains("-32602") == true, "\(args): \(output)")
        }
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

    @Test("tools/call (switch_model) still loads a local path: MCP stdio is the same user (#72)")
    func testToolsCallSwitchModelLocalPath() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine())
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orch) { line in
            Task { await capturer.capture(line) }
        }
        await server.dispatch("""
        {"jsonrpc": "2.0", "id": 14, "method": "tools/call", "params": {"name": "switch_model", "arguments": {"modelId": "\(dir.path)"}}}
        """)
        try? await Task.sleep(for: .milliseconds(50))
        #expect(await orch.currentModelId == dir.path)
        #expect(await capturer.lines.first?.contains("Successfully switched") == true)
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

    @Test("tools/call (gemma_generate) passes minP and dynamicTemperature to the engine")
    func samplingArguments() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "test-model")
        let server = MCPServer(orchestrator: orch) { _ in }

        await server.dispatch("""
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "gemma_generate", "arguments": {"prompt": "Hello", "maxTokens": 10, "minP": 0.1, "dynamicTemperature": true}}}
        """)

        let sent = try #require(await engine.lastRequest)
        #expect(sent.minP == 0.1)
        #expect(sent.dynamicTemperature == true)
        #expect(sent.cacheKey == PromptCacheKey.shared)
    }

    private actor OutputCapturer {
        var lines: [String] = []
        func capture(_ line: String) { lines.append(line) }
    }
}
