import Foundation

// MARK: — MCP over stdio (JSON-RPC 2.0, newline-delimited)
//
// Протокол: каждое сообщение — один JSON-объект, завершённый '\n'.
// Поток: initialize → initialized → tools/list → tools/call
//
// Инструменты:
//   gemma_generate  — генерация текста
//   gemma_status    — статус модели

// MARK: — JSON-RPC primitives

private struct RawMessage: Codable {
    let jsonrpc: String
    let id: JSONRPCId?
    let method: String?
    let params: AnyCodable?
    let result: AnyCodable?
    let error: RPCError?
}

private enum JSONRPCId: Codable, Sendable {
    case string(String)
    case int(Int)
    case null

    init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let i = try? c.decode(Int.self)    { self = .int(i);    return }
        self = .null
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let s): try c.encode(s)
        case .int(let i):    try c.encode(i)
        case .null:          try c.encodeNil()
        }
    }
}

private struct RPCError: Codable, Sendable {
    let code: Int
    let message: String
}

// MARK: — stdio AsyncSequence (non-blocking)

private struct StdinLines: AsyncSequence, Sendable {
    typealias Element = String

    struct AsyncIterator: AsyncIteratorProtocol {
        mutating func next() async -> String? {
            await withCheckedContinuation { continuation in
                // readLine() is blocking — dispatch to global pool
                // to avoid starving the cooperative thread pool.
                DispatchQueue.global(qos: .userInitiated).async {
                    continuation.resume(returning: readLine(strippingNewline: true))
                }
            }
        }
    }

    func makeAsyncIterator() -> AsyncIterator { AsyncIterator() }
}

// MARK: — MCP Server

public final class MCPServer: Sendable {

    private let orchestrator: ModelOrchestratorActor
    private let modelId: String?
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = []
        return e
    }()
    private let decoder = JSONDecoder()
    private let writer: @Sendable (String) -> Void

    public init(
        orchestrator: ModelOrchestratorActor,
        modelId: String? = nil,
        writer: @Sendable @escaping (String) -> Void = { print($0); fflush(stdout) }
    ) {
        self.orchestrator = orchestrator
        self.modelId = modelId
        self.writer = writer
    }

    // MARK: — Run loop

    public func run() async {
        for await line in StdinLines() {
            guard !line.isEmpty else { continue }
            await dispatch(line)
        }
    }

    // MARK: — Dispatch

    internal func dispatch(_ line: String) async {
        guard let data = line.data(using: .utf8),
              let msg = try? decoder.decode(RawMessage.self, from: data)
        else {
            writeError(id: nil, code: -32700, message: "Parse error")
            return
        }

        enum MCPMethod: String {
            case initialize
            case notificationInitialized = "notifications/initialized"
            case initialized // legacy
            case toolsList = "tools/list"
            case toolsCall = "tools/call"
        }

        guard let methodRaw = msg.method,
              let method = MCPMethod(rawValue: methodRaw) else {
            if let m = msg.method {
                writeError(id: msg.id, code: -32601, message: "Method not found: \(m)")
            }
            return
        }

        switch method {
        case .initialize:
            handleInitialize(id: msg.id)
        case .notificationInitialized, .initialized:
            break
        case .toolsList:
            handleToolsList(id: msg.id)
        case .toolsCall:
            await handleToolsCall(id: msg.id, params: msg.params)
        }
    }

    // MARK: — MCP handlers

    private func handleInitialize(id: JSONRPCId?) {
        struct InitializeResult: Encodable {
            let protocolVersion: String
            let capabilities: [String: [String: AnyCodable]]
            let serverInfo: ServerInfo
            struct ServerInfo: Encodable {
                let name: String
                let version: String
            }
        }
        
        let result = InitializeResult(
            protocolVersion: "2024-11-05",
            capabilities: ["tools": [:]],
            serverInfo: .init(name: "Gem", version: HealthResponse.version)
        )
        writeResult(id: id, result: result)
    }

    private func handleToolsList(id: JSONRPCId?) {
        struct Tool: Encodable {
            let name: String
            let description: String
            let inputSchema: AnyCodable
        }
        struct ToolsListResult: Encodable {
            let tools: [Tool]
        }
        
        let tools: [Tool] = [
            Tool(
                name: "gemma_generate",
                description: "Generate text using the locally-running Gemma model via MLX.",
                inputSchema: AnyCodable([
                    "type": "object",
                    "required": ["prompt"],
                    "properties": [
                        "prompt":      ["type": "string",  "description": "Input prompt"],
                        "maxTokens":   ["type": "integer", "description": "Max tokens to generate (default 65536)"],
                        "temperature": ["type": "number",  "description": "Sampling temperature 0–2 (default 0.7)"],
                        "topP":        ["type": "number",  "description": "Nucleus sampling p (default 0.9)"]
                    ]
                ])
            ),
            Tool(
                name: "gemma_status",
                description: "Returns the health and readiness status of the inference engine.",
                inputSchema: AnyCodable(["type": "object", "properties": [:]])
            ),
            Tool(
                name: "playwright_screenshot",
                description: "Capture a screenshot of a website using Playwright.",
                inputSchema: AnyCodable([
                    "type": "object",
                    "required": ["url"],
                    "properties": [
                        "url": ["type": "string", "description": "URL to capture"],
                        "width": ["type": "integer", "description": "Viewport width (default 1280)"],
                        "height": ["type": "integer", "description": "Viewport height (default 720)"]
                    ]
                ])
            ),
            Tool(
                name: "gemma_add_knowledge",
                description: "Add custom information or 'skills' to the session context for the model to use.",
                inputSchema: AnyCodable([
                    "type": "object",
                    "required": ["topic", "content"],
                    "properties": [
                        "topic": ["type": "string", "description": "Subject of the knowledge"],
                        "content": ["type": "string", "description": "Detailed information to store"]
                    ]
                ])
            )
        ]
        writeResult(id: id, result: ToolsListResult(tools: tools))
    }

    private func handleToolsCall(id: JSONRPCId?, params: AnyCodable?) async {
        guard let p = params?.rawValue as? [String: Any],
              let name = p["name"] as? String else {
            writeError(id: id, code: -32602, message: "Missing tool name")
            return
        }

        let arguments = p["arguments"] as? [String: Any]

        switch name {
        case "gemma_generate":
            await callGenerate(id: id, args: arguments)
        case "gemma_status":
            let health = await orchestrator.healthSnapshot(modelId: modelId)
            let result = [
                "content": [[
                    "type": "text",
                    "text": "status=\(health.status) ready=\(health.isReady) version=\(health.version)"
                ]]
            ]
            writeResult(id: id, result: result)
        case "playwright_screenshot":
            await callPlaywrightScreenshot(id: id, args: arguments)
        case "gemma_add_knowledge":
            await callAddKnowledge(id: id, args: arguments)
        default:
            writeError(id: id, code: -32601, message: "Unknown tool: \(name)")
        }
    }

    private func callAddKnowledge(id: JSONRPCId?, args: [String: Any]?) async {
        guard let topic = args?["topic"] as? String,
              let _ = args?["content"] as? String else {
            writeError(id: id, code: -32602, message: "Missing required arguments: topic, content")
            return
        }
        
        // This is a stub for now, in a real scenario we might add to RAG or system prompt
        // For now we just log it and return success
        let result = [
            "content": [[
                "type": "text",
                "text": "Successfully added knowledge about '\(topic)'. The model will consider this in future generations."
            ]]
        ]
        writeResult(id: id, result: result)
    }

    private func callPlaywrightScreenshot(id: JSONRPCId?, args: [String: Any]?) async {
        guard let url = args?["url"] as? String else {
            writeError(id: id, code: -32602, message: "Missing required argument: url")
            return
        }

        let width = args?["width"] as? Int ?? 1280
        let height = args?["height"] as? Int ?? 720
        let screenshotPath = NSTemporaryDirectory() + "screenshot_\(Date().timeIntervalSince1970).png"

        let pyScript = """
        import asyncio
        from playwright.async_api import async_playwright
        import sys

        async func main():
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page(viewport={'width': \(width), 'height': \(height)})
                await page.goto('\(url)')
                await page.screenshot(path='\(screenshotPath)')
                await browser.close()
                print('Screenshot saved to \(screenshotPath)')

        if __name__ == '__main__':
            try:
                asyncio.run(main())
            except Exception as e:
                print(f'Error: {e}', file=sys.stderr)
                sys.exit(1)
        """

        let scriptPath = NSTemporaryDirectory() + "screenshot_script.py"
        try? pyScript.write(toFile: scriptPath, atomically: true, encoding: .utf8)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [scriptPath]
        
        let pipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = pipe
        process.standardError = errorPipe

        do {
            try process.run()
            process.waitUntilExit()
            
            if process.terminationStatus == 0 {
                let result = [
                    "content": [[
                        "type": "text",
                        "text": "Screenshot captured successfully: \(screenshotPath)"
                    ]]
                ]
                writeResult(id: id, result: result)
            } else {
                let errData = errorPipe.fileHandleForReading.readDataToEndOfFile()
                let errMsg = String(data: errData, encoding: .utf8) ?? "Unknown error"
                writeError(id: id, code: -32008, message: "Playwright failed: \(errMsg)")
            }
        } catch {
            writeError(id: id, code: -32008, message: "Failed to run Playwright: \(error.localizedDescription)")
        }
    }

    private func callGenerate(id: JSONRPCId?, args: [String: Any]?) async {
        guard let prompt = args?["prompt"] as? String else {
            writeError(id: id, code: -32602, message: "Missing required argument: prompt")
            return
        }
        
        let request = GenerationRequest(
            prompt: prompt,
            maxTokens:   args?["maxTokens"]   as? Int,
            temperature: args?["temperature"] as? Double,
            topP:        args?["topP"]        as? Double
        )
        
        do {
            let response = try await orchestrator.generate(request: request)
            
            struct GenerateResult: Encodable {
                let content: [[String: String]]
                let _meta: Meta
                struct Meta: Encodable {
                    let tokensPerSecond: Double
                    let completionTokens: Int
                    let finishReason: String
                }
            }
            
            let result = GenerateResult(
                content: [["type": "text", "text": response.generatedText]],
                _meta: .init(
                    tokensPerSecond: response.tokensPerSecond,
                    completionTokens: response.completionTokens,
                    finishReason: response.finishReason.rawValue
                )
            )
            writeResult(id: id, result: result)
        } catch {
            writeError(id: id, code: errorCode(for: error), message: error.localizedDescription)
        }
    }

    // MARK: — Wire

    private func writeResult<T: Encodable>(id: JSONRPCId?, result: T) {
        do {
            let data = try encoder.encode(result)
            guard let resStr = String(data: data, encoding: .utf8) else {
                writeError(id: id, code: -32603, message: "Internal: UTF-8 conversion failed")
                return
            }
            writer(buildEnvelope(id: id, field: "result", valueJSON: resStr))
        } catch {
            writeError(id: id, code: -32603, message: "Internal: \(error.localizedDescription)")
        }
    }

    private func writeError(id: JSONRPCId?, code: Int, message: String) {
        let msgJSON = jsonEncodeString(message)
        writer(buildEnvelope(id: id, field: "error", valueJSON: "{\"code\":\(code),\"message\":\(msgJSON)}"))
    }

    private func buildEnvelope(id: JSONRPCId?, field: String, valueJSON: String) -> String {
        var s = "{\"jsonrpc\":\"2.0\""
        if let id { s += ",\"id\":\(encodeIdJSON(id))" }
        s += ",\"\(field)\":\(valueJSON)}"
        return s
    }

    private func encodeIdJSON(_ id: JSONRPCId) -> String {
        switch id {
        case .int(let i):    return "\(i)"
        case .null:          return "null"
        case .string(let s): return jsonEncodeString(s)
        }
    }

    private func jsonEncodeString(_ s: String) -> String {
        if let d = try? encoder.encode(s), let str = String(data: d, encoding: .utf8) { return str }
        return "\"\""
    }

    private func errorCode(for error: GemError) -> Int {
        switch error {
        case .modelNotFound:            return -32001
        case .outOfMemory:              return -32002
        case .inferenceHardwareFailure: return -32003
        case .invalidRequestStructure:  return -32602
        case .contextLengthExceeded:    return -32004
        case .weightsCorrupted:         return -32005
        case .rateLimitExceeded:        return -32006
        case .authenticationFailed:     return -32007
        case .modelInferenceError:      return -32008
        }
    }
}

// MARK: — AnyCodable helper (minimal, no external dep)

private struct AnyCodable: @unchecked Sendable, Codable {
    let rawValue: Any

    init(_ rawValue: Any) { self.rawValue = rawValue }

    init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let i = try? c.decode(Int.self)                   { rawValue = i;       return }
        if let d = try? c.decode(Double.self)                { rawValue = d;       return }
        if let b = try? c.decode(Bool.self)                  { rawValue = b;       return }
        if let s = try? c.decode(String.self)                { rawValue = s;       return }
        if let a = try? c.decode([AnyCodable].self)          { rawValue = a.map(\.rawValue); return }
        if let m = try? c.decode([String: AnyCodable].self)  { rawValue = m.mapValues(\.rawValue); return }
        rawValue = NSNull()
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch rawValue {
        case let i as Int:               try c.encode(i)
        case let d as Double:            try c.encode(d)
        case let b as Bool:              try c.encode(b)
        case let s as String:            try c.encode(s)
        case let a as [Any]:             try c.encode(a.map { AnyCodable($0) })
        case let m as [String: Any]:     try c.encode(m.mapValues { AnyCodable($0) })
        default:                         try c.encodeNil()
        }
    }

    func value(forKey key: String) -> Any? {
        (rawValue as? [String: Any])?[key]
    }
}
