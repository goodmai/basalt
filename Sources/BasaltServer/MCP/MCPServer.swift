import Foundation
import BasaltCore
import BasaltHarness

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
    private let library: HarnessExperienceLibrary?
    private let harnessRoots: [URL]
    private let harnessLimits: HarnessLoadLimits
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys]  // stable output: AGENTS.md / AI_SETUP.md show sorted keys
        return e
    }()
    private let decoder = JSONDecoder()
    private let writer: @Sendable (String) -> Void

    public init(
        orchestrator: ModelOrchestratorActor,
        modelId: String? = nil,
        library: HarnessExperienceLibrary? = nil,
        harnessRoots: [URL] = HarnessLoader.allowedRoots(),
        harnessLimits: HarnessLoadLimits = .standard,
        writer: @Sendable @escaping (String) -> Void = { print($0); fflush(stdout) }
    ) {
        self.orchestrator = orchestrator
        self.modelId = modelId
        self.library = library
        self.harnessRoots = harnessRoots
        self.harnessLimits = harnessLimits
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
                name: "status",
                description: "Returns the health, loaded model ID, memory snapshot, and readiness status of the inference engine.",
                inputSchema: AnyCodable([
                    "type": "object",
                    "properties": [
                        "modelId": ["type": "string", "description": "Optional model repo ID to check"]
                    ]
                ])
            ),
            Tool(
                name: "check_hardware",
                description: "Detects Apple Silicon hardware resources (chip, available unified memory, GPU, disk) without downloading models.",
                inputSchema: AnyCodable(["type": "object", "properties": [:]])
            ),
            Tool(
                name: "inspect_model",
                description: "Verifies weight shard integrity and detects the actual bit width of MLX/Safetensors weights.",
                inputSchema: AnyCodable([
                    "type": "object",
                    "required": ["modelId"],
                    "properties": [
                        "modelId": ["type": "string", "description": "Hugging Face repo ID or local path to inspect"]
                    ]
                ])
            ),
            Tool(
                name: "switch_model",
                description: "Switches the currently loaded model to a different model in the local cache.",
                inputSchema: AnyCodable([
                    "type": "object",
                    "required": ["modelId"],
                    "properties": [
                        "modelId": ["type": "string", "description": "Model ID in local cache to switch to"]
                    ]
                ])
            ),
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
                        "topP":        ["type": "number",  "description": "Nucleus sampling p (default 0.9)"],
                        "minP":        ["type": "number",  "description": "Min-P 0–1, overrides the server default (0 = off)"],
                        "dynamicTemperature": ["type": "boolean", "description": "Entropy-based dynamic temperature (EDT)"]
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
                description: "Store a lesson in the persistent Harness Experience Library (~/.basalt/harness-library/<domain>/): EXPERIENCE.md, entries.json and the domain INDEX.md that future harness_optimize proposers read. Limits: topic 200 characters, cue 500, domain 64, content 32 KiB, 1000 entries.",
                inputSchema: AnyCodable([
                    "type": "object",
                    "required": ["topic", "content"],
                    "properties": [
                        "topic": ["type": "string", "description": "Subject of the knowledge"],
                        "content": ["type": "string", "description": "Detailed information to store"],
                        "domain": ["type": "string", "description": "Optional domain bucket in the Experience Library"],
                        "cue": ["type": "string", "description": "Optional trigger cue for retrieving this lesson"],
                        "confidence": ["type": "string", "enum": ["low", "medium", "high"], "description": "Confidence level (default medium)"]
                    ]
                ])
            ),
            Tool(
                name: "harness_validate",
                description: "Validate an agent harness: a Claude Code (CLAUDE.md, .claude/agents, .claude/skills, .claude/settings.json hooks), opencode (AGENTS.md, .opencode/agent, .opencode/skill, .opencode/plugin) or AGENTS.md + .agents/skills layout. Reports the always-on budget (instruction files and their @imports), skill and agent frontmatter, hook events, missing script references and untriggered skills. A directory must lie under the server's working directory or a root listed in BASALT_HARNESS_ROOTS (colon-separated); loading is capped at 2000 files, 256 KiB per file, 16 MiB in total and 8 levels deep.",
                inputSchema: AnyCodable([
                    "type": "object",
                    "properties": [
                        "directory": ["type": "string", "description": "Repository root that contains the harness (e.g. the folder holding CLAUDE.md and .claude/)"],
                        "files": [
                            "type": "object",
                            "description": "Alternative to directory: the harness as {\"path relative to the repository root\": \"file content\"}",
                            "additionalProperties": ["type": "string"]
                        ],
                        "surface": ["type": "string", "enum": HarnessSurfaceSpec.all.map(\.name), "description": "Harness layout (default claude)"],
                        "alwaysOnMaxChars": ["type": "integer", "description": "Budget for the always-on files in characters (default 8000)"]
                    ]
                ])
            ),
            Tool(
                name: "harness_library_query",
                description: "Query stored lessons and component attributions from the Harness Experience Library.",
                inputSchema: AnyCodable([
                    "type": "object",
                    "properties": [
                        "query": ["type": "string", "description": "Optional substring query for topic, cue, or content"],
                        "domain": ["type": "string", "description": "Optional domain filter"]
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

        var arguments = p["arguments"] as? [String: Any]
        if arguments == nil, let raw = p["arguments"], !(raw is NSNull), !(raw is String) {
            writeError(id: id, code: -32602, message: "Tool arguments must be a JSON object")
            return
        }
        if arguments == nil, let rawArgs = p["arguments"] as? String {
            guard let repaired = ToolCallRepairer.repairJSON(rawArgs),
                  let dict = try? JSONSerialization.jsonObject(with: Data(repaired.utf8)) as? [String: Any] else {
                // Running the tool without its arguments would be a guess; say what is wrong.
                writeError(id: id, code: -32602, message: "Tool arguments are not a JSON object and could not be repaired")
                return
            }
            arguments = dict
        }

        switch name {
        case "status":
            await callStatus(id: id, args: arguments)
        case "check_hardware":
            await callCheckHardware(id: id)
        case "inspect_model":
            await callInspectModel(id: id, args: arguments)
        case "switch_model":
            await callSwitchModel(id: id, args: arguments)
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
        case "harness_validate":
            callHarnessValidate(id: id, args: arguments)
        case "harness_library_query":
            await callHarnessLibraryQuery(id: id, args: arguments)
        default:
            writeError(id: id, code: -32601, message: "Unknown tool: \(name)")
        }
    }

    private func callStatus(id: JSONRPCId?, args: [String: Any]?) async {
        // Health describes what is actually loaded; a requested id is only compared against it,
        // so `status {modelId}` can tell an agent the model it wants is not the one serving.
        let loaded = await orchestrator.currentModelId
        let requestedId = args?["modelId"] as? String
        let health = await orchestrator.healthSnapshot(modelId: loaded.isEmpty ? modelId : loaded)
        // One snapshot feeds both blocks, so diagnostics and telemetry cannot disagree
        // (the orchestrator's own counters skip streaming tokens).
        let telemetry = await orchestrator.telemetrySnapshot()
        
        struct StatusOutput: Encodable {
            let status: String
            let ready: Bool
            let version: String
            let loadedModel: String
            let requestedModel: String?
            let requestedModelLoaded: Bool?
            let diagnostics: Diagnostics
            let telemetry: ServerTelemetrySnapshot
            
            struct Diagnostics: Encodable {
                let requestCount: Int
                let totalTokensGenerated: Int
            }
        }
        
        let statusObj = StatusOutput(
            status: health.status,
            ready: health.isReady,
            version: health.version,
            loadedModel: loaded.isEmpty ? "none" : loaded,
            requestedModel: requestedId,
            requestedModelLoaded: requestedId.map { $0 == loaded },
            diagnostics: .init(requestCount: telemetry.totalRequests, totalTokensGenerated: telemetry.totalTokensGenerated),
            telemetry: telemetry
        )
        
        writeJSONResult(id: id, statusObj)
    }

    private func callCheckHardware(id: JSONRPCId?) async {
        writeJSONResult(id: id, await SystemProfiler().generateReport())
    }

    /// A tool result whose single text block is `value` as JSON. `isError` flags a result the
    /// agent must not treat as success (MCP tool-result convention).
    private struct ToolResult: Encodable {
        let content: [[String: String]]
        let isError: Bool?
    }

    private func writeJSONResult<T: Encodable>(id: JSONRPCId?, _ value: T, isError: Bool = false) {
        let text = (try? String(data: encoder.encode(value), encoding: .utf8)) ?? "{}"
        writeResult(id: id, result: ToolResult(content: [["type": "text", "text": text]], isError: isError ? true : nil))
    }

    private func callInspectModel(id: JSONRPCId?, args: [String: Any]?) async {
        guard let targetModel = args?["modelId"] as? String, !targetModel.isEmpty else {
            writeError(id: id, code: -32602, message: "Missing required argument: modelId")
            return
        }
        let report = await ModelInspector().inspect(modelIdOrPath: targetModel)
        // Same signal the CLI gives with its exit status: a broken model is an error result.
        writeJSONResult(id: id, report, isError: !report.intact)
    }

    private func callSwitchModel(id: JSONRPCId?, args: [String: Any]?) async {
        guard let targetModel = args?["modelId"] as? String, !targetModel.isEmpty else {
            writeError(id: id, code: -32602, message: "Missing required argument: modelId")
            return
        }
        do {
            try await orchestrator.switchModel(to: targetModel)
            let result = [
                "content": [[
                    "type": "text",
                    "text": "Successfully switched model to '\(targetModel)'"
                ]]
            ]
            writeResult(id: id, result: result)
        } catch {
            writeError(id: id, code: errorCode(for: error), message: error.localizedDescription)
        }
    }

    private func callAddKnowledge(id: JSONRPCId?, args: [String: Any]?) async {
        guard let topic = args?["topic"] as? String, !topic.isEmpty,
              let content = args?["content"] as? String, !content.isEmpty else {
            writeError(id: id, code: -32602, message: "Missing required arguments: topic, content")
            return
        }
        guard let library else {
            writeError(id: id, code: -32603, message: "Experience Library is not configured on this server")
            return
        }
        for key in ["domain", "cue", "confidence"] where args?[key] != nil && !(args?[key] is String) {
            writeError(id: id, code: -32602, message: "\(key) must be a string")
            return
        }
        do {
            let entry = try library.addKnowledge(
                topic: topic,
                content: content,
                domain: args?["domain"] as? String ?? "general",
                cue: args?["cue"] as? String ?? "",
                confidence: args?["confidence"] as? String ?? "medium"
            )
            let result = [
                "content": [[
                    "type": "text",
                    "text": "Stored '\(entry.topic)' in the Experience Library as \(entry.domain)/harnesses/\(entry.slug)."
                ]]
            ]
            writeResult(id: id, result: result)
        } catch let error as HarnessLibraryError {
            if case .invalidArgument = error {
                writeError(id: id, code: -32602, message: error.description)
            } else {
                writeError(id: id, code: -32603, message: "Failed to persist knowledge: \(error)")
            }
        } catch {
            writeError(id: id, code: -32603, message: "Failed to persist knowledge: \(error)")
        }
    }

    private func callHarnessValidate(id: JSONRPCId?, args: [String: Any]?) {
        let surfaceName = args?["surface"] as? String ?? "claude"
        guard var spec = HarnessSurfaceSpec.named(surfaceName) else {
            let known = HarnessSurfaceSpec.all.map(\.name).joined(separator: ", ")
            writeError(id: id, code: -32602, message: "Unknown surface '\(surfaceName)'; expected one of: \(known)")
            return
        }
        if let limit = args?["alwaysOnMaxChars"] {
            guard let limit = limit as? Int, limit > 0 else {
                writeError(id: id, code: -32602, message: "alwaysOnMaxChars must be a positive integer")
                return
            }
            spec = spec.withAlwaysOnMaxChars(limit)
        }
        var skipped: [String] = []
        let tree: HarnessTree
        if let directory = args?["directory"] as? String, !directory.isEmpty {
            do {
                let loaded = try HarnessLoader.load(
                    root: URL(fileURLWithPath: directory, isDirectory: true), spec: spec,
                    allowedRoots: harnessRoots, limits: harnessLimits
                )
                tree = loaded.tree
                skipped = loaded.skipped
            } catch {
                writeError(id: id, code: -32602, message: "Cannot load harness: \(error)")
                return
            }
        } else if let files = args?["files"] as? [String: String] {
            tree = HarnessTree(files: files)
        } else {
            writeError(id: id, code: -32602, message: "Missing required argument: directory or files")
            return
        }
        var report = HarnessValidator.validate(tree, spec: spec)
        report.warnings += skipped.map { "skipped \($0)" }
        writeJSONResult(id: id, report, isError: !report.isValid)
    }

    private func callHarnessLibraryQuery(id: JSONRPCId?, args: [String: Any]?) async {
        guard let library else {
            writeError(id: id, code: -32603, message: "Experience Library is not configured on this server")
            return
        }
        do {
            writeJSONResult(id: id, try library.query(args?["query"] as? String, domain: args?["domain"] as? String))
        } catch {
            writeError(id: id, code: -32603, message: "Cannot read the Experience Library: \(error)")
        }
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
            topP:        args?["topP"]        as? Double,
            minP:        args?["minP"]        as? Double,
            dynamicTemperature: args?["dynamicTemperature"] as? Bool,
            cacheKey: PromptCacheKey.resolve(isService: false)
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

    private func errorCode(for error: BasaltError) -> Int {
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
        case .inferenceError:           return -32009
        case .modelNotCached:           return -32010
        case .modelSwitching:           return -32011
        }
    }
}

// MARK: — AnyCodable helper (minimal, no external dep)

struct AnyCodable: @unchecked Sendable, Codable {
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
