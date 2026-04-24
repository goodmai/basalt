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

    public init(orchestrator: ModelOrchestratorActor, modelId: String? = nil) {
        self.orchestrator = orchestrator
        self.modelId = modelId
    }

    // MARK: — Run loop

    public func run() async {
        for await line in StdinLines() {
            guard !line.isEmpty else { continue }
            await dispatch(line)
        }
    }

    // MARK: — Dispatch

    private func dispatch(_ line: String) async {
        guard let data = line.data(using: .utf8),
              let msg = try? decoder.decode(RawMessage.self, from: data)
        else {
            writeError(id: nil, code: -32700, message: "Parse error")
            return
        }

        switch msg.method {
        case "initialize":
            handleInitialize(id: msg.id)
        case "notifications/initialized", "initialized":
            break   // no-op notification
        case "tools/list":
            handleToolsList(id: msg.id)
        case "tools/call":
            await handleToolsCall(id: msg.id, params: msg.params)
        default:
            if let method = msg.method {
                writeError(id: msg.id, code: -32601, message: "Method not found: \(method)")
            }
        }
    }

    // MARK: — MCP handlers

    private func handleInitialize(id: JSONRPCId?) {
        let result: [String: Any] = [
            "protocolVersion": "2024-11-05",
            "capabilities": ["tools": [:]],
            "serverInfo": [
                "name": "GemmaServer",
                "version": HealthResponse.version
            ]
        ]
        writeResult(id: id, result: result)
    }

    private func handleToolsList(id: JSONRPCId?) {
        let tools: [[String: Any]] = [
            [
                "name": "gemma_generate",
                "description": "Generate text using the locally-running Gemma model via MLX.",
                "inputSchema": [
                    "type": "object",
                    "required": ["prompt"],
                    "properties": [
                        "prompt":      ["type": "string",  "description": "Input prompt"],
                        "maxTokens":   ["type": "integer", "description": "Max tokens to generate (default 1024)"],
                        "temperature": ["type": "number",  "description": "Sampling temperature 0–2 (default 0.7)"],
                        "topP":        ["type": "number",  "description": "Nucleus sampling p (default 0.9)"]
                    ]
                ]
            ],
            [
                "name": "gemma_status",
                "description": "Returns the health and readiness status of the inference engine.",
                "inputSchema": ["type": "object", "properties": [:]]
            ]
        ]
        writeResult(id: id, result: ["tools": tools])
    }

    private func handleToolsCall(id: JSONRPCId?, params: AnyCodable?) async {
        guard let name = params?.value(forKey: "name") as? String else {
            writeError(id: id, code: -32602, message: "Missing tool name")
            return
        }

        switch name {
        case "gemma_generate":
            await callGenerate(id: id, args: params?.value(forKey: "arguments") as? [String: Any])
        case "gemma_status":
            let health = await orchestrator.healthSnapshot(modelId: modelId)
            writeResult(id: id, result: [
                "content": [[
                    "type": "text",
                    "text": "status=\(health.status) ready=\(health.isReady) version=\(health.version)"
                ]]
            ])
        default:
            writeError(id: id, code: -32601, message: "Unknown tool: \(name)")
        }
    }

    private func callGenerate(id: JSONRPCId?, args: [String: Any]?) async {
        guard let prompt = args?["prompt"] as? String else {
            writeError(id: id, code: -32602, message: "Missing required argument: prompt")
            return
        }
        let request = GenerationRequest(
            prompt: prompt,
            maxTokens:   args?["maxTokens"]   as? Int    ?? 1024,
            temperature: args?["temperature"] as? Double ?? 0.7,
            topP:        args?["topP"]        as? Double ?? 0.9
        )
        do {
            let response = try await orchestrator.generate(request: request)
            writeResult(id: id, result: [
                "content": [[
                    "type": "text",
                    "text": response.generatedText
                ]],
                "_meta": [
                    "tokensPerSecond": response.tokensPerSecond,
                    "completionTokens": response.completionTokens,
                    "finishReason": response.finishReason.rawValue
                ]
            ])
        } catch {
            writeError(id: id, code: errorCode(for: error), message: error.localizedDescription)
        }
    }

    // MARK: — Wire

    private func writeResult(id: JSONRPCId?, result: [String: Any]) {
        var obj: [String: Any] = ["jsonrpc": "2.0", "result": result]
        if let id { obj["id"] = encodeId(id) }
        writeLine(obj)
    }

    private func writeError(id: JSONRPCId?, code: Int, message: String) {
        var obj: [String: Any] = [
            "jsonrpc": "2.0",
            "error": ["code": code, "message": message]
        ]
        if let id { obj["id"] = encodeId(id) }
        writeLine(obj)
    }

    private func writeLine(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let line = String(data: data, encoding: .utf8)
        else { return }
        // stdout must be written atomically per message
        print(line)
        fflush(stdout)
    }

    private func encodeId(_ id: JSONRPCId) -> Any {
        switch id {
        case .string(let s): return s
        case .int(let i):    return i
        case .null:          return NSNull()
        }
    }

    private func errorCode(for error: GemmaServerError) -> Int {
        switch error {
        case .modelNotFound:            return -32001
        case .outOfMemory:              return -32002
        case .inferenceHardwareFailure: return -32003
        case .invalidRequestStructure:  return -32602
        case .contextLengthExceeded:    return -32004
        case .weightsCorrupted:         return -32005
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
