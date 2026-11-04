import Foundation
import BasaltCore
import BasaltHarness

enum MCPTool: Sendable {
    case status, checkHardware, inspectModel, switchModel, generate, health, addKnowledge
    case harnessValidate, harnessLibraryQuery, harnessOptimize

    init?(name: String) {
        switch name {
        case "status": self = .status
        case "check_hardware": self = .checkHardware
        case "inspect_model": self = .inspectModel
        case "switch_model": self = .switchModel
        case "basalt_generate", "gemma_generate": self = .generate
        case "basalt_status", "gemma_status": self = .health
        case "basalt_add_knowledge", "gemma_add_knowledge": self = .addKnowledge
        case "harness_validate": self = .harnessValidate
        case "harness_library_query": self = .harnessLibraryQuery
        case "harness_optimize": self = .harnessOptimize
        default: return nil
        }
    }
}

enum MCPArguments {
    static func toolCall(_ params: AnyCodable?) throws(MCPFailure) -> (name: String, arguments: [String: Any]?) {
        guard let p = params?.rawValue as? [String: Any], let name = p["name"] as? String else {
            throw .invalidParams("Missing tool name")
        }
        return (name, try arguments(p["arguments"]))
    }

    static func arguments(_ raw: Any?) throws(MCPFailure) -> [String: Any]? {
        if let object = raw as? [String: Any] { return object }
        guard let raw, !(raw is NSNull) else { return nil }
        guard let text = raw as? String else { throw .invalidParams("Tool arguments must be a JSON object") }
        guard let repaired = ToolCallRepairer.repairJSON(text),
              let object = try? JSONSerialization.jsonObject(with: Data(repaired.utf8)) as? [String: Any] else {
            throw .invalidParams("Tool arguments are not a JSON object and could not be repaired")
        }
        return object
    }

    static func modelId(_ args: [String: Any]?) throws(MCPFailure) -> String {
        guard let modelId = args?["modelId"] as? String, !modelId.isEmpty else {
            throw .invalidParams("Missing required argument: modelId")
        }
        return modelId
    }

    static func generation(_ args: [String: Any]?) throws(MCPFailure) -> GenerationRequest {
        guard let prompt = args?["prompt"] as? String else {
            throw .invalidParams("Missing required argument: prompt")
        }
        let speculativeLookup = try optionalBoolean("speculativeLookup", in: args)
        let grammarJumpForward = try optionalBoolean("grammarJumpForward", in: args)
        return GenerationRequest(
            prompt: prompt,
            maxTokens:   args?["maxTokens"]   as? Int,
            temperature: args?["temperature"] as? Double,
            topP:        args?["topP"]        as? Double,
            minP:        args?["minP"]        as? Double,
            dynamicTemperature: args?["dynamicTemperature"] as? Bool,
            cacheKey: PromptCacheKey.resolve(isService: false)
        ).withSpeculativeLookup(speculativeLookup)
            .withGrammarJumpForward(grammarJumpForward)
            .withOrigin(RequestOrigin(source: "mcp", endpoint: "tools/call generate", stream: false))
    }

    static func optionalBoolean(_ key: String, in args: [String: Any]?) throws(MCPFailure) -> Bool? {
        let value = args?[key].flatMap { $0 is NSNull ? nil : $0 }
        if let value, !isJSONBoolean(value) {
            throw .invalidParams("\(key) must be a boolean")
        }
        return value as? Bool
    }

    static func requireStrings(_ keys: [String], in args: [String: Any]?) throws(MCPFailure) {
        for key in keys where args?[key] != nil && !(args?[key] is String) {
            throw .invalidParams("\(key) must be a string")
        }
    }

    static func isJSONBoolean(_ value: Any) -> Bool {
        (value as? NSNumber).map { CFGetTypeID($0) == CFBooleanGetTypeID() } ?? false
    }
}

struct KnowledgeArguments: Sendable {
    let topic: String
    let content: String
    let domain: String
    let cue: String
    let confidence: String

    init(_ args: [String: Any]?, libraryConfigured: Bool) throws(MCPFailure) {
        guard let topic = args?["topic"] as? String, !topic.isEmpty,
              let content = args?["content"] as? String, !content.isEmpty else {
            throw .invalidParams("Missing required arguments: topic, content")
        }
        guard libraryConfigured else { throw .libraryMissing }
        try MCPArguments.requireStrings(["domain", "cue", "confidence"], in: args)
        self.topic = topic
        self.content = content
        domain = args?["domain"] as? String ?? "general"
        cue = args?["cue"] as? String ?? ""
        confidence = args?["confidence"] as? String ?? "medium"
    }
}

struct HarnessInputArguments: Sendable {
    enum Source: Sendable {
        case directory(URL)
        case files([String: String])
    }

    let spec: HarnessSurfaceSpec
    let source: Source

    init(_ args: [String: Any]?) throws(MCPFailure) {
        let surfaceName = args?["surface"] as? String ?? "claude"
        guard var spec = HarnessSurfaceSpec.named(surfaceName) else {
            let known = HarnessSurfaceSpec.all.map(\.name).joined(separator: ", ")
            throw .invalidParams("Unknown surface '\(surfaceName)'; expected one of: \(known)")
        }
        if let limit = args?["alwaysOnMaxChars"] {
            guard let limit = limit as? Int, limit > 0 else {
                throw .invalidParams("alwaysOnMaxChars must be a positive integer")
            }
            spec = spec.withAlwaysOnMaxChars(limit)
        }
        self.spec = spec
        source = try Self.source(args)
    }

    private static func source(_ args: [String: Any]?) throws(MCPFailure) -> Source {
        if let directory = args?["directory"] as? String, !directory.isEmpty {
            return .directory(URL(fileURLWithPath: directory, isDirectory: true))
        }
        if let files = args?["files"] as? [String: String] {
            return .files(files)
        }
        throw .invalidParams("Missing required argument: directory or files")
    }
}

struct HarnessOptimizeArguments: Sendable {
    let task: HarnessTask
    let k: Int
    let proposerMaxTokens: Int?

    init(_ args: [String: Any]?) throws(MCPFailure) {
        guard let prompt = args?["task"] as? String, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw .invalidParams("Missing required argument: task")
        }
        let k = args?["k"] ?? 1
        guard !MCPArguments.isJSONBoolean(k), let k = k as? Int, HarnessLedger.kRange.contains(k) else {
            throw .invalidParams("k must be an integer in \(HarnessLedger.kRange)")
        }
        if let expectsJSON = args?["expectsJSON"], !MCPArguments.isJSONBoolean(expectsJSON) {
            throw .invalidParams("expectsJSON must be a boolean")
        }
        try MCPArguments.requireStrings(["domain", "slug"], in: args)
        let proposerMaxTokens = try Self.proposerMaxTokens(args)
        guard let terms = (args?["requiredTerms"] ?? [String]()) as? [String] else {
            throw .invalidParams("requiredTerms must be an array of strings")
        }
        self.k = k
        self.proposerMaxTokens = proposerMaxTokens
        task = HarnessTask(
            slug: HarnessExperienceLibrary.slugify((args?["slug"] as? String) ?? String(prompt.prefix(48))),
            prompt: prompt,
            domain: (args?["domain"] as? String) ?? "general",
            requiredTerms: terms,
            expectsJSON: (args?["expectsJSON"] as? Bool) ?? false
        )
    }

    private static func proposerMaxTokens(_ args: [String: Any]?) throws(MCPFailure) -> Int? {
        let range = HarnessOptimizer.TokenBudgets.proposerRange
        guard let value = args?["proposerMaxTokens"] else { return nil }
        guard !MCPArguments.isJSONBoolean(value), let tokens = value as? Int, range.contains(tokens) else {
            throw .invalidParams("proposerMaxTokens must be an integer in \(range)")
        }
        return tokens
    }
}

struct ToolResult: Encodable, Sendable {
    struct Meta: Encodable, Sendable {
        let tokensPerSecond: Double
        let completionTokens: Int
        let finishReason: String
    }

    let content: [[String: String]]
    var isError: Bool?
    var _meta: Meta?

    static func text(_ text: String, isError: Bool = false) -> ToolResult {
        ToolResult(content: [["type": "text", "text": text]], isError: isError ? true : nil)
    }

    static func json(_ value: some Encodable, isError: Bool = false) -> ToolResult {
        text((try? String(data: MCPWire.encoder.encode(value), encoding: .utf8)) ?? "{}", isError: isError)
    }

    static func generated(_ response: GenerationResponse) -> ToolResult {
        var result = text(response.generatedText)
        result._meta = Meta(
            tokensPerSecond: response.tokensPerSecond,
            completionTokens: response.completionTokens,
            finishReason: response.finishReason.rawValue
        )
        return result
    }
}

struct StatusOutput: Encodable, Sendable {
    struct Diagnostics: Encodable, Sendable {
        let requestCount: Int
        let totalTokensGenerated: Int
    }

    let status: String
    let ready: Bool
    let version: String
    let loadedModel: String
    let requestedModel: String?
    let requestedModelLoaded: Bool?
    let diagnostics: Diagnostics
    let telemetry: ServerTelemetrySnapshot

    init(health: HealthResponse, loaded: String, requested: String?, telemetry: ServerTelemetrySnapshot) {
        status = health.status
        ready = health.isReady
        version = health.version
        loadedModel = loaded.isEmpty ? "none" : loaded
        requestedModel = requested
        requestedModelLoaded = requested.map { $0 == loaded }
        diagnostics = Diagnostics(requestCount: telemetry.totalRequests, totalTokensGenerated: telemetry.totalTokensGenerated)
        self.telemetry = telemetry
    }
}
