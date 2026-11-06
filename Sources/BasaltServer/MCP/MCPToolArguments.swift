import Foundation
import BasaltCore
import BasaltHarness

enum MCPTool: Sendable, Equatable {
    case status, checkHardware, inspectModel, switchModel, generate, health, addKnowledge
    case harnessValidate, harnessLibraryQuery, harnessOptimize
    case research(ResearchTool)
    case introspect

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
        case "introspect": self = .introspect
        default:
            guard let tool = ResearchTool(rawValue: name) else { return nil }
            self = .research(tool)
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

    static func generation(_ args: [String: Any]?, images: [ImageInput] = []) throws(MCPFailure) -> GenerationRequest {
        guard let prompt = args?["prompt"] as? String else {
            throw .invalidParams("Missing required argument: prompt")
        }
        let speculativeLookup = try optionalBoolean("speculativeLookup", in: args)
        let grammarJumpForward = try optionalBoolean("grammarJumpForward", in: args)
        var request = GenerationRequest(
            prompt: prompt,
            maxTokens:   args?["maxTokens"]   as? Int,
            temperature: args?["temperature"] as? Double,
            topP:        args?["topP"]        as? Double,
            messages:    images.isEmpty ? nil : [ChatMessageInput(role: .user, text: prompt, images: images)]
        )
        request.minP = args?["minP"] as? Double
        request.dynamicTemperature = args?["dynamicTemperature"] as? Bool
        request.cacheKey = PromptCacheKey.resolve(isService: false)
        let (regex, grammar) = (try MCPArguments.string("regex", in: args), try MCPArguments.string("grammar", in: args))
        guard regex == nil || grammar == nil else { throw .invalidParams("give regex or grammar, not both") }
        request.responseFormat = regex.map(ResponseFormat.regex) ?? grammar.map(ResponseFormat.grammar)
        return request.withSpeculativeLookup(speculativeLookup)
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

enum ResearchTool: String, Sendable, CaseIterable {
    case modelDescribe = "model_describe"
    case modelTensors = "model_tensors"
    case modelTensor = "model_tensor"
    case kvInspect = "kv_inspect"
    case traceRun = "trace_run"
    case intervene
    case weightsDelta = "weights_delta"
    case overlaysList = "overlays_list"
    case overlayUndo = "overlay_undo"
    case experimentCompare = "experiment_compare"
    case adapterApply = "adapter_apply"
    case gateCheck = "gate_check"
    case tensorPack = "tensor_pack"
    case tensorUnpack = "tensor_unpack"
    case trainExperts = "train_experts"
    case steerSearch = "steer_search"
    case steerCommit = "steer_commit"

    var isExperiment: Bool {
        switch self {
        case .modelDescribe, .modelTensors, .modelTensor, .kvInspect, .traceRun: false
        case .intervene, .weightsDelta, .overlaysList, .overlayUndo, .experimentCompare, .adapterApply, .gateCheck: true
        case .tensorPack, .tensorUnpack, .trainExperts, .steerSearch, .steerCommit: true
        }
    }
}

enum ResearchCall: Sendable, Equatable {
    case describe
    case tensors(filter: String?)
    case tensor(TensorRequest)
    case promptCache(namespace: String?)
    case trace(TraceRequest)
    case intervene(Intervention)
    case delta(WeightDelta)
    case overlays
    case undo(OverlayID)
    case compare(CompareRequest)
    case adapter(AdapterSource, gate: GateRequest?)
    case gate(GateRequest)
    case pack(PackRequest)
    case train(TrainRequest)
    case contrast(ContrastRequest)
    case commitSteering(SteeringCommit)

    static let maxValues = 0...4096
    static let topK = 1...100
    static let comparePrompts = 1...16
    static let compareTokens = 1...512
    static let nonNegative = 0...Int.max

    init(_ tool: ResearchTool, _ args: [String: Any]?) throws(MCPFailure) {
        switch tool {
        case .modelDescribe: self = .describe
        case .modelTensors: self = .tensors(filter: try MCPArguments.string("filter", in: args))
        case .modelTensor: self = .tensor(try TensorRequest(mcp: args))
        case .kvInspect: self = .promptCache(namespace: try MCPArguments.string("namespace", in: args))
        case .traceRun: self = .trace(try TraceRequest(mcp: args))
        case .intervene: self = .intervene(try Intervention(mcp: args))
        case .weightsDelta: self = .delta(try WeightDelta(mcp: args))
        case .overlaysList: self = .overlays
        case .overlayUndo: self = .undo(OverlayID(try MCPArguments.required("id", MCPArguments.integer("id", in: args, Self.nonNegative))))
        case .experimentCompare: self = .compare(try CompareRequest(mcp: args))
        case .adapterApply, .gateCheck, .tensorPack, .tensorUnpack, .trainExperts, .steerSearch, .steerCommit: self = try Self(lab: tool, args)
        }
    }

    init(lab tool: ResearchTool, _ args: [String: Any]?) throws(MCPFailure) {
        switch tool {
        case .adapterApply: self = .adapter(try AdapterSource(mcp: args), gate: try MCPArguments.present("gate", in: args).map { value throws(MCPFailure) in try GateRequest(mcp: value) })
        case .gateCheck: self = .gate(try GateRequest(mcp: args))
        case .tensorPack: self = .pack(try PackRequest(mcp: args, formats: PackRequest.packFormats))
        case .tensorUnpack: self = .pack(try PackRequest(mcp: args, formats: PackRequest.unpackFormats))
        case .trainExperts: self = .train(try TrainRequest(mcp: args))
        case .steerSearch: self = .contrast(try ContrastRequest(mcp: args))
        case .steerCommit: self = .commitSteering(try SteeringCommit(mcp: args))
        case .modelDescribe, .modelTensors, .modelTensor, .kvInspect, .traceRun, .intervene, .weightsDelta, .overlaysList, .overlayUndo, .experimentCompare: throw .invalidParams("\(tool.rawValue) is not a lab tool")
        }
    }
}

extension MCPArguments {
    static func present(_ key: String, in args: [String: Any]?) -> Any? {
        args?[key].flatMap { $0 is NSNull ? nil : $0 }
    }

    static func required<T>(_ key: String, _ value: T?) throws(MCPFailure) -> T {
        guard let value else { throw .invalidParams("Missing required argument: \(key)") }
        return value
    }

    static func string(_ key: String, in args: [String: Any]?) throws(MCPFailure) -> String? {
        guard let value = present(key, in: args) else { return nil }
        guard let text = value as? String else { throw .invalidParams("\(key) must be a string") }
        return text
    }

    static func integer(_ key: String, in args: [String: Any]?, _ range: ClosedRange<Int>) throws(MCPFailure) -> Int? {
        guard let value = present(key, in: args) else { return nil }
        guard !isJSONBoolean(value), let number = value as? Int, range.contains(number) else {
            throw .invalidParams("\(key) must be an integer in \(range.upperBound == .max ? "\(range.lowerBound)..." : "\(range)")")
        }
        return number
    }

    static func number(_ key: String, in args: [String: Any]?) throws(MCPFailure) -> Float? {
        guard let value = present(key, in: args) else { return nil }
        guard !isJSONBoolean(value), let number = (value as? Double) ?? (value as? Int).map(Double.init), number.isFinite else {
            throw .invalidParams("\(key) must be a number")
        }
        return Float(number)
    }

    static func numbers(_ key: String, in args: [String: Any]?) throws(MCPFailure) -> [Float]? {
        guard let value = present(key, in: args) else { return nil }
        let numbers = (value as? [Any])?.compactMap { ($0 as? Double) ?? ($0 as? Int).map(Double.init) }
        guard let numbers, numbers.count == (value as? [Any])?.count, !numbers.isEmpty, numbers.allSatisfy(\.isFinite) else {
            throw .invalidParams("\(key) must be a non-empty array of numbers")
        }
        return numbers.map { Float($0) }
    }

    static func integers(_ key: String, in args: [String: Any]?) throws(MCPFailure) -> [Int]? {
        guard let value = present(key, in: args) else { return nil }
        guard let numbers = value as? [Int], numbers.allSatisfy({ $0 >= 0 }) else {
            throw .invalidParams("\(key) must be an array of non-negative integers")
        }
        return numbers
    }

    static func slice(_ args: [String: Any]?) throws(MCPFailure) -> [Range<Int>]? {
        guard let value = present("slice", in: args) else { return nil }
        guard let pairs = value as? [[Int]], pairs.allSatisfy({ $0.count == 2 && 0 <= $0[0] && $0[0] < $0[1] }) else {
            throw .invalidParams("slice must be an array of [start, end] integer pairs with 0 <= start < end")
        }
        return pairs.map { $0[0] ..< $0[1] }
    }
}

extension TensorRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        self.init(
            name: try MCPArguments.required("name", MCPArguments.string("name", in: args)),
            slice: try MCPArguments.slice(args),
            maxValues: try MCPArguments.integer("maxValues", in: args, ResearchCall.maxValues) ?? 256)
        spectrum = try MCPArguments.optionalBoolean("spectrum", in: args) ?? false
    }
}

extension TraceRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        var request = TraceRequest(
            prompt: try MCPArguments.required("prompt", MCPArguments.string("prompt", in: args)),
            position: try MCPArguments.integer("position", in: args, ResearchCall.nonNegative),
            layers: try MCPArguments.integers("layers", in: args),
            topK: try MCPArguments.integer("topK", in: args, ResearchCall.topK) ?? 10)
        request.attention = try MCPArguments.optionalBoolean("attention", in: args) ?? true
        request.router = try MCPArguments.optionalBoolean("router", in: args) ?? true
        request.logitLens = try MCPArguments.optionalBoolean("logitLens", in: args) ?? true
        self = request
    }
}

extension Intervention {
    static let kinds = ["steer", "ablate_head", "ablate_expert"]

    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let layer = try MCPArguments.required("layer", MCPArguments.integer("layer", in: args, ResearchCall.nonNegative))
        switch try MCPArguments.string("kind", in: args) {
        case "steer"?:
            let vector = try MCPArguments.required("vector", MCPArguments.numbers("vector", in: args))
            self = .steer(layer: layer, vector: vector, scale: try MCPArguments.number("scale", in: args) ?? 1)
        case "ablate_head"?:
            self = .ablateHead(layer: layer, head: try MCPArguments.required("head", MCPArguments.integer("head", in: args, ResearchCall.nonNegative)))
        case "ablate_expert"?:
            self = .ablateExpert(layer: layer, expert: try MCPArguments.required("expert", MCPArguments.integer("expert", in: args, ResearchCall.nonNegative)))
        default:
            throw .invalidParams("kind must be one of: \(Self.kinds.joined(separator: ", "))")
        }
    }
}

extension WeightDelta {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        self.init(
            tensor: try MCPArguments.required("tensor", MCPArguments.string("tensor", in: args)),
            slice: try MCPArguments.slice(args),
            values: try MCPArguments.numbers("values", in: args),
            scale: try MCPArguments.number("scale", in: args) ?? 1)
    }
}

extension CompareRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let prompts = MCPArguments.present("prompts", in: args)
        guard let prompts = prompts as? [String], ResearchCall.comparePrompts.contains(prompts.count) else {
            throw .invalidParams("prompts must be an array of \(ResearchCall.comparePrompts) strings")
        }
        self.init(
            prompts: prompts,
            overlays: (try MCPArguments.integers("overlays", in: args) ?? []).map { OverlayID($0) },
            maxTokens: try MCPArguments.integer("maxTokens", in: args, ResearchCall.compareTokens) ?? 32)
    }
}

extension AdapterSource {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        if let directory = try MCPArguments.string("directory", in: args) {
            self = .directory(directory)
            return
        }
        self = .rankOne(
            module: try MCPArguments.required("module", MCPArguments.string("module", in: args)),
            input: try MCPArguments.required("input", MCPArguments.numbers("input", in: args)),
            output: try MCPArguments.required("output", MCPArguments.numbers("output", in: args)),
            scale: try MCPArguments.number("scale", in: args) ?? 1)
    }
}

extension GateRequest {
    init(mcp value: Any?) throws(MCPFailure) {
        let args = value as? [String: Any]
        guard value == nil || args != nil else { throw .invalidParams("gate must be an object") }
        guard let heldOut = MCPArguments.present("heldOut", in: args) as? [String], ResearchCall.comparePrompts.contains(heldOut.count) else {
            throw .invalidParams("heldOut must be an array of \(ResearchCall.comparePrompts) strings")
        }
        self.init(heldOut: heldOut)
        try readLimits(args)
        let targets = MCPArguments.present("target", in: args) ?? [Any]()
        guard let targets = targets as? [[String: Any]], targets.count <= ResearchCall.comparePrompts.upperBound else {
            throw .invalidParams("target must be an array of at most \(ResearchCall.comparePrompts.upperBound) {prompt, expected} objects")
        }
        for item in targets {
            target.append(GateTarget(
                prompt: try MCPArguments.required("prompt", MCPArguments.string("prompt", in: item)),
                expected: try MCPArguments.required("expected", MCPArguments.string("expected", in: item))))
        }
    }

    mutating func readLimits(_ args: [String: Any]?) throws(MCPFailure) {
        maxKL = try MCPArguments.number("maxKL", in: args) ?? maxKL
        maxPerplexityRise = try MCPArguments.number("maxPerplexityRise", in: args) ?? maxPerplexityRise
        guard maxKL >= 0, maxPerplexityRise >= 0 else { throw .invalidParams("maxKL and maxPerplexityRise must not be negative") }
    }
}

extension PackRequest {
    static let packFormats = ["affine", "mxfp4", "nf4:32", "nf4:64", "ternary"]
    static let unpackFormats = ["bfloat16", "float16"]
    static let packBits = 2...8
    static let packGroupSizes = 32...128

    init(mcp args: [String: Any]?, formats: [String]) throws(MCPFailure) {
        let tensor = try MCPArguments.required("tensor", MCPArguments.string("tensor", in: args))
        let format: PackFormat
        let name = try MCPArguments.string("format", in: args) ?? formats[0]
        switch name {
        case "affine" where formats.contains("affine"):
            format = .affine(
                bits: try MCPArguments.integer("bits", in: args, Self.packBits) ?? 4,
                groupSize: try MCPArguments.integer("groupSize", in: args, Self.packGroupSizes) ?? 64)
        case "mxfp4" where formats.contains("mxfp4"): format = .mxfp4
        case _ where name.hasPrefix("nf4:") && formats.contains(name): format = .nf4(groupSize: Int(name.dropFirst(4)) ?? 0)
        case "ternary" where formats.contains("ternary"): format = .ternary
        case "bfloat16" where formats.contains("bfloat16"): format = .bfloat16
        case "float16" where formats.contains("float16"): format = .float16
        default: throw .invalidParams("format must be one of: \(formats.joined(separator: ", "))")
        }
        self.init(tensor: tensor, format: format)
    }
}

extension TrainRequest {
    static let taskCounts = 2...10_000

    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let heldOut = MCPArguments.present("heldOut", in: args)
        guard let heldOut = heldOut as? [String], ResearchCall.comparePrompts.contains(heldOut.count) else {
            throw .invalidParams("heldOut must be an array of \(ResearchCall.comparePrompts) general prompts the quality gate checks the adapter on")
        }
        var gate = GateRequest(heldOut: heldOut)
        try gate.readLimits(args)
        let count = try MCPArguments.integer("count", in: args, Self.taskCounts) ?? 200
        let seed = try MCPArguments.integer("seed", in: args, ResearchCall.nonNegative) ?? 0
        let file = try MCPArguments.string("file", in: args)
        self.init(tasks: file.map(TrainingTasks.file) ?? .arithmetic(count: count, seed: UInt64(seed)), gate: gate)
        teacher = try MCPArguments.string("teacher", in: args)
        teacherModel = try MCPArguments.string("teacherModel", in: args)
        layers = try MCPArguments.integers("layers", in: args)
        experts = try MCPArguments.integers("experts", in: args)
        rank = try MCPArguments.integer("rank", in: args, ResearchCall.nonNegative) ?? rank
        iterations = try MCPArguments.integer("iterations", in: args, ResearchCall.nonNegative) ?? iterations
        learningRate = try MCPArguments.number("learningRate", in: args) ?? learningRate
        output = try MCPArguments.string("output", in: args)
    }
}

extension ContrastRequest {
    static let headLimit = 0...64

    static func prompts(_ key: String, in args: [String: Any]?, _ range: ClosedRange<Int>) throws(MCPFailure) -> [String] {
        guard let prompts = MCPArguments.present(key, in: args) as? [String], range.contains(prompts.count) else {
            throw .invalidParams("\(key) must be an array of \(range) strings")
        }
        return prompts
    }

    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let positive = try Self.prompts("positive", in: args, Self.classPrompts)
        let negative = try Self.prompts("negative", in: args, Self.classPrompts)
        let heldOut = try Self.prompts("heldOut", in: args, Self.heldOutPrompts)
        self.init(positive: positive, negative: negative, heldOut: heldOut, layers: try MCPArguments.integers("layers", in: args))
        topHeads = try MCPArguments.integer("topHeads", in: args, Self.headLimit) ?? topHeads
        if let scales = try MCPArguments.numbers("scales", in: args) {
            guard Self.scaleCount.contains(scales.count) else { throw .invalidParams("scales must hold \(Self.scaleCount) numbers") }
            self.scales = scales
        }
    }
}

extension SteeringCommit {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        var gate = GateRequest(heldOut: try ContrastRequest.prompts("heldOut", in: args, ContrastRequest.heldOutPrompts))
        try gate.readLimits(args)
        self.init(
            layer: try MCPArguments.required("layer", MCPArguments.integer("layer", in: args, ResearchCall.nonNegative)),
            direction: try MCPArguments.required("direction", MCPArguments.numbers("direction", in: args)),
            scale: try MCPArguments.required("scale", MCPArguments.number("scale", in: args)),
            gate: gate)
        positive = try ContrastRequest.prompts("positive", in: args, 1...ContrastRequest.classPrompts.upperBound)
    }
}
