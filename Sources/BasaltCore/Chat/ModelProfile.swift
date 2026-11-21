import Foundation

public protocol ModelProfile: Sendable {
    var name: String { get }

    func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable]

    var thinkingMarkers: ThinkingMarkers { get }

    var defaultMaxTokens: Int { get }

    var defaultMinP: Float? { get }

    func turns(_ turns: [ChatMessageInput], for reasoning: ReasoningPolicy) -> [ChatMessageInput]

    var writesToolCallsAsText: Bool { get }
}

extension ModelProfile {
    public func turns(_ turns: [ChatMessageInput], for reasoning: ReasoningPolicy) -> [ChatMessageInput] { turns }

    public var writesToolCallsAsText: Bool { false }
}

public struct ThinkingMarkers: Sendable {
    public let open: [String]
    public let close: [String]

    public init(open: [String], close: [String]) {
        self.open = open
        self.close = close
    }

    public static let none = ThinkingMarkers(open: [], close: [])
}

public struct QwenProfile: ModelProfile {
    public let name = "qwen3.x"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] {
        switch reasoning {
        case .modelDefault:
            return [:]
        case .disabled:
            return ["enable_thinking": false]
        case .effort(let level):
            let mapped = ["minimal": "low", "high": "xhigh"][level] ?? level
            guard ["xhigh", "medium", "low"].contains(mapped) else { return [:] }
            return ["reasoning_effort": mapped]
        }
    }

    public var thinkingMarkers: ThinkingMarkers {
        ThinkingMarkers(open: ["<think>"], close: ["</think>"])
    }

    public var defaultMaxTokens: Int { 8192 }

    public var defaultMinP: Float? { nil }
}

public struct GemmaProfile: ModelProfile {
    public let name = "gemma"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] {
        switch reasoning {
        case .modelDefault: return [:]
        case .disabled:     return ["enable_thinking": false]
        case .effort:       return ["enable_thinking": true]
        }
    }

    public var thinkingMarkers: ThinkingMarkers { .none }

    public var defaultMaxTokens: Int { 4096 }

    public var defaultMinP: Float? { 0.9 }
}

public struct NemotronProfile: ModelProfile {
    public let name = "nemotron"
    public let switchesThinkingInConversation: Bool

    public init(switchesThinkingInConversation: Bool) {
        self.switchesThinkingInConversation = switchesThinkingInConversation
    }

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] {
        reasoning == .disabled ? ["enable_thinking": false] : [:]
    }

    public func turns(_ turns: [ChatMessageInput], for reasoning: ReasoningPolicy) -> [ChatMessageInput] {
        guard switchesThinkingInConversation else { return turns }
        let directive: String
        switch reasoning {
        case .modelDefault: return turns
        case .disabled:     directive = "/no_think"
        case .effort:       directive = "/think"
        }
        guard let first = turns.first, first.role == .system else {
            return [ChatMessageInput(role: .system, text: directive)] + turns
        }
        return [ChatMessageInput(role: .system, text: first.text + "\n" + directive)] + turns.dropFirst()
    }

    public var thinkingMarkers: ThinkingMarkers {
        ThinkingMarkers(open: ["<think>"], close: ["</think>"])
    }

    public var defaultMaxTokens: Int { 4096 }

    public var defaultMinP: Float? { nil }

    public var writesToolCallsAsText: Bool { true }
}

public struct LiquidProfile: ModelProfile {
    public let name = "lfm2"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] { [:] }

    public var thinkingMarkers: ThinkingMarkers {
        ThinkingMarkers(open: ["<think>"], close: ["</think>"])
    }

    public var defaultMaxTokens: Int { 8192 }

    public var defaultMinP: Float? { nil }
}

public struct FlintProfile: ModelProfile {
    public let name = "flint"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] { [:] }

    public var thinkingMarkers: ThinkingMarkers {
        ThinkingMarkers(open: ["<think>"], close: ["</think>"])
    }

    public var defaultMaxTokens: Int { 32_768 }

    public var defaultMinP: Float? { nil }

    public var writesToolCallsAsText: Bool { false }
}

public struct RicursiveProfile: ModelProfile {
    public let name = "ricursive"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] { [:] }

    public var thinkingMarkers: ThinkingMarkers {
        ThinkingMarkers(open: ["<think>"], close: ["</think>"])
    }

    public var defaultMaxTokens: Int { 32_768 }

    public var defaultMinP: Float? { nil }

    public var writesToolCallsAsText: Bool { false }
}

public struct MuseGlimmerProfile: ModelProfile {
    public let name = "muse_glimmer"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] {
        switch reasoning {
        case .modelDefault:
            return [:]
        case .disabled:
            return ["reasoning_strength": "low"]
        case .effort(let level):
            let mapped = level == "minimal" ? "low" : level
            guard ["low", "medium", "high", "xhigh"].contains(mapped) else { return [:] }
            return ["reasoning_strength": mapped]
        }
    }

    public var thinkingMarkers: ThinkingMarkers { .none }

    public var defaultMaxTokens: Int { 16_384 }

    public var defaultMinP: Float? { nil }

    public var writesToolCallsAsText: Bool { false }
}

public struct GPTOSSProfile: ModelProfile {
    public let name = "gpt_oss"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] {
        switch reasoning {
        case .modelDefault:
            return [:]
        case .disabled:
            return ["reasoning_effort": "low"]
        case .effort(let level):
            let mapped = ["minimal": "low", "xhigh": "high"][level] ?? level
            guard ["low", "medium", "high"].contains(mapped) else { return [:] }
            return ["reasoning_effort": mapped]
        }
    }

    public var thinkingMarkers: ThinkingMarkers { .none }

    public var defaultMaxTokens: Int { 16_384 }

    public var defaultMinP: Float? { nil }
}

public struct GenericProfile: ModelProfile {
    public let name = "generic"

    public init() {}

    public func templateContext(for reasoning: ReasoningPolicy) -> [String: any Sendable] {
        switch reasoning {
        case .modelDefault:        return [:]
        case .disabled:            return ["enable_thinking": false]
        case .effort(let level):   return ["reasoning_effort": level]
        }
    }
    public var thinkingMarkers: ThinkingMarkers { .none }
    public var defaultMaxTokens: Int { 4096 }
    public var defaultMinP: Float? { nil }
}

public struct SamplingDefaults: Sendable, Equatable {
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let repetitionPenalty: Double?

    public init(temperature: Double? = nil, topP: Double? = nil, topK: Int? = nil, repetitionPenalty: Double? = nil) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
    }

    public init(generationConfig json: [String: Any]?) {
        self.init(
            temperature: (json?["temperature"] as? NSNumber)?.doubleValue,
            topP: (json?["top_p"] as? NSNumber)?.doubleValue,
            topK: (json?["top_k"] as? NSNumber)?.intValue,
            repetitionPenalty: (json?["repetition_penalty"] as? NSNumber)?.doubleValue)
    }
}

public enum ModelProfileRegistry {
    public static func profile(forModelType modelType: String?, chatTemplate: String? = nil) -> any ModelProfile {
        guard let t = modelType?.lowercased() else { return GenericProfile() }
        if t.hasPrefix("qwen3") { return QwenProfile() }
        if t.hasPrefix("gemma") { return GemmaProfile() }
        if t.hasPrefix("lfm2") { return LiquidProfile() }
        if t == "flint" { return FlintProfile() }
        if t == "ricursive" || t == "ripple" { return RicursiveProfile() }
        if t.hasPrefix("muse_glimmer") || t == "muse-glimmer" { return MuseGlimmerProfile() }
        if t == "gpt_oss" { return GPTOSSProfile() }
        if t == "nemotron_h" || t == "nemotron_lightning" || t == "nemotron-lightning" {
            return NemotronProfile(switchesThinkingInConversation: chatTemplate?.contains("/no_think") == true)
        }
        return GenericProfile()
    }

    public static func profile(forModelAt path: String) -> any ModelProfile {
        let directory = URL(fileURLWithPath: path)
        guard let json = ModelCache.config(at: directory) else { return GenericProfile() }
        return profile(forModelType: json["model_type"] as? String, chatTemplate: chatTemplate(at: directory))
    }

    static func chatTemplate(at directory: URL) -> String? {
        if let jinja = try? String(contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8) {
            return jinja
        }
        if let json = jsonObject(at: directory.appendingPathComponent("chat_template.json")),
           let template = json["chat_template"] as? String {
            return template
        }
        return jsonObject(at: directory.appendingPathComponent("tokenizer_config.json"))?["chat_template"] as? String
    }

    private static func jsonObject(at url: URL) -> [String: Any]? {
        (try? Data(contentsOf: url))
            .flatMap { try? JSONSerialization.jsonObject(with: $0, options: .json5Allowed) as? [String: Any] }
    }
}
