import Foundation

public enum AgentHarnessKind: String, Sendable, Equatable, Hashable, CaseIterable, Codable, CodingKeyRepresentable {
    case claudeCode = "claude-code"
    case openCode = "opencode"
    case pi = "pi"
    case hermes = "hermes"
}

public struct HarnessRequestFingerprint: Sendable, Equatable {
    public let userAgent: String?
    public let endpointPath: String
    public let explicitAdapterHeader: String?
    public let systemPromptPrefix: String?
    public let anthropicVersionHeader: String?

    public init(
        userAgent: String? = nil,
        endpointPath: String = "",
        explicitAdapterHeader: String? = nil,
        systemPromptPrefix: String? = nil,
        anthropicVersionHeader: String? = nil
    ) {
        self.userAgent = userAgent
        self.endpointPath = endpointPath
        self.explicitAdapterHeader = explicitAdapterHeader
        self.systemPromptPrefix = systemPromptPrefix
        self.anthropicVersionHeader = anthropicVersionHeader
    }
}

public struct HarnessRoutingTable: Sendable, Equatable, Codable {
    public var harnessToAdapter: [AgentHarnessKind: String]
    public var defaultAdapter: String?
    public var thinkingOffWhenHarnessAdapterActive: Bool

    public static let empty = HarnessRoutingTable()

    public var isEmpty: Bool {
        harnessToAdapter.isEmpty && defaultAdapter == nil
    }

    public init(
        harnessToAdapter: [AgentHarnessKind: String] = [:],
        defaultAdapter: String? = nil,
        thinkingOffWhenHarnessAdapterActive: Bool = true
    ) {
        self.harnessToAdapter = harnessToAdapter
        self.defaultAdapter = defaultAdapter
        self.thinkingOffWhenHarnessAdapterActive = thinkingOffWhenHarnessAdapterActive
    }

    public static func parseMappings(_ specs: [String]) throws(BasaltError) -> [AgentHarnessKind: String] {
        var result: [AgentHarnessKind: String] = [:]
        let known = AgentHarnessKind.allCases.map(\.rawValue).joined(separator: ", ")
        for spec in specs {
            let parts = spec.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false).map(String.init)
            guard parts.count == 2 else {
                throw .invalidRequestStructure(details: "--harness-adapter takes harness=adapter, got '\(spec)'")
            }
            let rawHarness = parts[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let adapter = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
            guard let harness = AgentHarnessKind(rawValue: rawHarness) else {
                throw .invalidRequestStructure(details: "--harness-adapter unknown harness '\(rawHarness)'; expected one of: \(known)")
            }
            guard !adapter.isEmpty else {
                throw .invalidRequestStructure(details: "--harness-adapter \(harness.rawValue) requires a non-empty adapter name")
            }
            guard result[harness] == nil else {
                throw .invalidRequestStructure(details: "--harness-adapter \(harness.rawValue) is specified more than once")
            }
            result[harness] = adapter
        }
        return result
    }

    public func validate(registeredAdapters: Set<String>) throws(BasaltError) {
        for (harness, adapter) in harnessToAdapter.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            guard !adapter.isEmpty else {
                throw .invalidRequestStructure(details: "--harness-adapter \(harness.rawValue) requires a non-empty adapter name")
            }
            guard registeredAdapters.contains(adapter) else {
                throw .invalidRequestStructure(details: "--harness-adapter \(harness.rawValue)=\(adapter) references unregistered adapter '\(adapter)'; register it with --adapter \(adapter)=<path-or-hf:org/repo>")
            }
        }
        if let defaultAdapter {
            let trimmed = defaultAdapter.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                throw .invalidRequestStructure(details: "--default-adapter must not be empty")
            }
            guard registeredAdapters.contains(trimmed) else {
                throw .invalidRequestStructure(details: "--default-adapter \(trimmed) references unregistered adapter '\(trimmed)'; register it with --adapter \(trimmed)=<path-or-hf:org/repo>")
            }
        }
    }
}

public struct HarnessRouteDecision: Sendable, Equatable {
    public let adapterName: String?
    public let detectedHarness: AgentHarnessKind?
    public let defaultThinkingOff: Bool
    public let promptCacheNamespace: String

    public init(
        adapterName: String?,
        detectedHarness: AgentHarnessKind?,
        defaultThinkingOff: Bool,
        promptCacheNamespace: String
    ) {
        self.adapterName = adapterName
        self.detectedHarness = detectedHarness
        self.defaultThinkingOff = defaultThinkingOff
        self.promptCacheNamespace = promptCacheNamespace
    }
}

public enum HarnessRoute: Sendable {
    public static let baseNamespace = "base"

    public static func detectHarness(_ fingerprint: HarnessRequestFingerprint) -> AgentHarnessKind? {
        if let fromUA = detectFromUserAgent(fingerprint.userAgent) {
            return fromUA
        }
        return detectFromSystemAndEndpoint(fingerprint)
    }

    public static func resolve(
        _ fingerprint: HarnessRequestFingerprint,
        table: HarnessRoutingTable
    ) -> HarnessRouteDecision {
        let detected = detectHarness(fingerprint)
        let adapterName = resolveAdapterName(fingerprint: fingerprint, detected: detected, table: table)
        let defaultThinkingOff = adapterName != nil && table.thinkingOffWhenHarnessAdapterActive
        let namespace = adapterName.map { "adapter:\($0)" } ?? baseNamespace
        return HarnessRouteDecision(
            adapterName: adapterName,
            detectedHarness: detected,
            defaultThinkingOff: defaultThinkingOff,
            promptCacheNamespace: namespace
        )
    }

    private static func resolveAdapterName(
        fingerprint: HarnessRequestFingerprint,
        detected: AgentHarnessKind?,
        table: HarnessRoutingTable
    ) -> String? {
        if let explicit = fingerprint.explicitAdapterHeader?.trimmingCharacters(in: .whitespacesAndNewlines), !explicit.isEmpty {
            let lowered = explicit.lowercased()
            if lowered == "none" || lowered == "base" { return nil }
            return explicit
        }
        if let detected, let mapped = table.harnessToAdapter[detected] {
            return mapped
        }
        if let fallback = table.defaultAdapter?.trimmingCharacters(in: .whitespacesAndNewlines), !fallback.isEmpty {
            return fallback
        }
        return nil
    }

    private static func detectFromUserAgent(_ userAgent: String?) -> AgentHarnessKind? {
        guard let raw = userAgent?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return nil }
        let ua = raw.lowercased()
        if ua.contains("claude-cli") || ua.contains("claude-code") { return .claudeCode }
        if ua.contains("opencode") { return .openCode }
        if ua.contains("pi-coding-agent") || ua.contains("pi-agent") || hasProductToken(ua, named: "pi") { return .pi }
        if ua.contains("hermes") { return .hermes }
        return nil
    }

    private static func detectFromSystemAndEndpoint(_ fingerprint: HarnessRequestFingerprint) -> AgentHarnessKind? {
        let system = fingerprint.systemPromptPrefix?.lowercased() ?? ""
        guard !system.isEmpty else { return nil }
        let isAnthropicMessages = fingerprint.endpointPath.lowercased().hasSuffix("/messages")
        let hasAnthropicVersion = !(fingerprint.anthropicVersionHeader?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
        if isAnthropicMessages, hasAnthropicVersion, isClaudeCodeSystem(system) {
            return .claudeCode
        }
        if isOpenCodeSystem(system) { return .openCode }
        if isPiSystem(system) { return .pi }
        if isHermesSystem(system) { return .hermes }
        return nil
    }

    private static func isClaudeCodeSystem(_ system: String) -> Bool {
        system.contains("you are claude code")
            || system.contains("claude code, anthropic's official cli")
            || system.contains("anthropic's official cli for claude")
    }

    private static func isOpenCodeSystem(_ system: String) -> Bool {
        system.contains("you are opencode")
            || system.contains("opencode, the best coding agent")
            || system.contains("opencode, an interactive cli")
    }

    private static func isPiSystem(_ system: String) -> Bool {
        system.contains("operating inside pi, a coding agent harness")
            || system.contains("you are pi, a coding agent")
    }

    private static func isHermesSystem(_ system: String) -> Bool {
        system.contains("you are hermes agent")
            || system.contains("you are hermes, an ai agent")
    }

    private static func hasProductToken(_ userAgent: String, named name: String) -> Bool {
        let prefix = name + "/"
        let delimiters = CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: "(),;"))
        for token in userAgent.components(separatedBy: delimiters) where !token.isEmpty {
            if token == name || token.hasPrefix(prefix) { return true }
        }
        return false
    }
}
