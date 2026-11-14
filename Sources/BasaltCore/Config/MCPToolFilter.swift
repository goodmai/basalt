import Foundation

public struct MCPToolFilter: Sendable, Equatable {
    public enum Group: String, Sendable, CaseIterable {
        case serving
        case gemma
        case media
        case harness
        case xray
        case surgery
        case research
        case all

        public var tools: Set<String> {
            switch self {
            case .serving:
                ["status", "check_hardware", "inspect_model", "switch_model", "basalt_generate", "basalt_status", "telemetry_query"]
            case .gemma:
                ["gemma_generate", "gemma_status", "gemma_add_knowledge"]
            case .media:
                ["basalt_transcribe", "basalt_systemone_decide", "basalt_ocr", "basalt_pipeline"]
            case .harness:
                ["basalt_add_knowledge", "harness_validate", "harness_optimize", "harness_library_query"]
            case .xray:
                ["model_describe", "model_tensors", "model_tensor", "kv_inspect", "trace_run"]
            case .surgery:
                [
                    "weights_delta", "adapter_apply", "steer_search", "steer_commit",
                    "tensor_pack", "tensor_unpack", "gate_check", "overlays_list", "overlay_undo",
                ]
            case .research:
                Self.surgery.tools.union([
                    "intervene", "experiment_compare", "train_experts", "train_jepa",
                    "distill", "trace_tokens", "activation_contrast", "introspect",
                ])
            case .all:
                Self.serving.tools
                    .union(Self.gemma.tools)
                    .union(Self.media.tools)
                    .union(Self.harness.tools)
                    .union(Self.xray.tools)
                    .union(Self.research.tools)
            }
        }
    }

    public struct InvalidSelectorError: Error, Sendable, Equatable, CustomStringConvertible, LocalizedError {
        public let unknown: [String]

        public init(unknown: [String]) {
            self.unknown = unknown
        }

        public var description: String {
            let names = unknown.joined(separator: ", ")
            let groups = Group.allCases.map(\.rawValue).joined(separator: ", ")
            return "unknown MCP tool or group: \(names) (groups: \(groups); or an MCP tool name such as basalt_generate, switch_model, model_describe, weights_delta)"
        }

        public var errorDescription: String? { description }
    }

    public static let allTools: Set<String> = Group.all.tools
    public static let experimentTools: Set<String> = Group.research.tools
    public static let defaultBaseTools: Set<String> = allTools.subtracting(experimentTools)
    public static let standard = MCPToolFilter()

    public let enabled: Set<String>
    public let disabled: Set<String>

    public init() {
        self.enabled = []
        self.disabled = []
    }

    public init(enable: [String] = [], disable: [String] = []) throws(InvalidSelectorError) {
        let enableTokens = Self.splitTokens(enable)
        let disableTokens = Self.splitTokens(disable)
        var bad: [String] = []
        let enabledSet = Self.expand(enableTokens, unknown: &bad)
        let disabledSet = Self.expand(disableTokens, unknown: &bad)
        if !bad.isEmpty { throw InvalidSelectorError(unknown: bad) }
        self.enabled = enabledSet
        self.disabled = disabledSet
    }

    public func isExplicitlyDisabled(_ tool: String) -> Bool {
        disabled.contains(tool)
    }

    public func allows(_ tool: String, researchTools: Bool) -> Bool {
        guard Self.allTools.contains(tool), !disabled.contains(tool) else { return false }
        if Self.defaultBaseTools.contains(tool) { return true }
        return researchTools || enabled.contains(tool)
    }

    public func allowsSpectrum(researchTools: Bool) -> Bool {
        guard allows("model_tensor", researchTools: researchTools),
              !disabled.isSuperset(of: Self.experimentTools) else { return false }
        return researchTools || !enabled.isDisjoint(with: Self.experimentTools)
    }

    public static func splitTokens(_ raw: [String]) -> [String] {
        raw.flatMap { $0.split(separator: ",", omittingEmptySubsequences: false) }
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    private static func expand(_ tokens: [String], unknown: inout [String]) -> Set<String> {
        var out: Set<String> = []
        for token in tokens {
            if let group = Group(rawValue: token) {
                out.formUnion(group.tools)
            } else if allTools.contains(token) {
                out.insert(token)
            } else if !unknown.contains(token) {
                unknown.append(token)
            }
        }
        return out
    }
}
