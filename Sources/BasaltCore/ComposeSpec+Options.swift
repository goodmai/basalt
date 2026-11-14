import Foundation

extension ComposeSpec {
    public struct Experimental: Sendable, Equatable {
        public var zeroCopyBody = false
        public var smallMKernel = false
        public var speechAnalyzer = false
        public var imageFallback: String?
        public var aneVisionTower: String?
        public var telemetryConfidenceEvery: Int?
        public var adaptiveThinkingBudget = false
        public var samplers = false
        public var mtpDraftHead = false
        public var expertStreaming = false
        public var expertCacheGB: Double?
        public var kvHeadSplit: Double?
        public var questTopK: Int?

        public init() {}
    }

    public struct MCP: Sendable, Equatable {
        public var enabled = false
        public var rest = false
        public var researchTools = false
        public var enable: [String] = []
        public var disable: [String] = []
        public var imageDirs: [String] = []

        public init() {}
    }

    public var usesSpeechAnalyzer: Bool {
        experimental.speechAnalyzer || audio.contains(Self.speechAnalyzer)
    }

    var experimentalIssues: [Issue] {
        var issues: [Issue] = []
        if let fallback = experimental.imageFallback, fallback != "ocr" {
            issues.append(Issue("experimental.image_fallback", "must be ocr"))
        }
        if let every = experimental.telemetryConfidenceEvery, every < 0 {
            issues.append(Issue("experimental.telemetry_confidence_every", "must be 0 or a positive integer"))
        }
        if let tower = experimental.aneVisionTower, !Self.isVisionTowerBundle(tower) {
            issues.append(Issue("experimental.ane_vision_tower", "\(tower) is not a Core AI vision tower bundle (metadata.json and tower.aimodel)"))
        }
        if let cacheGB = experimental.expertCacheGB {
            if !experimental.expertStreaming {
                issues.append(Issue("experimental.expert_cache_gb", "needs experimental.expert_streaming: true"))
            }
            if !(cacheGB > 0) || !cacheGB.isFinite {
                issues.append(Issue("experimental.expert_cache_gb", "must be positive"))
            }
        }
        if let split = experimental.kvHeadSplit, !(0.05 ... 1).contains(split) {
            issues.append(Issue("experimental.kv_head_split", "must lie between 0.05 and 1"))
        }
        if let topK = experimental.questTopK {
            if !(1 ... 4096).contains(topK) {
                issues.append(Issue("experimental.quest_top_k", "must lie between 1 and 4096"))
            }
            if experimental.kvHeadSplit != nil {
                issues.append(Issue("experimental.quest_top_k", "does not combine with experimental.kv_head_split"))
            }
        }
        return issues
    }

    var mcpIssues: [Issue] {
        var issues: [Issue] = []
        do {
            _ = try MCPToolFilter(enable: mcp.enable)
        } catch {
            issues.append(Issue("mcp.enable", error.description))
        }
        do {
            _ = try MCPToolFilter(disable: mcp.disable)
        } catch {
            issues.append(Issue("mcp.disable", error.description))
        }
        return issues
    }

    static func isVisionTowerBundle(_ path: String) -> Bool {
        let directory = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        return FileManager.default.fileExists(atPath: directory.appending(component: "metadata.json").path)
            && FileManager.default.fileExists(atPath: directory.appending(component: "tower.aimodel").path)
    }
}

extension ComposeSpec {
    var optionArguments: [String] {
        var arguments: [String] = []
        for (enabled, flag) in [
            (experimental.zeroCopyBody, "--experimental-zero-copy-body"),
            (experimental.smallMKernel, "--experimental-smallm-kernel"),
            (experimental.adaptiveThinkingBudget, "--experimental-adaptive-thinking-budget"),
            (experimental.samplers, "--experimental-samplers"),
            (experimental.mtpDraftHead, "--experimental-mtp-draft-head"),
            (experimental.expertStreaming, "--experimental-expert-streaming"),
            (mcp.enabled, "--mcp"),
            (mcp.rest, "--rest"),
            (mcp.researchTools, "--research-tools"),
        ] where enabled {
            arguments.append(flag)
        }
        if let fallback = experimental.imageFallback { arguments += ["--image-fallback", fallback] }
        if let tower = experimental.aneVisionTower { arguments += ["--experimental-ane-vision-tower", (tower as NSString).expandingTildeInPath] }
        if let every = experimental.telemetryConfidenceEvery { arguments += ["--telemetry-confidence-every", String(every)] }
        if let cacheGB = experimental.expertCacheGB { arguments += ["--expert-cache-gb", String(cacheGB)] }
        if let split = experimental.kvHeadSplit { arguments += ["--kv-head-split", String(split)] }
        if let topK = experimental.questTopK { arguments += ["--quest-top-k", String(topK)] }
        arguments += mcp.enable.flatMap { ["--mcp-enable", $0] }
        arguments += mcp.disable.flatMap { ["--mcp-disable", $0] }
        arguments += mcp.imageDirs.flatMap { ["--image-dir", ($0 as NSString).expandingTildeInPath] }
        return arguments
    }

    func hotReloadedExperimentalKeys(to next: ComposeSpec) -> [String] {
        var changed: [String] = []
        if usesSpeechAnalyzer != next.usesSpeechAnalyzer { changed.append("speech_analyzer") }
        if experimental.imageFallback != next.experimental.imageFallback { changed.append("image_fallback") }
        if (experimental.telemetryConfidenceEvery ?? 0) != (next.experimental.telemetryConfidenceEvery ?? 0) {
            changed.append("telemetry_confidence_every")
        }
        if experimental.adaptiveThinkingBudget != next.experimental.adaptiveThinkingBudget {
            changed.append("adaptive_thinking_budget")
        }
        return changed
    }

    func experimentalRestartReason(to next: ComposeSpec) -> String? {
        let checks: [(Bool, String)] = [
            (experimental.zeroCopyBody != next.experimental.zeroCopyBody, "experimental.zero_copy_body is configured when the HTTP server starts"),
            (experimental.smallMKernel != next.experimental.smallMKernel, "experimental.smallm_kernel is enabled once for the process at start"),
            (experimental.aneVisionTower != next.experimental.aneVisionTower, "experimental.ane_vision_tower is bound to the engine at start"),
            (experimental.samplers != next.experimental.samplers, "experimental.samplers is bound to the engine configuration at start"),
            (experimental.mtpDraftHead != next.experimental.mtpDraftHead, "experimental.mtp_draft_head is bound to the engine configuration at start"),
            (experimental.expertStreaming != next.experimental.expertStreaming || experimental.expertCacheGB != next.experimental.expertCacheGB,
             "experimental.expert_streaming is bound to the engine configuration at start"),
            (experimental.kvHeadSplit != next.experimental.kvHeadSplit, "experimental.kv_head_split is bound to the engine configuration at start"),
            (experimental.questTopK != next.experimental.questTopK, "experimental.quest_top_k is bound to the engine configuration at start"),
            (mcp != next.mcp, "mcp options are bound to the server transport at start"),
        ]
        return checks.first(where: \.0)?.1
    }
}
