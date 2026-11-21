import Foundation
import BasaltCore
import Yams

extension ComposeFile.Walk {
    mutating func experimental(_ node: Node?, into spec: inout ComposeSpec) {
        let path = "experimental"
        guard let node, let field = fields(node, path: path, section: "experimental") else { return }
        var out = ComposeSpec.Experimental()
        out.zeroCopyBody = flag(field["zero_copy_body"], "\(path).zero_copy_body") ?? false
        out.smallMKernel = flag(field["smallm_kernel"], "\(path).smallm_kernel") ?? false
        out.q3Kernel = flag(field["q3_kernel"], "\(path).q3_kernel") ?? false
        out.speechAnalyzer = flag(field["speech_analyzer"], "\(path).speech_analyzer") ?? false
        out.imageFallback = text(field["image_fallback"], "\(path).image_fallback")
        out.aneVisionTower = text(field["ane_vision_tower"], "\(path).ane_vision_tower")
        out.telemetryConfidenceEvery = integer(field["telemetry_confidence_every"], "\(path).telemetry_confidence_every")
        out.adaptiveThinkingBudget = flag(field["adaptive_thinking_budget"], "\(path).adaptive_thinking_budget") ?? false
        out.samplers = flag(field["samplers"], "\(path).samplers") ?? false
        out.mtpDraftHead = flag(field["mtp_draft_head"], "\(path).mtp_draft_head") ?? false
        out.expertStreaming = flag(field["expert_streaming"], "\(path).expert_streaming") ?? false
        out.expertCacheGB = number(field["expert_cache_gb"], "\(path).expert_cache_gb")
        out.kvHeadSplit = number(field["kv_head_split"], "\(path).kv_head_split")
        out.questTopK = integer(field["quest_top_k"], "\(path).quest_top_k")
        spec.experimental = out
    }

    mutating func mcp(_ node: Node?, into spec: inout ComposeSpec) {
        let path = "mcp"
        guard let node, let field = fields(node, path: path, section: "mcp") else { return }
        var out = ComposeSpec.MCP()
        out.enabled = flag(field["enabled"], "\(path).enabled") ?? false
        out.rest = flag(field["rest"], "\(path).rest") ?? false
        out.researchTools = flag(field["research_tools"], "\(path).research_tools") ?? false
        out.enable = list(field["enable"], "\(path).enable")
        out.disable = list(field["disable"], "\(path).disable")
        out.imageDirs = list(field["image_dirs"], "\(path).image_dirs")
        spec.mcp = out
    }

    mutating func integer(_ node: Node?, _ path: String) -> Int? {
        guard let node else { return nil }
        guard let value = node.int else {
            issue(path, node, "must be an integer")
            return nil
        }
        return value
    }

    mutating func epistemic(_ node: Node, path: String) -> EpistemicGateSpec? {
        guard let field = fields(node, path: path, section: "epistemic") else { return nil }
        var spec = EpistemicGateSpec()
        if let enabled = flag(field["enabled"], "\(path).enabled") { spec.enabled = enabled }
        if let support = number(field["support_threshold"], "\(path).support_threshold") { spec.supportThreshold = support }
        if let refutation = number(field["refutation_threshold"], "\(path).refutation_threshold") { spec.refutationThreshold = refutation }
        if let tool = number(field["tool_threshold"], "\(path).tool_threshold") { spec.toolThreshold = tool }
        if let noEvidence = text(field["on_no_evidence"], "\(path).on_no_evidence") {
            if let policy = EpistemicGateSpec.NoEvidencePolicy(rawValue: noEvidence) {
                spec.onNoEvidence = policy
            } else {
                issue("\(path).on_no_evidence", field["on_no_evidence"]!, "must be abstain or escalate")
            }
        }
        if let conflict = text(field["on_conflict"], "\(path).on_conflict") {
            if let policy = EpistemicGateSpec.ConflictPolicy(rawValue: conflict) {
                spec.onConflict = policy
            } else {
                issue("\(path).on_conflict", field["on_conflict"]!, "must be report or escalate")
            }
        }
        spec.abstainMessage = text(field["abstain_message"], "\(path).abstain_message")
        return spec
    }
}

extension ComposeFile {
    static func resolvedPath(_ path: String, base: URL?) -> String {
        if path == "~" || path.hasPrefix("~/") {
            return (path as NSString).expandingTildeInPath
        }
        if let base, path.hasPrefix("./") || path.hasPrefix("../") {
            return base.appending(path: path).standardizedFileURL.path
        }
        return path
    }
}
