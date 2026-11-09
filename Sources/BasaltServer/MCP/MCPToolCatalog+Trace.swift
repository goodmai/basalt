import Foundation
import BasaltCore

extension ResearchTool {
    static let traceTokensSummary = "Experiment: per-token trace over the prompt and a greedy generation, on the lab path. For every token and traced layer it records each head's output norm before the output projection (fused attention stays on), the routed experts with their gate weights, router entropy and top-two gate margin, optional residual projections on given directions, optional attention entropy (slower) and an optional sparse logit lens. Writes ~/.basalt/traces/<time>-<id>.trace (mode 0600): one JSON header line with the layout, one JSON line per token (id, text, integer columns), then a float32 little-endian block of tokens x floatsPerToken. Returns the file path and a summary: per layer the most active heads, the most used experts, mean router entropy and margin, and the positions where each projection crosses its threshold. Token, layer and file-size caps are checked before the run."

    static var traceTokensProperties: [String: Any] {
        [
            "prompt": ["type": "string", "description": "Prompt text, tokenized as is (apply the chat template yourself)"],
            "maxTokens": ["type": "integer", "minimum": TokenTraceRequest.generatedTokens.lowerBound, "maximum": TokenTraceRequest.generatedTokens.upperBound, "description": "Greedy tokens generated and traced after the prompt (default 32); prompt plus generation at most \(TokenTraceRequest.tokenLimit)"],
            "layers": ["type": "array", "items": ["type": "integer", "minimum": 0], "description": "Layers to trace (default: all)"],
            "heads": ["type": "boolean", "description": "Per-head output L2 norms (default true)"],
            "experts": ["type": "boolean", "description": "Routed expert ids, gates, router entropy and top-two margin (default true)"],
            "attentionEntropy": ["type": "boolean", "description": "Per-head attention entropy; computes the attention weights explicitly, so it is slower (default false)"],
            "projections": [
                "type": "array", "maxItems": TokenTraceRequest.projectionLimit.upperBound,
                "description": "Residual · direction after a traced layer, e.g. the direction from steer_search; positions above threshold are summarised",
                "items": ["type": "object", "properties": ["layer": ["type": "integer", "minimum": 0], "vector": numbers, "threshold": ["type": "number"]], "required": ["layer", "vector"]],
            ],
            "lensTopK": ["type": "integer", "minimum": TokenTraceRequest.lensLimit.lowerBound, "maximum": TokenTraceRequest.lensLimit.upperBound, "description": "Logit-lens token ids and probabilities per traced layer and token (default 0: off)"],
            "topK": ["type": "integer", "minimum": TokenTraceRequest.summaryTopK.lowerBound, "maximum": TokenTraceRequest.summaryTopK.upperBound, "description": "Heads and experts per layer in the summary (default 5)"],
            "maxFileBytes": ["type": "integer", "minimum": TokenTraceRequest.fileBytes.lowerBound, "maximum": TokenTraceRequest.fileBytes.upperBound, "description": "Refuse the run when the trace file would be larger (default \(TokenTraceRequest.defaultFileBytes))"],
        ]
    }
}
