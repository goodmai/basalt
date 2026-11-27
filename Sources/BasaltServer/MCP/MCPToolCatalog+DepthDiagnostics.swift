import BasaltCore
import Foundation

extension MCPToolCatalog {
    static let depthDiagnosticsDescriptor = MCPToolDescriptor(
        name: "depth_diagnostics",
        description: "Read-only X-ray depth diagnostics over one prompt forward pass without mutating weights: per-layer residual L2 norm growth, attention and MLP sublayer contribution norms and ratio, per-head pre-softmax maximum attention logit S_max, attention logit softcap saturation rate when present, and optional Roy-Vetterli effective rank of each layer's residual matrix (bounded to seqLen <= 512).",
        inputSchema: ResearchTool.object(
            [
                "prompt": ["type": "string", "description": "Prompt text to inspect"],
                "includeRank": ["type": "boolean", "description": "Compute Roy-Vetterli effective rank via CPU SVD (bounded to seqLen <= 512, default true)"],
                "softcapThresholdFraction": ["type": "number", "minimum": 0, "maximum": 1, "description": "Saturation threshold fraction of softcap for |tanh(z/softcap)| (default 0.95)"],
                "maxSequenceLength": [
                    "type": "integer",
                    "minimum": DepthDiagnosticsRequest.sequenceLengthRange.lowerBound,
                    "maximum": DepthDiagnosticsRequest.sequenceLengthRange.upperBound,
                    "description": "Maximum prompt tokens evaluated (default 512)",
                ],
                "layers": ["type": "array", "items": ["type": "integer", "minimum": 0], "description": "Layer indices to report (default: all)"],
            ],
            required: ["prompt"]
        )
    )

    static var depthDiagnosticsTool: MCPToolDescriptor { depthDiagnosticsDescriptor }

    static var depthDiagnosticsToolDefinition: ToolDefinition {
        let json = (try? MCPWire.encoder.encode(depthDiagnosticsDescriptor.inputSchema))
            .map { String(decoding: $0, as: UTF8.self) }
        return ToolDefinition(
            name: depthDiagnosticsDescriptor.name,
            description: depthDiagnosticsDescriptor.description,
            parametersJSON: json
        )
    }

    static func parseDepthDiagnosticsRequest(_ arguments: [String: Any]?) throws(MCPFailure) -> DepthDiagnosticsRequest {
        try DepthDiagnosticsRequest(mcp: arguments)
    }
}

extension DepthDiagnosticsRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let prompt = try MCPArguments.required("prompt", MCPArguments.string("prompt", in: args))
        guard !prompt.isEmpty else {
            throw .invalidParams("prompt must be non-empty")
        }
        let includeRank = try MCPArguments.optionalBoolean("includeRank", in: args) ?? true
        let threshold = try MCPArguments.double("softcapThresholdFraction", in: args) ?? 0.95
        guard threshold.isFinite, threshold > 0, threshold <= 1.0 else {
            throw .invalidParams("softcapThresholdFraction must be in (0, 1]")
        }
        let maxSeqLen = try MCPArguments.integer("maxSequenceLength", in: args, Self.sequenceLengthRange) ?? 512
        let layers = try MCPArguments.integers("layers", in: args)
        if let layers, layers.isEmpty {
            throw .invalidParams("layers must list at least one layer index")
        }
        self.init(
            prompt: prompt,
            includeRank: includeRank,
            softcapThresholdFraction: threshold,
            maxSequenceLength: maxSeqLen,
            layers: layers
        )
    }
}
