import Foundation
import BasaltCore

extension MCPToolCatalog {
    static let outlierProfileToolName = "outlier_profile"

    static var outlierProfileDescriptor: MCPToolDescriptor {
        MCPToolDescriptor(
            name: outlierProfileToolName,
            description: "Measure per-layer activation excess kurtosis, crest factor, quantizability risk tier (safe, watch, sensitive), SwiGLU/GeGLU dead-neuron fraction, and per-head attention sink mass across prompts via ForwardProbe without mutating weights.",
            inputSchema: ResearchTool.object([
                "prompts": [
                    "type": "array",
                    "items": ["type": "string"],
                    "minItems": OutlierProfileRequest.promptCount.lowerBound,
                    "maxItems": OutlierProfileRequest.promptCount.upperBound,
                    "description": "Prompts to profile across layers",
                ],
                "sinkTokenCount": [
                    "type": "integer",
                    "minimum": OutlierProfileRequest.sinkTokenRange.lowerBound,
                    "maximum": OutlierProfileRequest.sinkTokenRange.upperBound,
                    "description": "Leading sink token positions to sum attention mass over (default 1)",
                ],
                "deadNeuronThreshold": [
                    "type": "number",
                    "minimum": 0,
                    "description": "Absolute magnitude threshold below which an MLP neuron is counted dead (default 1e-3)",
                ],
                "layers": [
                    "type": "array",
                    "items": ["type": "integer", "minimum": 0],
                    "description": "Layer indices to profile (default: all)",
                ],
            ], required: ["prompts"])
        )
    }
}

extension OutlierProfileRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let prompts = try Self.parsePrompts(in: args)
        let sinkTokenCount = try MCPArguments.integer("sinkTokenCount", in: args, Self.sinkTokenRange) ?? 1
        let deadNeuronThreshold = try Self.parseDeadNeuronThreshold(in: args)
        let layers = try MCPArguments.integers("layers", in: args)
        if let layers, layers.isEmpty {
            throw .invalidParams("layers must be non-empty when specified")
        }
        self.init(
            prompts: prompts,
            sinkTokenCount: sinkTokenCount,
            deadNeuronThreshold: deadNeuronThreshold,
            layers: layers
        )
    }

    private static func parsePrompts(in args: [String: Any]?) throws(MCPFailure) -> [String] {
        guard let prompts = MCPArguments.present("prompts", in: args) as? [String],
              Self.promptCount.contains(prompts.count) else {
            throw .invalidParams("prompts must be an array of \(Self.promptCount) strings")
        }
        guard prompts.allSatisfy({ !$0.isEmpty }) else {
            throw .invalidParams("prompts must contain non-empty strings")
        }
        return prompts
    }

    private static func parseDeadNeuronThreshold(in args: [String: Any]?) throws(MCPFailure) -> Double {
        guard let value = try MCPArguments.double("deadNeuronThreshold", in: args) else {
            return 1e-3
        }
        guard value >= 0 else {
            throw .invalidParams("deadNeuronThreshold must be non-negative")
        }
        return value
    }
}
