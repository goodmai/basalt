import Foundation
import BasaltCore

extension SteerGeometry {
    static let names = Self.allCases.map(\.rawValue)

    static func parse(in args: [String: Any]?) throws(MCPFailure) -> SteerGeometry {
        guard let raw = try MCPArguments.string("geometry", in: args) else { return .additive }
        guard let geometry = SteerGeometry(rawValue: raw) else {
            throw .invalidParams("geometry must be one of: \(names.joined(separator: ", "))")
        }
        return geometry
    }
}

extension ContextSteerGate {
    static func parse(in args: [String: Any]?) throws(MCPFailure) -> ContextSteerGate? {
        guard let raw = MCPArguments.present("contextGate", in: args) else { return nil }
        guard let obj = raw as? [String: Any] else {
            throw .invalidParams("contextGate must be an object")
        }
        let vector = try MCPArguments.numbers("vector", in: obj)
        let threshold = try MCPArguments.number("threshold", in: obj) ?? 0.25
        let sharpness = try MCPArguments.number("sharpness", in: obj) ?? 20
        guard sharpness > 0 else {
            throw .invalidParams("contextGate.sharpness must be positive")
        }
        return ContextSteerGate(contextVector: vector, threshold: threshold, sharpness: sharpness)
    }
}

extension PolarityProbe {
    static let kindNames = PolarityProbeKind.allCases.map(\.rawValue)

    static func parseList(in args: [String: Any]?) throws(MCPFailure) -> [PolarityProbe]? {
        guard let raw = MCPArguments.present("polarityProbes", in: args) else { return nil }
        guard let items = raw as? [[String: Any]], (1 ... maxProbes).contains(items.count) else {
            throw .invalidParams("polarityProbes must be an array of 1...\(maxProbes) {kind, prompt} objects")
        }
        var probes: [PolarityProbe] = []
        for item in items {
            let kindRaw = try MCPArguments.required("polarityProbes.kind", MCPArguments.string("kind", in: item))
            guard let kind = PolarityProbeKind(rawValue: kindRaw) else {
                throw .invalidParams("polarityProbes.kind must be one of: \(kindNames.joined(separator: ", "))")
            }
            let prompt = try MCPArguments.required("polarityProbes.prompt", MCPArguments.string("prompt", in: item))
            guard !prompt.isEmpty else {
                throw .invalidParams("polarityProbes.prompt must be non-empty")
            }
            probes.append(PolarityProbe(kind: kind, prompt: prompt))
        }
        return probes
    }
}

extension Intervention {
    static let kinds = ["steer", "ablate_head", "ablate_expert"]

    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let layer = try MCPArguments.required("layer", MCPArguments.integer("layer", in: args, ResearchCall.nonNegative))
        switch try MCPArguments.string("kind", in: args) {
        case "steer"?:
            let vector = try MCPArguments.required("vector", MCPArguments.numbers("vector", in: args))
            let scale = try MCPArguments.number("scale", in: args) ?? 1
            let geometry = try SteerGeometry.parse(in: args)
            let contextGate = try ContextSteerGate.parse(in: args)
            self = .steer(layer: layer, vector: vector, scale: scale, geometry: geometry, contextGate: contextGate)
        case "ablate_head"?:
            self = .ablateHead(layer: layer, head: try MCPArguments.required("head", MCPArguments.integer("head", in: args, ResearchCall.nonNegative)))
        case "ablate_expert"?:
            self = .ablateExpert(layer: layer, expert: try MCPArguments.required("expert", MCPArguments.integer("expert", in: args, ResearchCall.nonNegative)))
        default:
            throw .invalidParams("kind must be one of: \(Self.kinds.joined(separator: ", "))")
        }
        if let when = MCPArguments.present("when", in: args) { self = .conditional(try ConditionalIntervention(self, mcp: when)) }
    }
}

extension ResearchTool {
    static var contextGateProperty: [String: Any] {
        [
            "type": "object",
            "description": "Context-gated steering scale alpha(h_t) = alpha_0 * sigmoid(sharpness * (<h_hat, v_ctx_hat> - threshold))",
            "properties": [
                "vector": numbers.merging(["description": "Context cone direction (default: steering vector)"]) { $1 },
                "threshold": ["type": "number", "description": "Cosine threshold theta (default 0.25)"],
                "sharpness": ["type": "number", "exclusiveMinimum": 0, "description": "Sigmoid sharpness gamma (default 20)"],
            ],
        ]
    }

    static var polarityProbesProperty: [String: Any] {
        [
            "type": "array",
            "minItems": 1,
            "maxItems": PolarityProbe.maxProbes,
            "description": "Polarity-flipped probes ({kind: user_pos|user_neg|sys, prompt}) for anti-steering measurement",
            "items": [
                "type": "object",
                "required": ["kind", "prompt"],
                "properties": [
                    "kind": ["type": "string", "enum": PolarityProbe.kindNames],
                    "prompt": ["type": "string"],
                ],
            ],
        ]
    }

    static func steerPromptsSchema(_ range: ClosedRange<Int>, _ description: String) -> [String: Any] {
        ["type": "array", "items": ["type": "string"], "minItems": range.lowerBound, "maxItems": range.upperBound, "description": description]
    }

    static var steerSearchProperties: [String: Any] {
        [
            "positive": steerPromptsSchema(ContrastRequest.classPrompts, "Prompts showing the behaviour to find"),
            "negative": steerPromptsSchema(ContrastRequest.classPrompts, "Prompts without it"),
            "heldOut": steerPromptsSchema(ContrastRequest.heldOutPrompts, "Prompts steered in the trials"),
            "layers": ["type": "array", "items": ["type": "integer", "minimum": 0], "description": "Layers to search (default: all)"],
            "topHeads": ["type": "integer", "minimum": ContrastRequest.headLimit.lowerBound, "maximum": ContrastRequest.headLimit.upperBound, "description": "Heads reported, ranked by ablation effect (default 8; 0 skips the ablations)"],
            "scales": numbers.merging(["maxItems": ContrastRequest.scaleCount.upperBound, "description": "Steering scales tried on the held-out prompts (default [2, 4, 8])"]) { $1 },
            "geometry": geometryProperty,
            "protectedPrompts": steerPromptsSchema(1...ContrastRequest.classPrompts.upperBound, "Anchor reasoning/math prompts whose cognitive subspace U_cog is projected out"),
            "nullspaceMaxRank": ["type": "integer", "minimum": 1, "description": "Maximum basis rank r for U_cog"],
            "contextGate": contextGateProperty,
            "evaluatePolarityFlips": ["type": "boolean", "description": "Expand heldOut into user_pos, user_neg and sys anti-steering probes (default false)"],
            "polarityProbes": polarityProbesProperty,
        ]
    }

    static var steerCommitProperties: [String: Any] {
        [
            "layer": ["type": "integer", "minimum": 0, "description": "bestLayer from steer_search"],
            "direction": numbers.merging(["description": "direction from steer_search (hidden size)"]) { $1 },
            "scale": ["type": "number", "description": "Scale chosen from the steer_search trials"],
            "positive": steerPromptsSchema(1...ContrastRequest.classPrompts.upperBound, "Positive prompts; their mean module input becomes the rank-1 input"),
            "negative": steerPromptsSchema(1...ContrastRequest.classPrompts.upperBound, "Optional negative prompts anchoring the contrastive logit-diff tokens"),
            "protectedPrompts": steerPromptsSchema(1...ContrastRequest.classPrompts.upperBound, "Protected reasoning/math prompts whose cognitive subspace is preserved"),
            "nullspaceMaxRank": ["type": "integer", "minimum": 1, "description": "Maximum basis rank r for U_cog"],
            "polarityProbes": polarityProbesProperty,
            "maxAntiSteeringRate": ["type": "number", "minimum": 0, "maximum": 1, "description": "Reject and undo commit when anti-steering rate exceeds this threshold"],
            "heldOut": steerPromptsSchema(ContrastRequest.heldOutPrompts, "Held-out prompts for the quality gate"),
            "maxKL": ["type": "number", "minimum": 0, "description": "Gate: largest mean KL against the baseline (default 0.5)"],
            "maxPerplexityRise": ["type": "number", "minimum": 0, "description": "Gate: largest relative perplexity rise (default 0.1)"],
        ]
    }
}
