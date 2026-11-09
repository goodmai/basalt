import Foundation
import BasaltCore

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

extension TokenTraceRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        self.init(prompt: try MCPArguments.required("prompt", MCPArguments.string("prompt", in: args)))
        maxTokens = try MCPArguments.integer("maxTokens", in: args, Self.generatedTokens) ?? maxTokens
        layers = try MCPArguments.integers("layers", in: args)
        heads = try MCPArguments.optionalBoolean("heads", in: args) ?? heads
        experts = try MCPArguments.optionalBoolean("experts", in: args) ?? experts
        attentionEntropy = try MCPArguments.optionalBoolean("attentionEntropy", in: args) ?? attentionEntropy
        lensTopK = try MCPArguments.integer("lensTopK", in: args, Self.lensLimit) ?? lensTopK
        topK = try MCPArguments.integer("topK", in: args, Self.summaryTopK) ?? topK
        maxFileBytes = try MCPArguments.integer("maxFileBytes", in: args, Self.fileBytes) ?? maxFileBytes
        projections = try Self.directions(args)
    }

    static func directions(_ args: [String: Any]?) throws(MCPFailure) -> [TraceDirection] {
        guard let value = MCPArguments.present("projections", in: args) else { return [] }
        guard let items = value as? [[String: Any]], projectionLimit.contains(items.count) else {
            throw .invalidParams("projections must be an array of at most \(projectionLimit.upperBound) {layer, vector, threshold} objects")
        }
        var directions: [TraceDirection] = []
        for item in items {
            directions.append(TraceDirection(
                layer: try MCPArguments.required("layer", MCPArguments.integer("layer", in: item, ResearchCall.nonNegative)),
                vector: try MCPArguments.required("vector", MCPArguments.numbers("vector", in: item)),
                threshold: try MCPArguments.number("threshold", in: item) ?? 0))
        }
        return directions
    }
}

extension ActivationContrastRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        self.init(
            a: try ContrastRequest.prompts("a", in: args, ContrastRequest.classPrompts),
            b: try ContrastRequest.prompts("b", in: args, ContrastRequest.classPrompts))
        maxTokens = try MCPArguments.integer("maxTokens", in: args, Self.generatedTokens) ?? maxTokens
        layers = try MCPArguments.integers("layers", in: args)
        onsetWindow = try MCPArguments.integer("onsetWindow", in: args, Self.onsetTokens) ?? onsetWindow
        topK = try MCPArguments.integer("topK", in: args, Self.ranked) ?? topK
    }
}
