import BasaltCore

extension LeJEPATrainRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        self.init(
            datasetPath: try MCPArguments.required("datasetPath", MCPArguments.string("datasetPath", in: args)),
            iterations: try MCPArguments.integer("iterations", in: args, ResearchCall.nonNegative) ?? 100)
        output = try MCPArguments.string("output", in: args)
        layers = try MCPArguments.integers("layers", in: args)
        rank = try MCPArguments.integer("rank", in: args, ResearchCall.nonNegative) ?? rank
        batchSize = try MCPArguments.integer("batchSize", in: args, ResearchCall.nonNegative) ?? batchSize
        maxSeqLen = try MCPArguments.integer("maxSeqLen", in: args, ResearchCall.nonNegative) ?? maxSeqLen
        learningRate = try MCPArguments.number("learningRate", in: args) ?? learningRate
        weightDecay = try MCPArguments.number("weightDecay", in: args) ?? weightDecay
        warmupRatio = try MCPArguments.number("warmupRatio", in: args) ?? warmupRatio
        lambda = try MCPArguments.number("lambda", in: args) ?? lambda
        maxGradNorm = try MCPArguments.number("maxGradNorm", in: args) ?? maxGradNorm
        isCausal = try MCPArguments.optionalBoolean("isCausal", in: args) ?? isCausal
    }
}
