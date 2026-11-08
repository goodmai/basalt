import BasaltCore

extension DistillRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let heldOut = MCPArguments.present("heldOut", in: args)
        guard let heldOut = heldOut as? [String], ResearchCall.comparePrompts.contains(heldOut.count) else {
            throw .invalidParams("heldOut must be an array of \(ResearchCall.comparePrompts) general prompts the quality gate checks the adapter on")
        }
        var gate = GateRequest(heldOut: heldOut)
        try gate.readLimits(args)
        var teacher = TeacherSpec(
            url: try MCPArguments.required("teacherUrl", MCPArguments.string("teacherUrl", in: args)),
            model: try MCPArguments.required("teacherModel", MCPArguments.string("teacherModel", in: args)))
        if let provider = try MCPArguments.string("provider", in: args) {
            guard let parsed = TeacherProvider(rawValue: provider) else { throw .invalidParams("provider must be openai or anthropic") }
            teacher.provider = parsed
        }
        teacher.maxTokens = try MCPArguments.integer("maxTeacherTokens", in: args, ResearchCall.nonNegative) ?? teacher.maxTokens
        teacher.temperature = try MCPArguments.number("temperature", in: args) ?? teacher.temperature
        teacher.concurrency = try MCPArguments.integer("concurrency", in: args, ResearchCall.nonNegative) ?? teacher.concurrency
        teacher.inputPricePerMillion = try MCPArguments.number("inputPricePerMillion", in: args).map(Double.init)
        teacher.outputPricePerMillion = try MCPArguments.number("outputPricePerMillion", in: args).map(Double.init)
        self.init(prompts: try MCPArguments.required("prompts", MCPArguments.string("prompts", in: args)), teacher: teacher, gate: gate)
        try readTraining(args)
    }

    mutating func readTraining(_ args: [String: Any]?) throws(MCPFailure) {
        if let filter = try MCPArguments.string("filter", in: args) {
            guard let parsed = DistillFilter(rawValue: filter) else { throw .invalidParams("filter must be none or verify") }
            self.filter = parsed
        }
        samplesPerPrompt = try MCPArguments.integer("samplesPerPrompt", in: args, ResearchCall.nonNegative) ?? samplesPerPrompt
        heldOutFraction = try MCPArguments.number("heldOutFraction", in: args) ?? heldOutFraction
        budgetUSD = try MCPArguments.number("budgetUSD", in: args).map(Double.init)
        layers = try MCPArguments.integers("layers", in: args)
        rank = try MCPArguments.integer("rank", in: args, ResearchCall.nonNegative) ?? rank
        iterations = try MCPArguments.integer("iterations", in: args, ResearchCall.nonNegative) ?? iterations
        batchSize = try MCPArguments.integer("batchSize", in: args, ResearchCall.nonNegative) ?? batchSize
        learningRate = try MCPArguments.number("learningRate", in: args) ?? learningRate
        maxSeqLen = try MCPArguments.integer("maxSeqLen", in: args, ResearchCall.nonNegative) ?? maxSeqLen
        output = try MCPArguments.string("output", in: args)
    }
}
