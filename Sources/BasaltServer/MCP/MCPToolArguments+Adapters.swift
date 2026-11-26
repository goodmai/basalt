import BasaltCore

private struct MCPAdapterOptions {
    let oftBlock: Int?
    let oftParamText: String?
    let oftShared: Bool?
    let phmDim: Int?
    let phmShared: Bool?
}

extension AdapterMethod {
    static func fromMCP(_ args: [String: Any]?, fallbackRank: Int) throws(MCPFailure) -> AdapterMethod? {
        let methodText = try MCPArguments.string("adapterMethod", in: args)
        let options = MCPAdapterOptions(
            oftBlock: try MCPArguments.integer("oftBlockSize", in: args, Self.allowedBlockSizes),
            oftParamText: try MCPArguments.string("oftParameterization", in: args),
            oftShared: try MCPArguments.optionalBoolean("oftBlockShared", in: args),
            phmDim: try MCPArguments.integer("phmHypercomplexDim", in: args, Self.allowedHypercomplexDims),
            phmShared: try MCPArguments.optionalBoolean("phmSharedBasis", in: args)
        )
        guard let chosen = resolvedMethodName(
            explicit: methodText,
            hasOFT: options.oftBlock != nil || options.oftParamText != nil || options.oftShared != nil,
            hasPHM: options.phmDim != nil || options.phmShared != nil
        ) else {
            return nil
        }
        let method = try buildFromMCP(chosen: chosen, fallbackRank: fallbackRank, options: options)
        do {
            try method.validate(prefix: "adapter")
        } catch {
            throw .invalidParams(error.localizedDescription)
        }
        return method
    }

    private static func resolvedMethodName(explicit: String?, hasOFT: Bool, hasPHM: Bool) -> String? {
        if let explicit { return explicit }
        if hasOFT { return "oft" }
        if hasPHM { return "phm" }
        return nil
    }

    private static func buildFromMCP(
        chosen: String,
        fallbackRank: Int,
        options: MCPAdapterOptions
    ) throws(MCPFailure) -> AdapterMethod {
        switch chosen {
        case "lora":
            return .lora(rank: fallbackRank, scale: 10.0)
        case "phm":
            return .phm(
                hypercomplexDim: options.phmDim ?? 8,
                factorRank: fallbackRank,
                sharedBasis: options.phmShared ?? true
            )
        case "oft":
            let param = try parseOFTParameterization(options.oftParamText)
            return .oft(
                blockSize: options.oftBlock ?? 16,
                parameterization: param,
                blockShared: options.oftShared ?? false
            )
        default:
            throw .invalidParams("adapterMethod must be lora, phm or oft")
        }
    }

    private static func parseOFTParameterization(_ text: String?) throws(MCPFailure) -> OrthogonalParameterization {
        guard let text else { return .cayley }
        guard let parsed = OrthogonalParameterization(rawValue: text) else {
            throw .invalidParams("oftParameterization must be cayley or matrixExp")
        }
        return parsed
    }
}

extension TrainingOptimizerKind {
    static func fromMCP(_ args: [String: Any]?) throws(MCPFailure) -> TrainingOptimizerKind? {
        guard let raw = try MCPArguments.string("optimizer", in: args) else { return nil }
        guard let parsed = TrainingOptimizerKind(rawValue: raw) else {
            throw .invalidParams("optimizer must be adamw or muon")
        }
        return parsed
    }
}

extension TrainRequest {
    static let taskCounts = 2...10_000

    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let heldOut = MCPArguments.present("heldOut", in: args)
        guard let heldOut = heldOut as? [String], ResearchCall.comparePrompts.contains(heldOut.count) else {
            throw .invalidParams("heldOut must be an array of \(ResearchCall.comparePrompts) general prompts the quality gate checks the adapter on")
        }
        var gate = GateRequest(heldOut: heldOut)
        try gate.readLimits(args)
        let count = try MCPArguments.integer("count", in: args, Self.taskCounts) ?? 200
        let seed = try MCPArguments.integer("seed", in: args, ResearchCall.nonNegative) ?? 0
        let file = try MCPArguments.string("file", in: args)
        self.init(tasks: file.map(TrainingTasks.file) ?? .arithmetic(count: count, seed: UInt64(seed)), gate: gate)
        teacher = try MCPArguments.string("teacher", in: args)
        teacherModel = try MCPArguments.string("teacherModel", in: args)
        layers = try MCPArguments.integers("layers", in: args)
        experts = try MCPArguments.integers("experts", in: args)
        rank = try MCPArguments.integer("rank", in: args, ResearchCall.nonNegative) ?? rank
        method = try AdapterMethod.fromMCP(args, fallbackRank: rank)
        iterations = try MCPArguments.integer("iterations", in: args, ResearchCall.nonNegative) ?? iterations
        learningRate = try MCPArguments.number("learningRate", in: args) ?? learningRate
        optimizer = try TrainingOptimizerKind.fromMCP(args) ?? optimizer
        output = try MCPArguments.string("output", in: args)
    }
}

extension ResearchTool {
    static var adapterMethodProperties: [String: Any] {
        [
            "adapterMethod": ["type": "string", "enum": ["lora", "phm", "oft"], "description": "Adapter parameterization (default lora)"],
            "oftBlockSize": [
                "type": "integer",
                "minimum": AdapterMethod.allowedBlockSizes.lowerBound,
                "maximum": AdapterMethod.allowedBlockSizes.upperBound,
                "description": "oft: diagonal block size dividing adapted input dimensions (default 16)",
            ],
            "oftParameterization": [
                "type": "string",
                "enum": OrthogonalParameterization.allCases.map(\.rawValue),
                "description": "oft: orthogonal map from skew-symmetric generators (default cayley)",
            ],
            "oftBlockShared": [
                "type": "boolean",
                "description": "oft: share one orthogonal block across all blocks within each projection (default false)",
            ],
            "phmHypercomplexDim": [
                "type": "integer",
                "minimum": AdapterMethod.allowedHypercomplexDims.lowerBound,
                "maximum": AdapterMethod.allowedHypercomplexDims.upperBound,
                "description": "phm: hypercomplex algebra dimension (default 8)",
            ],
            "phmSharedBasis": [
                "type": "boolean",
                "description": "phm: share hypercomplex basis across adapted projections (default true)",
            ],
        ]
    }
}
