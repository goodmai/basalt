import BasaltCore

extension TrainingOptimizerKind {
    static func fromMCP(_ args: [String: Any]?) throws(MCPFailure) -> TrainingOptimizerKind? {
        guard let raw = try MCPArguments.string("optimizer", in: args) else { return nil }
        guard let parsed = TrainingOptimizerKind(rawValue: raw) else {
            let valid = Self.allCases.map(\.rawValue).joined(separator: ", ")
            throw .invalidParams("optimizer must be one of: \(valid)")
        }
        return parsed
    }
}

extension TrainingOptimizerOptions {
    static func fromMCP(
        _ args: [String: Any]?,
        kind: TrainingOptimizerKind,
        weightDecay: Float = defaultWeightDecay
    ) throws(MCPFailure) -> TrainingOptimizerOptions? {
        var options = TrainingOptimizerOptions(kind: kind, weightDecay: weightDecay)
        let overrides = try applyOverrides(from: args, to: &options)
        guard overrides || kind == .muonclip || kind == .soap else { return nil }
        do {
            try options.validateAdapterCompatible(prefix: "optimizer")
        } catch {
            throw .invalidParams(error.localizedDescription)
        }
        return options
    }

    private static func applyOverrides(
        from args: [String: Any]?,
        to options: inout TrainingOptimizerOptions
    ) throws(MCPFailure) -> Bool {
        var qkSet = false
        var soapSet = false
        if let v = try MCPArguments.number("qkClipTau", in: args) { options.qkClipTau = v; qkSet = true }
        if let v = try MCPArguments.number("qkClipAlpha", in: args) { options.qkClipAlpha = v; qkSet = true }
        if let v = try MCPArguments.integer("soapPrecondFreq", in: args, Self.precondFreqRange) { options.soapPrecondFreq = v; soapSet = true }
        if let v = try MCPArguments.integer("soapMaxPrecondDim", in: args, Self.maxPrecondDimRange) { options.soapMaxPrecondDim = v; soapSet = true }
        if let v = try MCPArguments.number("soapShampooBeta", in: args) { options.soapShampooBeta = v; soapSet = true }
        if let v = try MCPArguments.number("soapBeta1", in: args) { options.soapBeta1 = v; soapSet = true }
        if let v = try MCPArguments.number("soapBeta2", in: args) { options.soapBeta2 = v; soapSet = true }
        if let v = try MCPArguments.number("soapEps", in: args) { options.soapEps = v; soapSet = true }
        if qkSet && options.kind != .muonclip {
            throw .invalidParams("qkClipTau and qkClipAlpha require optimizer 'muonclip'")
        }
        if soapSet && options.kind != .soap {
            throw .invalidParams("soap* parameters require optimizer 'soap'")
        }
        return qkSet || soapSet
    }
}

extension ResearchTool {
    static var optimizerProperties: [String: Any] {
        [
            "optimizer": [
                "type": "string",
                "enum": TrainingOptimizerKind.allCases.map(\.rawValue),
                "description": "Optimizer: adamw, muon, muonclip or soap (default: adamw)",
            ],
            "qkClipTau": [
                "type": "number",
                "description": "muonclip: max pre-softmax attention logit threshold tau (> 0, default 100.0)",
            ],
            "qkClipAlpha": [
                "type": "number",
                "description": "muonclip: Q/K clip split exponent alpha in [0, 1] (default 0.5)",
            ],
            "soapPrecondFreq": [
                "type": "integer",
                "minimum": TrainingOptimizerOptions.precondFreqRange.lowerBound,
                "maximum": TrainingOptimizerOptions.precondFreqRange.upperBound,
                "description": "soap: eigenbasis refresh interval in steps (default 10)",
            ],
            "soapMaxPrecondDim": [
                "type": "integer",
                "minimum": TrainingOptimizerOptions.maxPrecondDimRange.lowerBound,
                "maximum": TrainingOptimizerOptions.maxPrecondDimRange.upperBound,
                "description": "soap: max matrix axis dimension to precondition before AdamW fallback (default 1024)",
            ],
            "soapShampooBeta": [
                "type": "number",
                "description": "soap: preconditioner EMA decay in [0, 1) (default 0.95)",
            ],
            "soapBeta1": [
                "type": "number",
                "description": "soap: first-moment Adam beta1 in [0, 1) (default 0.95)",
            ],
            "soapBeta2": [
                "type": "number",
                "description": "soap: rotated second-moment Adam beta2 in [0, 1) (default 0.95)",
            ],
            "soapEps": [
                "type": "number",
                "description": "soap: numerical stability epsilon (> 0, default 1e-8)",
            ],
        ]
    }
}
