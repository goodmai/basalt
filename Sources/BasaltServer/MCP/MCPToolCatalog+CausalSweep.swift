import Foundation
import BasaltCore

extension MCPToolCatalog {
    static var causalSweepDescriptor: MCPToolDescriptor {
        MCPToolDescriptor(
            name: "causal_sweep",
            description: "Experiment: Anthropic-style Direct Logit Attribution (DLA) and automated single-layer residual causal patching sweep across clean vs corrupted prompts. Computes exact linearized per-layer logit-difference attributions with additive conservation error, and optionally patches each clean layer's last-token residual into the corrupted forward pass to measure causal recovery without mutating model weights. Requires serve --research-tools.",
            inputSchema: ResearchTool.object([
                "cleanPrompt": ["type": "string", "description": "Clean prompt producing the target next-token preference"],
                "corruptedPrompt": ["type": "string", "description": "Corrupted prompt producing the foil or baseline preference"],
                "targetToken": ["type": "string", "description": "Single target token string (alternative to targetTokenID)"],
                "foilToken": ["type": "string", "description": "Single foil token string (alternative to foilTokenID)"],
                "targetTokenID": ["type": "integer", "minimum": 0, "description": "Target vocabulary token ID (alternative to targetToken)"],
                "foilTokenID": ["type": "integer", "minimum": 0, "description": "Foil vocabulary token ID (alternative to foilToken)"],
                "layers": ["type": "array", "items": ["type": "integer", "minimum": 0], "description": "Layer indices to report and patch (default: all)"],
                "includePatching": ["type": "boolean", "description": "Run single-layer residual patching passes into the corrupted prompt (default true)"],
            ], required: ["cleanPrompt", "corruptedPrompt"])
        )
    }

    static var causalSweepToolDefinition: ToolDefinition {
        let descriptor = causalSweepDescriptor
        let json = (try? MCPWire.encoder.encode(descriptor.inputSchema)).map { String(decoding: $0, as: UTF8.self) }
        return ToolDefinition(name: descriptor.name, description: descriptor.description, parametersJSON: json)
    }
}

extension CausalSweepRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let cleanPrompt = try MCPArguments.required("cleanPrompt", MCPArguments.string("cleanPrompt", in: args))
        let corruptedPrompt = try MCPArguments.required("corruptedPrompt", MCPArguments.string("corruptedPrompt", in: args))
        let targetToken = try MCPArguments.string("targetToken", in: args)
        let foilToken = try MCPArguments.string("foilToken", in: args)
        let targetTokenID = try MCPArguments.integer("targetTokenID", in: args, ResearchCall.nonNegative)
        let foilTokenID = try MCPArguments.integer("foilTokenID", in: args, ResearchCall.nonNegative)
        let layers = try MCPArguments.integers("layers", in: args)
        let includePatching = try MCPArguments.optionalBoolean("includePatching", in: args) ?? true
        var request = CausalSweepRequest(
            cleanPrompt: cleanPrompt,
            corruptedPrompt: corruptedPrompt,
            targetToken: targetToken,
            foilToken: foilToken,
            layers: layers,
            includePatching: includePatching
        )
        request.targetTokenID = targetTokenID
        request.foilTokenID = foilTokenID
        do throws(BasaltError) {
            try request.validate()
        } catch {
            throw MCPFailure(error)
        }
        self = request
    }
}
