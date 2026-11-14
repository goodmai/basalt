import Foundation
import BasaltCore

extension Introspector {
    init(
        orchestrator: ModelOrchestratorActor,
        interventions: Bool,
        filter: MCPToolFilter = .standard,
        researchTools: Bool = true,
        journal: URL? = OverlayRegistry.defaultJournal
    ) {
        let allowed = ResearchTool.introspection(interventions: interventions).filter { filter.allows($0.rawValue, researchTools: researchTools) }
        self.init(orchestrator: orchestrator, tools: allowed.map(\.definition), journal: journal) { call, model throws(BasaltError) in
            try await ResearchCall(introspected: call).run(on: model).text
        }
    }
}

extension ResearchTool {
    static func introspection(interventions: Bool) -> [ResearchTool] {
        [.modelDescribe, .modelTensors, .modelTensor, .kvInspect, .traceRun] + (interventions ? [.intervene, .experimentCompare, .overlayUndo] : [])
    }

    var definition: ToolDefinition {
        let json = (try? MCPWire.encoder.encode(schema)).map { String(decoding: $0, as: UTF8.self) }
        return ToolDefinition(name: rawValue, description: summary, parametersJSON: json)
    }
}

extension ResearchCall {
    init(introspected call: ToolCallOutput) throws(BasaltError) {
        guard let tool = ResearchTool(rawValue: call.name) else {
            throw .invalidRequestStructure(details: "\(call.name) is not a research tool")
        }
        do throws(MCPFailure) {
            self = try ResearchCall(tool, MCPArguments.arguments(call.arguments.isEmpty ? nil : call.arguments))
        } catch {
            throw .invalidRequestStructure(details: error.message)
        }
    }
}

extension ToolResult {
    var text: String {
        content.compactMap { $0["text"] }.joined(separator: "\n")
    }
}

extension IntrospectRequest {
    init(mcp args: [String: Any]?) throws(MCPFailure) {
        let goal = try MCPArguments.required("goal", MCPArguments.string("goal", in: args))
        let maxSteps = try MCPArguments.integer("maxSteps", in: args, Self.steps) ?? 8
        let allowInterventions = try MCPArguments.optionalBoolean("allowInterventions", in: args) ?? false
        self.init(goal: goal, maxSteps: maxSteps, allowInterventions: allowInterventions)
        resultBudget = try MCPArguments.integer("resultBudget", in: args, Self.budgets) ?? resultBudget
    }
}

extension MCPToolCatalog {
    static var introspectDescriptor: MCPToolDescriptor {
        MCPToolDescriptor(
            name: "introspect",
            description: "Experiment: the served model studies itself towards a goal with the X-ray tools, one call per step, writing hypotheses (statement, test, outcome). Returns every step with its tool, arguments and truncated result, and the hypotheses. Interventions are offered only with allowInterventions and are undone when the loop ends.",
            inputSchema: AnyCodable([
                "type": "object",
                "required": ["goal"],
                "properties": [
                    "goal": ["type": "string", "description": "What to find out, e.g. which heads attend to the last tool call"],
                    "maxSteps": ["type": "integer", "minimum": IntrospectRequest.steps.lowerBound, "maximum": IntrospectRequest.steps.upperBound, "description": "Model turns before the loop stops (default 8)"],
                    "allowInterventions": ["type": "boolean", "description": "Also offer intervene, experiment_compare and overlay_undo (default false)"],
                    "resultBudget": ["type": "integer", "minimum": IntrospectRequest.budgets.lowerBound, "maximum": IntrospectRequest.budgets.upperBound, "description": "Characters of each tool result fed back to the model (default 2000)"],
                ],
            ]))
    }
}
