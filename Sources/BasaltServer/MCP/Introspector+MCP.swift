import Foundation
import BasaltCore

extension Introspector {
    init(
        orchestrator: any ModelOrchestrating,
        interventions: Bool,
        filter: MCPToolFilter = .standard,
        researchTools: Bool = true,
        journal: URL? = OverlayRegistry.defaultJournal
    ) {
        let allowed = ResearchTool.introspection(interventions: interventions).filter { filter.allows($0.rawValue, researchTools: researchTools) }
        let definitions = allowed.map(\.definition) + Self.extraXRayDefinitions(interventions: interventions, filter: filter, researchTools: researchTools)
        self.init(orchestrator: orchestrator, tools: definitions, journal: journal) { call, model throws(BasaltError) in
            if let diagnostic = try await Self.runExtraXRay(call, on: model) { return diagnostic }
            let parsed = try ResearchCall(introspected: call)
            if case .tensor(let req) = parsed, req.spectrum, !filter.allowsSpectrum(researchTools: researchTools) {
                throw .invalidRequestStructure(details: "model_tensor spectrum requires --research-tools or an enabled experiment tool")
            }
            return try await parsed.run(on: model).text
        }
    }

    private static func extraXRayDefinitions(
        interventions: Bool,
        filter: MCPToolFilter,
        researchTools: Bool
    ) -> [ToolDefinition] {
        guard interventions else { return [] }
        var extra: [ToolDefinition] = []
        if filter.allows("depth_diagnostics", researchTools: researchTools) {
            extra.append(MCPToolCatalog.depthDiagnosticsToolDefinition)
        }
        if filter.allows(MCPToolCatalog.outlierProfileToolName, researchTools: researchTools) {
            extra.append(MCPToolCatalog.outlierProfileDescriptor.asToolDefinition)
        }
        if filter.allows("causal_sweep", researchTools: researchTools) {
            extra.append(MCPToolCatalog.causalSweepToolDefinition)
        }
        return extra
    }

    private static func runExtraXRay(_ call: ToolCallOutput, on model: any ModelResearch) async throws(BasaltError) -> String? {
        switch call.name {
        case "depth_diagnostics":
            let request: DepthDiagnosticsRequest
            do throws(MCPFailure) {
                let args = try MCPArguments.arguments(call.arguments.isEmpty ? nil : call.arguments)
                request = try MCPToolCatalog.parseDepthDiagnosticsRequest(args)
            } catch {
                throw .invalidRequestStructure(details: error.message)
            }
            return ToolResult.json(try await ResearchCall.lab(model).runDepthDiagnostics(request)).text
        case MCPToolCatalog.outlierProfileToolName:
            let request: OutlierProfileRequest
            do throws(MCPFailure) {
                request = try OutlierProfileRequest(mcp: MCPArguments.arguments(call.arguments.isEmpty ? nil : call.arguments))
            } catch {
                throw .invalidRequestStructure(details: error.message)
            }
            return ToolResult.json(try await ResearchCall.lab(model).runOutlierProfile(request)).text
        case "causal_sweep":
            let request: CausalSweepRequest
            do throws(MCPFailure) {
                request = try CausalSweepRequest(mcp: MCPArguments.arguments(call.arguments.isEmpty ? nil : call.arguments))
            } catch {
                throw .invalidRequestStructure(details: error.message)
            }
            return ToolResult.json(try await ResearchCall.lab(model).runCausalSweep(request)).text
        default:
            return nil
        }
    }
}

private extension MCPToolDescriptor {
    var asToolDefinition: ToolDefinition {
        let json = (try? MCPWire.encoder.encode(inputSchema)).map { String(decoding: $0, as: UTF8.self) }
        return ToolDefinition(name: name, description: description, parametersJSON: json)
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
