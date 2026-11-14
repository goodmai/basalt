import ArgumentParser
import Foundation
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltServer

@Suite("MCPToolFilter expands groups, enforces deny-overrides precedence, and matches the full MCP catalog (#484)")
struct MCPToolFilterValueTests {
    @Test("Group.all covers the exact 40 tools in MCPToolCatalog, partitioned into 23 base and 17 experiment tools")
    func catalogCoverage() {
        let fullCatalog = Set(MCPToolCatalog.catalog(researchTools: true).tools.map(\.name))
        let baseCatalog = Set(MCPToolCatalog.catalog(researchTools: false).tools.map(\.name))
        #expect(fullCatalog == MCPToolFilter.allTools)
        #expect(fullCatalog.count == 40)
        #expect(baseCatalog == MCPToolFilter.defaultBaseTools)
        #expect(baseCatalog.count == 23)
        #expect(MCPToolFilter.experimentTools.count == 17)
        #expect(MCPToolFilter.Group.surgery.tools.isSubset(of: MCPToolFilter.Group.research.tools))
        #expect(MCPToolFilter.Group.surgery.tools.count == 9)
    }

    @Test("Comma-separated and repeatable selectors expand groups and single tools, and disable overrides enable and researchTools")
    func precedenceAndExpansion() throws {
        let surgeryOnly = try MCPToolFilter(enable: ["surgery,intervene"], disable: ["weights_delta", "gemma"])
        #expect(surgeryOnly.allows("adapter_apply", researchTools: false))
        #expect(surgeryOnly.allows("intervene", researchTools: false))
        #expect(!surgeryOnly.allows("weights_delta", researchTools: false))
        #expect(!surgeryOnly.allows("weights_delta", researchTools: true))
        #expect(!surgeryOnly.allows("distill", researchTools: false))
        #expect(!surgeryOnly.allows("gemma_generate", researchTools: true))
        #expect(surgeryOnly.allows("basalt_generate", researchTools: false))
        #expect(surgeryOnly.isExplicitlyDisabled("weights_delta"))
        #expect(!surgeryOnly.isExplicitlyDisabled("distill"))

        let readOnlyLocked = try MCPToolFilter(disable: ["research", "switch_model,basalt_generate"])
        #expect(!readOnlyLocked.allows("overlays_list", researchTools: true))
        #expect(!readOnlyLocked.allows("switch_model", researchTools: true))
        #expect(!readOnlyLocked.allows("basalt_generate", researchTools: false))
        #expect(readOnlyLocked.allows("model_describe", researchTools: false))
    }

    @Test("model_tensor spectrum is allowed when researchTools is set or any experiment tool is enabled, unless all experiment tools or model_tensor are disabled")
    func spectrumAllowance() throws {
        #expect(!MCPToolFilter.standard.allowsSpectrum(researchTools: false))
        #expect(MCPToolFilter.standard.allowsSpectrum(researchTools: true))
        #expect(try MCPToolFilter(enable: ["surgery"]).allowsSpectrum(researchTools: false))
        #expect(try !MCPToolFilter(disable: ["research"]).allowsSpectrum(researchTools: true))
        #expect(try !MCPToolFilter(enable: ["surgery"], disable: ["model_tensor"]).allowsSpectrum(researchTools: true))
    }

    @Test("Unknown tool or group names throw InvalidSelectorError and fail ServeCommand validation")
    func unknownSelectorsRejected() throws {
        let error = try #require(throws: MCPToolFilter.InvalidSelectorError.self) {
            try MCPToolFilter(enable: ["surgery,not_a_tool"], disable: ["bogus_group", "not_a_tool"])
        }
        #expect(error.unknown == ["not_a_tool", "bogus_group"])
        #expect(error.description.contains("not_a_tool, bogus_group"))
        #expect(error.description.contains("surgery"))
        #expect(throws: (any Error).self) {
            try ServeCommand.parse(["--mcp", "--mcp-enable", "not_a_tool"])
        }
        #expect(throws: (any Error).self) {
            try ServeCommand.parse(["--mcp", "--mcp-disable", "bogus_group"])
        }
        let valid = try ServeCommand.parse(["--mcp", "--mcp-enable", "surgery,intervene", "--mcp-disable", "switch_model"])
        #expect(try valid.mcpOptions.toolFilter().allows("steer_search", researchTools: false))
        #expect(try !valid.mcpOptions.toolFilter().allows("switch_model", researchTools: false))
    }
}

@Suite("MCPServer enforces MCPToolFilter on tools/list, tools/call, model_tensor spectrum, and Introspector (#484)")
struct MCPToolFilterServerTests {
    @Test("tools/list omits disabled base tools and lists selectively enabled experiment tools without --research-tools")
    func wireListAndCallEnforcement() async throws {
        let engine = ScriptedResearchEngine([])
        let filter = try MCPToolFilter(enable: ["overlays_list"], disable: ["status", "gemma"])
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: ModelOrchestratorActor(engine: engine)) { capturer.capture($0) }
            .withToolFilter(filter)

        await server.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#)
        await server.dispatch(#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"status","arguments":{}}}"#)
        await server.dispatch(#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"gemma_status","arguments":{}}}"#)
        await server.dispatch(#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"overlays_list","arguments":{}}}"#)
        await server.dispatch(#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"weights_delta","arguments":{"tensor":"w","values":[1]}}}"#)

        let lines = capturer.lines
        #expect(lines.count == 5)
        #expect(lines[0].contains("overlays_list") && !lines[0].contains("\"name\":\"status\"") && !lines[0].contains("gemma_status"))
        #expect(lines[1].contains("-32601") && lines[1].contains("disabled by server configuration (--mcp-disable)"))
        #expect(lines[2].contains("-32601") && lines[2].contains("disabled by server configuration (--mcp-disable)"))
        #expect(lines[3].contains("\"result\"") && !lines[3].contains("\"error\""))
        #expect(lines[4].contains("-32601") && lines[4].contains("--mcp-enable weights_delta"))
    }

    @Test("Explicitly disabling research overrides --research-tools on tools/call and model_tensor spectrum")
    func disableOverridesResearchToolsFlag() async throws {
        let engine = ScriptedResearchEngine([])
        let filter = try MCPToolFilter(disable: ["research"])
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: ModelOrchestratorActor(engine: engine)) { capturer.capture($0) }
            .withResearchTools(true)
            .withToolFilter(filter)

        await server.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"overlays_list","arguments":{}}}"#)
        await server.dispatch(#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"model_tensor","arguments":{"name":"w","spectrum":true}}}"#)

        let lines = capturer.lines
        #expect(lines[0].contains("-32601") && lines[0].contains("disabled by server configuration"))
        #expect(lines[1].contains("-32601") && lines[1].contains("model_tensor spectrum"))
    }

    @Test("Introspector filters disabled X-ray and intervention sub-tools from both tool definitions and execution")
    func introspectorRespectsFilter() async throws {
        let engine = ScriptedResearchEngine([
            .call("intervene", #"{"kind":"ablate_head","layer":0,"head":0}"#),
            .answer("Hypothesis: h\nTest: t\nOutcome: inconclusive"),
        ])
        let journal = FileManager.default.temporaryDirectory.appending(path: "introspect-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: journal) }
        let filter = try MCPToolFilter(enable: ["introspect"], disable: ["intervene", "trace_run"])
        let introspector = Introspector(
            orchestrator: ModelOrchestratorActor(engine: engine),
            interventions: true,
            filter: filter,
            researchTools: true,
            journal: journal
        )
        let report = try await introspector.introspect(IntrospectRequest(goal: "check filter", maxSteps: 2, allowInterventions: true))
        let offered = await engine.requests.first?.tools?.map(\.name) ?? []
        #expect(!offered.contains("intervene") && !offered.contains("trace_run"))
        #expect(offered.contains("model_describe") && offered.contains("experiment_compare"))
        #expect(report.steps.first?.result?.contains("not one of the tools offered in this session") == true)
        #expect(await engine.overlays().isEmpty)
    }

    @Test("ServeCommand.makeMCPServer wires --mcp-enable, --mcp-disable, and --research-tools into MCPServer")
    func serveCommandWiresMCPServer() async throws {
        let command = try ServeCommand.parse(["--mcp", "--mcp-enable", "overlays_list", "--mcp-disable", "switch_model"])
        let orchestrator = ModelOrchestratorActor(engine: ScriptedResearchEngine([]))
        let server = try command.makeMCPServer(orchestrator: orchestrator, resolvedPath: "/tmp/mock-model", decisions: nil, stages: PipelineStages())
        #expect(server.researchTools == false)
        #expect(server.toolFilter.allows("overlays_list", researchTools: false))
        #expect(!server.toolFilter.allows("switch_model", researchTools: false))
    }
}
