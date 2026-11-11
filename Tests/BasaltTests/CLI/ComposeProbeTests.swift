import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltServer

@Suite("compose startup probes: answer models generate one token, deciders decide once, on-demand files are checked without a load (#423 H5)", .timeLimit(.minutes(1)))
struct ComposeProbeTests {
    @Test("Answer models, deciders and on-demand models are probed; perceivers and cloud models are not")
    func selection() async throws {
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        #expect(ComposeHost.probes(.resident(chat), answering: true) && !ComposeHost.probes(.resident(chat), answering: false))
        #expect(ComposeHost.probes(.decision(StubDecisionEngine()), answering: false) && !ComposeHost.probes(.cloud("x"), answering: true))
    }

    @Test("The live probe passes a loaded model and a decider, and names what failed otherwise")
    func live() async throws {
        let (_, loaded) = try await PipelineSwapEdgeTests.chat()
        let (_, empty) = try await PipelineSwapEdgeTests.chat(loaded: false)
        #expect(await ComposeProbe.live(.resident(loaded)) == nil)
        #expect(await ComposeProbe.live(.decision(StubDecisionEngine())) == nil)
        #expect(await ComposeProbe.live(.resident(empty))?.contains("startup probe failed") == true)
        let missing = OnDemandModel(model: empty, path: "/nonexistent/basalt-probe-model", idle: .seconds(60))
        #expect(await ComposeProbe.live(.onDemand(missing))?.contains("incomplete") == true)
    }

    @Test("A failed probe marks the model failed and keeps ready at 503; replacing the model probes the new one")
    func failures() async throws {
        let probe: @Sendable (ComposeHeld) async -> String? = { model in
            guard case .decision(let engine) = model, engine.modelID.hasSuffix("laya") else { return nil }
            return "startup probe failed: no decision"
        }
        let yaml = ComposePipelineDeciderTests.yaml(ComposePipelineDeciderTests.laya)
        let (host, _, _) = try await ComposeRoleSwapTests.host(yaml, probe: probe)
        await host.probe(try ComposeFile.parse(yaml, file: "p.yaml").models.keys.sorted(), in: try ComposeFile.parse(yaml, file: "p.yaml"))
        let status = await host.status()
        #expect(status.models.first { $0.name == "laya" }?.state == .failed && status.waiting == ["laya (failed)"])
        #expect(status.models.first { $0.name == "laya" }?.refusal == "startup probe failed: no decision")
        let judged = ComposePipelineDeciderTests.yaml(ComposePipelineDeciderTests.judge, pipelines: ComposePipelineDeciderTests.twoPipelines.replacingOccurrences(of: "laya", with: "judge"))
        _ = try await host.apply(ComposeApply(yaml: judged))
        #expect(await host.status().waiting.isEmpty)
    }
}
