import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltServer

actor TrackedDecider: DecisionEngine {
    nonisolated let modelID: String
    private(set) var loads = 0
    private(set) var unloads = 0
    private var failNext = false

    init(_ modelID: String = "org/tracked-jev") { self.modelID = modelID }

    func failOnce() { failNext = true }

    func recordLoad() throws(BasaltError) -> TrackedDecider {
        if failNext {
            failNext = false
            throw .modelNotFound(identifier: modelID)
        }
        loads += 1
        return self
    }

    func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try await StubDecisionEngine().decide(request)
    }

    func unload() async { unloads += 1 }
}

@Suite("On-demand deciders: compose schema, idle unload lifecycle, disk-only startup probe, and live apply (#423 J5)", .timeLimit(.minutes(1)))
struct ComposeOnDemandDeciderTests {
    static let onDemandLaya = "  laya:\n    source: ./laya\n    keep: on-demand\n    idle_unload: 45s\n"

    static func waitUntil(_ decision: OnDemandDecision, _ phase: OnDemandCore.Phase) async {
        guard await decision.phase != phase else { return }
        for await next in decision.phases where next == phase { return }
    }

    @Test("A local decider accepts keep: on-demand and idle_unload, counts as resident in the plan, and refuses sequential modes")
    func parsesAndPlans() throws {
        let gb = ComposeFileTests.gb
        let yaml = ComposePipelineDeciderTests.yaml(Self.onDemandLaya)
        let spec = try ComposeFile.parse(yaml, file: "od.yaml")
        let plan = spec.plan(ComposeFileTests.sizes(["mlx-community/gemma-4-12B-it-qat-4bit": 7 * gb, spec.local("laya") ?? "": 2 * gb]), budgetBytes: 19 * gb)
        #expect(plan.verdict.residentBytes == 9 * gb && plan.verdict.memory == .resident && spec.models["laya"]?.idleUnload == 45)
        #expect(spec.onDemandLines.contains { $0.contains("laya on demand") && $0.contains("45 s idle") })
        let defaultDecider = ComposeRoleSwapTests.laya.replacingOccurrences(of: "source: ./laya", with: "source: ./laya\n    keep: on-demand")
        let defaultSpec = try ComposeFile.parse(defaultDecider, file: "def.yaml")
        #expect(!defaultSpec.serveArguments.contains("--decision-model") && !defaultSpec.serveArguments.contains("--decision-escalate"))
        let seqDecider = ComposePipelineDeciderTests.yaml("  laya:\n    source: ./laya\n    keep: sequential\n")
        let seqError = #expect(throws: (any Error).self) { try ComposeFile.parse(seqDecider, file: "seq.yaml") }
        #expect("\(seqError.map { "\($0)" } ?? "")".contains("applies only to a perceiver"))
        let seqMemory = "memory: sequential\n" + yaml
        let memError = #expect(throws: (any Error).self) { try ComposeFile.parse(seqMemory, file: "mem.yaml") }
        #expect("\(memError.map { "\($0)" } ?? "")".contains("on-demand decider"))
    }

    @Test("OnDemandDecision loads on first decide, reuses while warm, unloads on idle expiry, retries after failure, and explicit unload clears state")
    func lifecycle() async throws {
        let (tracked, clock) = (TrackedDecider(), ManualSleep())
        let decider = OnDemandDecision(modelID: "org/tracked-jev", path: "/models/tracked", idle: .seconds(45), loader: { try await tracked.recordLoad() }, sleep: { _ in try await clock.sleep() })
        let (initialPhase, initialUsed) = (await decider.phase, await decider.lastUsed)
        #expect(initialPhase == .unloaded && initialUsed == nil)
        await tracked.failOnce()
        await #expect(throws: BasaltError.self) { _ = try await decider.decide(ComposeProbe.request) }
        let (failedPhase, failedLoads) = (await decider.phase, await tracked.loads)
        #expect(failedPhase == .unloaded && failedLoads == 0)
        _ = try await decider.decide(ComposeProbe.request)
        await clock.waitForSleeper()
        _ = try await decider.decide(ComposeProbe.request)
        await clock.waitForSleeper()
        let (loads1, phase1, used1) = (await tracked.loads, await decider.phase, await decider.lastUsed)
        #expect(loads1 == 1 && phase1 == .loaded && used1 != nil)
        await clock.fire()
        await Self.waitUntil(decider, .unloaded)
        #expect(await tracked.unloads == 1)
        _ = try await decider.decide(ComposeProbe.request)
        await clock.waitForSleeper()
        #expect(await tracked.loads == 2)
        await decider.unload()
        let (finalPhase, finalUnloads) = (await decider.phase, await tracked.unloads)
        #expect(finalPhase == .unloaded && finalUnloads == 2)
    }

    @Test("ComposeHost registers an on-demand decider without loading it, probes its bundle on disk, loads on first decision, and reloads on apply")
    func hostProbeAndApply() async throws {
        let yaml = ComposePipelineDeciderTests.yaml(Self.onDemandLaya)
        let (host, _, asked) = try await ComposeRoleSwapTests.host(yaml)
        #expect(asked.all.isEmpty)
        let initial = await host.status()
        let entry = try #require(initial.models.first { $0.name == "laya" })
        #expect(entry.state == .idleUnloaded && entry.idleUnload == 45 && initial.waiting.isEmpty)
        let decisions = try #require(await host.registry.system("support")?.decisions)
        _ = try await decisions.decide(ComposeProbe.request)
        #expect(asked.all.count == 1)
        let loaded = try #require(await host.status().models.first { $0.name == "laya" })
        #expect(loaded.state == .ready && loaded.lastUsed != nil)
        let residentYaml = ComposePipelineDeciderTests.yaml(ComposePipelineDeciderTests.laya)
        let diff = try await host.apply(ComposeApply(yaml: residentYaml))
        #expect(diff.reloaded == ["laya"] && diff.restart == nil)
        #expect(await host.status().models.first { $0.name == "laya" }?.state == .decision)
    }

    @Test("ComposeProbe.live checks on-demand decision bundles on disk without invoking the loader")
    func diskProbe() async throws {
        let calls = KeyLog()
        let missing = OnDemandDecision(modelID: "m", path: "/nonexistent/basalt-jev-probe", idle: .seconds(60)) {
            calls.record("load")
            return StubDecisionEngine()
        }
        #expect(ComposeHost.probes(.onDemandDecision(missing), answering: false))
        #expect(await ComposeProbe.live(.onDemandDecision(missing))?.contains("incomplete") == true)
        let dir = FileManager.default.temporaryDirectory.appending(path: "jev-probe-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appending(component: "openjev_config.json"))
        try Data().write(to: dir.appending(component: "head.safetensors"))
        let brokenJev = OnDemandDecision(modelID: "j", path: dir.path, idle: .seconds(60)) { calls.record("load"); return StubDecisionEngine() }
        #expect(await ComposeProbe.live(.onDemandDecision(brokenJev))?.contains("model_name") == true)
        try Data("{\"model_name\":\"Qwen/Qwen3-0.6B\"}".utf8).write(to: dir.appending(component: "openjev_config.json"))
        let validJev = OnDemandDecision(modelID: "j", path: dir.path, idle: .seconds(60)) { calls.record("load"); return StubDecisionEngine() }
        #expect(await ComposeProbe.live(.onDemandDecision(validJev)) == nil && calls.all.isEmpty)
    }
}
