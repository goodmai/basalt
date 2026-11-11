import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltServer

@Suite("Each named pipeline may decide with its own local model, loaded once however many pipelines share it (#423 A1)", .timeLimit(.minutes(1)))
struct ComposePipelineDeciderTests {
    static let laya = "  laya:\n    source: ./laya\n"
    static let judge = "  judge:\n    source: ./judge\n"
    static let twoPipelines = ComposeOperationsTests.support.replacingOccurrences(of: "cascade: [chat]", with: "cascade: [laya]")
        + "  triage:\n    decide:\n      cascade: [jev, laya]\n    answer: chat\n"

    static func yaml(_ models: String, pipelines: String = twoPipelines) -> String {
        ComposeOperationsTests.with(ComposeOperationsTests.haiku + models, pipelines: pipelines)
    }

    @Test("A named pipeline's own decider parses, is planned once, and a decider that also answers is refused")
    func parses() throws {
        let gb = ComposeFileTests.gb
        let spec = try ComposeFile.parse(Self.yaml(Self.laya), file: "p.yaml")
        #expect(spec.deciderModels == ["laya"] && spec.decider == "self")
        let sizes = ComposeFileTests.sizes(["mlx-community/gemma-4-12B-it-qat-4bit": 7 * gb, spec.local("laya") ?? "": 2 * gb])
        #expect(spec.workloads(sizes).workloads.filter { $0.role == "decide" }.map(\.name) == ["laya"])
        let answering = Self.yaml(Self.laya, pipelines: Self.twoPipelines.replacingOccurrences(of: "fallback: chat", with: "fallback: laya"))
        do {
            _ = try ComposeFile.parse(answering, file: "q.yaml")
            Issue.record("a decision model that also answers was accepted")
        } catch {
            #expect(error.description.contains("a decision model only decides"), "\(error.description)")
        }
    }

    @Test("A host loads a shared pipeline decider once, and each pipeline cascades over its own local model")
    func start() async throws {
        let (host, _, asked) = try await ComposeRoleSwapTests.host(Self.yaml(Self.laya))
        #expect(asked.all.filter { $0?.hasSuffix("laya") == true }.count == 1)
        #expect(await host.registry.system("support")?.decisions?.modelID.hasSuffix("laya") == true)
        #expect(await host.registry.system("triage")?.decisions?.modelID.hasSuffix("laya") == true)
        #expect(await host.registry.system(ComposeSpec.defaultPipeline)?.decisions?.modelID == "self")
        #expect(await host.status().models.first { $0.name == "laya" }?.state == .decision)
    }

    @Test("apply swaps one pipeline's decider; a model that stops deciding is reloaded in its new role")
    func apply() async throws {
        let (host, _, _) = try await ComposeRoleSwapTests.host(Self.yaml(Self.laya))
        let judged = Self.yaml(Self.laya + Self.judge, pipelines: Self.twoPipelines.replacingOccurrences(of: "cascade: [jev, laya]", with: "cascade: [jev, judge]"))
        let diff = try await host.apply(ComposeApply(yaml: judged))
        #expect(diff.added == ["judge"] && diff.kept.contains("laya") && diff.restart == nil)
        #expect(await host.registry.system("triage")?.decisions?.modelID.hasSuffix("judge") == true)
        #expect(await host.registry.system("support")?.decisions?.modelID.hasSuffix("laya") == true)
        let retired = Self.yaml(Self.laya + Self.judge, pipelines: ComposeOperationsTests.support.replacingOccurrences(of: "fallback: chat", with: "fallback: laya")
            + "  triage:\n    decide:\n      cascade: [jev, judge]\n    answer: chat\n")
        #expect(ComposeDiff.between(try ComposeFile.parse(judged, file: "j.yaml"), try ComposeFile.parse(retired, file: "r.yaml")).reloaded == ["laya"])
    }
}
