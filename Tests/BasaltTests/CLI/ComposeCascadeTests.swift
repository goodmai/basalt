import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltServer

@Suite("compose 4e-1: cascades in any local and cloud order, every local stage a held decider, calibration read from the bundle (#423 J1 J4 J6)", .timeLimit(.minutes(1)))
struct ComposeCascadeTests {
    static let localFirst = ComposeOperationsTests.base
        .replacingOccurrences(of: "pipeline:\n", with: "  laya:\n    source: ./laya\npipeline:\n")
        .replacingOccurrences(of: "cascade: [jev, chat]", with: "cascade: [laya, jev, chat]")

    @Test("A local-first cascade parses; more than four stages or a cloud last stage is refused; every local stage but the chat model is a decider")
    func rules() throws {
        let spec = try ComposeFile.parse(Self.localFirst, file: "c.yaml")
        #expect(spec.cascade == ["laya", "jev", "chat"] && spec.deciderModels == ["laya"] && spec.decider == "self")
        for (yaml, reason) in [
            (Self.localFirst.replacingOccurrences(of: "cascade: [laya, jev, chat]", with: "cascade: [laya, jev, laya2, jev2, chat]")
                .replacingOccurrences(of: "pipeline:\n", with: "  laya2:\n    source: ./laya2\n  jev2:\n    source: cloud:openrouter/x/y\n    api_key_env: HOME\npipeline:\n"), "at most four stages"),
            (Self.localFirst.replacingOccurrences(of: "cascade: [laya, jev, chat]", with: "cascade: [laya, jev]"), "ends with a local model"),
        ] {
            do {
                _ = try ComposeFile.parse(yaml, file: "c.yaml")
                Issue.record("accepted a cascade that should be refused for \(reason)")
            } catch {
                #expect(error.description.contains(reason), "\(error.description)")
            }
        }
    }

    @Test("Stages are built in the order the file names them; without the last local stage there are no decisions")
    func stages() throws {
        let spec = try ComposeFile.parse(Self.localFirst, file: "c.yaml")
        let serve = try ServeCommand.parse(spec.serveArguments)
        let locals: [String: any DecisionEngine] = ["laya": NamedDecider(modelID: "./laya"), "chat": NamedDecider(modelID: "self")]
        let cascade = try #require(try serve.composedDecisions(spec, locals: locals, gate: GenerationGate(), besideDecode: true, cloudGate: CloudGate(limits: [:], storage: nil)) as? CascadeDecisionEngine)
        #expect(cascade.stages.map(\.modelID) == ["./laya", "~typesafe/jev-latest", "self"])
        #expect(cascade.onDevice?.modelID == "./laya+self")
        #expect(try serve.composedDecisions(spec, locals: ["laya": NamedDecider(modelID: "./laya")], gate: GenerationGate(), besideDecode: true, cloudGate: CloudGate(limits: [:], storage: nil)) == nil)
    }

    @Test("A running host loads a local first stage as a decider and cascades it in front of the cloud stage and the chat model")
    func host() async throws {
        let (host, _, asked) = try await ComposeRoleSwapTests.host(Self.localFirst)
        #expect(asked.all.contains { $0?.hasSuffix("laya") == true })
        #expect(await host.status().models.first { $0.name == "laya" }?.state == .decision)
    }

    @Test("A bundle's basalt_readout.json wins over the name table, which wins over uncalibrated; bad values are ignored")
    func calibration() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "readout-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(SystemOneReadout.calibration(for: "openjev/OpenJev-Flash-9B-MLX", in: folder).temperature == 1.07)
        #expect(SystemOneReadout.calibration(for: "org/unknown", in: folder) == .uncalibrated)
        try Data(#"{"temperature": 0.9, "noul_temperature": 1.3, "noul_bias": -0.1}"#.utf8).write(to: folder.appending(path: SystemOneReadout.bundleFile))
        #expect(SystemOneReadout.calibration(for: "openjev/OpenJev-Flash-9B-MLX", in: folder) == SystemOneReadout(temperature: 0.9, noulTemperature: 1.3, noulBias: -0.1))
        try Data(#"{"temperature": 0}"#.utf8).write(to: folder.appending(path: SystemOneReadout.bundleFile))
        #expect(SystemOneReadout.bundled(in: folder) == nil)
        #expect(ComposeCommand.jevBaseBytes(folder) == 0)
    }
}
