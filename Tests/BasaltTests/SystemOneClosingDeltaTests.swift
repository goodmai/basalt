import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCLI

@Suite("System One closing delta (#234): self readout calibration loading, decide preset sizing, and Laya Core AI export manifest contract")
struct SystemOneClosingDeltaTests {
    static func runScript(_ path: String, args: [String]) -> Int32 {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = ["swift", URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appending(path: path).path] + args
        task.standardOutput = FileHandle.nullDevice
        task.standardError = FileHandle.nullDevice
        try? task.run()
        task.waitUntilExit()
        return task.terminationStatus
    }

    @Test("serve --decision-model self loads basalt_readout.json temperature and noulBias from the chat model directory and falls back to uncalibrated when absent")
    func selfCalibrationLoadsBundledReadoutAndFallsBack() async throws {
        let chatDir = FileManager.default.temporaryDirectory.appending(path: "self-readout-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: chatDir) }
        try FileManager.default.createDirectory(at: chatDir, withIntermediateDirectories: true)

        let serve = try ServeCommand.parse(["--model", "org/custom-chat", "--decision-model", "self"])
        #expect(serve.selfCalibration(chatPath: chatDir.path) == .uncalibrated)
        let uncalibrated = try #require(try await serve.loadDecisions(chat: MLXInferenceEngine(), chatPath: chatDir.path) as? ReadoutDecisionEngine)
        #expect(uncalibrated.modelID == "self" && uncalibrated.readout == .uncalibrated)

        let expected = SystemOneReadout(temperature: 1.35, noulTemperature: 1.15, noulBias: -0.42)
        try expected.encode().write(to: chatDir.appending(path: SystemOneReadout.bundleFile))
        #expect(serve.selfCalibration(chatPath: chatDir.path) == expected)

        let localServe = try ServeCommand.parse(["--model-path", chatDir.path, "--decision-model", "self"])
        #expect(localServe.selfCalibration() == expected)

        let calibrated = try #require(try await serve.loadDecisions(chat: MLXInferenceEngine(), chatPath: chatDir.path) as? ReadoutDecisionEngine)
        #expect(calibrated.modelID == "self")
        #expect(calibrated.readout.temperature == 1.35 && calibrated.readout.noulBias == -0.42)
    }

    @Test("PipelineMember(.decide) produces nil for self, an ANE encoder workload for Laya bundles, and a gated GPU encoder workload for standalone decision models")
    func pipelineMemberDecideWorkloadsAndAdmission() throws {
        #expect(PipelineMember(.decide, "self").workload == nil)
        #expect(PipelineMember(.decide, "self", bytes: 500_000_000).workload == nil)
        #expect(PipelineMember(.decide, "basalt-community/Laya-Decision-395M-ANE", bytes: 0).workload == nil)

        let laya = PipelineMember(.decide, "basalt-community/Laya-Decision-395M-ANE", bytes: 410_000_000)
        let layaWorkload = try #require(laya.workload)
        #expect(layaWorkload == Workload(name: "basalt-community/Laya-Decision-395M-ANE", role: "decide", unit: .ane, kind: .encoder, residentBytes: 410_000_000))

        let gpuDecider = PipelineMember(.decide, "openjev/OpenJev-Flash-9B-MLX-4bit", bytes: 5_200_000_000)
        let gpuWorkload = try #require(gpuDecider.workload)
        #expect(gpuWorkload == Workload(name: "openjev/OpenJev-Flash-9B-MLX-4bit", role: "decide", unit: .gpu, kind: .encoder, residentBytes: 5_200_000_000))
        #expect(PipelineMember(.decide, "org/arcane-judge-4bit", bytes: 2_000_000_000).workload?.unit == .gpu)
        #expect(PipelineMember(.decide, "org/custom-judge", bytes: 300_000_000, unit: .ane).workload?.unit == .ane)

        let chat = PipelinePreset.gemma
        let aneVerdict = PipelineProfile(members: [chat, laya], largerChat: nil, abstainBelow: 0.75, minMargin: 0.1, questions: [:])
            .plan(budgetBytes: 18_000_000_000, memory: .resident)
        #expect(aneVerdict.refusal == nil && aneVerdict.residentBytes == chat.bytes + 410_000_000)
        #expect(aneVerdict.runsBesideDecode("decide"))

        let gpuVerdict = PipelineProfile(members: [chat, gpuDecider], largerChat: nil, abstainBelow: 0.75, minMargin: 0.1, questions: [:])
            .plan(budgetBytes: 18_000_000_000, memory: .resident)
        #expect(gpuVerdict.refusal == nil && gpuVerdict.residentBytes == chat.bytes + 5_200_000_000)
        #expect(!gpuVerdict.runsBesideDecode("decide"))

        let refused = ComputePlan.admit([try #require(chat.workload), gpuWorkload], budgetBytes: 15_000_000_000)
        #expect(refused.refusal?.contains("OpenJev-Flash-9B-MLX-4bit") == true)
    }

    @Test("scripts/laya_decision_export.swift --emit-sample-manifest writes a canonical laya_decision.json accepted by CoreAIDecisionBundle and CoreAIDecisionEngine")
    func exporterSampleManifestMatchesCoreAIDecisionBundle() throws {
        let bundleDir = FileManager.default.temporaryDirectory.appending(path: "laya-export-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: bundleDir) }
        try FileManager.default.createDirectory(at: bundleDir, withIntermediateDirectories: true)
        let manifestURL = bundleDir.appending(path: CoreAIDecisionBundle.manifest)

        #expect(Self.runScript("scripts/laya_decision_export.swift", args: ["--help"]) == 0)
        #expect(Self.runScript("scripts/laya_decision_export.swift", args: ["--emit-sample-manifest", manifestURL.path, "0x256"]) == 2)
        #expect(Self.runScript("scripts/laya_decision_export.swift", args: ["--emit-sample-manifest", manifestURL.path]) == 0)

        let data = try Data(contentsOf: manifestURL)
        let root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let body = try #require(root["laya_decision"] as? [String: Any])
        let parsed = try CoreAIDecisionBundle(manifest: body, path: bundleDir.path)
        #expect(parsed.modelID == "basalt-community/Laya-Decision-395M-ANE")
        #expect((parsed.cls, parsed.sep, parsed.mask, parsed.pad) == (50281, 50282, 50284, 50283))
        #expect(parsed.functions.map(\.name) == ["b1_n256", "b4_n256", "b1_n512", "b4_n512", "b1_n1024", "b4_n1024"])
        #expect(parsed.plan([120]).map(\.function.name) == ["b1_n256"])
        #expect(parsed.plan([120, 200, 240]).map(\.function.name) == ["b4_n256"])
        #expect(parsed.plan([300, 450]).map(\.function.name) == ["b4_n512"])
        #expect(parsed.plan([700]).map(\.function.name) == ["b1_n1024"])

        #expect(CoreAIDecisionEngine.isBundle(bundleDir))
        #expect(PipelineMember(.decide, bundleDir.path, bytes: 410_000_000).workload?.unit == .ane)
        for part in CoreAIDecisionBundle.parts {
            let partURL = bundleDir.appending(path: part)
            try FileManager.default.createDirectory(at: partURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: partURL)
        }
        try CoreAIDecisionEngine.validateBundle(bundleDir)
        #expect(try CoreAIDecisionBundle(directory: bundleDir) == parsed)
    }
}
