import Foundation
import ArgumentParser
import Testing
@testable import BasaltCore
@testable import BasaltCLI

@Suite("serve.pipeline profile: config keys and flags, OCR recipe check, the OCR model as a GPU encoder in the compute plan (#402)")
struct PipelineConfigTests {
    static func directory(modelType: String, weightBytes: Int) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: "pipeline-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(#"{"model_type":"\#(modelType)"}"#.utf8).write(to: directory.appending(path: "config.json"))
        try Data(count: weightBytes).write(to: directory.appending(path: "model.safetensors"))
        return directory
    }

    @Test("serve.pipeline keys in the config file give the same options as the flags, JSON syntax included; decide maps to --decision-model")
    func configMatchesFlags() throws {
        let read: (ServeCommand) -> String = { "\($0.pipeline.ocrModel ?? "") \($0.pipeline.audioModel ?? "") \($0.decisionModel ?? "") \($0.pipeline.abstainBelow ?? -1) \($0.pipeline.minMargin ?? -1)" }
        let yaml = "defaults:\n  serve:\n    pipeline: { ocr_model: zai-org/GLM-OCR, audio_model: nvidia/Transcribe, decide: self, abstain_below: 0.75, min_margin: 0.1 }\n"
        let json = #"{"version": 1, "defaults": {"serve": {"pipeline": {"ocr_model": "zai-org/GLM-OCR", "audio_model": "nvidia/Transcribe", "decide": "self", "abstain_below": 0.75, "min_margin": 0.1}}}}"#
        let flags = try ServeCommand.parse(["--pipeline-ocr-model", "zai-org/GLM-OCR", "--pipeline-audio-model", "nvidia/Transcribe", "--decision-model", "self", "--pipeline-abstain-below", "0.75", "--pipeline-min-margin", "0.1"])
        #expect(read(try ServeConfigFileTests.serve(yaml)) == read(flags))
        #expect(read(try ServeConfigFileTests.serve(json)) == read(flags))
        #expect(read(flags) == "zai-org/GLM-OCR nvidia/Transcribe self 0.75 0.1")
        #expect(ServeConfigFileTests.issues("defaults:\n  serve:\n    pipeline: { ocr: x, abstain_below: high }\n").map(\.path) == ["defaults.serve.pipeline.ocr", "defaults.serve.pipeline.abstain_below"])
    }

    @Test("Thresholds outside 0...1 are refused at parse time")
    func thresholdsRefused() {
        #expect(throws: (any Error).self) { try ServeCommand.parse(["--pipeline-abstain-below", "1.5"]) }
        #expect(throws: (any Error).self) { try ServeCommand.parse(["--pipeline-min-margin", "-0.1"]) }
    }

    @Test("A pipeline OCR model without an OCR recipe is refused before it loads; GLM-OCR and MinerU2.5 pass")
    func ocrRecipeChecked() throws {
        let llama = try Self.directory(modelType: "llama", weightBytes: 1)
        #expect(ServeCommand.ocrRefusal("org/chat", path: llama.path)?.contains("no OCR recipe") == true)
        #expect(ServeCommand.ocrRefusal("zai-org/GLM-OCR", path: try Self.directory(modelType: "glm_ocr", weightBytes: 1).path) == nil)
        #expect(ServeCommand.ocrRefusal("opendatalab/MinerU2.5-2509-1.2B", path: try Self.directory(modelType: "qwen2_vl", weightBytes: 1).path) == nil)
    }

    @Test("The OCR model is admitted as a GPU encoder gated between generations, and counts against the memory budget")
    func planWorkload() throws {
        let path = try Self.directory(modelType: "glm_ocr", weightBytes: 4096).path
        let workload = ServeCommand.perceiveWorkload("zai-org/GLM-OCR", path: path)
        #expect(workload == Workload(name: "zai-org/GLM-OCR", role: "perceive", unit: .gpu, kind: .encoder, residentBytes: 4096))
        let chat = Workload(name: "chat", role: "chat", unit: .gpu, kind: .decode, residentBytes: 10_000)
        let admitted = ComputePlan.admit([chat, workload], budgetBytes: 20_000)
        #expect(admitted.refusal == nil && admitted.placements.last?.concurrentWithDecode == false)
        #expect(ComputePlan.admit([chat, workload], budgetBytes: 12_000).refusal?.contains("zai-org/GLM-OCR") == true)
    }
}
