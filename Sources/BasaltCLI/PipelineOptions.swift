import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltServer

struct PipelineOptions: ParsableArguments {
    @Option(name: .customLong("pipeline-ocr-model"), help: "OCR model for /v1/pipeline and basalt_pipeline (GLM-OCR, MinerU2.5 or PaddleOCR-VL): repo ID or local path, loaded in its own engine beside the chat model so OCR never swaps it out")
    var ocrModel: String?

    @Option(name: .customLong("pipeline-abstain-below"), help: "In /v1/pipeline, System One abstains, and the question goes to the chat model, when its confidence is below this value (0 to 1; default: never)")
    var abstainBelow: Double?

    @Option(name: .customLong("pipeline-min-margin"), help: "In /v1/pipeline, System One abstains when the margin between its top two options is below this value (0 to 1; default: never)")
    var minMargin: Double?

    func validate() throws {
        for (flag, value) in [("--pipeline-abstain-below", abstainBelow), ("--pipeline-min-margin", minMargin)] {
            if let value, !(0 ... 1).contains(value) { throw ValidationError("\(flag) must lie between 0 and 1, got \(value)") }
        }
    }
}

extension ServeCommand {
    static func perceiveWorkload(_ model: String, path: String) -> Workload {
        Workload(name: model, role: "perceive", unit: .gpu, kind: .encoder, residentBytes: weightBytes(path))
    }

    static func ocrRefusal(_ model: String, path: String) -> String? {
        let modelType = ModelCache.config(at: URL(fileURLWithPath: path))?["model_type"] as? String
        guard OCRRecipe.of(modelType: modelType, hints: [path, model]) == nil else { return nil }
        return "--pipeline-ocr-model \(model) (model_type \(modelType ?? "unknown")) has no OCR recipe; use GLM-OCR (glm_ocr), MinerU2.5 (qwen2_vl) or PaddleOCR-VL (paddleocr_vl)"
    }

    func pipelineOCRPath() async throws -> String? {
        guard let ocrModel = pipeline.ocrModel else { return nil }
        let path = try await resolvedPath(ocrModel, quant: nil)
        if let refusal = Self.ocrRefusal(ocrModel, path: path) {
            log("\(red("[error]")) \(refusal)")
            throw ExitCode.failure
        }
        return path
    }

    func loadPipelineStages(ocrPath: String?) async throws -> PipelineStages {
        guard let ocrPath, let ocrModel = pipeline.ocrModel else {
            return PipelineStages(abstainBelow: pipeline.abstainBelow, minMargin: pipeline.minMargin)
        }
        var configuration = EngineConfiguration.standard
        configuration.promptCacheBytes = 0
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine(configuration), telemetry: TelemetryTracker())
        do {
            try await orchestrator.loadModel(path: ocrPath)
        } catch {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
        await orchestrator.setModelId(ocrModel)
        log("Serving \(ocrModel) (perceive, /v1/pipeline OCR) in its own engine")
        return PipelineStages(ocr: orchestrator, abstainBelow: pipeline.abstainBelow, minMargin: pipeline.minMargin)
    }
}
