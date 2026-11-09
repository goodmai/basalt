import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltServer
import BasaltSpeech

struct PipelineOptions: ParsableArguments {
    @Option(name: .customLong("pipeline-preset"), help: "A shipped pipeline profile: document-qa (PaddleOCR-VL, System One by readout of the chat model, the chat model), voice-agent (Nemotron Transcribe and the chat model) or support-triage (System One with the ticket questions on department, lateness and severity, asked when a /v1/pipeline request sends none, and the chat model). The chat model is Gemma 4 12B, or Qwen3.8-27B 3-bit when it fits beside the other models with 8 GB to spare. Sets --model, --pipeline-ocr-model, --pipeline-audio-model, --decision-model, --pipeline-abstain-below and --pipeline-min-margin; a flag or config key given explicitly wins over the preset")
    var preset: PipelinePreset?

    @Option(name: .customLong("pipeline-ocr-model"), help: "OCR model for /v1/pipeline and basalt_pipeline (GLM-OCR, MinerU2.5 or PaddleOCR-VL): repo ID or local path, loaded in its own engine beside the chat model so OCR never swaps it out; without it, documents are read by Apple Vision")
    var ocrModel: String?

    @Option(name: .customLong("pipeline-audio-model"), help: "Speech model for audio in /v1/pipeline and basalt_pipeline (Nemotron Transcribe): repo ID or local path, loaded in its own engine beside the chat model; without it, audio goes to the chat model when it has an audio encoder, otherwise to macOS SpeechAnalyzer when serve runs with --experimental-speech-analyzer")
    var audioModel: String?

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

extension PipelinePreset: ExpressibleByArgument {}

extension ServeCommand {
    mutating func applyPipelinePreset(budgetBytes: Int64? = nil) async {
        guard let preset = pipeline.preset else { return }
        let budget = if let budgetBytes { budgetBytes } else { await SystemProfiler().detectResources().gpuMemory }
        let explicit = PipelineChoices(
            chat: model ?? modelPath, ocr: pipeline.ocrModel, audio: pipeline.audioModel, decide: decisionModel,
            abstainBelow: pipeline.abstainBelow, minMargin: pipeline.minMargin)
        let chosen = preset.choices(budgetBytes: budget).overridden(by: explicit)
        if modelPath == nil { model = chosen.chat }
        (pipeline.ocrModel, pipeline.audioModel, decisionModel) = (chosen.ocr, chosen.audio, chosen.decide)
        (pipeline.abstainBelow, pipeline.minMargin) = (chosen.abstainBelow, chosen.minMargin)
        log("Pipeline preset \(preset.rawValue) for a \(ComputePlan.gigabytes(budget)) working set: " + chosen.summary)
    }

    static func perceiveWorkload(_ model: String, path: String) -> Workload {
        Workload(name: model, role: "perceive", unit: .gpu, kind: .encoder, residentBytes: weightBytes(path))
    }

    static func ocrRefusal(_ model: String, path: String) -> String? {
        let modelType = ModelCache.config(at: URL(fileURLWithPath: path))?["model_type"] as? String
        guard OCRRecipe.of(modelType: modelType, hints: [path, model]) == nil else { return nil }
        return "--pipeline-ocr-model \(model) (model_type \(modelType ?? "unknown")) has no OCR recipe; use GLM-OCR (glm_ocr), MinerU2.5 (qwen2_vl) or PaddleOCR-VL (paddleocr_vl)"
    }

    struct PerceivePaths: Sendable {
        var ocr: String?
        var audio: String?
    }

    func pipelinePerceivePaths() async throws -> PerceivePaths {
        var paths = PerceivePaths()
        if let audioModel = pipeline.audioModel { paths.audio = try await resolvedPath(audioModel, quant: nil) }
        guard let ocrModel = pipeline.ocrModel else { return paths }
        let path = try await resolvedPath(ocrModel, quant: nil)
        if let refusal = Self.ocrRefusal(ocrModel, path: path) {
            log("\(red("[error]")) \(refusal)")
            throw ExitCode.failure
        }
        paths.ocr = path
        return paths
    }

    func loadPipelineStages(_ paths: PerceivePaths) async throws -> PipelineStages {
        let ocr = try await loadPerceiver(pipeline.ocrModel, path: paths.ocr, role: "OCR")
        let audio = try await loadPerceiver(pipeline.audioModel, path: paths.audio, role: "speech")
        if let audio, await !audio.transcribesAudio {
            log("\(red("[error]")) --pipeline-audio-model \(pipeline.audioModel ?? "") has no audio encoder; use Nemotron Transcribe")
            throw ExitCode.failure
        }
        return PipelineStages(ocr: ocr, audio: audio, speech: experimental.speechAnalyzer ? SpeechAnalyzerTranscriber() : nil, abstainBelow: pipeline.abstainBelow, minMargin: pipeline.minMargin,
            questions: pipeline.preset?.questions ?? [:])
    }

    func loadPerceiver(_ model: String?, path: String?, role: String) async throws -> ModelOrchestratorActor? {
        guard let model, let path else { return nil }
        var configuration = EngineConfiguration.standard
        configuration.promptCacheBytes = 0
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine(configuration), telemetry: TelemetryTracker())
        do {
            try await orchestrator.loadModel(path: path)
        } catch {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
        await orchestrator.setModelId(model)
        log("Serving \(model) (perceive, /v1/pipeline \(role)) in its own engine")
        return orchestrator
    }
}
