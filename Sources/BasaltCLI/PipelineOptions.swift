import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltServer
import BasaltSpeech

struct PipelineOptions: ParsableArguments {
    @Option(name: .customLong("pipeline-preset"), help: "A shipped pipeline profile: document-qa (PaddleOCR-VL, System One by readout of the chat model, the chat model), voice-agent (Nemotron Transcribe and the chat model) or support-triage (System One with the ticket questions on department, lateness and severity, asked when a /v1/pipeline request sends none, and the chat model). The chat model is Gemma 4 12B, or Qwen3.8-27B 3-bit when it fits beside the other models with 8 GB to spare. Sets --model, --pipeline-ocr-model, --pipeline-audio-model, --decision-model, --pipeline-abstain-below and --pipeline-min-margin; a flag or config key given explicitly wins over the preset")
    var preset: PipelinePreset?

    @Option(name: .customLong("pipeline-memory"), help: "How pipeline models share memory: resident keeps every model loaded, so their sum must fit the working set; sequential loads each perceiver (OCR, speech) for its stage and unloads it after, keeping the chat model loaded when it fits beside the largest perceiver and unloading it around them otherwise, so only the largest stage plus 2 GB for KV and activations must fit; sequential requests to /v1/pipeline run one at a time and plain chat requests wait while models swap. auto picks resident when the sum fits, otherwise sequential, and refuses only when one stage alone does not fit (default: auto)")
    var memory: PipelineMemory = .auto

    @Option(name: .customLong("pipeline-ocr-model"), help: "OCR model for /v1/pipeline and basalt_pipeline (GLM-OCR, MinerU2.5, PaddleOCR-VL or DeepSeek-OCR): repo ID or local path, loaded in its own engine beside the chat model so OCR never swaps it out; without it, documents are read by Apple Vision")
    var ocrModel: String?

    @Option(name: .customLong("pipeline-audio-model"), help: "Speech model for audio in /v1/pipeline and basalt_pipeline (Nemotron Transcribe): repo ID or local path, loaded in its own engine beside the chat model; without it, audio goes to the chat model when it has an audio encoder, otherwise to macOS SpeechAnalyzer when serve runs with --experimental-speech-analyzer")
    var audioModel: String?

    @Option(name: .customLong("pipeline-abstain-below"), help: "In /v1/pipeline, System One abstains, and the question goes to the chat model, when its confidence is below this value (0 to 1; default: never)")
    var abstainBelow: Double?

    @Option(name: .customLong("pipeline-min-margin"), help: "In /v1/pipeline, System One abstains when the margin between its top two options is below this value (0 to 1; default: never)")
    var minMargin: Double?

    @Option(name: .customLong("pipeline-ocr-idle-unload"), help: "Keep --pipeline-ocr-model on demand: load it on the first /v1/pipeline request that reads a document and unload it after this many seconds without one. The compute plan still counts it as resident, since it may be loaded at any time (default: always loaded)")
    var ocrIdleUnload: Int?

    @Option(name: .customLong("pipeline-audio-idle-unload"), help: "Keep --pipeline-audio-model on demand: load it on the first /v1/pipeline request with audio and unload it after this many seconds without one; counted as resident by the compute plan (default: always loaded)")
    var audioIdleUnload: Int?

    func validate() throws {
        for (flag, value) in [("--pipeline-abstain-below", abstainBelow), ("--pipeline-min-margin", minMargin)] {
            if let value, !(0 ... 1).contains(value) { throw ValidationError("\(flag) must lie between 0 and 1, got \(value)") }
        }
        for (flag, value, model) in [("--pipeline-ocr-idle-unload", ocrIdleUnload, ocrModel), ("--pipeline-audio-idle-unload", audioIdleUnload, audioModel)] {
            guard let value else { continue }
            guard value > 0 else { throw ValidationError("\(flag) must be a positive number of seconds, got \(value)") }
            guard model != nil || preset != nil else { throw ValidationError("\(flag) needs the model it unloads (\(flag.replacingOccurrences(of: "-idle-unload", with: "-model")))") }
        }
    }
}

extension PipelinePreset: ExpressibleByArgument {}

extension PipelineMemory: ExpressibleByArgument {}

extension ServeCommand {
    mutating func applyPipelinePreset(budgetBytes: Int64? = nil) async {
        guard let preset = pipeline.preset else { return }
        let budget = if let budgetBytes { budgetBytes } else { await SystemProfiler().detectResources().gpuMemory }
        let explicit = PipelineChoices(
            chat: model ?? modelPath, ocr: pipeline.ocrModel, audio: pipeline.audioModel, decide: decision.model,
            abstainBelow: pipeline.abstainBelow, minMargin: pipeline.minMargin)
        let chosen = preset.choices(budgetBytes: budget).overridden(by: explicit)
        if modelPath == nil { model = chosen.chat }
        (pipeline.ocrModel, pipeline.audioModel, decision.model) = (chosen.ocr, chosen.audio, chosen.decide)
        (pipeline.abstainBelow, pipeline.minMargin) = (chosen.abstainBelow, chosen.minMargin)
        log("Pipeline preset \(preset.rawValue) for a \(ComputePlan.gigabytes(budget)) working set: " + chosen.summary)
    }

    static func perceiveWorkload(_ model: String, path: String) -> Workload {
        Workload(name: model, role: "perceive", unit: .gpu, kind: .encoder, residentBytes: weightBytes(path))
    }

    static func ocrRefusal(_ model: String, path: String) -> String? {
        let modelType = ModelCache.config(at: URL(fileURLWithPath: path))?["model_type"] as? String
        guard OCRRecipe.of(modelType: modelType, hints: [path, model]) == nil else { return nil }
        return "--pipeline-ocr-model \(model) (model_type \(modelType ?? "unknown")) has no OCR recipe; use GLM-OCR (glm_ocr), MinerU2.5 (qwen2_vl), PaddleOCR-VL (paddleocr_vl) or DeepSeek-OCR (deepseekocr)"
    }

    static func audioRefusal(_ model: String, path: String) -> String? {
        let modelType = ModelCache.config(at: URL(fileURLWithPath: path))?["model_type"] as? String
        guard !MLXInferenceEngine.transcribesAudio(modelType: modelType) else { return nil }
        return "--pipeline-audio-model \(model) (model_type \(modelType ?? "unknown")) has no audio encoder; use Nemotron Transcribe (nemotron_transcribe)"
    }

    static func adapterRefusal(_ adapters: [String], plan: PlanVerdict?) -> String? {
        guard !adapters.isEmpty, plan?.keep(ofKind: .decode) == .sequential else { return nil }
        return "--adapter needs the chat model to stay loaded, but this pipeline plan unloads it while a perceiver runs; drop --adapter, or serve with --pipeline-memory resident on a machine where the set fits"
    }

    struct PerceivePaths: Sendable {
        var ocr: String?
        var audio: String?
    }

    func pipelinePerceivePaths() async throws -> PerceivePaths {
        var paths = PerceivePaths()
        if let audioModel = pipeline.audioModel {
            let path = try await resolvedPath(audioModel, quant: nil)
            if let refusal = Self.audioRefusal(audioModel, path: path) {
                log("\(red("[error]")) \(refusal)")
                throw ExitCode.failure
            }
            paths.audio = path
        }
        guard let ocrModel = pipeline.ocrModel else { return paths }
        let path = try await resolvedPath(ocrModel, quant: nil)
        if let refusal = Self.ocrRefusal(ocrModel, path: path) {
            log("\(red("[error]")) \(refusal)")
            throw ExitCode.failure
        }
        paths.ocr = path
        return paths
    }

    func loadPipelineStages(_ paths: PerceivePaths, plan: PlanVerdict? = nil) async throws -> PipelineStages {
        if let refusal = Self.adapterRefusal(adapters, plan: plan) {
            log("\(red("[error]")) \(refusal)")
            throw ExitCode.failure
        }
        let sequential = plan?.keep(of: Workload.perceiveRole) == .sequential
        let swap = sequential ? PipelineSwap(ocrPath: paths.ocr, audioPath: paths.audio, unloadsChat: plan?.keep(ofKind: .decode) == .sequential, maxQueue: maxQueue) : nil
        if sequential, pipeline.ocrIdleUnload != nil || pipeline.audioIdleUnload != nil {
            log("\(red("[error]")) an on-demand perceiver (--pipeline-*-idle-unload) counts as resident, but this plan swaps perceivers per request because the set does not fit resident; drop the idle unload or serve smaller models")
            throw ExitCode.failure
        }
        let ocr = try await loadPerceiver(pipeline.ocrModel, path: paths.ocr, role: "OCR", loads: !sequential, onDemand: pipeline.ocrIdleUnload != nil)
        let audio = try await loadPerceiver(pipeline.audioModel, path: paths.audio, role: "speech", loads: !sequential, onDemand: pipeline.audioIdleUnload != nil)
        var stages = PipelineStages(ocr: ocr, audio: audio, speech: experimental.speechAnalyzer ? SpeechAnalyzerTranscriber() : nil, abstainBelow: pipeline.abstainBelow, minMargin: pipeline.minMargin,
            questions: pipeline.preset?.questions ?? [:])
        stages.swap = swap
        stages.onDemand = onDemand([(.ocrModel, ocr, paths.ocr, pipeline.ocrIdleUnload, pipeline.ocrModel), (.speechModel, audio, paths.audio, pipeline.audioIdleUnload, pipeline.audioModel)])
        return stages
    }

    func loadPerceiver(_ model: String?, path: String?, role: String, loads: Bool = true, onDemand: Bool = false) async throws -> ModelOrchestratorActor? {
        guard let model, let path else { return nil }
        var configuration = EngineConfiguration.standard
        configuration.promptCacheBytes = 0
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine(configuration), telemetry: TelemetryTracker())
        guard loads, !onDemand else {
            await orchestrator.setModelId(model)
            if !onDemand { log("Serving \(model) (perceive, /v1/pipeline \(role)) sequentially: loaded for each request's \(role) stage and unloaded after it") }
            return orchestrator
        }
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

extension ServeCommand {
    func onDemand(_ perceivers: [(PipelineCore.Perceiver, ModelOrchestratorActor?, String?, Int?, String?)]) -> [PipelineCore.Perceiver: OnDemandModel] {
        var held: [PipelineCore.Perceiver: OnDemandModel] = [:]
        for case let (perceiver, model?, path?, seconds?, name) in perceivers {
            held[perceiver] = OnDemandModel(model: model, path: path, idle: .seconds(seconds))
            log("Serving \(name ?? path) on demand: loaded by the first request that needs it, unloaded after \(seconds) s idle")
        }
        return held
    }
}
