import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation

struct TrainJepaSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "train-jepa",
        abstract: "Train a JEPA LoRA adapter and projector on a multi-view dataset"
    )

    @Argument(help: "Model repo id from the local cache, or a local model directory")
    var model: String

    @Option(name: .long, help: "Path to a JSONL dataset of multi-view scenes")
    var dataset: String

    @Option(name: .long, help: "Output directory for adapters.safetensors and jepa_projector.safetensors")
    var output: String?

    @Option(name: .long, help: "Training iterations")
    var iterations: Int = 100

    @Option(name: .long, help: "LoRA rank")
    var rank: Int = 16

    @Option(name: .long, help: "Maximum sequence length")
    var maxSeqLen: Int = 64

    @Option(name: .long, help: "Scenes per batch")
    var batchSize: Int = 256

    mutating func run() async throws {
        guard let modelPath = FileManager.default.fileExists(atPath: model) ? model : ModelCache.resolve(repoId: model)?.path else {
            throw ValidationError("Model '\(model)' is not in the local cache; run `basalt models download \(model)` first")
        }
        let engine = MLXInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)
        defer { Task { await orchestrator.shutdown() } }
        try await orchestrator.loadModel(path: modelPath)
        let request = LeJEPATrainRequest(
            datasetPath: dataset, output: output, rank: rank, batchSize: batchSize,
            maxSeqLen: maxSeqLen, iterations: iterations)
        let report = try await engine.trainJepa(request)
        print("adapter: \(report.adapterDirectory)")
        print("projector: \(report.projectorPath)")
        print("losses: \(report.losses.count)")
    }
}
