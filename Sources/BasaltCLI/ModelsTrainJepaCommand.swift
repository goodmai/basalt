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

    @Option(name: .long, help: "JSONL file: one scene per line, a string array or {\"views\": [...]} with at least two views")
    var dataset: String

    @Option(name: .long, help: "New output directory for adapters.safetensors and jepa_projector.safetensors (default: ~/.basalt/adapters)")
    var output: String?

    @Option(name: .long, help: "Training iterations")
    var iterations: Int = 100

    @Option(name: .long, help: "LoRA rank")
    var rank: Int = 16

    @Option(name: .long, help: "Maximum tokens per view")
    var maxSeqLen: Int = 64

    @Option(name: .long, help: "Scenes per batch, eight views each")
    var batchSize: Int = 256

    mutating func run() async throws {
        guard let modelPath = FileManager.default.fileExists(atPath: model) ? model : ModelCache.resolve(repoId: model)?.path else {
            throw ValidationError("Model '\(model)' is not in the local cache; run `basalt models download \(model)` first")
        }
        let engine = MLXInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)
        defer { await orchestrator.shutdown() }
        try await orchestrator.loadModel(path: modelPath)
        var request = LeJEPATrainRequest(datasetPath: dataset, iterations: iterations)
        (request.output, request.rank, request.batchSize, request.maxSeqLen) = (output, rank, batchSize, maxSeqLen)
        let report = try await engine.trainJEPA(request)
        print("adapter: \(report.adapterDirectory ?? "-")")
        print("projector: \(report.projectorPath ?? "-")")
        print("loss: \(report.losses.first ?? .nan) -> \(report.losses.last ?? .nan)")
        print("embedding rank: \(report.embeddingRankBefore.map(String.init) ?? "-") -> \(report.embeddingRankAfter.map(String.init) ?? "-")")
    }
}
