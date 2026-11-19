import ArgumentParser
import BasaltBench
import BasaltCore
import BasaltInference
import Foundation

struct TrainSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "train",
        abstract: "Train a LoRA, PHM or OFT adapter directly on a Hugging Face Nemotron / SFT corpus or local JSONL file (#510)"
    )

    @Argument(help: "Model repo id from the local cache, or a local model directory")
    var model: String

    @Option(
        name: .long,
        help: "Dataset preset (nemotron:post-v2, nemotron:post-v1, nemotron:code, nemotron:math, nemotron:crossthink, nemotron:personas), hf:<org>/<repo>, or a local .jsonl file"
    )
    var dataset: String

    @Option(name: .long, help: "Optional Hugging Face dataset subset/config name")
    var subset: String?

    @Option(name: .long, help: "Optional dataset split name (defaults to preset split or auto-detected split)")
    var split: String?

    @Option(name: .long, help: "Maximum corpus examples to sample deterministically (2...50000)")
    var maxRows: Int = 512

    @Option(name: .long, help: "Minimum pass_rate / score filter in [0, 1] for verified code or math traces")
    var minPassRate: Double?

    @Flag(name: .long, inversion: .prefixedNo, help: "Keep <think> reasoning spans in target completions")
    var includeReasoning = true

    @Option(name: .long, help: "Deterministic SplitMix64 sampling seed")
    var seed: UInt64 = 42

    @Option(name: .long, help: "Training iterations")
    var iterations: Int = 200

    @Option(name: .long, help: "Batch size (1...64)")
    var batchSize: Int = 2

    @Option(name: .long, help: "Adam learning rate")
    var learningRate: Float = 1e-4

    @Option(name: .long, help: "Maximum tokenized sequence length (16...32768)")
    var maxSeqLen: Int = 2048

    @Option(name: .long, help: "LoRA rank (or PHM factor rank)")
    var rank: Int = 16

    @OptionGroup
    var adapter: AdapterTrainingOptions

    @Option(name: .long, help: "Output adapter directory (default ~/.basalt/adapters/<model fingerprint>/<timestamp>)")
    var output: String?

    @Option(name: .long, help: "Write resumable checkpoint every N steps; 0 disables")
    var checkpointEvery: Int = 50

    @Flag(name: .long, help: "Resume an interrupted training run from --output/checkpoint")
    var resume = false

    @Flag(name: .long, help: "Run quality gate check after saving")
    var checkGate = false

    @Option(name: .customLong("dpo-pairs"), help: "Optional preference pairs JSONL for two-stage SFT->DPO refinement on adapter weights (#617)")
    var dpoPairs: String?

    @Option(name: .customLong("quantized-base-bits"), help: "Quantized base weight bit-width (e.g. 4 or 8) to apply 40x LR scaling policy (#617)")
    var quantizedBaseBits: Int?

    @Option(name: .customLong("adalora-target-rank"), help: "Optional AdaLoRA target average rank for importance-based rank pruning (#617)")
    var adaLoRATargetRank: Int?

    mutating func run() async throws {
        guard let modelPath = FileManager.default.fileExists(atPath: model) ? model : ModelCache.resolve(repoId: model)?.path else {
            throw ValidationError("Model '\(model)' is not in the local cache; run `basalt models download \(model)` first")
        }
        var corpus = CorpusSourceSpec(source: dataset, subset: subset, split: split, maxRows: maxRows, includeReasoning: includeReasoning)
        corpus.minPassRate = minPassRate
        corpus.seed = seed
        var request = CorpusTrainRequest(corpus: corpus)
        request.rank = rank
        request.method = try adapter.resolveMethod(fallbackRank: rank)
        (request.iterations, request.batchSize, request.learningRate, request.maxSeqLen) = (iterations, batchSize, learningRate, maxSeqLen)
        (request.output, request.checkpointEvery, request.resume, request.applyOverlay) = (output, checkpointEvery, resume, checkGate)
        request.dpoPairsPath = dpoPairs
        request.quantizedBaseBits = quantizedBaseBits
        request.adaLoRATargetRank = adaLoRATargetRank
        request.learningRate = request.effectiveLearningRate

        let report = try await GPULock.withLock(timeout: .seconds(300)) { [request] in
            let engine = MLXInferenceEngine()
            let orchestrator = ModelOrchestratorActor(engine: engine)
            defer { await orchestrator.shutdown() }
            try await orchestrator.loadModel(path: modelPath)
            return try await engine.trainCorpus(request, lease: nil)
        }
        print("corpus: \(report.fetchedRows) fetched, \(report.keptRows) train, \(report.heldOutRows) held-out")
        print("held-out NLL: \(report.nllBefore.map { String(format: "%.3f", $0) } ?? "-") -> \(report.nllAfter.map { String(format: "%.3f", $0) } ?? "-")")
        if let step = report.resumedAt { print("resumed at step \(step) of \(iterations)") }
        print("adapter: \(report.adapterDirectory ?? "-")")
    }
}
