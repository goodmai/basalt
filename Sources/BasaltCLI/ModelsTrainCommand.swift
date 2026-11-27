import ArgumentParser
import BasaltBench
import BasaltCore
import BasaltInference
import Foundation

struct TrainSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "train",
        abstract: "Train a LoRA, PHM or OFT adapter with SFT (--dataset), offline DPO (--dpo-pairs), or GRPO (--grpo) (#510, #822)"
    )

    @Argument(help: "Model repo id from the local cache, or a local model directory")
    var model: String

    @Option(
        name: .long,
        help: "Dataset preset (nemotron:post-v2, nemotron:post-v1, nemotron:code, nemotron:math, nemotron:crossthink, nemotron:personas), hf:<org>/<repo>, or a local .jsonl file"
    )
    var dataset: String?

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

    @Option(name: .long, help: "Optimizer learning rate")
    var learningRate: Float = 1e-4

    @OptionGroup
    var optimizerOptions: OptimizerTrainingOptions

    var optimizer: TrainingOptimizerKind { optimizerOptions.optimizer }

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

    @Flag(name: .long, help: "Resume an interrupted SFT training run from --output/checkpoint")
    var resume = false

    @Flag(name: .long, help: "Run quality gate check after saving")
    var checkGate = false

    @Option(name: .customLong("dpo-pairs"), help: "Preference pairs JSONL for standalone offline DPO or two-stage SFT -> DPO refinement (#822)")
    var dpoPairs: String?

    @Option(name: .customLong("dpo-beta"), help: "DPO inverse temperature beta (> 0, default 0.1)")
    var dpoBeta: Float = 0.1

    @Flag(name: .customLong("grpo"), help: "Train a LoRA adapter with DeepSeekMath GRPO on verifiable tasks (#822)")
    var grpo = false

    @Option(name: .customLong("grpo-group-size"), help: "Rollouts per prompt for GRPO (2...64, default 4)")
    var grpoGroupSize: Int = 4

    @Option(name: .customLong("grpo-beta-kl"), help: "Schulman KL penalty weight for GRPO (>= 0, default 0.04)")
    var grpoBetaKL: Float = 0.04

    @Option(name: .customLong("grpo-rounds"), help: "Number of GRPO rollout/update rounds (1...10000, default 8)")
    var grpoRounds: Int = 8

    @Option(name: .customLong("grpo-tasks"), help: "Number of verifiable arithmetic tasks for GRPO (2...10000, default 8)")
    var grpoTasks: Int = 8

    @Option(name: .customLong("quantized-base-bits"), help: "Quantized base weight bit-width (e.g. 4 or 8) to apply 40x LR scaling policy (#617)")
    var quantizedBaseBits: Int?

    @Option(name: .customLong("adalora-target-rank"), help: "Optional AdaLoRA target average rank for importance-based rank pruning (#617)")
    var adaLoRATargetRank: Int?

    @Option(name: .customLong("stability-telemetry-every"), help: "Sample training stability telemetry every N steps; 0 disables (#861)")
    var stabilityTelemetryEvery: Int = 0

    func validate() throws {
        if adaLoRATargetRank != nil { throw ValidationError("--adalora-target-rank is not wired to corpus training yet (#617): the run would keep a fixed rank") }
        try optimizerOptions.validate()
        try validateModes()
        try validatePostTrainingOptions()
    }

    mutating func run() async throws {
        guard let modelPath = FileManager.default.fileExists(atPath: model) ? model : ModelCache.resolve(repoId: model)?.path else {
            throw ValidationError("Model '\(model)' is not in the local cache; run `basalt models download \(model)` first")
        }
        let resolvedMethod = try adapter.resolveMethod(fallbackRank: rank)
        if grpo {
            try await runGRPO(modelPath: modelPath, method: resolvedMethod)
        } else if let dataset, let pairsPath = dpoPairs {
            try await runTwoStageSFTDPO(modelPath: modelPath, dataset: dataset, pairsPath: pairsPath, method: resolvedMethod)
        } else if let pairsPath = dpoPairs {
            try await runOfflineDPO(modelPath: modelPath, pairsPath: pairsPath, method: resolvedMethod)
        } else if let dataset {
            try await runCorpusSFT(modelPath: modelPath, dataset: dataset, method: resolvedMethod)
        }
    }
}

extension TrainSubcommand {
    private func validateModes() throws {
        if dataset == nil && dpoPairs == nil && !grpo {
            throw ValidationError("Provide at least one training mode: --dataset, --dpo-pairs, or --grpo")
        }
        if grpo && (dataset != nil || dpoPairs != nil) {
            throw ValidationError("--grpo cannot be combined with --dataset or --dpo-pairs")
        }
        if resume && dataset == nil {
            throw ValidationError("--resume requires --dataset")
        }
    }

    private func validatePostTrainingOptions() throws {
        try requireValid(dpoBeta.isFinite && dpoBeta > 0 && dpoBeta <= 100, "--dpo-beta must be finite and in (0, 100]")
        try requireValid((2 ... 64).contains(grpoGroupSize), "--grpo-group-size must be in 2...64")
        try requireValid(grpoBetaKL.isFinite && grpoBetaKL >= 0 && grpoBetaKL <= 100, "--grpo-beta-kl must be finite and in [0, 100]")
        try requireValid((1 ... 10_000).contains(grpoRounds), "--grpo-rounds must be in 1...10000")
        try requireValid((2 ... 10_000).contains(grpoTasks), "--grpo-tasks must be in 2...10000")
        try requireValid(stabilityTelemetryEvery >= 0, "--stability-telemetry-every must be 0 or positive")
    }

    private func requireValid(_ condition: Bool, _ message: String) throws {
        guard condition else { throw ValidationError(message) }
    }

    func buildCorpusRequest(
        dataset: String,
        method: AdapterMethod?,
        effectiveLR: Float,
        outputPath: String?,
        applyOverlay: Bool
    ) -> CorpusTrainRequest {
        var corpus = CorpusSourceSpec(source: dataset, subset: subset, split: split, maxRows: maxRows, includeReasoning: includeReasoning)
        corpus.minPassRate = minPassRate
        corpus.seed = seed
        var request = CorpusTrainRequest(corpus: corpus)
        request.rank = rank
        request.method = method
        request.optimizer = optimizer
        request.optimizerOptions = optimizerOptions.buildOptions()
        (request.iterations, request.batchSize, request.learningRate, request.maxSeqLen) = (iterations, batchSize, effectiveLR, maxSeqLen)
        (request.output, request.checkpointEvery, request.resume, request.applyOverlay) = (outputPath, checkpointEvery, resume, applyOverlay)
        request.dpoPairsPath = dpoPairs
        request.quantizedBaseBits = quantizedBaseBits
        request.adaLoRATargetRank = adaLoRATargetRank
        request.stabilityTelemetryEvery = stabilityTelemetryEvery > 0 ? stabilityTelemetryEvery : nil
        return request
    }

    func buildDPOConfig(
        pairsPath: String,
        method: AdapterMethod?,
        effectiveLR: Float,
        outputPath: String?,
        initialAdapterPath: String?
    ) -> OfflineDPOConfig {
        var config = OfflineDPOConfig(
            pairsPath: pairsPath,
            beta: dpoBeta,
            iterations: iterations,
            learningRate: effectiveLR,
            batchSize: batchSize,
            maxSeqLen: maxSeqLen
        )
        config.rank = rank
        config.method = method
        config.optimizer = optimizer
        config.optimizerOptions = optimizerOptions.buildOptions(weightDecay: config.weightDecay)
        config.seed = seed
        config.initialAdapterPath = initialAdapterPath
        config.output = outputPath
        config.applyOverlay = checkGate
        config.gate = checkGate ? GateRequest(heldOut: CorpusTrainRequest.defaultHeldOutPrompts) : nil
        config.stabilityTelemetryEvery = stabilityTelemetryEvery > 0 ? stabilityTelemetryEvery : nil
        return config
    }

    func buildGRPOConfig(method: AdapterMethod?, effectiveLR: Float) -> GRPOConfig {
        var config = GRPOConfig(
            groupSize: grpoGroupSize,
            betaKL: grpoBetaKL,
            clipEpsilon: 0.2,
            rounds: grpoRounds,
            learningRate: effectiveLR,
            seed: seed
        )
        config.optimizer = optimizer
        config.optimizerOptions = optimizerOptions.buildOptions(weightDecay: config.weightDecay)
        config.tasks = .arithmetic(count: grpoTasks, seed: seed)
        config.rank = rank
        config.method = method
        config.output = output
        config.applyOverlay = checkGate
        config.stabilityTelemetryEvery = stabilityTelemetryEvery > 0 ? stabilityTelemetryEvery : nil
        return config
    }
}

extension TrainSubcommand {
    private func runCorpusSFT(modelPath: String, dataset: String, method: AdapterMethod?) async throws {
        let effectiveLR = try QuantizedLRPolicy.effectiveLearningRate(baseLR: learningRate, weightBits: quantizedBaseBits)
        let request = buildCorpusRequest(dataset: dataset, method: method, effectiveLR: effectiveLR, outputPath: output, applyOverlay: checkGate)
        let report = try await GPULock.withLock(timeout: .seconds(300)) { [modelPath, request] in
            let engine = MLXInferenceEngine()
            let orchestrator = ModelOrchestratorActor(engine: engine)
            defer { await orchestrator.shutdown() }
            try await orchestrator.loadModel(path: modelPath)
            return try await engine.trainCorpus(request, lease: nil)
        }
        printCorpusSummary(report)
    }

    private func runOfflineDPO(modelPath: String, pairsPath: String, method: AdapterMethod?) async throws {
        let effectiveLR = try QuantizedLRPolicy.effectiveLearningRate(baseLR: learningRate, weightBits: quantizedBaseBits)
        let config = buildDPOConfig(pairsPath: pairsPath, method: method, effectiveLR: effectiveLR, outputPath: output, initialAdapterPath: nil)
        let report = try await GPULock.withLock(timeout: .seconds(300)) { [modelPath, config] in
            let engine = MLXInferenceEngine()
            let orchestrator = ModelOrchestratorActor(engine: engine)
            defer { await orchestrator.shutdown() }
            try await orchestrator.loadModel(path: modelPath)
            return try await engine.trainOfflineDPO(config)
        }
        printDPOSummary(report)
    }

    private func runTwoStageSFTDPO(
        modelPath: String,
        dataset: String,
        pairsPath: String,
        method: AdapterMethod?
    ) async throws {
        let policy = TwoStageSFTDPOPolicy(baseLearningRate: learningRate, dpoLearningRateRatio: 0.2, dpoBeta: dpoBeta, dpoPairsPath: pairsPath)
        let stages = try policy.resolvedStages(weightBits: quantizedBaseBits)
        let sftOutput = output.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath).standardizedFileURL.path + "-sft" }
        let sftRequest = buildCorpusRequest(dataset: dataset, method: method, effectiveLR: stages.sftLR, outputPath: sftOutput, applyOverlay: false)
        let dpoTemplate = buildDPOConfig(pairsPath: pairsPath, method: method, effectiveLR: stages.dpoLR, outputPath: output, initialAdapterPath: nil)
        let (sftReport, dpoReport) = try await GPULock.withLock(timeout: .seconds(300)) { [modelPath, sftRequest, dpoTemplate] in
            let engine = MLXInferenceEngine()
            let orchestrator = ModelOrchestratorActor(engine: engine)
            defer { await orchestrator.shutdown() }
            try await orchestrator.loadModel(path: modelPath)
            let sft = try await engine.trainCorpus(sftRequest, lease: nil)
            var dpoConfig = dpoTemplate
            dpoConfig.initialAdapterPath = sft.adapterDirectory
            if dpoConfig.output == nil { dpoConfig.output = sft.adapterDirectory.map { "\($0)-dpo" } }
            let dpo = try await engine.trainOfflineDPO(dpoConfig)
            return (sft, dpo)
        }
        printCorpusSummary(sftReport)
        printDPOSummary(dpoReport)
    }

    private func runGRPO(modelPath: String, method: AdapterMethod?) async throws {
        let effectiveLR = try QuantizedLRPolicy.effectiveLearningRate(baseLR: learningRate, weightBits: quantizedBaseBits)
        let config = buildGRPOConfig(method: method, effectiveLR: effectiveLR)
        let report = try await GPULock.withLock(timeout: .seconds(300)) { [modelPath, config] in
            let engine = MLXInferenceEngine()
            let orchestrator = ModelOrchestratorActor(engine: engine)
            defer { await orchestrator.shutdown() }
            try await orchestrator.loadModel(path: modelPath)
            return try await engine.trainGRPO(config)
        }
        print("grpo accuracy: \(String(format: "%.3f", report.accuracyBefore)) -> \(String(format: "%.3f", report.accuracyAfter)), updates: \(report.updatesApplied), skipped: \(report.skippedGroups), mean KL: \(String(format: "%.4f", report.meanKL)), gate: \(report.gatePassed ? "passed" : "failed")")
        if let stability = report.stabilitySummary { print(stability.formattedSummaryLine()) }
        print("adapter: \(report.outputPath)")
    }

    private func printCorpusSummary(_ report: CorpusTrainReport) {
        print("corpus: \(report.fetchedRows) fetched, \(report.keptRows) train, \(report.heldOutRows) held-out")
        print("held-out NLL: \(report.nllBefore.map { String(format: "%.3f", $0) } ?? "-") -> \(report.nllAfter.map { String(format: "%.3f", $0) } ?? "-")")
        if let step = report.resumedAt { print("resumed at step \(step) of \(iterations)") }
        if let stability = report.provenance?.stabilitySummary { print(stability.formattedSummaryLine()) }
        print("adapter: \(report.adapterDirectory ?? "-")")
    }

    private func printDPOSummary(_ report: OfflineDPOReport) {
        print("dpo margin: \(String(format: "%.3f", report.meanMarginBefore)) -> \(String(format: "%.3f", report.meanMarginAfter)), accuracy: \(String(format: "%.3f", report.preferenceAccuracyBefore)) -> \(String(format: "%.3f", report.preferenceAccuracyAfter)), KL: \(String(format: "%.4f", report.klDivergence))")
        if let stability = report.stabilitySummary { print(stability.formattedSummaryLine()) }
        print("adapter: \(report.outputPath)")
    }
}
