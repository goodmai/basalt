import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import MLXOptimizers
import BasaltCore

extension MLXInferenceEngine {
    static let jepaProjection = 512
    static let jepaHidden = 2048
    static let jepaSlices = 1024
    static let jepaBatchScenes = 1...1024
    static let jepaSequence = 4...512

    public func trainJEPA(_ request: LeJEPATrainRequest) async throws(BasaltError) -> LeJEPATrainReport {
        guard let container else { throw .modelNotLoaded }
        try Self.validate(request)
        let dataset = try GoodViewDataset.load(from: request.datasetPath)
        let directory = try adapterDirectory(request.output)
        let report = try await Self.labRun(container, "JEPA training") { context in
            try Self.trainJEPA(request, dataset: dataset, directory: directory, context: context)
        }
        guard self.container === container else { throw .inferenceError("the model changed during JEPA training; the adapter in \(directory.path) is saved but belongs to the previous model") }
        return report
    }

    static func validate(_ request: LeJEPATrainRequest) throws(BasaltError) {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "train_jepa: \(reason)") }
        guard trainingRanks.contains(request.rank) else { throw refused("rank must be in \(trainingRanks)") }
        guard trainingIterations.contains(request.iterations) else { throw refused("iterations must be in \(trainingIterations)") }
        guard jepaBatchScenes.contains(request.batchSize) else { throw refused("batchSize must be in \(jepaBatchScenes)") }
        guard jepaSequence.contains(request.maxSeqLen) else { throw refused("maxSeqLen must be in \(jepaSequence)") }
        guard request.learningRate.isFinite, request.learningRate > 0, request.learningRate <= 1 else { throw refused("learningRate must be in (0, 1]") }
        guard request.weightDecay.isFinite, request.weightDecay >= 0 else { throw refused("weightDecay must be finite and non-negative") }
        guard request.warmupRatio.isFinite, (0 ... 1).contains(request.warmupRatio) else { throw refused("warmupRatio must be in [0, 1]") }
        guard request.lambda.isFinite, request.lambda >= 0 else { throw refused("lambda must be finite and non-negative") }
        guard request.maxGradNorm.isFinite, request.maxGradNorm > 0 else { throw refused("maxGradNorm must be finite and positive") }
    }

    static func jepaSchedule(_ request: LeJEPATrainRequest) -> (Int) -> Float {
        let warmup = max(1, Int(Float(request.iterations) * request.warmupRatio))
        return joinSchedules(
            [linearSchedule(0, end: request.learningRate, steps: warmup),
             cosineDecay(request.learningRate, decaySteps: max(1, request.iterations - warmup))],
            boundaries: [warmup])
    }

    static func jepaScenes(_ scenes: [GoodViewScene], batchSize: Int, step: Int) -> [GoodViewScene] {
        let size = min(batchSize, scenes.count)
        return (0 ..< size).map { scenes[(step * size + $0) % scenes.count] }
    }

    static func embeddingRank(_ head: JEPAHead, _ batch: GoodViewBatch) -> Int? {
        let projected = head(batch.inputIds, attentionMask: batch.attentionMask).projected
        eval(projected)
        return (try? WeightSpectrum.measure(rows: projected.dim(0), columns: projected.dim(1)) { projected[$0] })?.rank
    }

    static func trainJEPA(
        _ request: LeJEPATrainRequest, dataset: GoodViewDataset, directory: URL, context: ModelContext
    ) throws -> LeJEPATrainReport {
        let collator = GoodViewCollator(maxSeqLen: request.maxSeqLen)
        let encode: (String) -> [Int] = { context.tokenizer.encode(text: $0, addSpecialTokens: true) }
        var rng = SplitMix64(seed: 0)
        let probe = try collator.collate(scenes: jepaScenes(dataset.scenes, batchSize: request.batchSize, step: 0), encode: encode, rng: &rng)
        guard let hidden = LeJEPABackbone.encode(model: context.model, inputIds: probe.inputIds[0 ..< 1], isCausal: request.isCausal).hiddenStates else {
            throw BasaltError.invalidRequestStructure(details: "train_jepa: this architecture has no attention-only encoder path (hybrid and recurrent layers are not supported)")
        }
        let configuration = try attachLoRA(to: context.model, layers: request.layers, rank: request.rank)
        defer { detachLoRA(from: context.model) }
        let head = JEPAHead(
            backbone: context.model, inputDim: hidden.dim(-1), hiddenDim: jepaHidden, outputDim: jepaProjection,
            isCausal: request.isCausal, poolingStrategy: .mean)
        head.projector.train()
        let rankBefore = embeddingRank(head, probe)
        var history = try jepaSteps(request, dataset: dataset, collator: collator, encode: encode, head: head)
        history.embeddingRankBefore = rankBefore
        history.embeddingRankAfter = embeddingRank(head, probe)
        try withTaskCancellationShield {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try JSONEncoder().encode(configuration).write(to: directory.appending(component: "adapter_config.json"))
            try LoRATrain.saveLoRAWeights(model: context.model, url: directory.appending(component: "adapters.safetensors"))
            try MLX.save(arrays: Dictionary(uniqueKeysWithValues: head.projector.parameters().flattened()), url: directory.appending(component: "jepa_projector.safetensors"))
        }
        history.adapterDirectory = directory.path
        history.projectorPath = directory.appending(component: "jepa_projector.safetensors").path
        return history
    }

    static func jepaSteps(
        _ request: LeJEPATrainRequest, dataset: GoodViewDataset, collator: GoodViewCollator, encode: (String) -> [Int], head: JEPAHead
    ) throws -> LeJEPATrainReport {
        let views = GoodViewCollator.totalViewsPerScene
        let loss = LeJEPALoss(baseLambda: request.lambda, referenceSamples: request.batchSize * views, sliceCount: jepaSlices, viewsPerScene: views)
        let directions = LeJEPALoss.unitDirections(projDim: jepaProjection, sliceCount: jepaSlices, seed: 42)
        let optimizer = AdamW(learningRate: request.learningRate, weightDecay: request.weightDecay)
        let schedule = jepaSchedule(request)
        let step = valueAndGrad(model: head) { head, arrays in
            let out = loss(head(arrays[0], attentionMask: arrays[1]).projected, directions: arrays[2])
            return [out.total, out.invariance, out.sigReg]
        }
        var report = LeJEPATrainReport()
        for index in 0 ..< request.iterations {
            try Task.checkCancellation()
            var rng = SplitMix64(seed: UInt64(index) &+ 42)
            let batch = try collator.collate(scenes: jepaScenes(dataset.scenes, batchSize: request.batchSize, step: index), encode: encode, rng: &rng)
            let (values, gradients) = step(head, [batch.inputIds, batch.attentionMask, directions])
            let (clipped, norm) = clipGradNorm(gradients: gradients, maxNorm: request.maxGradNorm)
            optimizer.learningRate = schedule(index)
            optimizer.update(model: head, gradients: clipped)
            eval(head, optimizer, values, norm)
            let total = values[0].item(Float.self)
            guard total.isFinite else { throw BasaltError.inferenceError("train_jepa: the loss became non-finite at step \(index); nothing was saved") }
            report.losses.append(total)
            report.invarianceLosses.append(values[1].item(Float.self))
            report.sigRegLosses.append(values[2].item(Float.self))
            report.gradNorms.append(norm.item(Float.self))
        }
        return LeJEPATrainReport(
            losses: reported(report.losses), invarianceLosses: reported(report.invarianceLosses),
            sigRegLosses: reported(report.sigRegLosses), gradNorms: reported(report.gradNorms))
    }
}
