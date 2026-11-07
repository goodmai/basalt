import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
import BasaltCore

extension MLXInferenceEngine {
    public func trainJepa(_ request: LeJEPATrainRequest) async throws(BasaltError) -> LeJEPATrainReport {
        try await trainJEPA(request)
    }

    public func trainJEPA(_ request: LeJEPATrainRequest) async throws(BasaltError) -> LeJEPATrainReport {
        guard let container else { throw .modelNotLoaded }
        let epoch = modelEpoch
        let anatomy = try await self.anatomy()
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while the JEPA training was prepared; retry") }
        try Self.validate(request, layers: anatomy.layers.count)
        let dataset = try GoodViewDataset.load(from: request.datasetPath)
        let directory = try adapterDirectory(request.output)
        guard self.container === container else { throw .inferenceError("the model changed; retry") }
        let report: LeJEPATrainReport
        do {
            report = try await container.performEvaluating(nonSendable: ()) { context, _ in
                try Self.trainJepa(request, dataset: dataset, directory: directory, context: context)
            }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("JEPA training failed: \(Self.briefly(error))")
        }
        guard self.container === container else { throw .inferenceError("the model changed during JEPA training; the adapter in \(directory.path) was not applied") }
        return report
    }

    static func validate(_ request: LeJEPATrainRequest, layers: Int) throws(BasaltError) {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "train_jepa: \(reason)") }
        guard request.iterations > 0 else { throw refused("iterations must be positive") }
        guard request.rank > 0 else { throw refused("rank must be positive") }
        guard request.batchSize > 0 else { throw refused("batchSize must be positive") }
        guard request.maxSeqLen >= 4 else { throw refused("maxSeqLen must be at least 4") }
        guard request.learningRate.isFinite, request.learningRate > 0 else { throw refused("learningRate must be finite and positive") }
        guard request.weightDecay.isFinite, request.weightDecay >= 0 else { throw refused("weightDecay must be finite and non-negative") }
        guard request.warmupRatio.isFinite, request.warmupRatio >= 0, request.warmupRatio <= 1 else { throw refused("warmupRatio must be in [0, 1]") }
        guard request.lambda.isFinite, request.lambda >= 0 else { throw refused("lambda must be finite and non-negative") }
        guard request.maxGradNorm.isFinite, request.maxGradNorm > 0 else { throw refused("maxGradNorm must be finite and positive") }
        if let chosen = request.layers, chosen.isEmpty || chosen.contains(where: { !(0 ..< layers).contains($0) }) {
            throw refused("layers must be a non-empty subset of 0..<\(layers)")
        }
    }

    static func hiddenSize(of model: Module) -> Int {
        for (path, module) in model.namedModules() {
            if path.hasSuffix("embed_tokens"), let embedding = module as? Embedding {
                return embedding.weight.dim(-1)
            }
        }
        return 64
    }

    static func trainJepa(
        _ request: LeJEPATrainRequest,
        dataset: GoodViewDataset,
        directory: URL,
        context: ModelContext
    ) throws -> LeJEPATrainReport {
        let hiddenSize = Self.hiddenSize(of: context.model)
        let configuration = try attachLoRA(to: context.model, layers: request.layers, rank: request.rank)
        defer { detachLoRA(from: context.model) }

        let collator = GoodViewCollator(maxSeqLen: request.maxSeqLen, maskRatio: 0.15, maskTokenId: 0, padTokenId: 0)
        let encode: (String) -> [Int] = { text in context.tokenizer.encode(text: text, addSpecialTokens: true) }

        let head = JEPAHead(backbone: context.model, inputDim: hiddenSize, hiddenDim: 2048, outputDim: 512, isCausal: request.isCausal, poolingStrategy: .mean)
        head.train()
        eval(head)

        let optimizer = AdamW(learningRate: request.learningRate, weightDecay: request.weightDecay)
        let warmup = max(1, Int(Float(request.iterations) * request.warmupRatio))
        let decay = max(1, request.iterations - warmup)
        let schedule = joinSchedules(
            [linearSchedule(0, end: request.learningRate, steps: warmup),
             cosineDecay(request.learningRate, decaySteps: decay)],
            boundaries: [warmup])
        optimizer.learningRate = schedule(0)

        let lossFn = LeJEPALoss(baseLambda: request.lambda, referenceSamples: max(1, request.batchSize * GoodViewCollator.totalViewsPerScene), sliceCount: 1024, knotCount: 17, viewsPerScene: GoodViewCollator.totalViewsPerScene)
        let directions = LeJEPALoss.unitDirections(projDim: 512, sliceCount: 1024, seed: 42)
        eval(directions)

        let scenes = dataset.scenes
        let chunkCount = max(1, (scenes.count + request.batchSize - 1) / request.batchSize)

        func batchScenes(for iteration: Int) -> [GoodViewScene] {
            let start = (iteration % chunkCount) * request.batchSize
            let end = min(start + request.batchSize, scenes.count)
            if start >= scenes.count { return Array(scenes.prefix(min(request.batchSize, scenes.count))) }
            return Array(scenes[start ..< end])
        }

        var rng0 = SplitMix64(seed: 0)
        let firstBatch = try collator.collate(scenes: batchScenes(for: 0), encode: encode, rng: &rng0)
        eval(firstBatch.inputIds, firstBatch.attentionMask)
        let embeddingsBefore = head.encode(firstBatch.inputIds, attentionMask: firstBatch.attentionMask)
        let (_, projectedBefore) = head(firstBatch.inputIds, attentionMask: firstBatch.attentionMask)
        eval(embeddingsBefore, projectedBefore)
        let rankBefore = (try? WeightSpectrum.measure(rows: projectedBefore.dim(0), columns: projectedBefore.dim(1)) { projectedBefore[$0] })?.rank ?? 0

        var losses: [Float] = []
        var invarianceLosses: [Float] = []
        var sigRegLosses: [Float] = []
        var gradNorms: [Float] = []

        for step in 0 ..< request.iterations {
            try Task.checkCancellation()
            var rng = SplitMix64(seed: UInt64(step) &+ 42)
            let batch = try collator.collate(scenes: batchScenes(for: step), encode: encode, rng: &rng)
            eval(batch.inputIds, batch.attentionMask)

            let (_, projected) = head(batch.inputIds, attentionMask: batch.attentionMask)
            let out = lossFn(projected, directions: directions)
            eval(out.total, out.invariance, out.sigReg)

            let vg = valueAndGrad(model: head) { model, arrays in
                let (_, proj) = model(arrays[0], attentionMask: arrays[1])
                return [lossFn(proj, directions: arrays[2]).total]
            }
            let (values, grads) = vg(head, [batch.inputIds, batch.attentionMask, directions])
            let totalLoss = values[0]
            let (clipped, gradNorm) = clipGradNorm(gradients: grads, maxNorm: request.maxGradNorm)

            optimizer.learningRate = schedule(step)
            optimizer.update(model: head, gradients: clipped)
            eval(head, optimizer, totalLoss, gradNorm, out.invariance, out.sigReg)

            losses.append(totalLoss.item(Float.self))
            invarianceLosses.append(out.invariance.item(Float.self))
            sigRegLosses.append(out.sigReg.item(Float.self))
            gradNorms.append(min(gradNorm.item(Float.self), request.maxGradNorm))
        }

        let (_, projectedAfter) = head(firstBatch.inputIds, attentionMask: firstBatch.attentionMask)
        eval(projectedAfter)
        let rankAfter = (try? WeightSpectrum.measure(rows: projectedAfter.dim(0), columns: projectedAfter.dim(1)) { projectedAfter[$0] })?.rank ?? 0

        try withTaskCancellationShield {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try JSONEncoder().encode(configuration).write(to: directory.appending(component: "adapter_config.json"))
            let loraParams = Dictionary(uniqueKeysWithValues: context.model.trainableParameters().flattened())
            try MLX.save(arrays: loraParams, url: directory.appending(component: "adapters.safetensors"))
            let projParams = Dictionary(uniqueKeysWithValues: head.projector.parameters().flattened())
            try MLX.save(arrays: projParams, url: directory.appending(component: "jepa_projector.safetensors"))
        }

        return LeJEPATrainReport(
            losses: losses,
            invarianceLosses: invarianceLosses,
            sigRegLosses: sigRegLosses,
            gradNorms: gradNorms,
            embeddingRankBefore: rankBefore,
            embeddingRankAfter: rankAfter,
            adapterDirectory: directory.path,
            projectorPath: directory.appending(component: "jepa_projector.safetensors").path
        )
    }
}
