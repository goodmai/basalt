import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import MLXOptimizers
import BasaltCore

extension MLXInferenceEngine {
    static func distillAdapter(
        _ train: [TrainingTask], heldOut: [TrainingTask], request: DistillRequest, directory: URL, steps: AdapterSteps
    ) async throws(BasaltError) -> DistillRun {
        let (rows, layerCount) = try await steps.read { context in
            (try distillRows(train, heldOut: heldOut, maxSeqLen: request.maxSeqLen, context), loraBlocks(of: context.model).count)
        }
        let identity = AdapterProgress(request, model: steps.fingerprint, data: AdapterProgress.digest(rows.train + rows.heldOut), layerCount: layerCount)
        var state = AdapterState(optimizer: Adam(learningRate: request.learningRate))
        var progress = try await started(&state, identity, request: request, directory: directory, steps: steps)
        let resumedAt = request.resume ? progress.step : nil
        var before: Float = 0
        for row in rows.heldOut { before += try await steps.read { context in nll([row], context.model) } }
        while progress.step < request.iterations {
            progress = try await steps.section(&state) { [progress, rows] context, state in
                let next = progress.advanced(by: try steps.withLoRA(&state, context: context) { context, optimizer, _ in
                    try distillStep(rows.train, index: progress.step, batchSize: request.batchSize, optimizer: optimizer, model: context.model)
                })
                if next.checkpointDue(every: request.checkpointEvery, of: request.iterations) { try AdapterCheckpoint.write(state, next, to: directory) }
                return next
            }
        }
        var after: Float = 0
        for row in rows.heldOut { after += try await steps.trained(&state) { context, _, _ in nll([row], context.model) } }
        try await steps.trained(&state) { context, _, configuration in try saveAdapter(configuration, model: context.model, to: directory) }
        AdapterCheckpoint.discard(in: directory)
        let count = Float(rows.heldOut.count)
        return DistillRun(losses: reported(progress.losses), nllBefore: before / count, nllAfter: after / count, tooLong: rows.tooLong, resumedAt: resumedAt)
    }

    static func started(
        _ state: inout sending AdapterState, _ identity: AdapterProgress, request: DistillRequest, directory: URL, steps: AdapterSteps
    ) async throws(BasaltError) -> AdapterProgress {
        if request.resume {
            let progress = try AdapterCheckpoint.progress(in: directory, as: identity, iterations: request.iterations)
            try await steps.section(&state) { context, state in try steps.restore(&state, from: directory, context: context) }
            return progress
        }
        if request.checkpointEvery > 0 {
            try await steps.section(&state) { context, state in
                try steps.withLoRA(&state, context: context) { _, _, _ in }
                try AdapterCheckpoint.write(state, identity, to: directory)
            }
        }
        return identity
    }

    static func distillRows(
        _ train: [TrainingTask], heldOut: [TrainingTask], maxSeqLen: Int, _ context: ModelContext
    ) throws -> (train: [DistillRow], heldOut: [DistillRow], tooLong: Int) {
        let all = try (train + heldOut).map { try distillRow($0, context) }
        let fits = all.map { $0.tokens.count <= maxSeqLen }
        let trainRows = zip(all.prefix(train.count), fits.prefix(train.count)).filter(\.1).map(\.0)
        let heldOutRows = zip(all.suffix(heldOut.count), fits.suffix(heldOut.count)).filter(\.1).map(\.0)
        guard !trainRows.isEmpty, !heldOutRows.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "distill: every training or held-out example is longer than maxSeqLen \(maxSeqLen) tokens")
        }
        return (trainRows, heldOutRows, fits.filter { !$0 }.count)
    }

    static func saveAdapter(_ configuration: LoRAConfiguration, model: Module, to directory: URL) throws {
        try withTaskCancellationShield {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try JSONEncoder().encode(configuration).write(to: directory.appending(component: "adapter_config.json"))
            try LoRATrain.saveLoRAWeights(model: model, url: directory.appending(component: "adapters.safetensors"))
        }
    }

    static func distillRow(_ pair: TrainingTask, _ context: ModelContext) throws -> DistillRow {
        let thinking = opensThinking(pair.answer)
        let prompt = try labPrompt(pair.prompt, context, thinking: thinking)
        var answer = Substring(pair.answer)
        if thinking, tailOpensThinking(context.tokenizer.decode(tokenIds: prompt, skipSpecialTokens: false)),
           let opener = thinkingOpeners.first(where: { answer.hasPrefix($0) }) {
            answer = answer.dropFirst(opener.count).drop(while: \.isNewline)
        }
        let completion = context.tokenizer.encode(text: String(answer), addSpecialTokens: false) + [context.tokenizer.eosTokenId].compactMap { $0 }
        return DistillRow(tokens: prompt + completion, promptCount: prompt.count)
    }

    static func distillBatch(_ rows: [DistillRow]) -> (inputs: MLXArray, targets: MLXArray, mask: MLXArray) {
        let width = (rows.map(\.tokens.count).max() ?? 1) - 1
        var inputs = [Int32](), targets = [Int32](), mask = [Float]()
        for row in rows {
            for position in 0 ..< width {
                let inside = position + 1 < row.tokens.count
                inputs.append(Int32(inside ? row.tokens[position] : 0))
                targets.append(Int32(inside ? row.tokens[position + 1] : 0))
                mask.append(inside && position + 1 >= row.promptCount ? 1 : 0)
            }
        }
        let shape = [rows.count, width]
        return (MLXArray(inputs).reshaped(shape), MLXArray(targets).reshaped(shape), MLXArray(mask).reshaped(shape))
    }

    static func maskedLoss(_ model: any LanguageModel, _ inputs: MLXArray, _ targets: MLXArray, _ mask: MLXArray) -> MLXArray {
        let logits = model(inputs, cache: nil).asType(.float32)
        return (crossEntropy(logits: logits, targets: targets, reduction: .none) * mask).sum() / maximum(mask.sum(), 1)
    }

    static func nll(_ rows: [DistillRow], _ model: any LanguageModel) -> Float {
        let total = rows.reduce(Float(0)) { sum, row in
            let (inputs, targets, mask) = distillBatch([row])
            return sum + maskedLoss(model, inputs, targets, mask).item(Float.self)
        }
        return total / Float(rows.count)
    }

    static func distillStep(_ rows: [DistillRow], index: Int, batchSize: Int, optimizer: Adam, model: any LanguageModel) throws -> Float {
        try Task.checkCancellation()
        let step = valueAndGrad(model: model as Module) { _, arrays in [maskedLoss(model, arrays[0], arrays[1], arrays[2])] }
        let size = min(batchSize, rows.count)
        let (inputs, targets, mask) = distillBatch((0 ..< size).map { rows[(index * size + $0) % rows.count] })
        let (values, gradients) = step(model as Module, [inputs, targets, mask])
        optimizer.update(model: model as Module, gradients: gradients)
        eval(model as Module, optimizer, values)
        let loss = values[0].item(Float.self)
        guard loss.isFinite else { throw BasaltError.inferenceError("distill: the loss became non-finite at step \(index); nothing was saved") }
        return loss
    }
}
