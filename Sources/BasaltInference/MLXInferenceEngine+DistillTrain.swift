import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import MLXOptimizers
import BasaltCore

extension MLXInferenceEngine {
    static func distillAdapter(
        _ train: [TrainingTask], heldOut: [TrainingTask], request: DistillRequest, directory: URL, context: ModelContext
    ) throws -> DistillRun {
        let all = try (train + heldOut).map { try distillRow($0, context) }
        let fits = all.map { $0.tokens.count <= request.maxSeqLen }
        let trainRows = zip(all.prefix(train.count), fits.prefix(train.count)).filter(\.1).map(\.0)
        let heldOutRows = zip(all.suffix(heldOut.count), fits.suffix(heldOut.count)).filter(\.1).map(\.0)
        let tooLong = fits.filter { !$0 }.count
        guard !trainRows.isEmpty, !heldOutRows.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "distill: every training or held-out example is longer than maxSeqLen \(request.maxSeqLen) tokens")
        }
        let before = nll(heldOutRows, context.model)
        let configuration = try attachLoRA(to: context.model, layers: request.layers, rank: request.rank)
        defer { detachLoRA(from: context.model) }
        let losses = try distillSteps(trainRows, request: request, model: context.model)
        let after = nll(heldOutRows, context.model)
        try withTaskCancellationShield {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try JSONEncoder().encode(configuration).write(to: directory.appending(component: "adapter_config.json"))
            try LoRATrain.saveLoRAWeights(model: context.model, url: directory.appending(component: "adapters.safetensors"))
        }
        return DistillRun(losses: reported(losses), nllBefore: before, nllAfter: after, tooLong: tooLong)
    }

    static func distillRow(_ pair: TrainingTask, _ context: ModelContext) throws -> DistillRow {
        let prompt = try labPrompt(pair.prompt, context)
        let completion = context.tokenizer.encode(text: pair.answer, addSpecialTokens: false) + [context.tokenizer.eosTokenId].compactMap { $0 }
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

    static func distillSteps(_ rows: [DistillRow], request: DistillRequest, model: any LanguageModel) throws -> [Float] {
        let optimizer = Adam(learningRate: request.learningRate)
        let step = valueAndGrad(model: model as Module) { _, arrays in [maskedLoss(model, arrays[0], arrays[1], arrays[2])] }
        let size = min(request.batchSize, rows.count)
        var losses: [Float] = []
        for index in 0 ..< request.iterations {
            try Task.checkCancellation()
            let (inputs, targets, mask) = distillBatch((0 ..< size).map { rows[(index * size + $0) % rows.count] })
            let (values, gradients) = step(model as Module, [inputs, targets, mask])
            optimizer.update(model: model as Module, gradients: gradients)
            eval(model as Module, optimizer, values)
            let loss = values[0].item(Float.self)
            guard loss.isFinite else { throw BasaltError.inferenceError("distill: the loss became non-finite at step \(index); nothing was saved") }
            losses.append(loss)
        }
        return losses
    }
}
