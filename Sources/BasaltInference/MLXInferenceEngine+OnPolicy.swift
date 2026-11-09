import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
import BasaltCore

struct OnPolicyPrompt: Sendable, Equatable {
    var text: String
    var ids: [Int]
}

struct OnPolicyPlan: Sendable {
    var train: [OnPolicyPrompt]
    var heldOut: [OnPolicyPrompt]
    var tokenizer: String
    var vocabulary: Int
    var tooLong: Int
}

struct ScoredRow: Sendable {
    var row: DistillRow
    var teacher: [TopLogprobs]
}

struct OnPolicyOutcome: Sendable {
    var losses: [Float]
    var before: Float
    var after: Float
}

extension MLXInferenceEngine {
    static let onPolicyTopK = 20
    static let tailFloor: Float = 1e-6

    func onPolicyDistill(_ request: DistillRequest, examples: [DistillExample], teacher: TeacherClient, directory: URL, steps: AdapterSteps) async throws(BasaltError) -> DistillReport {
        guard request.teacher.provider == .openai else {
            throw .invalidRequestStructure(details: "distill: onPolicy needs a basalt teacher that serves /v1/score, so the provider must be openai")
        }
        guard Self.scoreTokens.contains(request.teacher.maxTokens) else {
            throw .invalidRequestStructure(details: "distill: in onPolicy mode maxTeacherTokens caps the student's samples, which the teacher scores up to \(Self.scoreTokens.upperBound) tokens")
        }
        let heldOutStart = examples.count - DistillCore.heldOutCount(examples.count, fraction: request.heldOutFraction)
        let maxSeqLen = request.maxSeqLen
        let plan = try await steps.read { context in try Self.onPolicyPlan(examples, heldOutStart: heldOutStart, maxSeqLen: maxSeqLen, context) }
        var report = DistillReport()
        (report.prompts, report.kept, report.heldOut) = (examples.count, plan.train.count, plan.heldOut.count)
        if plan.tooLong > 0 { report.rejected["too_long"] = plan.tooLong }
        guard !plan.train.isEmpty, !plan.heldOut.isEmpty else {
            throw .invalidRequestStructure(details: "distill: no training or no held-out prompt fits in maxSeqLen \(maxSeqLen) tokens")
        }
        let outcome = try await OnPolicyRun(request: request, plan: plan, teacher: teacher, steps: steps).train(saving: directory)
        (report.losses, report.reverseKLBefore, report.reverseKLAfter) = (Self.reported(outcome.losses), outcome.before, outcome.after)
        report.teacherCalls = request.iterations * min(request.batchSize, plan.train.count) + plan.heldOut.count
        report.adapterDirectory = directory.path
        report.verdict = try await applyDistilled(directory, gate: request.gate, steps: steps)
        return report
    }

    static func onPolicyPlan(_ examples: [DistillExample], heldOutStart: Int, maxSeqLen: Int, _ context: ModelContext) throws -> OnPolicyPlan {
        let prompts = try examples.map { OnPolicyPrompt(text: $0.prompt, ids: try labPrompt($0.prompt, context)) }
        let fits = prompts.map { $0.ids.count < maxSeqLen }
        func kept(_ range: Range<Int>) -> [OnPolicyPrompt] { range.filter { fits[$0] }.map { prompts[$0] } }
        return OnPolicyPlan(
            train: kept(0 ..< heldOutStart), heldOut: kept(heldOutStart ..< prompts.count), tokenizer: tokenizerFingerprint(context.tokenizer),
            vocabulary: vocabularySize(context.model), tooLong: fits.filter { !$0 }.count)
    }

    static func checked(_ reply: ScoreReport, completion: Int, plan: OnPolicyPlan) throws(BasaltError) -> [TopLogprobs] {
        guard reply.tokenizer == plan.tokenizer else {
            throw .invalidRequestStructure(details: "distill: the teacher's tokenizer \(reply.tokenizer.prefix(32)) is not the student's \(plan.tokenizer); on-policy distillation sends the student's token ids, so both need the same tokenizer")
        }
        let vocabulary = 0 ..< plan.vocabulary
        guard reply.positions.count == completion, reply.positions.allSatisfy({
            $0.ids.count == onPolicyTopK && $0.logprobs.count == onPolicyTopK && $0.ids.allSatisfy(vocabulary.contains) && $0.logprobs.allSatisfy(\.isFinite)
        }) else {
            throw .invalidRequestStructure(details: "distill: the teacher scored \(reply.positions.count) of \(completion) positions, not \(onPolicyTopK) finite logprobs each within the student's vocabulary of \(plan.vocabulary)")
        }
        return reply.positions
    }

    static func reverseKL(_ logits: MLXArray, teacherIds: MLXArray, teacherLogprobs: MLXArray, mask: MLXArray) -> MLXArray {
        let top = takeAlong(logits - logSumExp(logits, axis: -1, keepDims: true), teacherIds, axis: -1)
        let studentTail = maximum(1 - exp(top).sum(axis: -1), tailFloor)
        let teacherTail = maximum(1 - exp(teacherLogprobs).sum(axis: -1), tailFloor)
        let kl = (exp(top) * (top - teacherLogprobs)).sum(axis: -1) + studentTail * (log(studentTail) - log(teacherTail))
        return (kl * mask).sum() / maximum(mask.sum(), 1)
    }

    static func teacherTensors(_ batch: [ScoredRow], width: Int) -> (ids: MLXArray, logprobs: MLXArray) {
        let k = onPolicyTopK
        var ids = [Int32](repeating: 0, count: batch.count * width * k), logprobs = [Float](repeating: 0, count: batch.count * width * k)
        for (row, scored) in batch.enumerated() {
            for (offset, top) in scored.teacher.enumerated() {
                let start = (row * width + scored.row.promptCount - 1 + offset) * k
                ids.replaceSubrange(start ..< start + k, with: top.ids.map(Int32.init))
                logprobs.replaceSubrange(start ..< start + k, with: top.logprobs)
            }
        }
        return (MLXArray(ids).reshaped([batch.count, width, k]), MLXArray(logprobs).reshaped([batch.count, width, k]))
    }

    static func onPolicyLoss(_ batch: [ScoredRow], model: any LanguageModel) -> MLXArray {
        let (inputs, _, mask) = distillBatch(batch.map(\.row))
        let (ids, logprobs) = teacherTensors(batch, width: inputs.dim(1))
        return reverseKL(model(inputs, cache: nil).asType(.float32), teacherIds: ids, teacherLogprobs: logprobs, mask: mask)
    }

    static func meanKL(_ batch: [ScoredRow], model: any LanguageModel) -> Float {
        batch.reduce(Float(0)) { $0 + onPolicyLoss([$1], model: model).item(Float.self) } / Float(batch.count)
    }

    static func onPolicyStep(_ batch: [ScoredRow], optimizer: Adam, model: any LanguageModel) throws -> Float {
        try Task.checkCancellation()
        let step = valueAndGrad(model: model as Module) { _, _ in [onPolicyLoss(batch, model: model)] }
        let (values, gradients) = step(model as Module, [])
        optimizer.update(model: model as Module, gradients: gradients)
        eval(model as Module, optimizer, values)
        let loss = values[0].item(Float.self)
        guard loss.isFinite else { throw BasaltError.inferenceError("distill: the reverse KL became non-finite; nothing was saved") }
        return loss
    }
}
