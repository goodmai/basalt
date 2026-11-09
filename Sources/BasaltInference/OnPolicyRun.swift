import Foundation
import MLX
import MLXLMCommon
import MLXOptimizers
import BasaltCore

struct OnPolicyRun: Sendable {
    let request: DistillRequest
    let plan: OnPolicyPlan
    let teacher: TeacherClient
    let container: ModelContainer

    func train(saving directory: URL) async throws(BasaltError) -> OnPolicyOutcome {
        var state = OnPolicyState(optimizer: Adam(learningRate: request.learningRate))
        let heldOutRows = try await adapted(&state) { context, _, _ in try sample(plan.heldOut, context) }
        let heldOut = try await scored(plan.heldOut, heldOutRows)
        let before = try await adapted(&state) { context, _, _ in MLXInferenceEngine.meanKL(heldOut, model: context.model) }
        let size = min(request.batchSize, plan.train.count)
        var losses: [Float] = []
        for step in 0 ..< request.iterations {
            let prompts = (0 ..< size).map { plan.train[(step * size + $0) % plan.train.count] }
            let rows = try await adapted(&state) { context, _, _ in try sample(prompts, context) }
            let batch = try await scored(prompts, rows)
            losses.append(try await adapted(&state) { context, optimizer, _ in try MLXInferenceEngine.onPolicyStep(batch, optimizer: optimizer, model: context.model) })
        }
        let after = try await adapted(&state) { context, _, configuration in
            try MLXInferenceEngine.saveAdapter(configuration, model: context.model, to: directory)
            return MLXInferenceEngine.meanKL(heldOut, model: context.model)
        }
        return OnPolicyOutcome(losses: losses, before: before, after: after)
    }

    func adapted<R: Sendable>(
        _ state: inout sending OnPolicyState, _ body: @Sendable (ModelContext, Adam, LoRAConfiguration) throws -> R
    ) async throws(BasaltError) -> R {
        let (layers, rank) = (request.layers, request.rank)
        let (next, outcome) = await container.performEvaluating(nonSendable: state) { context, state in
            var state = state
            let outcome = Result { () throws(BasaltError) -> R in
                do {
                    return try withError { try Self.withLoRA(&state, layers: layers, rank: rank, context: context, body) }
                } catch {
                    throw MLXInferenceEngine.labFailure(error, "on-policy distillation")
                }
            }
            return (state, outcome)
        }
        state = next
        return try outcome.get()
    }

    static func withLoRA<R>(
        _ state: inout OnPolicyState, layers: [Int]?, rank: Int, context: ModelContext, _ body: (ModelContext, Adam, LoRAConfiguration) throws -> R
    ) throws -> R {
        let configuration = try MLXInferenceEngine.attachLoRA(to: context.model, layers: layers, rank: rank)
        defer { MLXInferenceEngine.detachLoRA(from: context.model) }
        if state.weights.isEmpty {
            state.weights = context.model.trainableParameters().flattened()
        } else {
            context.model.update(parameters: .unflattened(state.weights))
        }
        let result = try body(context, state.optimizer, configuration)
        state.weights = context.model.trainableParameters().flattened()
        eval(state.weights.map(\.1))
        return result
    }

    func sample(_ prompts: [OnPolicyPrompt], _ context: ModelContext) throws -> [DistillRow] {
        let stops = MLXInferenceEngine.stopTokenIds(context)
        return try prompts.map { prompt in
            try Task.checkCancellation()
            let parameters = GenerateParameters(maxTokens: min(request.teacher.maxTokens, request.maxSeqLen - prompt.ids.count), temperature: request.teacher.temperature)
            var iterator = try TokenIterator(input: LMInput(tokens: MLXArray(prompt.ids.map(Int32.init))), model: context.model, parameters: parameters)
            var completion: [Int] = []
            while let token = iterator.next() {
                completion.append(token)
                if stops.contains(token) { break }
            }
            return DistillRow(tokens: prompt.ids + completion, promptCount: prompt.ids.count)
        }
    }

    func scored(_ prompts: [OnPolicyPrompt], _ rows: [DistillRow]) async throws(BasaltError) -> [ScoredRow] {
        let pairs = Array(zip(prompts, rows))
        var scored: [ScoredRow] = []
        for start in stride(from: 0, to: pairs.count, by: request.teacher.concurrency) {
            let wave = Array(pairs[start ..< min(start + request.teacher.concurrency, pairs.count)].enumerated())
            do {
                scored += try await withThrowingTaskGroup(of: (Int, ScoredRow).self) { group in
                    for (index, (prompt, row)) in wave { group.addTask { (index, try await score(prompt, row)) } }
                    return try await group.reduce(into: []) { $0.append($1) }.sorted { $0.0 < $1.0 }.map(\.1)
                }
            } catch let error as BasaltError {
                throw error
            } catch {
                throw .inferenceError("distill: \(MLXInferenceEngine.briefly(error))")
            }
        }
        return scored
    }

    func score(_ prompt: OnPolicyPrompt, _ row: DistillRow) async throws(BasaltError) -> ScoredRow {
        let completion = Array(row.tokens.dropFirst(row.promptCount))
        let reply = try await teacher.score(ScoreRequest(messages: [["role": "user", "content": prompt.text]], completion: completion, topLogprobs: MLXInferenceEngine.onPolicyTopK))
        return ScoredRow(row: row, teacher: try MLXInferenceEngine.checked(reply, completion: completion.count, plan: plan))
    }
}
