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
        let final = try await steps(try await heldOutKL(OnPolicyState(optimizer: Adam(learningRate: request.learningRate))), from: 0, saving: directory)
        return OnPolicyOutcome(losses: final.losses, before: final.heldOutKL.first ?? 0, after: final.heldOutKL.last ?? 0)
    }

    func steps(_ state: sending OnPolicyState, from step: Int, saving directory: URL) async throws(BasaltError) -> sending OnPolicyState {
        let size = min(request.batchSize, plan.train.count)
        let prompts = (0 ..< size).map { plan.train[(step * size + $0) % plan.train.count] }
        let (sampled, rows) = try await adapted(state) { context, _, _ in try sample(prompts, context) }
        let batch = try await scored(prompts, rows)
        var (trained, loss) = try await adapted(sampled) { context, optimizer, _ in try MLXInferenceEngine.onPolicyStep(batch, optimizer: optimizer, model: context.model) }
        trained.losses.append(loss)
        return step + 1 < request.iterations ? try await steps(trained, from: step + 1, saving: directory) : try await heldOutKL(trained, saving: directory)
    }

    func heldOutKL(_ state: sending OnPolicyState, saving directory: URL? = nil) async throws(BasaltError) -> sending OnPolicyState {
        let (state, rows) = try await adapted(state) { context, _, _ in try sample(plan.heldOut, context) }
        let batch = try await scored(plan.heldOut, rows)
        var (measured, kl) = try await adapted(state) { context, _, configuration in
            let kl = batch.reduce(Float(0)) { $0 + MLXInferenceEngine.onPolicyLoss([$1], model: context.model).item(Float.self) } / Float(batch.count)
            if let directory { try MLXInferenceEngine.saveAdapter(configuration, model: context.model, to: directory) }
            return kl
        }
        measured.heldOutKL.append(kl)
        return measured
    }

    func adapted<R: Sendable>(
        _ state: sending OnPolicyState, _ body: @Sendable (ModelContext, Adam, LoRAConfiguration) throws -> R
    ) async throws(BasaltError) -> sending (OnPolicyState, R) {
        let (layers, rank) = (request.layers, request.rank)
        return try await MLXInferenceEngine.labRun(container, "on-policy distillation", nonSendable: state) { context, state in
            var adapter = state
            let configuration = try MLXInferenceEngine.attachLoRA(to: context.model, layers: layers, rank: rank)
            defer { MLXInferenceEngine.detachLoRA(from: context.model) }
            if adapter.weights.isEmpty {
                adapter.weights = context.model.trainableParameters().flattened()
            } else {
                context.model.update(parameters: .unflattened(adapter.weights))
            }
            let result = try body(context, adapter.optimizer, configuration)
            adapter.weights = context.model.trainableParameters().flattened()
            eval(adapter.weights.map(\.1))
            return (adapter, result)
        }
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
