import Foundation
import MLX
import MLXLMCommon
import MLXOptimizers
import BasaltCore

struct OnPolicyRun: Sendable {
    let request: DistillRequest
    let plan: OnPolicyPlan
    let teacher: TeacherClient
    let steps: AdapterSteps

    func train(saving directory: URL) async throws(BasaltError) -> OnPolicyOutcome {
        var state = AdapterState(optimizer: Adam(learningRate: request.learningRate))
        let heldOutRows = try await sampled(plan.heldOut, round: 0, &state)
        let heldOut = try await scored(plan.heldOut, heldOutRows)
        let before = try await steps.trainedAdapter(&state) { context, _, _ in MLXInferenceEngine.meanKL(heldOut, model: context.model) }
        let size = min(request.batchSize, plan.train.count)
        var losses: [Float] = []
        for step in 0 ..< request.iterations {
            let prompts = (0 ..< size).map { plan.train[(step * size + $0) % plan.train.count] }
            let batch = try await scored(prompts, try await sampled(prompts, round: step + 1, &state))
            losses.append(try await steps.trainedAdapter(&state) { context, optimizer, _ in try MLXInferenceEngine.onPolicyStep(batch, optimizer: optimizer, model: context.model) })
        }
        let after = try await steps.trainedAdapter(&state) { context, _, configuration in
            try MLXInferenceEngine.saveConfiguredAdapter(configuration, model: context.model, to: directory)
            return MLXInferenceEngine.meanKL(heldOut, model: context.model)
        }
        return OnPolicyOutcome(losses: losses, before: before, after: after)
    }

    func sampled(_ prompts: [OnPolicyPrompt], round: Int, _ state: inout sending AdapterState) async throws(BasaltError) -> [DistillRow] {
        var rows: [DistillRow] = []
        for (index, prompt) in prompts.enumerated() {
            let seed = Self.rolloutSeed(prompt.ids, round: round, slot: index)
            rows.append(try await steps.trainedAdapter(&state) { context, _, _ in try sample(prompt, seed: seed, context) })
        }
        return rows
    }

    static func rolloutSeed(_ ids: [Int], round: Int, slot: Int) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for value in ids + [round, slot] {
            hash = (hash ^ UInt64(truncatingIfNeeded: value)) &* 0x0000_0100_0000_01b3
        }
        return hash
    }

    func sample(_ prompt: OnPolicyPrompt, seed: UInt64, _ context: ModelContext) throws -> DistillRow {
        try Task.checkCancellation()
        let stops = MLXInferenceEngine.stopTokenIds(context)
        let parameters = GenerateParameters(maxTokens: min(request.teacher.maxTokens, request.maxSeqLen - prompt.ids.count), temperature: request.teacher.temperature)
        let completion = try withRandomState(MLXRandom.RandomState(seed: seed)) {
            var iterator = try TokenIterator(input: LMInput(tokens: MLXArray(prompt.ids.map(Int32.init))), model: context.model, parameters: parameters)
            var tokens: [Int] = []
            while let token = iterator.next() {
                tokens.append(token)
                if stops.contains(token) { break }
            }
            return tokens
        }
        return DistillRow(tokens: prompt.ids + completion, promptCount: prompt.ids.count)
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
