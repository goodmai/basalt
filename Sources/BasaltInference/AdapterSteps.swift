import Foundation
import MLX
import MLXLMCommon
import MLXOptimizers
import BasaltCore

struct AdapterState {
    var weights: [(String, MLXArray)] = []
    let optimizer: Adam
}

struct AdapterSteps: Sendable {
    let engine: MLXInferenceEngine
    let container: ModelContainer
    let epoch: ModelEpoch
    let fingerprint: String
    let layers: [Int]?
    let rank: Int
    let lease: StepLease?
    let what: String

    func enter() async throws(BasaltError) -> Int? {
        let ticket = try await lease?.enter()
        guard await engine.trainsOn(epoch) else {
            await leave(ticket)
            throw .inferenceError("the model was switched, or changed by an overlay or weights delta, during \(what); training stopped and the last checkpoint is kept")
        }
        return ticket
    }

    func leave(_ ticket: Int?) async {
        guard let ticket, let lease else { return }
        await lease.leave(ticket)
    }

    func read<R: Sendable>(_ body: @Sendable (ModelContext) throws -> R) async throws(BasaltError) -> R {
        let ticket = try await enter()
        defer { await leave(ticket) }
        return try await MLXInferenceEngine.labRun(container, what, body)
    }

    func section<R: Sendable>(
        _ state: inout sending AdapterState, _ body: @Sendable (ModelContext, inout AdapterState) throws -> R
    ) async throws(BasaltError) -> R {
        let ticket = try await enter()
        defer { await leave(ticket) }
        let what = what
        let (next, outcome) = await container.performEvaluating(nonSendable: state) { context, state in
            var state = state
            let outcome = Result { () throws(BasaltError) -> R in
                do {
                    return try withError { try body(context, &state) }
                } catch {
                    throw MLXInferenceEngine.labFailure(error, what)
                }
            }
            return (state, outcome)
        }
        state = next
        return try outcome.get()
    }

    func trained<R: Sendable>(
        _ state: inout sending AdapterState, _ body: @Sendable (ModelContext, Adam, LoRAConfiguration) throws -> R
    ) async throws(BasaltError) -> R {
        try await section(&state) { context, state in try withLoRA(&state, context: context, body) }
    }

    func withLoRA<R>(_ state: inout AdapterState, context: ModelContext, _ body: (ModelContext, Adam, LoRAConfiguration) throws -> R) throws -> R {
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

    func restore(_ state: inout AdapterState, from adapter: URL, context: ModelContext) throws {
        let trainable = try withLoRA(&state, context: context) { context, _, _ in
            Dictionary(uniqueKeysWithValues: context.model.trainableParameters().flattened().map { ($0.0, $0.1.shape) })
        }
        try AdapterCheckpoint.restore(&state, from: adapter, trainable: trainable)
    }
}

extension MLXInferenceEngine {
    func adapterSteps(layers: [Int]?, rank: Int, lease: StepLease?, what: String) throws(BasaltError) -> AdapterSteps {
        guard let container else { throw .modelNotLoaded }
        guard !modelModified else {
            throw .invalidRequestStructure(details: "\(what): an overlay or weights delta changes the live model; undo it before training on the base weights")
        }
        return AdapterSteps(engine: self, container: container, epoch: modelEpoch, fingerprint: modelFingerprint ?? modelDescription, layers: layers, rank: rank, lease: lease, what: what)
    }

    func trainsOn(_ epoch: ModelEpoch) -> Bool { modelEpoch == epoch && !modelModified }
}
