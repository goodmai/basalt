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
    let layers: [Int]?
    let rank: Int
    let lease: StepLease?
    let what: String

    func enter() async throws(BasaltError) -> Int? {
        let ticket = try await lease?.enter()
        guard await engine.container === container else {
            await leave(ticket)
            throw .inferenceError("the model changed during \(what); training stopped")
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
}

struct AdapterProgress: Sendable, Codable, Equatable {
    var step = 0
    var losses: [Float] = []
    var rank: Int
    var layers: [Int]?
    var batchSize: Int
    var learningRate: Float

    init(_ request: DistillRequest) {
        (rank, layers, batchSize, learningRate) = (request.rank, request.layers, request.batchSize, request.learningRate)
    }

    func advanced(by loss: Float) -> AdapterProgress {
        var next = self
        next.step += 1
        next.losses.append(loss)
        return next
    }

    func checkpointDue(every: Int, of iterations: Int) -> Bool {
        every > 0 && step % every == 0 && step < iterations
    }

    func resumable(by request: DistillRequest) -> String? {
        var fresh = AdapterProgress(request)
        (fresh.step, fresh.losses) = (step, losses)
        guard fresh == self else { return "the checkpoint was trained with rank \(rank), layers \(layers.map { "\($0)" } ?? "all"), batch size \(batchSize) and learning rate \(learningRate); resume with the same settings" }
        guard step < request.iterations else { return "the checkpoint is at step \(step); iterations must be larger to resume" }
        return nil
    }
}

enum AdapterCheckpoint {
    static func directory(in adapter: URL) -> URL { adapter.appending(component: "checkpoint") }

    static func write(_ state: AdapterState, _ progress: AdapterProgress, to adapter: URL) throws {
        try withTaskCancellationShield {
            let files = FileManager.default
            let next = adapter.appending(component: "checkpoint.next")
            try? files.removeItem(at: next)
            try files.createDirectory(at: next, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try MLX.save(arrays: Dictionary(uniqueKeysWithValues: state.weights), url: next.appending(component: "adapter.safetensors"))
            try MLX.save(arrays: Dictionary(uniqueKeysWithValues: state.optimizer.state().flattened()), url: next.appending(component: "optimizer.safetensors"))
            try JSONEncoder().encode(progress).write(to: next.appending(component: "progress.json"))
            let current = directory(in: adapter)
            if files.fileExists(atPath: current.path) {
                _ = try files.replaceItemAt(current, withItemAt: next)
            } else {
                try files.moveItem(at: next, to: current)
            }
        }
    }

    static func progress(in adapter: URL, for request: DistillRequest) throws(BasaltError) -> AdapterProgress {
        let file = directory(in: adapter).appending(component: "progress.json")
        guard let data = try? Data(contentsOf: file), let progress = try? JSONDecoder().decode(AdapterProgress.self, from: data) else {
            throw .invalidRequestStructure(details: "distill: resume needs the checkpoint of an interrupted run in output; \(file.path) is missing or unreadable")
        }
        if let refusal = progress.resumable(by: request) { throw .invalidRequestStructure(details: "distill: \(refusal)") }
        return progress
    }

    static func restore(_ state: inout AdapterState, from adapter: URL) throws {
        let directory = directory(in: adapter)
        state.weights = try MLX.loadArrays(url: directory.appending(component: "adapter.safetensors")).sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
        state.optimizer.update(parameters: .unflattened(try MLX.loadArrays(url: directory.appending(component: "optimizer.safetensors"))))
        eval(state.weights.map(\.1), state.optimizer)
    }

    static func discard(in adapter: URL) {
        try? FileManager.default.removeItem(at: directory(in: adapter))
    }
}
