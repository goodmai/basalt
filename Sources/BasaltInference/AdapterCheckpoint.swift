import CryptoKit
import Foundation
import MLX
import MLXOptimizers
import BasaltCore

struct AdapterProgress: Sendable, Codable, Equatable {
    var step = 0
    var losses: [Float] = []
    var model: String
    var data: String
    var layers: [Int]
    var layerCount: Int
    var rank: Int
    var batchSize: Int
    var learningRate: Float

    init(_ request: DistillRequest, model: String, data: String, layerCount: Int) {
        (self.model, self.data, self.layerCount) = (model, data, layerCount)
        layers = request.layers ?? Array(0 ..< layerCount)
        (rank, batchSize, learningRate) = (request.rank, request.batchSize, request.learningRate)
    }

    static func digest(_ rows: [DistillRow]) -> String {
        var hash = SHA256()
        for row in rows {
            hash.update(data: Data("\(row.promptCount):\(row.tokens.map(String.init).joined(separator: ","))\n".utf8))
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
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

    func resumable(as identity: AdapterProgress, iterations: Int) -> String? {
        guard model == identity.model else { return "the checkpoint was trained on another model (\(model.prefix(40))), not \(identity.model.prefix(40))" }
        guard data == identity.data else { return "the training rows differ from the checkpoint's: prompts, teacher answers, filter, held-out split, maxSeqLen, samples or reasoning changed" }
        var same = identity
        (same.step, same.losses) = (step, losses)
        guard same == self else {
            return "the checkpoint was trained with rank \(rank), layers \(layers) of \(layerCount), batch size \(batchSize) and learning rate \(learningRate); resume with the same settings"
        }
        guard step < iterations else { return "the checkpoint is at step \(step); iterations must be larger to resume" }
        return nil
    }
}

enum AdapterCheckpoint {
    static func directory(in adapter: URL) -> URL { adapter.appending(component: "checkpoint") }
    static func staging(in adapter: URL) -> URL { adapter.appending(component: "checkpoint.next") }

    static func write(_ state: AdapterState, _ progress: AdapterProgress, to adapter: URL) throws {
        try withTaskCancellationShield {
            let files = FileManager.default
            let next = staging(in: adapter)
            try? files.removeItem(at: next)
            try files.createDirectory(at: next, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
            var written = [next.appending(component: "adapter.safetensors"), next.appending(component: "progress.json")]
            try MLX.save(arrays: Dictionary(uniqueKeysWithValues: state.weights), url: written[0])
            try JSONEncoder().encode(progress).write(to: written[1])
            let optimizer = Dictionary(uniqueKeysWithValues: state.optimizer.state().flattened())
            if !optimizer.isEmpty {
                written.append(next.appending(component: "optimizer.safetensors"))
                try MLX.save(arrays: optimizer, url: written[2])
            }
            try (written + [next]).forEach(fullSync)
            let current = directory(in: adapter)
            if files.fileExists(atPath: current.path) {
                _ = try files.replaceItemAt(current, withItemAt: next)
            } else {
                try files.moveItem(at: next, to: current)
            }
        }
    }

    static func fullSync(_ url: URL) throws {
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw CocoaError(.fileReadNoPermission, userInfo: [NSFilePathErrorKey: url.path]) }
        defer { close(descriptor) }
        guard fcntl(descriptor, F_FULLFSYNC) != -1 else { throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path]) }
    }

    static func progress(in adapter: URL, as identity: AdapterProgress, iterations: Int) throws(BasaltError) -> AdapterProgress {
        let progress = try saved(in: adapter)
        if let refusal = progress.resumable(as: identity, iterations: iterations) { throw .invalidRequestStructure(details: "distill: \(refusal)") }
        return progress
    }

    static func saved(in adapter: URL) throws(BasaltError) -> AdapterProgress {
        let file = directory(in: adapter).appending(component: "progress.json")
        guard let data = try? Data(contentsOf: file), let progress = try? JSONDecoder().decode(AdapterProgress.self, from: data) else {
            throw .invalidRequestStructure(details: "distill: resume needs the checkpoint of an interrupted run in output; \(file.path) is missing or unreadable")
        }
        return progress
    }

    static func restore(_ state: inout AdapterState, from adapter: URL, trainable: [String: [Int]]) throws {
        let directory = directory(in: adapter)
        let weights = try MLX.loadArrays(url: directory.appending(component: "adapter.safetensors"))
        let optimizerFile = directory.appending(component: "optimizer.safetensors")
        let optimizer = FileManager.default.fileExists(atPath: optimizerFile.path) ? try MLX.loadArrays(url: optimizerFile) : [:]
        if let problem = mismatch(weights: weights.mapValues(\.shape), optimizer: optimizer.mapValues(\.shape), trainable: trainable) {
            throw BasaltError.invalidRequestStructure(details: "distill: the checkpoint in \(directory.path) does not fit this adapter: \(problem)")
        }
        state.weights = weights.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
        if !optimizer.isEmpty { state.optimizer.update(parameters: .unflattened(optimizer)) }
        eval(state.weights.map(\.1), state.optimizer)
    }

    static func mismatch(weights: [String: [Int]], optimizer: [String: [Int]], trainable: [String: [Int]]) -> String? {
        guard weights == trainable else {
            let foreign = Set(weights.keys).subtracting(trainable.keys).sorted().prefix(3)
            return foreign.isEmpty ? "its LoRA weights miss keys or have other shapes" : "it holds keys that are not LoRA weights of this adapter: \(foreign.joined(separator: ", "))"
        }
        guard !optimizer.isEmpty else { return nil }
        let moments = Dictionary(uniqueKeysWithValues: trainable.flatMap { key, shape in [("\(key).m", shape), ("\(key).v", shape)] })
        let steps = optimizer.filter { $0.key.hasSuffix(".step") }
        guard optimizer.filter({ !$0.key.hasSuffix(".step") }) == moments, Set(steps.keys) == Set(trainable.keys.map { "\($0).step" }),
              steps.values.allSatisfy({ $0.reduce(1, *) == 1 }) else {
            return "its optimizer state is not Adam's m, v and step for each LoRA weight"
        }
        return nil
    }

    static func discard(in adapter: URL) {
        try? FileManager.default.removeItem(at: directory(in: adapter))
        try? FileManager.default.removeItem(at: staging(in: adapter))
    }
}

extension MLXInferenceEngine {
    func resumedDirectory(_ output: String?) throws(BasaltError) -> URL {
        guard let output else {
            throw .invalidRequestStructure(details: "distill: resume continues the run in output; give the adapter directory of the interrupted run")
        }
        let directory = URL(fileURLWithPath: (output as NSString).expandingTildeInPath)
        try refuseInsideModel(directory)
        _ = try AdapterCheckpoint.saved(in: directory)
        return directory
    }

    func reservedDirectory(_ output: String?) throws(BasaltError) -> URL {
        let directory = try adapterDirectory(output)
        try Self.reserve(directory)
        return directory
    }

    static func reserve(_ directory: URL) throws(BasaltError) {
        do {
            try FileManager.default.createDirectory(at: directory.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        } catch {
            throw .invalidRequestStructure(details: "distill: \(directory.path) could not be reserved (\(error.localizedDescription)); the adapter goes to a new directory, nothing is overwritten")
        }
    }

    static func releaseIfEmpty(_ directory: URL) {
        guard (try? FileManager.default.contentsOfDirectory(atPath: directory.path))?.isEmpty == true else { return }
        try? FileManager.default.removeItem(at: directory)
    }
}
