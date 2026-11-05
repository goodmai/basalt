import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

struct AppliedDelta: Sendable {
    var info: OverlayInfo?
    var slice: [Range<Int>] = []
    var original: [Float] = []
    var modules: [String] = []
}

extension MLXInferenceEngine {
    static let maxDeltaScale: Float = 10
    static let maxDeltaElements = 1 << 24
    static let writableDeltaTypes: [DType] = [.float16, .bfloat16, .float32]

    var modelModified: Bool { !modelFacts.deltas.isEmpty }

    public var activeDeltas: [OverlayInfo] {
        modelFacts.deltas.values.compactMap(\.info).sorted { $0.id < $1.id }
    }

    func modelModified(since lease: Int) -> Bool { modelModified || lease <= modelFacts.cacheFence }

    public func applyDelta(_ delta: WeightDelta) async throws(BasaltError) -> OverlayInfo {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        guard modelFacts.deltas[delta.tensor] == nil else {
            throw .invalidRequestStructure(details: "weights_delta: \(delta.tensor) already carries an active delta; undo it first")
        }
        record(.learned(modelEpoch) { $0.deltas[delta.tensor] = AppliedDelta() })
        let written = await container.performEvaluating(nonSendable: delta) { context, delta in
            Result { () throws(BasaltError) in try Self.add(delta, to: context.model) }
        }
        var applied: AppliedDelta
        switch written {
        case .success(let value):
            applied = value
        case .failure(let error):
            record(.learned(modelEpoch) { $0.deltas[delta.tensor] = nil })
            throw error
        }
        guard self.container === container else {
            throw .inferenceHardwareFailure(reason: "the model was replaced while the delta was written; nothing is applied")
        }
        let info = register(nil, description: Self.describe(delta))
        applied.info = info
        weightsChanged { [applied] in $0.deltas[delta.tensor] = applied }
        scheduleUndo(of: info)
        return info
    }

    @discardableResult
    func undoDelta(_ id: OverlayID) async throws(BasaltError) -> Bool {
        guard let container, let (tensor, applied) = modelFacts.deltas.first(where: { $0.value.info?.id == id }) else { return false }
        let restored = await container.performEvaluating(nonSendable: applied) { context, applied in
            Result { () throws(BasaltError) in try Self.restore(applied, of: tensor, in: context.model) }
        }
        if case .failure(let error) = restored { throw error }
        guard self.container === container else { return true }
        weightsChanged { $0.deltas[tensor] = nil }
        return true
    }

    func weightsChanged(_ change: @escaping (inout ModelFacts) -> Void) {
        let fence = max(modelFacts.cacheFence, leases.keys.max() ?? 0)
        record(.learned(modelEpoch) {
            change(&$0)
            $0.cacheFence = fence
        })
        dropPromptCaches()
    }

    func scheduleUndo(of info: OverlayInfo) {
        guard let expiresAt = info.expiresAt else { return }
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(max(0, expiresAt.timeIntervalSinceNow)))
            guard let self else { return }
            do {
                try await undoDelta(info.id)
                await remove(info.id)
            } catch {
                BasaltLogger(module: "MLXInferenceEngine").warn("Expired delta \(info.id.rawValue) could not be undone: \(error.localizedDescription)")
            }
        }
    }

    static func describe(_ delta: WeightDelta) -> String {
        let slice = delta.slice.map { "[" + $0.map { "\($0.lowerBound)..<\($0.upperBound)" }.joined(separator: ", ") + "]" } ?? ""
        return "weights_delta \(delta.tensor)\(slice) += \(delta.values?.count ?? 0) values × \(delta.scale)"
    }

    static func add(_ delta: WeightDelta, to model: Module) throws(BasaltError) -> AppliedDelta {
        let tensor = try writableTensor(delta.tensor, in: model)
        let ranges = try deltaRanges(delta, shape: tensor.shape)
        let indices: [any MLXArrayIndex] = ranges
        let slice = tensor[indices].asType(.float32)
        let original = slice.asArray(Float.self)
        let values = delta.values ?? []
        let added = values.count == 1 ? MLXArray(values[0]) : MLXArray(values, slice.shape)
        tensor[indices] = (slice + added * delta.scale).asType(tensor.dtype)
        eval(tensor)
        return AppliedDelta(slice: ranges, original: original)
    }

    static func restore(_ applied: AppliedDelta, of name: String, in model: Module) throws(BasaltError) {
        guard applied.modules.isEmpty else { return try detach(applied.modules, from: model) }
        let tensor = try writableTensor(name, in: model)
        let indices: [any MLXArrayIndex] = applied.slice
        tensor[indices] = MLXArray(applied.original, applied.slice.map(\.count)).asType(tensor.dtype)
        eval(tensor)
    }

    static func writableTensor(_ name: String, in model: Module) throws(BasaltError) -> MLXArray {
        guard let tensor = model.parameters().flattened().first(where: { $0.0 == name })?.1 else {
            throw .invalidRequestStructure(details: "weights_delta: the model has no tensor named \(name); model_tensors lists them")
        }
        let owner = name.split(separator: ".").dropLast().joined(separator: ".")
        let module = model.namedModules().first { $0.0 == owner }?.1
        guard writableDeltaTypes.contains(tensor.dtype), let module, !(module is Quantized) else {
            throw .invalidRequestStructure(details: "weights_delta: \(name) is \(tensor.dtype) in \(module.map { "a \(type(of: $0))" } ?? "no module"); deltas write only float tensors of unquantized layers (norms, biases, unquantized embeddings), a quantized write path is not implemented")
        }
        return tensor
    }

    static func deltaRanges(_ delta: WeightDelta, shape: [Int]) throws(BasaltError) -> [Range<Int>] {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "weights_delta on \(delta.tensor): \(reason)") }
        let slice = delta.slice ?? []
        guard slice.count <= shape.count else { throw refused("\(slice.count) slice ranges for a tensor of \(shape.count) dimensions") }
        let ranges = slice + shape.dropFirst(slice.count).map { 0 ..< $0 }
        guard zip(ranges, shape).allSatisfy({ $0.lowerBound >= 0 && $0.upperBound <= $1 && !$0.isEmpty }) else {
            throw refused("slice \(slice.map { "\($0.lowerBound)..<\($0.upperBound)" }) is empty or outside the shape \(shape)")
        }
        let count = ranges.reduce(1) { $0 * $1.count }
        guard count <= maxDeltaElements else { throw refused("the slice holds \(count) elements, more than \(maxDeltaElements)") }
        guard let values = delta.values, values.count == count || values.count == 1 else {
            throw refused("\(delta.values?.count ?? 0) values for a slice of \(count) elements; give \(count), or 1 to broadcast")
        }
        guard values.allSatisfy(\.isFinite) else { throw refused("the values must be finite") }
        guard delta.scale.isFinite, abs(delta.scale) <= maxDeltaScale else { throw refused("|scale| must be at most \(maxDeltaScale), got \(delta.scale)") }
        return ranges
    }
}
