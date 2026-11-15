import Foundation
import MLX
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    public func phasorProbe(_ request: PhasorProbeRequest) async throws(BasaltError) -> PhasorProbeSummary {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        let epoch = modelEpoch
        let anatomy = try await self.anatomy()
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while the phasor probe was prepared; retry") }
        try request.validate(layers: anatomy.layers.count)
        let extracted = try await Self.labRun(container, "the phasor probe") { context in
            try Self.extractPhasorStates(request, context: context)
        }
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while the phasor probe ran; retry") }
        return PhasorProbeEvaluator.evaluate(request: request, extracted: extracted)
    }

    static func extractPhasorStates(_ request: PhasorProbeRequest, context: ModelContext) throws -> ExtractedPhasorStates {
        let prompts = request.uniquePrompts
        let encoded = try encode(prompts, tokenizer: context.tokenizer)
        let requestedSet = request.layers.map(Set.init)
        var resolvedLayers = request.layers.map { Array(Set($0)).sorted() }
        var vectorsByPrompt: [String: [Int: [Float]]] = [:]
        for (prompt, ids) in zip(prompts, encoded) {
            try Task.checkCancellation()
            let probe = PhasorResidualProbe(layers: requestedSet, pooling: request.pooling)
            let logits = forward(context, probe: probe, ids, cache: try context.model.newCache(parameters: nil))
            guard probe.layerCount > 0 else {
                throw BasaltError.invalidRequestStructure(details: "this model's family does not report its layers to the forward probe")
            }
            let ordered = resolvedLayers ?? Array(0 ..< probe.layerCount)
            resolvedLayers = ordered
            if let missing = ordered.first(where: { probe.states[$0] == nil }) {
                throw BasaltError.invalidRequestStructure(details: "layer \(missing) is outside the \(probe.layerCount) layers the forward probe reports")
            }
            let matrix = stacked(ordered.compactMap { probe.states[$0] })
            eval(logits, matrix)
            let width = matrix.dim(-1)
            let flat = matrix.asArray(Float.self)
            var perLayer: [Int: [Float]] = [:]
            for (row, layer) in ordered.enumerated() {
                perLayer[layer] = Array(flat[row * width ..< (row + 1) * width])
            }
            vectorsByPrompt[prompt] = perLayer
        }
        return ExtractedPhasorStates(layers: resolvedLayers ?? [], vectorsByPrompt: vectorsByPrompt)
    }
}

final class PhasorResidualProbe: ForwardProbe {
    let layers: Set<Int>?
    let pooling: PhasorPooling
    private(set) var layerCount = 0
    private(set) var states: [Int: MLXArray] = [:]

    init(layers: Set<Int>?, pooling: PhasorPooling) {
        self.layers = layers
        self.pooling = pooling
    }

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        layerCount = max(layerCount, layer + 1)
        if layers?.contains(layer) ?? true {
            switch pooling {
            case .lastToken:
                states[layer] = hidden[0, hidden.dim(1) - 1].asType(.float32)
            case .meanPool:
                states[layer] = hidden[0].asType(.float32).mean(axis: 0)
            }
        }
        return hidden
    }

    func attentionWeights(layer: Int, _ weights: MLXArray) {}

    func wantsAttention(layer: Int) -> Bool { false }

    func headMask(layer: Int) -> MLXArray? { nil }

    func routerLogits(layer: Int, _ logits: MLXArray) {}

    func expertMask(layer: Int) -> MLXArray? { nil }
}
