import Foundation
import MLX
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    public func activationContrast(_ request: ActivationContrastRequest) async throws(BasaltError) -> ActivationContrastReport {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        let epoch = modelEpoch
        let anatomy = try await self.anatomy()
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while the contrast was prepared; retry") }
        try request.validate(layers: anatomy.layers.count)
        return try await Self.labRun(container, "the activation contrast") { context in try Self.activationContrast(request, context: context) }
    }

    static func activationContrast(_ request: ActivationContrastRequest, context: ModelContext) throws -> ActivationContrastReport {
        let (a, statesA) = try tracedSamples(request.a, request: request, context: context)
        let (b, statesB) = try tracedSamples(request.b, request: request, context: context)
        guard (a + b).allSatisfy({ $0.table.layout == a[0].table.layout }) else {
            throw BasaltError.inferenceError("the traced layout differs between prompts; trace fewer layers")
        }
        let (unit, norm, separation) = ContrastSearch.directions(positive: stacked(statesA), negative: stacked(statesB))
        let indices = request.trace.tracedLayers(of: statesA[0].dim(0))
        let norms = norm.asArray(Float.self), separations = separation.asArray(Float.self)
        let best = separations.indices.max { separations[$0] < separations[$1] } ?? 0
        return ActivationContrastReport(
            components: ActivationStatistics.rank(a: a, b: b, onsetWindow: request.onsetWindow, topK: request.topK),
            layers: indices.enumerated().map { DirectionLayer(index: $1, norm: norms[$0], separation: separations[$0]) },
            bestLayer: indices[best], direction: unit[best].asArray(Float.self))
    }

    static func tracedSamples(_ prompts: [String], request: ActivationContrastRequest, context: ModelContext) throws -> ([TracedSample], [MLXArray]) {
        var samples: [TracedSample] = [], states: [MLXArray] = []
        for ids in try encode(prompts, tokenizer: context.tokenizer) {
            guard ids.count + request.maxTokens <= TokenTraceRequest.tokenLimit else {
                throw BasaltError.contextLengthExceeded(maxTokens: TokenTraceRequest.tokenLimit, requested: ids.count + request.maxTokens)
            }
            let recorder = TokenTraceRecorder(request.trace, readout: nil)
            recorder.capturesPromptResiduals = true
            let generated = try tracedGreedy(ids, maxTokens: request.maxTokens, recorder: recorder, context: context)
            guard let layout = recorder.layout, let (floats, ints) = recorder.table(), !recorder.promptResiduals.isEmpty else {
                throw BasaltError.invalidRequestStructure(details: "this model's family reports none of the traced signals to the forward probe")
            }
            samples.append(TracedSample(
                table: TokenTraceTable(layout: layout, tokens: ids.count + generated.count, floats: floats.asArray(Float.self), ints: ints.asArray(Int32.self)),
                promptTokens: ids.count))
            states.append(stacked(recorder.promptResiduals.keys.sorted().compactMap { recorder.promptResiduals[$0] }))
        }
        return (samples, states)
    }
}
