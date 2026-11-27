import Foundation
import MLX
import MLXNN
import MLXLMCommon
@_spi(GemmaEncoder) import MLXLLM
import MLXVLM
import BasaltCore

extension MLXInferenceEngine {
    public func runCausalSweep(_ request: CausalSweepRequest) async throws(BasaltError) -> CausalSweepReport {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }
        let epoch = modelEpoch
        let anatomy = try await self.anatomy()
        guard modelEpoch == epoch else {
            throw .inferenceError("the model changed while the causal sweep was prepared; retry")
        }
        try request.validate(layers: anatomy.layers.count)
        let report = try await Self.labRun(container, "the causal sweep") { context in
            try Self.runCausalSweep(request, context: context)
        }
        guard modelEpoch == epoch else {
            throw .inferenceError("the model changed while the causal sweep ran; retry")
        }
        return report
    }

    package static func runCausalSweep(
        model: any LanguageModel,
        cleanTokens: MLXArray,
        corruptedTokens: MLXArray,
        targetTokenID: Int,
        foilTokenID: Int,
        options: CausalSweepOptions = CausalSweepOptions()
    ) throws(BasaltError) -> CausalSweepReport {
        try CausalSweepExecutor.run(
            model: model, cleanTokens: cleanTokens, corruptedTokens: corruptedTokens,
            targetTokenID: targetTokenID, foilTokenID: foilTokenID, options: options
        )
    }

    static func runCausalSweep(_ request: CausalSweepRequest, context: ModelContext) throws(BasaltError) -> CausalSweepReport {
        let encoded = try encode([request.cleanPrompt, request.corruptedPrompt], tokenizer: context.tokenizer)
        let targetID = try resolveTokenID(id: request.targetTokenID, text: request.targetToken, label: "target", tokenizer: context.tokenizer)
        let foilID = try resolveTokenID(id: request.foilTokenID, text: request.foilToken, label: "foil", tokenizer: context.tokenizer)
        guard targetID != foilID else {
            throw .invalidRequestStructure(details: "causal_sweep: target and foil resolved to the same token ID \(targetID)")
        }
        return try runCausalSweep(
            model: context.model,
            cleanTokens: MLXArray(encoded[0].map(Int32.init)).reshaped(1, -1),
            corruptedTokens: MLXArray(encoded[1].map(Int32.init)).reshaped(1, -1),
            targetTokenID: targetID,
            foilTokenID: foilID,
            options: CausalSweepOptions(layers: request.layers, includePatching: request.includePatching)
        )
    }

    static func resolveFinalLogitSoftcap(of object: Any) -> Double? {
        CausalSweepExecutor.resolveFinalLogitSoftcap(of: object)
    }

    static func resolveTokenID(id: Int?, text: String?, label: String, tokenizer: any MLXLMCommon.Tokenizer) throws(BasaltError) -> Int {
        if let id {
            guard id >= 0 else {
                throw .invalidRequestStructure(details: "causal_sweep: \(label)TokenID must be non-negative, got \(id)")
            }
            return id
        }
        guard let text, !text.isEmpty else {
            throw .invalidRequestStructure(details: "causal_sweep requires \(label)Token or \(label)TokenID")
        }
        if let direct = tokenizer.convertTokenToId(text) { return direct }
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        guard ids.count == 1, let only = ids.first else {
            throw .invalidRequestStructure(details: "causal_sweep: \(label)Token '\(text)' encoded to \(ids.count) tokens; pass \(label)TokenID instead")
        }
        return only
    }
}

private enum CausalSweepExecutor {
    struct ModelReadout {
        let embedding: Embedding
        let norm: (any UnaryLayer)?
        let head: Linear?
        let embedScale: Float?
        let finalLogitSoftcap: Double?
        var vocabSize: Int { head?.weight.dim(0) ?? embedding.weight.dim(0) }
    }

    struct LinearizedNorm {
        let scale: MLXArray
        let centerMean: Bool
        let bias: MLXArray?
    }

    struct PatchContext {
        let targetID: Int
        let foilID: Int
        let softcap: Double?
        let cleanDiff: Double
        let corruptedDiff: Double
        let layerDLAs: [Double]
        let includePatching: Bool
    }

    static func run(
        model: any LanguageModel,
        cleanTokens: MLXArray,
        corruptedTokens: MLXArray,
        targetTokenID: Int,
        foilTokenID: Int,
        options: CausalSweepOptions
    ) throws(BasaltError) -> CausalSweepReport {
        let (cleanBatch, cleanLastID, corruptedBatch, readout) = try prepareInputs(
            model: model, cleanTokens: cleanTokens, corruptedTokens: corruptedTokens,
            targetID: targetTokenID, foilID: foilTokenID
        )
        let cleanProbe = CausalCaptureProbe()
        let cleanLogits = try forwardLogits(model: model, tokens: cleanBatch, probe: cleanProbe)
        let layerCount = cleanProbe.layerCount
        guard layerCount > 0, (0 ..< layerCount).allSatisfy({ cleanProbe.residuals[$0] != nil }) else {
            throw .invalidRequestStructure(details: "this model's family does not report all layer residuals to the forward probe")
        }
        let selected = try resolveSelectedLayers(options.layers, total: layerCount)
        let cleanDiff = try logitDiff(
            logits: cleanLogits, targetID: targetTokenID, foilID: foilTokenID,
            softcap: readout.finalLogitSoftcap, extraEval: Array(cleanProbe.residuals.values)
        )
        let dla = computeDLA(
            cleanResiduals: cleanProbe.residuals, layerCount: layerCount, cleanLastID: cleanLastID,
            targetID: targetTokenID, foilID: foilTokenID, readout: readout
        )
        let corruptedLogits = try forwardLogits(model: model, tokens: corruptedBatch, probe: nil)
        let corruptedDiff = try logitDiff(
            logits: corruptedLogits, targetID: targetTokenID, foilID: foilTokenID,
            softcap: readout.finalLogitSoftcap
        )
        let context = PatchContext(
            targetID: targetTokenID, foilID: foilTokenID, softcap: readout.finalLogitSoftcap, cleanDiff: cleanDiff,
            corruptedDiff: corruptedDiff, layerDLAs: dla.layerDLAs, includePatching: options.includePatching
        )
        let scores = try runPatching(
            model: model, corruptedBatch: corruptedBatch, cleanResiduals: cleanProbe.residuals,
            selectedLayers: selected, context: context
        )
        return CausalSweepReport(
            targetTokenID: targetTokenID, foilTokenID: foilTokenID, cleanLogitDiff: cleanDiff,
            corruptedLogitDiff: corruptedDiff, embeddingDLA: dla.embeddingDLA, layers: scores
        ).withTotalDLASum(dla.totalSum)
    }

    static func prepareInputs(
        model: any LanguageModel,
        cleanTokens: MLXArray,
        corruptedTokens: MLXArray,
        targetID: Int,
        foilID: Int
    ) throws(BasaltError) -> (cleanBatch: MLXArray, cleanLastID: Int, corruptedBatch: MLXArray, readout: ModelReadout) {
        guard targetID >= 0, foilID >= 0, targetID != foilID else {
            throw .invalidRequestStructure(details: "causal_sweep requires distinct non-negative targetTokenID and foilTokenID")
        }
        let (cleanBatch, cleanLastID) = try normalizeTokens(cleanTokens, label: "clean")
        let (corruptedBatch, _) = try normalizeTokens(corruptedTokens, label: "corrupted")
        let readout = try extractReadout(of: model)
        guard (0 ..< readout.vocabSize).contains(targetID), (0 ..< readout.vocabSize).contains(foilID) else {
            throw .invalidRequestStructure(details: "causal_sweep: token ID is outside vocabulary size \(readout.vocabSize)")
        }
        guard cleanLastID < readout.embedding.weight.dim(0) else {
            throw .invalidRequestStructure(details: "causal_sweep: clean token ID \(cleanLastID) exceeds embedding size")
        }
        return (cleanBatch, cleanLastID, corruptedBatch, readout)
    }
}

private extension CausalSweepExecutor {
    static func normalizeTokens(_ tokens: MLXArray, label: String) throws(BasaltError) -> (batch: MLXArray, lastID: Int) {
        let batch: MLXArray
        if tokens.ndim == 1 {
            guard tokens.dim(0) > 0 else {
                throw .invalidRequestStructure(details: "causal_sweep: \(label) tokens must be non-empty")
            }
            batch = tokens.asType(.int32).reshaped(1, -1)
        } else if tokens.ndim == 2 {
            guard tokens.dim(0) == 1, tokens.dim(1) > 0 else {
                throw .invalidRequestStructure(details: "causal_sweep: \(label) tokens must have shape [1, L] with L > 0")
            }
            batch = tokens.asType(.int32)
        } else {
            throw .invalidRequestStructure(details: "causal_sweep: \(label) tokens must have shape [L] or [1, L]")
        }
        let last = batch[0, batch.dim(1) - 1]
        eval(last)
        let lastID = last.item(Int.self)
        guard lastID >= 0 else {
            throw .invalidRequestStructure(details: "causal_sweep: \(label) token ID \(lastID) must be non-negative")
        }
        return (batch, lastID)
    }

    static func resolveSelectedLayers(_ requested: [Int]?, total: Int) throws(BasaltError) -> [Int] {
        guard let requested else { return Array(0 ..< total) }
        guard !requested.isEmpty else {
            throw .invalidRequestStructure(details: "causal_sweep: layers must be non-empty when specified")
        }
        if let invalid = requested.first(where: { !(0 ..< total).contains($0) }) {
            throw .invalidRequestStructure(details: "causal_sweep: layer \(invalid) is outside 0..<\(total)")
        }
        return Array(Set(requested)).sorted()
    }

    static func extractReadout(of model: Module) throws(BasaltError) -> ModelReadout {
        let modules: [(path: String, name: String?, module: Module)] = model.namedModules().map {
            (path: $0.0, name: $0.0.split(separator: ".").last.map(String.init), module: $0.1)
        }.sorted { $0.path.split(separator: ".").count < $1.path.split(separator: ".").count }
        guard let embedding = ["embed_tokens", "embeddings"].lazy.compactMap({ name in
            modules.first { $0.name == name && $0.module is Embedding }
        }).first, let table = embedding.module as? Embedding else {
            throw .invalidRequestStructure(details: "causal_sweep: model exposes no embed_tokens or embeddings module")
        }
        let parent = embedding.path.split(separator: ".").dropLast().map { $0 + "." }.joined()
        let norm = ["norm", "norm_f", "embedding_norm", "final_norm"].lazy.compactMap { name in
            modules.first { $0.path == parent + name }?.module as? any UnaryLayer
        }.first
        let head = modules.first { $0.name == "lm_head" }?.module as? Linear
        let embedScale = resolveEmbedScale(model: model, modules: modules, embeddingPath: embedding.path, embedding: table)
        let softcap = resolveFinalLogitSoftcap(of: model)
        return ModelReadout(embedding: table, norm: norm, head: head, embedScale: embedScale, finalLogitSoftcap: softcap)
    }
}

extension CausalSweepExecutor {
    private static func positiveSoftcap(_ value: Float?) -> Double? {
        guard let value, value.isFinite, value > 0 else { return nil }
        return Double(value)
    }

    static func resolveFinalLogitSoftcap(of object: Any) -> Double? {
        if let m = object as? MLXVLM.Gemma4Unified {
            return positiveSoftcap(m.config.textConfiguration.finalLogitSoftcapping)
        }
        if let m = object as? MLXVLM.Gemma4 {
            return positiveSoftcap(m.config.textConfiguration.finalLogitSoftcapping)
        }
        if let m = object as? Inkling2Model {
            return positiveSoftcap(m.config.finalLogitSoftcap)
        }
        if let m = object as? MuseGlimmerTextModel {
            return positiveSoftcap(m.configuration.finalLogitSoftcapping)
        }
        if let m = object as? MLXLLM.Gemma4Model {
            return resolveFinalLogitSoftcap(of: m.languageModel)
        }
        if let m = object as? MLXLLM.Gemma4TextModel {
            return mirrorSoftcap(m, key: "config", field: "finalLogitSoftcapping")
        }
        if let m = object as? MLXLLM.Gemma2Model {
            return mirrorSoftcap(m, key: nil, field: "logitSoftCap")
        }
        return nil
    }

    private static func mirrorSoftcap(_ object: Any, key: String?, field: String) -> Double? {
        let target: Any
        if let key {
            guard let child = Mirror(reflecting: object).children.first(where: { $0.label == key }) else { return nil }
            target = child.value
        } else {
            target = object
        }
        guard let leaf = Mirror(reflecting: target).children.first(where: { $0.label == field }) else { return nil }
        if let f = leaf.value as? Float { return positiveSoftcap(f) }
        if let d = leaf.value as? Double, d.isFinite, d > 0 { return d }
        return nil
    }

    static func resolveEmbedScale(
        model: Module,
        modules: [(path: String, name: String?, module: Module)],
        embeddingPath: String,
        embedding: Embedding
    ) -> Float? {
        let parentPath = embeddingPath.split(separator: ".").dropLast().joined(separator: ".")
        let parentModule = parentPath.isEmpty ? model : modules.first { $0.path == parentPath }?.module
        if let parentModule {
            for child in Mirror(reflecting: parentModule).children {
                guard let label = child.label,
                      ["embedScale", "embed_scale", "hiddenScale", "hidden_scale"].contains(label) else { continue }
                if let f = child.value as? Float, f.isFinite, f > 0 { return f }
                if let d = child.value as? Double, d.isFinite, d > 0 { return Float(d) }
            }
        }
        if modules.contains(where: { $0.name == "pre_feedforward_layernorm" }) {
            return Float(embedding.weight.dim(-1)).squareRoot()
        }
        return nil
    }

    static func forwardLogits(model: any LanguageModel, tokens: MLXArray, probe: (any ForwardProbe)?) throws(BasaltError) -> MLXArray {
        let cache: [KVCache]
        do {
            cache = try model.newCache(parameters: nil)
        } catch {
            throw .modelInferenceError(details: MLXInferenceEngine.briefly(error))
        }
        if let probe, let probable = model as? any BasaltProbableModel {
            return probable.forwardWithProbe(tokens, cache: cache, probe: probe)
        }
        return withForwardProbe(probe) {
            model(LMInput.Text(tokens: tokens), cache: cache, state: nil).logits
        }
    }

    static func logitDiff(
        logits: MLXArray, targetID: Int, foilID: Int, softcap: Double? = nil, extraEval: [MLXArray] = []
    ) throws(BasaltError) -> Double {
        let last = logits[0, logits.dim(1) - 1].asType(.float32)
        let vocab = last.dim(0)
        guard (0 ..< vocab).contains(targetID), (0 ..< vocab).contains(foilID) else {
            throw .invalidRequestStructure(details: "causal_sweep: token ID is outside logits vocabulary \(vocab)")
        }
        let tArr = last[targetID]
        let fArr = last[foilID]
        eval([tArr, fArr] + extraEval)
        let tRaw = Double(tArr.item(Float.self))
        let fRaw = Double(fArr.item(Float.self))
        let value: Double
        if let cap = softcap, cap > 0 {
            let tClamped = max(-0.999999, min(0.999999, tRaw / cap))
            let fClamped = max(-0.999999, min(0.999999, fRaw / cap))
            value = cap * (Foundation.atanh(tClamped) - Foundation.atanh(fClamped))
        } else {
            value = tRaw - fRaw
        }
        guard value.isFinite else { throw .nonFiniteLogits }
        return value
    }
}

private extension CausalSweepExecutor {
    static func unembedRow(index: Int, readout: ModelReadout) -> (row: MLXArray, bias: MLXArray?) {
        if let qHead = readout.head as? QuantizedLinear {
            let slice = index ..< (index + 1)
            let row = dequantized(
                qHead.weight[slice], scales: qHead.scales[slice], biases: qHead.biases?[slice],
                groupSize: qHead.groupSize, bits: qHead.bits, mode: qHead.mode, globalScale: qHead.globalScale
            )[0].asType(.float32)
            return (row, qHead.bias.map { $0[index].asType(.float32) })
        }
        if let nf4 = readout.head as? NF4Linear {
            let slice = index ..< (index + 1)
            let row = NF4Metal.dequantized(nf4.weight[slice], scales: nf4.scales[slice], groupSize: nf4.groupSize)[0].asType(.float32)
            return (row, nf4.bias.map { $0[index].asType(.float32) })
        }
        if let linear = readout.head {
            return (linear.weight[index].asType(.float32), linear.bias.map { $0[index].asType(.float32) })
        }
        return (readout.embedding(MLXArray([Int32(index)]))[0].asType(.float32), nil)
    }

    static func linearizeNorm(_ norm: (any UnaryLayer)?, at hFinal: MLXArray) -> LinearizedNorm {
        let hiddenDim = hFinal.dim(0)
        guard let norm, !(norm is PassthroughRMSNorm) else {
            return LinearizedNorm(scale: MLXArray.ones([hiddenDim], dtype: .float32), centerMean: false, bias: nil)
        }
        if let rms = norm as? RMSNorm {
            let rmsValue = sqrt(square(hFinal).mean() + rms.eps)
            return LinearizedNorm(scale: rms.weight.asType(.float32) / rmsValue, centerMean: false, bias: nil)
        }
        if let ln = norm as? LayerNorm {
            let variance = square(hFinal - hFinal.mean()).mean()
            let weight = ln.weight?.asType(.float32) ?? MLXArray.ones([hiddenDim], dtype: .float32)
            return LinearizedNorm(scale: weight * rsqrt(variance + ln.eps), centerMean: true, bias: ln.bias?.asType(.float32))
        }
        let radiusArr = sqrt(square(hFinal).mean())
        eval(radiusArr)
        let radius = radiusArr.item(Float.self)
        guard radius > 0, radius.isFinite else {
            return LinearizedNorm(scale: MLXArray.ones([hiddenDim], dtype: .float32), centerMean: false, bias: nil)
        }
        let probeInput = MLXArray.ones([1, 1, hiddenDim], dtype: .float32) * radius
        return LinearizedNorm(scale: norm(probeInput)[0, 0].asType(.float32) / radius, centerMean: false, bias: nil)
    }

    static func applyLinearNorm(_ vector: [Double], scale: [Double], centerMean: Bool) -> [Double] {
        let count = min(vector.count, scale.count)
        let mean = centerMean && count > 0 ? vector.prefix(count).reduce(0.0, +) / Double(count) : 0.0
        var out: [Double] = []
        out.reserveCapacity(count)
        for index in 0 ..< count {
            out.append((vector[index] - mean) * scale[index])
        }
        return out
    }
}

private extension CausalSweepExecutor {
    static func computeDLA(
        cleanResiduals: [Int: MLXArray],
        layerCount: Int,
        cleanLastID: Int,
        targetID: Int,
        foilID: Int,
        readout: ModelReadout
    ) -> (embeddingDLA: Double, layerDLAs: [Double], totalSum: Double) {
        var h0Arr = readout.embedding(MLXArray([Int32(cleanLastID)]))[0].asType(.float32)
        if let embedScale = readout.embedScale {
            h0Arr = h0Arr * embedScale
        }
        let stackedStates = stacked((0 ..< layerCount).compactMap { cleanResiduals[$0]?.asType(.float32) })
        let normOp = linearizeNorm(readout.norm, at: stackedStates[layerCount - 1])
        let target = unembedRow(index: targetID, readout: readout)
        let foil = unembedRow(index: foilID, readout: readout)
        var toEval: [MLXArray] = [h0Arr, stackedStates, normOp.scale, target.row, foil.row]
        if let nb = normOp.bias { toEval.append(nb) }
        if let tb = target.bias { toEval.append(tb) }
        if let fb = foil.bias { toEval.append(fb) }
        eval(toEval)
        let h0 = h0Arr.asArray(Float.self).map(Double.init)
        let scale = normOp.scale.asArray(Float.self).map(Double.init)
        let direction = CausalAttributionMath.unembedDirection(
            targetRow: target.row.asArray(Float.self).map(Double.init),
            foilRow: foil.row.asArray(Float.self).map(Double.init)
        )
        let normedDeltas = buildNormedDeltas(
            h0: h0, flatStates: stackedStates.asArray(Float.self), layerCount: layerCount,
            scale: scale, centerMean: normOp.centerMean
        )
        let normedH0 = applyLinearNorm(h0, scale: scale, centerMean: normOp.centerMean)
        let headBiasDiff = Double(target.bias?.item(Float.self) ?? 0) - Double(foil.bias?.item(Float.self) ?? 0)
        var normBiasDiff = 0.0
        if let biasArr = normOp.bias {
            let biasVec = biasArr.asArray(Float.self).map(Double.init)
            normBiasDiff = CausalAttributionMath.dot(biasVec.span, direction.span)
        }
        return CausalAttributionMath.directLogitAttributions(
            embeddingState: normedH0, layerDeltas: normedDeltas, direction: direction, biasDiff: headBiasDiff + normBiasDiff
        )
    }

    static func buildNormedDeltas(
        h0: [Double],
        flatStates: [Float],
        layerCount: Int,
        scale: [Double],
        centerMean: Bool
    ) -> [[Double]] {
        let width = h0.count
        var normedDeltas: [[Double]] = []
        normedDeltas.reserveCapacity(layerCount)
        var previous = h0
        for layer in 0 ..< layerCount {
            let base = layer * width
            var current: [Double] = []
            var delta: [Double] = []
            current.reserveCapacity(width)
            delta.reserveCapacity(width)
            for col in 0 ..< width {
                let value = Double(flatStates[base + col])
                current.append(value)
                delta.append(value - previous[col])
            }
            normedDeltas.append(applyLinearNorm(delta, scale: scale, centerMean: centerMean))
            previous = current
        }
        return normedDeltas
    }

    static func runPatching(
        model: any LanguageModel,
        corruptedBatch: MLXArray,
        cleanResiduals: [Int: MLXArray],
        selectedLayers: [Int],
        context: PatchContext
    ) throws(BasaltError) -> [LayerCausalScore] {
        guard context.includePatching else {
            return selectedLayers.map { LayerCausalScore(layerIndex: $0, directLogitAttribution: context.layerDLAs[$0]) }
        }
        var scores: [LayerCausalScore] = []
        scores.reserveCapacity(selectedLayers.count)
        for layer in selectedLayers {
            if Task.isCancelled { throw .cancelled }
            guard let cleanResidual = cleanResiduals[layer] else {
                throw .invalidRequestStructure(details: "causal_sweep: missing clean residual for layer \(layer)")
            }
            let probe = CausalPatchProbe(targetLayer: layer, cleanResidual: cleanResidual)
            let patchedLogits = try forwardLogits(model: model, tokens: corruptedBatch, probe: probe)
            let patchedDiff = try logitDiff(
                logits: patchedLogits, targetID: context.targetID, foilID: context.foilID, softcap: context.softcap
            )
            let recovery = CausalAttributionMath.causalRecovery(
                cleanDiff: context.cleanDiff, corruptedDiff: context.corruptedDiff, patchedDiff: patchedDiff
            )
            scores.append(LayerCausalScore(
                layerIndex: layer, directLogitAttribution: context.layerDLAs[layer],
                patchedLogitDiff: patchedDiff, causalRecovery: recovery
            ))
        }
        return scores
    }
}

private final class CausalCaptureProbe: ForwardProbe {
    private(set) var layerCount = 0
    private(set) var residuals: [Int: MLXArray] = [:]

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        layerCount = max(layerCount, layer + 1)
        if hidden.ndim == 3 {
            residuals[layer] = hidden[0, hidden.dim(1) - 1]
        } else if hidden.ndim == 2 {
            residuals[layer] = hidden[hidden.dim(0) - 1]
        }
        return hidden
    }

    func attentionWeights(layer: Int, _ weights: MLXArray) {}
    func wantsAttention(layer: Int) -> Bool { false }
    func headMask(layer: Int) -> MLXArray? { nil }
    func routerLogits(layer: Int, _ logits: MLXArray) {}
    func expertMask(layer: Int) -> MLXArray? { nil }
}

private final class CausalPatchProbe: ForwardProbe {
    let targetLayer: Int
    let cleanResidual: MLXArray

    init(targetLayer: Int, cleanResidual: MLXArray) {
        self.targetLayer = targetLayer
        self.cleanResidual = cleanResidual
    }

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        guard layer == targetLayer else { return hidden }
        if hidden.ndim == 3 {
            let seqLen = hidden.dim(1)
            let replacement = cleanResidual.asType(hidden.dtype).reshaped(1, 1, hidden.dim(2))
            guard seqLen > 1 else { return replacement }
            return concatenated([hidden[0..., 0 ..< (seqLen - 1), 0...], replacement], axis: 1)
        }
        guard hidden.ndim == 2 else { return hidden }
        let seqLen = hidden.dim(0)
        let replacement = cleanResidual.asType(hidden.dtype).reshaped(1, hidden.dim(1))
        guard seqLen > 1 else { return replacement }
        return concatenated([hidden[0 ..< (seqLen - 1), 0...], replacement], axis: 0)
    }

    func attentionWeights(layer: Int, _ weights: MLXArray) {}
    func wantsAttention(layer: Int) -> Bool { false }
    func headMask(layer: Int) -> MLXArray? { nil }
    func routerLogits(layer: Int, _ logits: MLXArray) {}
    func expertMask(layer: Int) -> MLXArray? { nil }
}
