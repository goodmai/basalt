import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    public func runOutlierProfile(_ request: OutlierProfileRequest) async throws(BasaltError) -> OutlierProfileReport {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        let epoch = modelEpoch
        let anatomy = try await self.anatomy()
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while outlier_profile was prepared; retry") }
        try request.validate(layerCount: anatomy.layers.count)
        let layerFilter = request.layers.map(Set.init)
        let report = try await Self.labRun(container, "outlier_profile") { context in
            let batches = try Self.encode(request.prompts, tokenizer: context.tokenizer)
                .map { MLXArray($0.map(Int32.init)).reshaped(1, -1) }
            return try Self.measureOutlierProfile(
                model: context.model, tokenBatches: batches, sinkTokenCount: request.sinkTokenCount,
                deadNeuronThreshold: request.deadNeuronThreshold, layerFilter: layerFilter
            )
        }
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while outlier_profile ran; retry") }
        return report
    }

    package static func measureOutlierProfile(
        model: any LanguageModel,
        tokenBatches: [MLXArray],
        sinkTokenCount: Int = 1,
        deadNeuronThreshold: Double = 1e-3,
        layerFilter: Set<Int>? = nil
    ) throws -> OutlierProfileReport {
        try OutlierModelInspector.validateInputs(
            tokenBatches: tokenBatches, sinkTokenCount: sinkTokenCount,
            deadNeuronThreshold: deadNeuronThreshold, layerFilter: layerFilter
        )
        let qkLayers = Dictionary(
            QKClipController.attentionLayers(of: model).map { ($0.layerIndex, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        let probe = OutlierProfileProbe(
            layerFilter: layerFilter, sinkTokenCount: sinkTokenCount,
            deadNeuronThreshold: Float(deadNeuronThreshold),
            layerSpecs: OutlierModelInspector.layerSpecs(of: model), qkLayers: qkLayers
        )
        var promptCount = 0
        var totalTokens = 0
        for rawBatch in tokenBatches {
            try Task.checkCancellation()
            let batch = rawBatch.ndim == 1 ? rawBatch.reshaped(1, -1) : rawBatch
            guard batch.ndim == 2, batch.dim(0) > 0, batch.dim(1) > 0 else { continue }
            promptCount += batch.dim(0)
            totalTokens += batch.dim(0) * batch.dim(1)
            probe.resetForward(initialResidual: OutlierModelInspector.initialResidual(model: model, inputs: batch))
            _ = try OutlierModelInspector.runForward(model: model, tokens: batch, probe: probe)
            probe.resetForward(initialResidual: nil)
        }
        return try probe.buildReport(promptCount: promptCount, totalTokens: totalTokens, layerFilter: layerFilter)
    }
}

private struct LayerProfileSpec {
    let oProj: (any UnaryLayer)?
    let postAttnNorm: (any UnaryLayer)?
    let preFFNorm: (any UnaryLayer)?
    let isGeGLU: Bool
    let gateProj: (any UnaryLayer)?
    let upProj: (any UnaryLayer)?
    let mlpUnary: (any UnaryLayer)?

    func mlpInput(prevResidual: MLXArray, perHead: MLXArray) -> MLXArray? {
        guard let oProj, prevResidual.ndim == 3, perHead.ndim == 4 else { return nil }
        let (b, l) = (perHead.dim(0), perHead.dim(2))
        guard prevResidual.dim(0) == b, prevResidual.dim(1) == l else { return nil }
        let flat = stopGradient(perHead).transposed(0, 2, 1, 3).reshaped(b, l, -1)
        guard Self.matchesLinearInput(oProj, width: flat.dim(-1)) else { return nil }
        let attnOut = stopGradient(oProj(flat))
        guard attnOut.shape == prevResidual.shape else { return nil }
        let residualAdd = isGeGLU ? (postAttnNorm.map { stopGradient($0(attnOut)) } ?? attnOut) : attnOut
        return applyMLPNorm(prevResidual + residualAdd)
    }

    func fallbackMLPInput(from residual: MLXArray) -> MLXArray {
        applyMLPNorm(stopGradient(residual))
    }

    func deadFraction(forMLPInput x: MLXArray, threshold: Float) -> (fraction: MLXArray, count: Int)? {
        guard let hidden = mlpHidden(for: x), hidden.size > 0 else { return nil }
        return ((MLX.abs(hidden.asType(.float32)) .< threshold).asType(.float32).mean(), hidden.size)
    }

    private func applyMLPNorm(_ x: MLXArray) -> MLXArray {
        guard let norm = preFFNorm ?? postAttnNorm else { return x }
        return stopGradient(norm(x))
    }

    private func mlpHidden(for x: MLXArray) -> MLXArray? {
        if let gateProj, let upProj {
            guard Self.matchesLinearInput(gateProj, width: x.dim(-1)) else { return nil }
            let g = stopGradient(gateProj(x))
            let u = stopGradient(upProj(x))
            return (isGeGLU ? MLXNN.geluApproximate(g) : MLXNN.silu(g)) * u
        }
        return mlpUnary.map { stopGradient($0(x)) }
    }

    private static func matchesLinearInput(_ layer: any UnaryLayer, width: Int) -> Bool {
        guard type(of: layer) == Linear.self, let linear = layer as? Linear else { return true }
        return linear.shape.1 == width
    }
}

private final class OutlierProfileProbe: ForwardProbe {
    let layerFilter: Set<Int>?
    let maxFilteredLayer: Int?
    let sinkTokenCount: Int
    let deadNeuronThreshold: Float
    let layerSpecs: [Int: LayerProfileSpec]
    let qkLayers: [Int: QKAttentionLayer]

    private(set) var layerCount = 0
    private(set) var moments: [Int: [(count: Int, packed: MLXArray)]] = [:]
    private(set) var deadStats: [Int: [(count: Int, fraction: MLXArray)]] = [:]
    private(set) var sinkStats: [Int: (headSum: MLXArray, headMax: MLXArray, queryCount: Int)] = [:]

    private var currentResidual: MLXArray?
    private var recordedSink: Set<Int> = []
    private var recordedDead: Set<Int> = []

    init(
        layerFilter: Set<Int>?, sinkTokenCount: Int, deadNeuronThreshold: Float,
        layerSpecs: [Int: LayerProfileSpec], qkLayers: [Int: QKAttentionLayer]
    ) {
        self.layerFilter = layerFilter
        self.maxFilteredLayer = layerFilter?.max()
        self.sinkTokenCount = sinkTokenCount
        self.deadNeuronThreshold = deadNeuronThreshold
        self.layerSpecs = layerSpecs
        self.qkLayers = qkLayers
    }

    func resetForward(initialResidual: MLXArray?) {
        currentResidual = initialResidual
        recordedSink.removeAll(keepingCapacity: true)
        recordedDead.removeAll(keepingCapacity: true)
    }

    func traces(_ layer: Int) -> Bool { layerFilter?.contains(layer) ?? true }

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        layerCount = Swift.max(layerCount, layer + 1)
        if traces(layer) {
            recordResidualMoments(layer: layer, hidden: hidden)
            recordFallbacks(layer: layer, hidden: hidden)
        }
        if let maxTarget = maxFilteredLayer, layer >= maxTarget {
            currentResidual = nil
        } else {
            currentResidual = stopGradient(hidden)
        }
        return hidden
    }

    private func recordFallbacks(layer: Int, hidden: MLXArray) {
        if !recordedSink.contains(layer), let prev = currentResidual, let qkLayer = qkLayers[layer],
           let stats = OutlierModelInspector.qkSinkStats(layer: qkLayer, hidden: prev, sinkTokenCount: sinkTokenCount) {
            recordSink(layer: layer, headSum: stats.headSum, headMax: stats.headMax, queryCount: stats.queryCount)
        }
        if !recordedDead.contains(layer), let spec = layerSpecs[layer],
           let stats = spec.deadFraction(forMLPInput: spec.fallbackMLPInput(from: hidden), threshold: deadNeuronThreshold) {
            recordDead(layer: layer, fraction: stats.fraction, count: stats.count)
        }
    }

    func wantsAttention(layer: Int) -> Bool { traces(layer) }

    func attentionWeights(layer: Int, _ weights: MLXArray) {
        guard traces(layer), weights.ndim == 4 else { return }
        let (b, heads, qLen, kLen) = (weights.dim(0), weights.dim(1), weights.dim(2), weights.dim(3))
        guard Swift.min(b, heads, qLen, kLen) > 0 else { return }
        let kCount = Swift.max(1, Swift.min(sinkTokenCount, kLen))
        let qStart = qLen > kCount ? kCount : 0
        let sinkMass = stopGradient(weights[0..., 0..., qStart ..< qLen, 0 ..< kCount]).asType(.float32).sum(axis: -1)
        recordSink(layer: layer, headSum: sinkMass.sum(axes: [0, 2]), headMax: sinkMass.max(axes: [0, 2]), queryCount: b * (qLen - qStart))
    }

    func headOutputs(layer: Int, _ perHead: MLXArray) {
        guard traces(layer), let prev = currentResidual, let spec = layerSpecs[layer],
              let mlpIn = spec.mlpInput(prevResidual: prev, perHead: perHead),
              let stats = spec.deadFraction(forMLPInput: mlpIn, threshold: deadNeuronThreshold) else { return }
        recordDead(layer: layer, fraction: stats.fraction, count: stats.count)
    }

    func headMask(layer: Int) -> MLXArray? { nil }
    func routerLogits(layer: Int, _ logits: MLXArray) {}
    func expertMask(layer: Int) -> MLXArray? { nil }

    private func recordResidualMoments(layer: Int, hidden: MLXArray) {
        guard hidden.size > 0 else { return }
        let x = stopGradient(hidden).asType(.float32)
        let mean = x.mean()
        let dx = x - mean
        let dx2 = dx * dx
        let packed = MLX.stacked([mean, dx2.sum(), (dx2 * dx).sum(), (dx2 * dx2).sum(), MLX.abs(x).max()], axis: 0)
        moments[layer, default: []].append((count: hidden.size, packed: packed))
    }

    private func recordSink(layer: Int, headSum: MLXArray, headMax: MLXArray, queryCount: Int) {
        guard queryCount > 0 else { return }
        recordedSink.insert(layer)
        if let existing = sinkStats[layer] {
            sinkStats[layer] = (existing.headSum + headSum, MLX.maximum(existing.headMax, headMax), existing.queryCount + queryCount)
        } else {
            sinkStats[layer] = (headSum, headMax, queryCount)
        }
    }

    private func recordDead(layer: Int, fraction: MLXArray, count: Int) {
        guard count > 0 else { return }
        recordedDead.insert(layer)
        deadStats[layer, default: []].append((count: count, fraction: fraction))
    }
}

extension OutlierProfileProbe {
    func buildReport(promptCount: Int, totalTokens: Int, layerFilter: Set<Int>?) throws -> OutlierProfileReport {
        guard promptCount > 0 else { throw BasaltError.invalidRequestStructure(details: "outlier_profile received only empty token batches") }
        guard layerCount > 0 else { throw BasaltError.invalidRequestStructure(details: "this model's family does not report its layers to the forward probe") }
        let orderedLayers = layerFilter.map { $0.sorted() } ?? Array(0 ..< layerCount)
        if let missing = orderedLayers.first(where: { moments[$0] == nil }) {
            throw BasaltError.invalidRequestStructure(details: "layer \(missing) is outside the \(layerCount) layers the forward probe reports")
        }
        let pending = pendingArrays(for: orderedLayers)
        if !pending.isEmpty { eval(pending) }
        return OutlierProfileReport(
            promptCount: promptCount, totalTokens: totalTokens,
            layers: orderedLayers.map { layerMetrics(for: $0) }, layerCount: layerCount
        )
    }

    private func pendingArrays(for layers: [Int]) -> [MLXArray] {
        var arrays: [MLXArray] = []
        for layer in layers {
            if let batches = moments[layer] { arrays.append(contentsOf: batches.map(\.packed)) }
            if let dead = deadStats[layer] { arrays.append(contentsOf: dead.map(\.fraction)) }
            if let sink = sinkStats[layer] { arrays.append(contentsOf: [sink.headSum, sink.headMax]) }
        }
        return arrays
    }

    private func layerMetrics(for layer: Int) -> LayerOutlierMetrics {
        var combined = OutlierMomentAccumulator()
        for batch in moments[layer] ?? [] {
            let v = batch.packed.asArray(Float.self)
            guard v.count == 5 else { continue }
            combined.merge(OutlierMomentAccumulator(
                count: batch.count, mean: Double(v[0]), m2: Double(v[1]),
                m3: Double(v[2]), m4: Double(v[3]), maxAbs: Double(v[4])
            ))
        }
        return LayerOutlierMetrics(
            layerIndex: layer, excessKurtosis: combined.excessKurtosis, crestFactor: combined.crestFactor,
            maxAbsActivation: combined.maxAbs, rmsActivation: combined.rms, deadNeuronFraction: deadFraction(for: layer)
        ).withHeads(headMetrics(for: layer))
    }

    private func deadFraction(for layer: Int) -> Double? {
        guard let batches = deadStats[layer], !batches.isEmpty else { return nil }
        var (deadTotal, neuronTotal) = (0.0, 0)
        for batch in batches {
            let frac = Double(batch.fraction.item(Float.self))
            if frac.isFinite {
                deadTotal += Swift.min(1.0, Swift.max(0.0, frac)) * Double(batch.count)
                neuronTotal += batch.count
            }
        }
        return neuronTotal > 0 ? Swift.min(1.0, Swift.max(0.0, deadTotal / Double(neuronTotal))) : nil
    }

    private func headMetrics(for layer: Int) -> [HeadSinkMetrics] {
        guard let sink = sinkStats[layer], sink.queryCount > 0 else { return [] }
        let (sums, maxes, denom) = (sink.headSum.asArray(Float.self), sink.headMax.asArray(Float.self), Double(sink.queryCount))
        return (0 ..< Swift.min(sums.count, maxes.count)).map { head in
            let meanRaw = Double(sums[head]) / denom
            let maxRaw = Double(maxes[head])
            return HeadSinkMetrics(
                headIndex: head,
                meanSinkMass: meanRaw.isFinite ? Swift.min(1.0, Swift.max(0.0, meanRaw)) : 0.0,
                maxSinkMass: maxRaw.isFinite ? Swift.min(1.0, Swift.max(0.0, maxRaw)) : 0.0
            )
        }
    }
}

private enum OutlierModelInspector {
    static func validateInputs(
        tokenBatches: [MLXArray], sinkTokenCount: Int, deadNeuronThreshold: Double, layerFilter: Set<Int>?
    ) throws {
        guard !tokenBatches.isEmpty else { throw BasaltError.invalidRequestStructure(details: "outlier_profile requires at least one token batch") }
        guard sinkTokenCount >= 1 else { throw BasaltError.invalidRequestStructure(details: "outlier_profile: sinkTokenCount must be at least 1") }
        guard deadNeuronThreshold.isFinite, deadNeuronThreshold >= 0 else {
            throw BasaltError.invalidRequestStructure(details: "outlier_profile: deadNeuronThreshold must be finite and non-negative")
        }
        if let layerFilter, layerFilter.isEmpty || layerFilter.contains(where: { $0 < 0 }) {
            throw BasaltError.invalidRequestStructure(details: "outlier_profile: layerFilter must contain non-negative indices")
        }
    }

    static func runForward(model: any LanguageModel, tokens: MLXArray, probe: any ForwardProbe) throws -> MLXArray {
        let cache = try model.newCache(parameters: nil)
        if let probable = model as? any BasaltProbableModel {
            return probable.forwardWithProbe(tokens, cache: cache, probe: probe)
        }
        return withForwardProbe(probe) { model(LMInput.Text(tokens: tokens), cache: cache, state: nil).logits }
    }

    static func transformerBlocks(of model: Module) -> [Module] {
        let direct = LeJEPABackbone.layers(of: model)
        return !direct.isEmpty ? direct : (childModule(of: model, key: "language_model").map(LeJEPABackbone.layers(of:)) ?? [])
    }

    static func layerSpecs(of model: Module) -> [Int: LayerProfileSpec] {
        Dictionary(uniqueKeysWithValues: transformerBlocks(of: model).enumerated().map { ($0.offset, layerSpec(for: $0.element)) })
    }

    private static func layerSpec(for block: Module) -> LayerProfileSpec {
        let attn = childModule(of: block, key: "self_attn") ?? block
        let preFeedforward = childUnary(of: block, key: "pre_feedforward_layernorm")
        let mlp = childModule(of: block, key: "mlp") ?? childModule(of: block, key: "feed_forward")
        return LayerProfileSpec(
            oProj: childUnary(of: attn, keys: ["o_proj", "out_proj"]),
            postAttnNorm: childUnary(of: block, key: "post_attention_layernorm"),
            preFFNorm: preFeedforward ?? childUnary(of: block, key: "ffn_norm"),
            isGeGLU: preFeedforward != nil,
            gateProj: mlp.flatMap { childUnary(of: $0, keys: ["gate_proj", "w1"]) },
            upProj: mlp.flatMap { childUnary(of: $0, keys: ["up_proj", "w3"]) },
            mlpUnary: mlp.flatMap(denseMLPUnary)
        )
    }

    private static func denseMLPUnary(of mlp: Module) -> (any UnaryLayer)? {
        guard childModule(of: mlp, key: "switch_mlp") == nil, childModule(of: mlp, key: "experts") == nil else { return nil }
        return mlp as? any UnaryLayer
    }

    private static func childUnary(of module: Module, keys: [String]) -> (any UnaryLayer)? {
        for key in keys { if let found = childUnary(of: module, key: key) { return found } }
        return nil
    }

    static func initialResidual(model: Module, inputs: MLXArray) -> MLXArray? {
        let modules: [(path: String, name: String?, module: Module)] = model.namedModules()
            .map { (path: $0.0, name: $0.0.split(separator: ".").last.map(String.init), module: $0.1) }
            .sorted { $0.path.split(separator: ".").count < $1.path.split(separator: ".").count }
        guard let found = modules.first(where: { ($0.name == "embed_tokens" || $0.name == "embeddings") && $0.module is Embedding }),
              let embed = found.module as? any UnaryLayer else { return nil }
        var h0 = stopGradient(embed(inputs))
        let parentPath = found.path.split(separator: ".").dropLast().joined(separator: ".")
        let parentModule = parentPath.isEmpty ? model : modules.first(where: { $0.path == parentPath })?.module
        if let explicitScale = parentModule.flatMap(readEmbedScale) {
            h0 = h0 * MLXArray(explicitScale).asType(h0.dtype)
        } else if let firstBlock = transformerBlocks(of: model).first, childUnary(of: firstBlock, key: "pre_feedforward_layernorm") != nil {
            h0 = h0 * MLXArray(Float(h0.dim(-1)).squareRoot()).asType(h0.dtype)
        }
        return stopGradient(h0)
    }

    private static func readEmbedScale(from module: Module) -> Float? {
        for child in Mirror(reflecting: module).children {
            if child.label == "embedScale" || child.label == "embed_scale", let value = child.value as? Float { return value }
        }
        return nil
    }

    private static func childModule(of module: Module, key: String) -> Module? {
        if case .value(let child) = module.children()[key] { return child }
        return nil
    }

    private static func childUnary(of module: Module, key: String) -> (any UnaryLayer)? {
        childModule(of: module, key: key) as? any UnaryLayer
    }
}

extension OutlierModelInspector {
    static func qkSinkStats(
        layer: QKAttentionLayer, hidden: MLXArray, sinkTokenCount: Int
    ) -> (headSum: MLXArray, headMax: MLXArray, queryCount: Int)? {
        guard hidden.ndim == 3, hidden.dim(0) > 0, hidden.dim(1) > 0 else { return nil }
        let (b, l) = (hidden.dim(0), hidden.dim(1))
        let (g, c) = (layer.geometry, layer.components)
        let normed = c.inputNorm.map { stopGradient($0(stopGradient(hidden))) } ?? stopGradient(hidden)
        let (qRaw, kRaw) = (stopGradient(c.qProj(normed)), stopGradient(c.kProj(normed)))
        var q = g.isGatedQuery
            ? qRaw.reshaped([b, l, g.qHeads, g.headDim * 2])[0..., 0..., 0..., 0 ..< g.headDim]
            : qRaw.reshaped([b, l, g.qHeads, g.headDim])
        var k = kRaw.reshaped([b, l, g.kvHeads, g.headDim])
        if let qNorm = c.qNorm { q = stopGradient(qNorm(q)) }
        if let kNorm = c.kNorm { k = stopGradient(kNorm(k)) }
        (q, k) = (q.transposed(0, 2, 1, 3), k.transposed(0, 2, 1, 3))
        if let rope = c.rope { (q, k) = (stopGradient(rope(q, offset: 0)), stopGradient(rope(k, offset: 0))) }
        return tiledCausalSinkMass(queries: q, keys: k, scale: g.scale, sinkTokenCount: sinkTokenCount)
    }

    private static func tiledCausalSinkMass(
        queries: MLXArray, keys: MLXArray, scale: Float, sinkTokenCount: Int
    ) -> (headSum: MLXArray, headMax: MLXArray, queryCount: Int)? {
        let (b, qHeads, qLen, headDim) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        let (kvHeads, kLen) = (keys.dim(1), keys.dim(2))
        guard Swift.min(b, qHeads, kvHeads, qLen, kLen) > 0, qHeads % kvHeads == 0 else { return nil }
        let (groups, kCount) = (qHeads / kvHeads, Swift.max(1, Swift.min(sinkTokenCount, kLen)))
        let qBegin = qLen > kCount ? kCount : 0
        var (sumAcc, maxAcc): (MLXArray?, MLXArray?) = (nil, nil)
        for qStart in stride(from: qBegin, to: qLen, by: QKClipController.defaultQueryTileSize) {
            let qRange = qStart ..< Swift.min(qStart + QKClipController.defaultQueryTileSize, qLen)
            let (tileLen, kEnd) = (qRange.count, Swift.min(kLen, qRange.upperBound))
            let qTile = queries[0..., 0..., qRange, 0...].asType(.float32).reshaped([b, kvHeads, groups, tileLen, headDim]) * scale
            let kSlice = keys[0..., 0..., 0 ..< kEnd, 0...].asType(.float32).transposed(0, 1, 3, 2).reshaped([b, kvHeads, 1, headDim, kEnd])
            let qPos = (MLXArray.arange(tileLen) + qStart).reshaped([1, 1, 1, tileLen, 1])
            let kPos = MLXArray.arange(kEnd).reshaped([1, 1, 1, 1, kEnd])
            let probs = softmax(MLX.where(kPos .<= qPos, matmul(qTile, kSlice), MLXArray(-Float.infinity)), axis: -1)
            let sink = probs[0..., 0..., 0..., 0..., 0 ..< Swift.min(kCount, kEnd)].sum(axis: -1).reshaped([b, qHeads, tileLen])
            let (tileSum, tileMax) = (sink.sum(axes: [0, 2]), sink.max(axes: [0, 2]))
            sumAcc = sumAcc.map { $0 + tileSum } ?? tileSum
            maxAcc = maxAcc.map { MLX.maximum($0, tileMax) } ?? tileMax
        }
        guard let headSum = sumAcc, let headMax = maxAcc else { return nil }
        return (headSum, headMax, b * (qLen - qBegin))
    }
}
