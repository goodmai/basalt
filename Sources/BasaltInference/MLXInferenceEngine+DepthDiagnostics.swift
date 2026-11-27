import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension MLXInferenceEngine {
    public func runDepthDiagnostics(_ request: DepthDiagnosticsRequest) async throws(BasaltError) -> DepthDiagnosticsReport {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }
        let epoch = modelEpoch
        let anatomy = try await self.anatomy()
        guard modelEpoch == epoch else {
            throw .inferenceError("the model changed while depth diagnostics were prepared; retry")
        }
        try request.validate(layerCount: anatomy.layers.count)
        let report = try await Self.labRun(container, "depth diagnostics") { context in
            try Self.captureDepthDiagnostics(request, context: context)
        }
        guard modelEpoch == epoch else {
            throw .inferenceError("the model changed while depth diagnostics ran; retry")
        }
        return report
    }

    package static func measureDepthDiagnostics(
        model: any LanguageModel,
        tokens: MLXArray,
        includeRank: Bool = true,
        softcapThresholdFraction: Double = 0.95,
        layerFilter: Set<Int>? = nil
    ) throws -> DepthDiagnosticsReport {
        try DepthDiagnosticsCollector.measure(
            model: model,
            tokens: tokens,
            includeRank: includeRank,
            softcapThresholdFraction: softcapThresholdFraction,
            layerFilter: layerFilter
        )
    }

    static func captureDepthDiagnostics(_ request: DepthDiagnosticsRequest, context: ModelContext) throws -> DepthDiagnosticsReport {
        let encoded = ModelProbe.encoded(request.prompt, context.tokenizer)
        guard !encoded.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "depth_diagnostics: the prompt encodes to no tokens")
        }
        let trimmed = Array(encoded.prefix(request.maxSequenceLength))
        let tokens = MLXArray(trimmed.map(Int32.init)).reshaped(1, -1)
        return try measureDepthDiagnostics(
            model: context.model,
            tokens: tokens,
            includeRank: request.includeRank,
            softcapThresholdFraction: request.softcapThresholdFraction,
            layerFilter: request.layers.map(Set.init)
        )
    }
}

private final class DepthDiagnosticsProbe: ForwardProbe {
    private(set) var layerCount = 0
    private(set) var residuals: [Int: MLXArray] = [:]
    private(set) var headOuts: [Int: MLXArray] = [:]

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        layerCount = max(layerCount, layer + 1)
        residuals[layer] = stopGradient(hidden)
        return hidden
    }

    func headOutputs(layer: Int, _ perHead: MLXArray) {
        headOuts[layer] = stopGradient(perHead)
    }

    func attentionWeights(layer: Int, _ weights: MLXArray) {}
    func wantsAttention(layer: Int) -> Bool { false }
    func headMask(layer: Int) -> MLXArray? { nil }
    func routerLogits(layer: Int, _ logits: MLXArray) {}
    func expertMask(layer: Int) -> MLXArray? { nil }
}

private struct DepthCaptureContext {
    let probe: DepthDiagnosticsProbe
    let blocks: [Module]
    let qkByLayer: [Int: QKAttentionLayer]
    let h0: MLXArray?
    let includeRank: Bool
    let softcapThreshold: Double
}

private struct DepthLayerPending {
    let layerIndex: Int
    let normIn: MLXArray
    let normOut: MLXArray
    let attnNorm: MLXArray?
    let mlpNorm: MLXArray?
    let sMax: MLXArray?
    let softcap: Double?
    let satRate: MLXArray?
    let singularValues: MLXArray?

    func appendPending(into arrays: inout [MLXArray]) {
        arrays.append(normIn)
        arrays.append(normOut)
        if let attnNorm { arrays.append(attnNorm) }
        if let mlpNorm { arrays.append(mlpNorm) }
        if let sMax { arrays.append(sMax) }
        if let satRate { arrays.append(satRate) }
        if let singularValues { arrays.append(singularValues) }
    }

    func materialize() -> LayerDepthMetrics {
        let sMaxValues = sMax.map { $0.asType(.float32).asArray(Float.self).map(Double.init) } ?? []
        let rank = singularValues.map {
            DepthDiagnosticsMath.effectiveRank(singularValues: $0.asType(.float32).asArray(Float.self).map(Double.init))
        }
        var metrics = LayerDepthMetrics(
            layerIndex: layerIndex,
            residualNormIn: Double(normIn.item(Float.self)),
            residualNormOut: Double(normOut.item(Float.self)),
            preSoftmaxSMax: sMaxValues,
            effectiveRank: rank
        )
        metrics.setSublayerNorms(
            attention: attnNorm.map { Double($0.item(Float.self)) },
            mlp: mlpNorm.map { Double($0.item(Float.self)) }
        )
        metrics.setSoftcap(
            value: softcap,
            saturationRate: satRate.map { Double($0.item(Float.self)) }
        )
        return metrics
    }
}

private enum DepthDiagnosticsCollector {
    static func measure(
        model: any LanguageModel,
        tokens: MLXArray,
        includeRank: Bool,
        softcapThresholdFraction: Double,
        layerFilter: Set<Int>?
    ) throws -> DepthDiagnosticsReport {
        guard softcapThresholdFraction.isFinite, softcapThresholdFraction > 0, softcapThresholdFraction <= 1.0 else {
            throw BasaltError.invalidRequestStructure(
                details: "depth_diagnostics: softcapThresholdFraction must be in (0, 1], got \(softcapThresholdFraction)"
            )
        }
        let input2D = tokens.ndim == 1 ? tokens.reshaped(1, -1) : tokens
        guard input2D.ndim == 2, input2D.dim(0) > 0, input2D.dim(1) > 0 else {
            throw BasaltError.invalidRequestStructure(details: "depth_diagnostics: tokens must be non-empty 1D or 2D")
        }
        let probe = DepthDiagnosticsProbe()
        try runForward(model: model, tokens: input2D, probe: probe)
        let selected = try resolveLayers(layerCount: probe.layerCount, residuals: probe.residuals, filter: layerFilter)
        let qkLayers = QKClipController.attentionLayers(of: model)
        let context = DepthCaptureContext(
            probe: probe,
            blocks: transformerBlocks(of: model),
            qkByLayer: Dictionary(uniqueKeysWithValues: qkLayers.map { ($0.layerIndex, $0) }),
            h0: layerZeroResidual(model: model, inputs: input2D),
            includeRank: includeRank,
            softcapThreshold: softcapThresholdFraction
        )
        let pending = selected.compactMap { buildLayerPending(layer: $0, context: context) }
        var toEval: [MLXArray] = []
        toEval.reserveCapacity(pending.count * 7)
        for item in pending { item.appendPending(into: &toEval) }
        if !toEval.isEmpty { eval(toEval) }
        return DepthDiagnosticsReport(
            promptTokenCount: input2D.dim(1),
            layerCount: probe.layerCount,
            layers: pending.map { $0.materialize() }
        )
    }

    private static func runForward(model: any LanguageModel, tokens: MLXArray, probe: DepthDiagnosticsProbe) throws {
        let cache = try model.newCache(parameters: nil)
        if let probable = model as? any BasaltProbableModel {
            _ = probable.forwardWithProbe(tokens, cache: cache, probe: probe)
        } else {
            _ = withForwardProbe(probe) {
                model(LMInput.Text(tokens: tokens), cache: cache, state: nil)
            }
        }
    }

    private static func resolveLayers(
        layerCount: Int,
        residuals: [Int: MLXArray],
        filter: Set<Int>?
    ) throws -> [Int] {
        guard layerCount > 0 else {
            throw BasaltError.invalidRequestStructure(
                details: "this model's family does not report its layers to the forward probe"
            )
        }
        let ordered = filter.map { Array($0).sorted() } ?? Array(0 ..< layerCount)
        guard !ordered.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "depth_diagnostics: layers must list at least one layer index")
        }
        if let bad = ordered.first(where: { $0 < 0 || $0 >= layerCount || residuals[$0] == nil }) {
            throw BasaltError.invalidRequestStructure(details: "layer \(bad) is outside the model's \(layerCount) layers")
        }
        return ordered
    }

    private static func buildLayerPending(layer: Int, context: DepthCaptureContext) -> DepthLayerPending? {
        guard let hOut = context.probe.residuals[layer] else { return nil }
        let hIn = layer == 0 ? (context.h0 ?? hOut) : (context.probe.residuals[layer - 1] ?? hOut)
        let block = context.blocks.indices.contains(layer) ? context.blocks[layer] : nil
        let sublayers = sublayerNorms(block: block, hIn: hIn, hOut: hOut, perHead: context.probe.headOuts[layer])
        let softcap = block.flatMap(detectAttentionSoftcap(block:))
        let preSoftmax = context.qkByLayer[layer].map {
            attentionPreSoftmax(qkLayer: $0, hIn: hIn, softcap: softcap, threshold: context.softcapThreshold)
        }
        let singularValues = context.includeRank ? residualSingularValues(hOut) : nil
        return DepthLayerPending(
            layerIndex: layer,
            normIn: meanTokenL2Norm(hIn),
            normOut: meanTokenL2Norm(hOut),
            attnNorm: sublayers.attn,
            mlpNorm: sublayers.mlp,
            sMax: preSoftmax?.sMax,
            softcap: preSoftmax != nil ? softcap : nil,
            satRate: preSoftmax?.satRate,
            singularValues: singularValues
        )
    }

    private static func residualSingularValues(_ hOut: MLXArray) -> MLXArray? {
        guard hOut.ndim == 3, hOut.dim(0) > 0, hOut.dim(1) > 0, hOut.dim(2) > 0 else { return nil }
        let rows = min(hOut.dim(1), DepthDiagnosticsRequest.rankSequenceCap)
        let matrix = stopGradient(hOut[0, 0 ..< rows, 0...]).asType(.float32)
        let values: MLXArray = MLXLinalg.svd(matrix, stream: .cpu)
        return values
    }

    private static func meanTokenL2Norm(_ tensor: MLXArray) -> MLXArray {
        let f = stopGradient(tensor).asType(.float32)
        return (f * f).sum(axis: -1).sqrt().mean()
    }
}

extension DepthDiagnosticsCollector {
    private static func sublayerNorms(
        block: Module?,
        hIn: MLXArray,
        hOut: MLXArray,
        perHead: MLXArray?
    ) -> (attn: MLXArray?, mlp: MLXArray?) {
        guard let block else { return (nil, nil) }
        let attnContrib = attentionContribution(block: block, perHead: perHead)
        let mlpContrib = mlpContribution(block: block, hIn: hIn, hOut: hOut, attnContrib: attnContrib)
        return (attnContrib.map(meanTokenL2Norm), mlpContrib.map(meanTokenL2Norm))
    }

    private static func attentionContribution(block: Module, perHead: MLXArray?) -> MLXArray? {
        guard let perHead, perHead.ndim == 4,
              let attn = childModule(of: block, key: "self_attn"),
              let oProj = childUnary(of: attn, key: "o_proj") ?? childUnary(of: attn, key: "out_proj") else {
            return nil
        }
        let combined = perHead.transposed(0, 2, 1, 3).reshaped(perHead.dim(0), perHead.dim(2), -1)
        if let linear = oProj as? Linear, linear.shape.1 != combined.dim(-1) { return nil }
        let raw = stopGradient(oProj(combined))
        if childUnary(of: block, key: "pre_feedforward_layernorm") != nil,
           let postAttn = childUnary(of: block, key: "post_attention_layernorm") {
            return stopGradient(postAttn(raw))
        }
        return raw
    }

    private static func mlpContribution(
        block: Module,
        hIn: MLXArray,
        hOut: MLXArray,
        attnContrib: MLXArray?
    ) -> MLXArray? {
        guard mlpChild(of: block) != nil, let attnContrib else { return nil }
        let hMid = stopGradient(hIn + attnContrib)
        guard hOut.shape == hMid.shape else { return nil }
        return stopGradient(hOut - hMid)
    }

    private static func mlpChild(of block: Module) -> Module? {
        for key in ["mlp", "feed_forward", "ffn", "block_sparse_moe"] {
            if let child = childModule(of: block, key: key) { return child }
        }
        return nil
    }
}

extension DepthDiagnosticsCollector {
    private static func attentionPreSoftmax(
        qkLayer: QKAttentionLayer,
        hIn: MLXArray,
        softcap: Double?,
        threshold: Double
    ) -> (sMax: MLXArray, satRate: MLXArray?) {
        guard let softcap else {
            return (QKClipController.layerMaxLogits(qkLayer, hidden: hIn, isCausal: true), nil)
        }
        let projected = projectQK(qkLayer, hidden: hIn)
        let sMax = QKClipController.maxHeadLogits(
            queries: projected.queries,
            keys: projected.keys,
            scale: qkLayer.geometry.scale,
            isCausal: true
        )
        let sat = tiledSoftcapSaturation(
            queries: projected.queries,
            keys: projected.keys,
            geometry: qkLayer.geometry,
            softcap: softcap,
            threshold: threshold
        )
        return (sMax, sat)
    }

    private static func projectQK(
        _ layer: QKAttentionLayer,
        hidden: MLXArray
    ) -> (queries: MLXArray, keys: MLXArray) {
        let (b, l) = (hidden.dim(0), hidden.dim(1))
        let (g, c) = (layer.geometry, layer.components)
        let h = stopGradient(hidden)
        let normed = c.inputNorm.map { stopGradient($0(h)) } ?? h
        let qRaw = stopGradient(c.qProj(normed))
        let kRaw = stopGradient(c.kProj(normed))
        var q = g.isGatedQuery
            ? qRaw.reshaped([b, l, g.qHeads, g.headDim * 2])[0..., 0..., 0..., 0 ..< g.headDim]
            : qRaw.reshaped([b, l, g.qHeads, g.headDim])
        var k = kRaw.reshaped([b, l, g.kvHeads, g.headDim])
        if let qNorm = c.qNorm { q = stopGradient(qNorm(q)) }
        if let kNorm = c.kNorm { k = stopGradient(kNorm(k)) }
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        if let rope = c.rope {
            q = stopGradient(rope(q, offset: 0))
            k = stopGradient(rope(k, offset: 0))
        }
        return (q, k)
    }

    private static func tiledSoftcapSaturation(
        queries: MLXArray,
        keys: MLXArray,
        geometry: QKAttentionLayer.Geometry,
        softcap: Double,
        threshold: Double
    ) -> MLXArray {
        let (b, qLen, kLen) = (queries.dim(0), queries.dim(2), keys.dim(2))
        guard b > 0, qLen > 0, kLen > 0, geometry.kvHeads > 0, geometry.qHeads % geometry.kvHeads == 0 else {
            return MLXArray(Float(0))
        }
        let groups = geometry.qHeads / geometry.kvHeads
        let tileSize = max(1, QKClipController.defaultQueryTileSize)
        let capScalar = Float(softcap)
        let threshScalar = Float(threshold)
        let headBatch = Float(b * geometry.qHeads)
        var satSum = MLXArray(Float(0))
        var validSum = MLXArray(Float(0))
        for qStart in stride(from: 0, to: qLen, by: tileSize) {
            let qEnd = min(qStart + tileSize, qLen)
            let kEnd = min(kLen, qEnd)
            let tileLen = qEnd - qStart
            let qTile = stopGradient(queries[0..., 0..., qStart ..< qEnd, 0...]).asType(.float32)
                .reshaped([b, geometry.kvHeads, groups, tileLen, geometry.headDim]) * geometry.scale
            let kSlice = stopGradient(keys[0..., 0..., 0 ..< kEnd, 0...]).asType(.float32)
                .transposed(0, 1, 3, 2)
                .reshaped([b, geometry.kvHeads, 1, geometry.headDim, kEnd])
            let scores = matmul(qTile, kSlice)
            let qPos = (MLXArray.arange(tileLen) + qStart).reshaped([1, 1, 1, tileLen, 1])
            let kPos = MLXArray.arange(kEnd).reshaped([1, 1, 1, 1, kEnd])
            let valid = kPos .<= qPos
            let saturated = (MLX.abs(MLX.tanh(scores / capScalar)) .>= threshScalar).asType(.float32)
            satSum = satSum + MLX.where(valid, saturated, MLXArray(Float(0))).sum()
            validSum = validSum + valid.asType(.float32).sum() * headBatch
        }
        return stopGradient(satSum / MLX.maximum(validSum, MLXArray(Float(1))))
    }
}

extension DepthDiagnosticsCollector {
    private static func transformerBlocks(of model: Module) -> [Module] {
        let direct = LeJEPABackbone.layers(of: model)
        if !direct.isEmpty { return direct }
        return childModule(of: model, key: "language_model").map(LeJEPABackbone.layers(of:)) ?? []
    }

    private static func layerZeroResidual(model: Module, inputs: MLXArray) -> MLXArray? {
        let modules: [(path: String, name: String?, module: Module)] = model.namedModules()
            .map { (path: $0.0, name: $0.0.split(separator: ".").last.map(String.init), module: $0.1) }
            .sorted { $0.path.split(separator: ".").count < $1.path.split(separator: ".").count }
        guard let found = modules.first(where: {
            ($0.name == "embed_tokens" || $0.name == "embeddings") && $0.module is Embedding
        }), let embed = found.module as? any UnaryLayer else { return nil }
        var h0 = stopGradient(embed(inputs))
        let parentPath = found.path.split(separator: ".").dropLast().joined(separator: ".")
        let parentModule = parentPath.isEmpty ? model : modules.first(where: { $0.path == parentPath })?.module
        if let explicitScale = parentModule.flatMap(readEmbedScale) {
            h0 = h0 * MLXArray(explicitScale).asType(h0.dtype)
        } else if let firstBlock = transformerBlocks(of: model).first,
                  childUnary(of: firstBlock, key: "pre_feedforward_layernorm") != nil {
            h0 = h0 * MLXArray(Float(h0.dim(-1)).squareRoot()).asType(h0.dtype)
        }
        return stopGradient(h0)
    }

    private static func readEmbedScale(from module: Module) -> Float? {
        let keys: Set<String> = ["embedScale", "embed_scale", "hiddenScale", "hidden_scale"]
        for child in Mirror(reflecting: module).children {
            if let label = child.label, keys.contains(label), let value = child.value as? Float, value.isFinite, value > 0 {
                return value
            }
        }
        return nil
    }

    private static func detectAttentionSoftcap(block: Module) -> Double? {
        let attn = childModule(of: block, key: "self_attn") ?? block
        if let direct = extractSoftcap(from: attn) { return direct }
        for child in Mirror(reflecting: attn).children where child.label == "args" || child.label == "config" {
            if let nested = extractSoftcap(from: child.value) { return nested }
        }
        return nil
    }

    private static func extractSoftcap(from value: Any) -> Double? {
        let labels: Set<String> = [
            "logitSoftCap", "logitSoftcap", "attnLogitSoftcapping", "attnLogitSoftcap",
            "attentionLogitSoftcapping", "attnSoftcap", "softcap", "softCap",
        ]
        for child in Mirror(reflecting: value).children {
            guard let label = child.label, labels.contains(label) else { continue }
            if let f = child.value as? Float, f.isFinite, f > 0 { return Double(f) }
            if let d = child.value as? Double, d.isFinite, d > 0 { return d }
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
