import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

public final class QKResidualProbe: ForwardProbe {
    public private(set) var residuals: [Int: MLXArray] = [:]

    public init() {}

    public func reset() { residuals.removeAll(keepingCapacity: true) }

    public func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        residuals[layer] = stopGradient(hidden)
        return hidden
    }

    public func attentionWeights(layer: Int, _ weights: MLXArray) {}
    public func wantsAttention(layer: Int) -> Bool { false }
    public func headMask(layer: Int) -> MLXArray? { nil }
    public func routerLogits(layer: Int, _ logits: MLXArray) {}
    public func expertMask(layer: Int) -> MLXArray? { nil }
}

public struct QKClipStats: Sendable, Equatable {
    public var clippedLayers: Int
    public var clippedHeads: Int
    public var maxObservedLogit: Float

    public static let empty = QKClipStats(clippedLayers: 0, clippedHeads: 0, maxObservedLogit: 0)

    public init(clippedLayers: Int, clippedHeads: Int, maxObservedLogit: Float) {
        (self.clippedLayers, self.clippedHeads, self.maxObservedLogit) = (clippedLayers, clippedHeads, maxObservedLogit)
    }
}

public struct QKAttentionLayer {
    public struct Geometry: Sendable, Equatable {
        public let qHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let scale: Float
        public let isGatedQuery: Bool

        public init(qHeads: Int, kvHeads: Int, headDim: Int, scale: Float, isGatedQuery: Bool = false) {
            (self.qHeads, self.kvHeads, self.headDim, self.scale, self.isGatedQuery) = (qHeads, kvHeads, headDim, scale, isGatedQuery)
        }
    }

    public struct Components {
        public let inputNorm: (any UnaryLayer)?
        public let qProj: any UnaryLayer
        public let kProj: any UnaryLayer
        public let qNorm: (any UnaryLayer)?
        public let kNorm: (any UnaryLayer)?
        public let rope: (any OffsetLayer)?

        public init(
            inputNorm: (any UnaryLayer)? = nil,
            qProj: any UnaryLayer,
            kProj: any UnaryLayer,
            qNorm: (any UnaryLayer)? = nil,
            kNorm: (any UnaryLayer)? = nil,
            rope: (any OffsetLayer)? = nil
        ) {
            self.inputNorm = inputNorm
            self.qProj = qProj
            self.kProj = kProj
            self.qNorm = qNorm
            self.kNorm = kNorm
            self.rope = rope
        }
    }

    public let layerIndex: Int
    public let components: Components
    public let geometry: Geometry

    public init(layerIndex: Int, components: Components, geometry: Geometry) {
        (self.layerIndex, self.components, self.geometry) = (layerIndex, components, geometry)
    }
}

public enum QKClipController {
    public static let defaultQueryTileSize = 128
    public static let defaultBatchTileSize = 8

    public static func forward(_ model: any LanguageModel, inputs: MLXArray, probe: QKResidualProbe) -> MLXArray {
        if let probable = model as? any BasaltProbableModel {
            return probable.forwardWithProbe(inputs, cache: nil, probe: probe)
        }
        return withForwardProbe(probe) { model(inputs, cache: nil) }
    }

    public static func maxHeadLogits(
        queries: MLXArray,
        keys: MLXArray,
        scale: Float,
        isCausal: Bool = true,
        tileSize: Int = defaultQueryTileSize,
        validLengths: [Int]? = nil
    ) -> MLXArray {
        let (b, qHeads, qLen) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let (kvHeads, kLen) = (keys.dim(1), keys.dim(2))
        guard b > 0, qHeads > 0, kvHeads > 0, qHeads % kvHeads == 0, qLen > 0, kLen > 0 else {
            return MLXArray.zeros([max(0, qHeads)], type: Float.self)
        }
        let groups = qHeads / kvHeads
        let qStep = max(1, tileSize)
        let batchStep = max(1, min(b, defaultBatchTileSize))
        let offset = kLen - qLen
        var runningMax: MLXArray?
        for bStart in stride(from: 0, to: b, by: batchStep) {
            let bRange = bStart ..< min(bStart + batchStep, b)
            let batchLengths = validLengths.map { Array($0[bRange]) }
            for qStart in stride(from: 0, to: qLen, by: qStep) {
                let qRange = qStart ..< min(qStart + qStep, qLen)
                let kEnd = isCausal ? min(kLen, qRange.upperBound + offset) : kLen
                guard kEnd > 0 else { continue }
                let tileMax = tileMaxLogits(
                    queries: queries[bRange, 0..., qRange, 0...],
                    keys: keys[bRange, 0..., 0 ..< kEnd, 0...],
                    groups: groups,
                    scale: scale,
                    causalOffset: isCausal ? (qStart + offset) : nil,
                    validLengths: batchLengths.map { (qStart: qStart, lengths: $0) }
                )
                runningMax = runningMax.map { MLX.maximum($0, tileMax) } ?? tileMax
            }
        }
        return stopGradient(runningMax ?? MLXArray.zeros([qHeads], type: Float.self))
    }

    private static func tileMaxLogits(
        queries: MLXArray,
        keys: MLXArray,
        groups: Int,
        scale: Float,
        causalOffset: Int?,
        validLengths: (qStart: Int, lengths: [Int])?
    ) -> MLXArray {
        let (bCount, qHeads, tileLen, headDim) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        let (kvHeads, kEnd) = (keys.dim(1), keys.dim(2))
        let qTile = stopGradient(queries).asType(.float32)
            .reshaped([bCount, kvHeads, groups, tileLen, headDim]) * scale
        let kSlice = stopGradient(keys).asType(.float32)
            .transposed(0, 1, 3, 2)
            .reshaped([bCount, kvHeads, 1, headDim, kEnd])
        var scores = matmul(qTile, kSlice)
        if let causalOffset {
            let qPos = (MLXArray.arange(tileLen) + causalOffset).reshaped([1, 1, 1, tileLen, 1])
            let kPos = MLXArray.arange(kEnd).reshaped([1, 1, 1, 1, kEnd])
            scores = MLX.where(kPos .<= qPos, scores, MLXArray(-Float.infinity))
        }
        if let valid = validLengths, valid.lengths.count == bCount {
            let lens = MLXArray(valid.lengths.map(Int32.init)).reshaped([bCount, 1, 1, 1, 1])
            let qIdx = (MLXArray.arange(tileLen) + valid.qStart).reshaped([1, 1, 1, tileLen, 1])
            let kIdx = MLXArray.arange(kEnd).reshaped([1, 1, 1, 1, kEnd])
            scores = MLX.where((qIdx .< lens) .&& (kIdx .< lens), scores, MLXArray(-Float.infinity))
        }
        return scores.max(axes: [0, 3, 4]).reshaped([qHeads])
    }

    public static func layerMaxLogits(
        _ layer: QKAttentionLayer,
        hidden: MLXArray,
        isCausal: Bool = true,
        tileSize: Int = defaultQueryTileSize,
        validLengths: [Int]? = nil
    ) -> MLXArray {
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
        return maxHeadLogits(
            queries: q, keys: k, scale: g.scale, isCausal: isCausal, tileSize: tileSize, validLengths: validLengths
        )
    }

    public static func layerMaxLogits(
        model: Module,
        layers: [QKAttentionLayer],
        inputs: MLXArray,
        probe: QKResidualProbe,
        isCausal: Bool = true,
        validLengths: [Int]? = nil
    ) -> [MLXArray] {
        guard !layers.isEmpty else { return [] }
        let inputZero = layerZeroResidual(model: model, inputs: inputs)
        return layers.map { layer in
            let residual = layer.layerIndex == 0 ? inputZero : probe.residuals[layer.layerIndex - 1]
            return residual.map {
                layerMaxLogits(layer, hidden: $0, isCausal: isCausal, validLengths: validLengths)
            } ?? MLXArray.zeros([layer.geometry.qHeads], type: Float.self)
        }
    }

    @discardableResult
    public static func measureAndClip(
        model: Module,
        inputs: MLXArray,
        isCausal: Bool = true,
        options: TrainingOptimizerOptions
    ) -> QKClipStats {
        measureAndClipSequences(model: model, sequences: [inputs], isCausal: isCausal, options: options)
    }

    @discardableResult
    public static func measureAndClipSequences(
        model: Module,
        sequences: [MLXArray],
        isCausal: Bool = true,
        options: TrainingOptimizerOptions
    ) -> QKClipStats {
        let layers = attentionLayers(of: model)
        guard !layers.isEmpty, !sequences.isEmpty else { return .empty }
        var combined: [MLXArray]?
        let probe = QKResidualProbe()
        for seq in sequences where seq.dim(-1) > 0 {
            probe.reset()
            if let lm = model as? any LanguageModel { _ = forward(lm, inputs: seq, probe: probe) }
            let cur = layerMaxLogits(model: model, layers: layers, inputs: seq, probe: probe, isCausal: isCausal)
            probe.reset()
            combined = combined.map { zip($0, cur).map { MLX.maximum($0, $1) } } ?? cur
        }
        guard let maxArrays = combined, !maxArrays.isEmpty else { return .empty }
        eval(maxArrays)
        return applyClip(model: model, layers: layers, layerMaxLogitArrays: maxArrays, tau: options.qkClipTau, alpha: options.qkClipAlpha)
    }
}

extension QKClipController {
    @discardableResult
    public static func applyClip(
        model: Module,
        layers: [QKAttentionLayer],
        layerMaxLogitArrays: [MLXArray],
        tau: Float,
        alpha: Float
    ) -> QKClipStats {
        guard !layers.isEmpty, layers.count == layerMaxLogitArrays.count else { return .empty }
        var (clippedLayers, clippedHeads) = (0, 0)
        var maxObserved: Float = 0
        for (layer, array) in zip(layers, layerMaxLogitArrays) {
            let headLogits = array.asType(.float32).asArray(Float.self)
            maxObserved = max(maxObserved, headLogits.filter(\.isFinite).max() ?? 0)
            let exceeded = headLogits.filter { $0.isFinite && $0 > tau }.count
            guard exceeded > 0 else { continue }
            if applyClip(to: layer, headMaxLogits: headLogits, tau: tau, alpha: alpha) {
                clippedLayers += 1
                clippedHeads += exceeded
            }
        }
        if clippedLayers > 0 { model.invalidateCompiledTraces() }
        return QKClipStats(clippedLayers: clippedLayers, clippedHeads: clippedHeads, maxObservedLogit: maxObserved)
    }

    @discardableResult
    public static func applyClip(to layer: QKAttentionLayer, headMaxLogits: [Float], tau: Float, alpha: Float) -> Bool {
        let g = layer.geometry
        guard headMaxLogits.count == g.qHeads else { return false }
        let (qScales, kScales) = MatrixOptimizerMath.gqaSafeQKClipScales(
            headMaxLogits: headMaxLogits, tau: tau, alpha: alpha, kvHeads: g.kvHeads
        )
        let hasQKNorm = layer.components.qNorm != nil || layer.components.kNorm != nil
        var changed = false
        if !hasQKNorm {
            let qChannels = channelScales(headScales: qScales, headDim: g.headDim, isGated: g.isGatedQuery)
            let kChannels = channelScales(headScales: kScales, headDim: g.headDim, isGated: false)
            changed = scaleProjection(layer.components.qProj, channelScales: qChannels)
            changed = scaleProjection(layer.components.kProj, channelScales: kChannels) || changed
        } else {
            let groups = g.qHeads / g.kvHeads
            let minGamma = (0 ..< g.qHeads).map { qScales[$0] * kScales[$0 / groups] }.min() ?? 1.0
            let (qNormShared, kNormShared) = sharedNormScales(
                minGamma: minGamma, alpha: alpha,
                qTrainable: hasTrainableWeight(layer.components.qNorm),
                kTrainable: hasTrainableWeight(layer.components.kNorm)
            )
            changed = scaleNorm(layer.components.qNorm, headScales: qScales, sharedScale: qNormShared) || changed
            changed = scaleNorm(layer.components.kNorm, headScales: kScales, sharedScale: kNormShared) || changed
        }
        return changed
    }

    private static func channelScales(headScales: [Float], headDim: Int, isGated: Bool) -> [Float] {
        let strideDim = isGated ? headDim * 2 : headDim
        var out = Array(repeating: Float(1.0), count: headScales.count * strideDim)
        for (head, scale) in headScales.enumerated() {
            let base = head * strideDim
            for offset in 0 ..< headDim { out[base + offset] = scale }
        }
        return out
    }

    public static func canBoundLogits(_ layer: QKAttentionLayer) -> Bool {
        let c = layer.components
        if c.qNorm != nil || c.kNorm != nil {
            return hasTrainableWeight(c.qNorm) || hasTrainableWeight(c.kNorm)
        }
        return hasTrainableDenseProjection(c.qProj) && hasTrainableDenseProjection(c.kProj)
    }

    public static func validateTrainableSupport(for model: Module, prefix: String = "train") throws(BasaltError) {
        let layers = attentionLayers(of: model)
        guard !layers.isEmpty, layers.allSatisfy(canBoundLogits) else {
            throw .invalidRequestStructure(
                details: "\(prefix): muonclip requires trainable Q/K projections (or trainable QK-norms on QK-norm models) to bound attention logits; frozen LoRA/QLoRA adapters cannot bound logits — use muon instead"
            )
        }
    }

    private static func hasTrainableDenseProjection(_ module: any UnaryLayer) -> Bool {
        guard !(module is QuantizedLinear), !(module is LoRALayer) else { return false }
        return module.trainableParameters().flattened().contains { $0.0 == "weight" }
    }

    private static func scaleProjection(_ module: any UnaryLayer, channelScales: [Float]) -> Bool {
        guard channelScales.contains(where: { $0 < 1.0 - 1e-7 }) else { return false }
        let trainable = Dictionary(uniqueKeysWithValues: module.trainableParameters().flattened())
        guard !(module is QuantizedLinear), !(module is LoRALayer),
              let weight = trainable["weight"], weight.ndim == 2, weight.dim(0) == channelScales.count else {
            return false
        }
        var updates: [(String, MLXArray)] = [("weight", weight * MLXArray(channelScales, [channelScales.count, 1]).asType(weight.dtype))]
        if let bias = trainable["bias"], bias.ndim == 1, bias.dim(0) == channelScales.count {
            updates.append(("bias", bias * MLXArray(channelScales, [channelScales.count]).asType(bias.dtype)))
        }
        eval(updates.map(\.1))
        module.update(parameters: ModuleParameters.unflattened(updates))
        return true
    }

    private static func hasTrainableWeight(_ norm: (any UnaryLayer)?) -> Bool {
        norm?.trainableParameters().flattened().contains { $0.0 == "weight" } ?? false
    }

    private static func sharedNormScales(
        minGamma: Float, alpha: Float, qTrainable: Bool, kTrainable: Bool
    ) -> (qShared: Float, kShared: Float) {
        let clampedAlpha = Double(min(max(alpha.isFinite ? alpha : 0.5, 0), 1))
        let gamma = Double(min(max(minGamma, 0), 1))
        if qTrainable && kTrainable {
            return (Float(Foundation.pow(gamma, clampedAlpha)), Float(Foundation.pow(gamma, 1.0 - clampedAlpha)))
        }
        if qTrainable { return (Float(gamma), 1.0) }
        if kTrainable { return (1.0, Float(gamma)) }
        return (1.0, 1.0)
    }

    private static func scaleNorm(_ norm: (any UnaryLayer)?, headScales: [Float], sharedScale: Float) -> Bool {
        guard let norm else { return false }
        let trainable = Dictionary(uniqueKeysWithValues: norm.trainableParameters().flattened())
        guard let weight = trainable["weight"] else { return false }
        if weight.ndim == 2, weight.dim(0) == headScales.count {
            guard headScales.contains(where: { $0 < 1.0 - 1e-7 }) else { return false }
            let scaled = weight * MLXArray(headScales, [headScales.count, 1]).asType(weight.dtype)
            eval(scaled)
            norm.update(parameters: ModuleParameters.unflattened([("weight", scaled)]))
            return true
        }
        guard weight.ndim == 1, sharedScale < 1.0 - 1e-7 else { return false }
        let scaled = weight * MLXArray(sharedScale).asType(weight.dtype)
        eval(scaled)
        norm.update(parameters: ModuleParameters.unflattened([("weight", scaled)]))
        return true
    }
}

extension QKClipController {
    public static func attentionLayers(of model: Module) -> [QKAttentionLayer] {
        let blocks = transformerBlocks(of: model)
        let kvCounts = (model as? KVCacheDimensionProvider)?.kvHeads ?? []
        return blocks.enumerated().compactMap { index, block in
            let modelKV = kvCounts.indices.contains(index) ? kvCounts[index] : nil
            return extractLayer(block: block, layerIndex: index, modelKVHeads: modelKV)
        }
    }

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
        for child in Mirror(reflecting: module).children {
            if child.label == "embedScale" || child.label == "embed_scale", let value = child.value as? Float {
                return value
            }
        }
        return nil
    }

    private static func extractLayer(block: Module, layerIndex: Int, modelKVHeads: Int?) -> QKAttentionLayer? {
        let inputNorm = childUnary(of: block, key: "input_layernorm")
        let attn = childModule(of: block, key: "self_attn") ?? block
        guard let qProj = childUnary(of: attn, key: "q_proj"), let kProj = childUnary(of: attn, key: "k_proj"),
              let qLinear = qProj as? Linear, let kLinear = kProj as? Linear else { return nil }
        let qNorm = childUnary(of: attn, key: "q_norm") ?? childUnary(of: attn, key: "q_layernorm")
        let kNorm = childUnary(of: attn, key: "k_norm") ?? childUnary(of: attn, key: "k_layernorm")
        let rope = (childModule(of: attn, key: "rope")
            ?? childModule(of: attn, key: "rotary_emb")
            ?? childModule(of: attn, key: "rotary")) as? any OffsetLayer
        let oProj = childModule(of: attn, key: "o_proj") as? Linear
        guard let geometry = resolveGeometry(
            attn: attn, qOut: qLinear.shape.0, kOut: kLinear.shape.0,
            oIn: oProj?.shape.1, qNorm: qNorm, modelKVHeads: modelKVHeads
        ) else { return nil }
        let components = QKAttentionLayer.Components(
            inputNorm: inputNorm, qProj: qProj, kProj: kProj, qNorm: qNorm, kNorm: kNorm, rope: rope
        )
        return QKAttentionLayer(layerIndex: layerIndex, components: components, geometry: geometry)
    }

    private static func childModule(of module: Module, key: String) -> Module? {
        if case .value(let child) = module.children()[key] { return child }
        return nil
    }

    private static func childUnary(of module: Module, key: String) -> (any UnaryLayer)? {
        childModule(of: module, key: key) as? any UnaryLayer
    }
}

extension QKClipController {
    private struct MirrorHints {
        var scale: Float?
        var qHeads: Int?
        var kvHeads: Int?
        var headDim: Int?
    }

    private static func resolveGeometry(
        attn: Module, qOut: Int, kOut: Int, oIn: Int?, qNorm: (any UnaryLayer)?, modelKVHeads: Int?
    ) -> QKAttentionLayer.Geometry? {
        let hints = inspectHints(attn)
        guard let headDim = resolveHeadDim(hints: hints, qOut: qOut, kOut: kOut, qNorm: qNorm, modelKVHeads: modelKVHeads),
              headDim > 0, kOut > 0, qOut > 0, kOut % headDim == 0, qOut % headDim == 0 else { return nil }
        let kvHeads = kOut / headDim
        let (qHeads, isGated) = resolveQueryHeads(qOut: qOut, headDim: headDim, oIn: oIn, hintHeads: hints.qHeads)
        guard qHeads > 0, kvHeads > 0, qHeads % kvHeads == 0 else { return nil }
        return QKAttentionLayer.Geometry(
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            scale: hints.scale ?? (1.0 / Float(headDim).squareRoot()), isGatedQuery: isGated
        )
    }

    private static func resolveHeadDim(
        hints: MirrorHints, qOut: Int, kOut: Int, qNorm: (any UnaryLayer)?, modelKVHeads: Int?
    ) -> Int? {
        if let d = hints.headDim, d > 0 { return d }
        if let d = qNorm?.parameters().flattened().first?.1.dim(-1), d > 0 { return d }
        if let kv = hints.kvHeads ?? modelKVHeads, kv > 0, kOut % kv == 0 { return kOut / kv }
        if let qh = hints.qHeads, qh > 0, qOut % qh == 0 { return qOut / qh }
        return nil
    }

    private static func resolveQueryHeads(
        qOut: Int, headDim: Int, oIn: Int?, hintHeads: Int?
    ) -> (qHeads: Int, isGated: Bool) {
        let raw = qOut / headDim
        if let mh = hintHeads, mh > 0, raw == mh * 2 { return (mh, true) }
        if let oIn, oIn > 0, qOut == oIn * 2, oIn % headDim == 0 { return (oIn / headDim, true) }
        return (raw, false)
    }

    private static func inspectHints(_ attn: Module) -> MirrorHints {
        var hints = MirrorHints()
        collectHints(from: attn, into: &hints)
        for child in Mirror(reflecting: attn).children where child.label == "args" || child.label == "config" {
            collectHints(from: child.value, into: &hints)
        }
        return hints
    }

    private static func collectHints(from value: Any, into hints: inout MirrorHints) {
        for child in Mirror(reflecting: value).children {
            guard let label = child.label else { continue }
            switch label {
            case "scale":
                if hints.scale == nil, let v = child.value as? Float { hints.scale = v }
            case "nHeads", "attentionHeads", "numHeads", "numAttentionHeads":
                if hints.qHeads == nil, let v = child.value as? Int { hints.qHeads = v }
            case "nKvHeads", "kvHeads", "numKeyValueHeads", "nKVHeads":
                if hints.kvHeads == nil, let v = child.value as? Int { hints.kvHeads = v }
            case "effectiveHeadDim", "headDim", "headDimensions":
                if hints.headDim == nil, let v = child.value as? Int { hints.headDim = v }
            default:
                break
            }
        }
    }
}
