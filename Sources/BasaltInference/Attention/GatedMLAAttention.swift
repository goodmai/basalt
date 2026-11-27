import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package enum GatedMLAOutputGate: String, Codable, Sendable, Equatable, CaseIterable {
    case none
    case headWise = "head_wise"
    case elementWise = "element_wise"
}

package enum GatedMLAValueGate: String, Codable, Sendable, Equatable, CaseIterable {
    case none
    case latentGate = "latent_gate"
    case valueHeadWise = "value_head_wise"
    case valueElementWise = "value_element_wise"
}

package struct GatedMLAConfiguration: Codable, Sendable, Equatable {
    package var hiddenSize: Int
    package var numHeads: Int
    package var qLoraRank: Int?
    package var kvLoraRank: Int
    package var qkNopeHeadDim: Int
    package var qkRopeHeadDim: Int
    package var vHeadDim: Int
    package var rmsNormEps: Float
    package var ropeTheta: Float
    package var ropeInterleave: Bool
    package var ropeScaling: [String: StringOrNumber]?
    package var maxPositionEmbeddings: Int
    package var useBias: Bool
    package var outputGate: GatedMLAOutputGate
    package var valueGate: GatedMLAValueGate
    package var useOutputNorm: Bool
    package var indexer: DSALightningIndexer.Config?

    package var qkHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }
    package var isNoPE: Bool { qkRopeHeadDim == 0 }

    package init(
        hiddenSize: Int,
        numHeads: Int,
        kvLoraRank: Int,
        qkNopeHeadDim: Int,
        qkRopeHeadDim: Int = 0,
        vHeadDim: Int
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.qLoraRank = nil
        self.kvLoraRank = kvLoraRank
        self.qkNopeHeadDim = qkNopeHeadDim
        self.qkRopeHeadDim = qkRopeHeadDim
        self.vHeadDim = vHeadDim
        self.rmsNormEps = 1e-6
        self.ropeTheta = 10_000.0
        self.ropeInterleave = true
        self.ropeScaling = nil
        self.maxPositionEmbeddings = 131_072
        self.useBias = false
        self.outputGate = .headWise
        self.valueGate = .none
        self.useOutputNorm = false
        self.indexer = nil
    }
}

package final class GatedMLAAttention: Module {
    package let config: GatedMLAConfiguration
    package let scale: Float
    let rope: (any RoPELayer)?

    @ModuleInfo(key: "q_proj") package var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") package var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") package var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") package var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") package var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") package var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") package var kvBProj: Linear
    @ModuleInfo(key: "latent_gate_proj") package var latentGateProj: Linear?
    @ModuleInfo(key: "v_gate_proj") package var vGateProj: Linear?
    @ModuleInfo(key: "o_norm") package var oNorm: RMSNorm?
    @ModuleInfo(key: "g_proj") package var gProj: Linear?
    @ModuleInfo(key: "dense") package var dense: Linear
    @ModuleInfo(key: "indexer") package var indexer: DSALightningIndexer?

    package var usesAbsorbedPath: Bool {
        config.valueGate != .valueElementWise && BailingHybridMLAAbsorption.canAbsorb(kvBProj)
    }

    package init(_ config: GatedMLAConfiguration) {
        precondition(config.kvLoraRank > 0 && config.qkHeadDim > 0 && config.vHeadDim > 0)
        precondition(config.qkRopeHeadDim >= 0 && config.qkRopeHeadDim.isMultiple(of: 2))
        self.config = config
        self.scale = Self.attentionScale(qkHeadDim: config.qkHeadDim, ropeScaling: config.ropeScaling)

        let hidden = config.hiddenSize
        let bias = config.useBias
        if let qLoraRank = config.qLoraRank {
            self._qAProj.wrappedValue = Linear(hidden, qLoraRank, bias: bias)
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: config.rmsNormEps)
            self._qBProj.wrappedValue = Linear(qLoraRank, config.numHeads * config.qkHeadDim, bias: false)
        } else {
            self._qProj.wrappedValue = Linear(hidden, config.numHeads * config.qkHeadDim, bias: false)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(hidden, config.kvLoraRank + config.qkRopeHeadDim, bias: bias)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank, eps: config.rmsNormEps)
        self._kvBProj.wrappedValue = Linear(
            config.kvLoraRank,
            config.numHeads * (config.qkNopeHeadDim + config.vHeadDim),
            bias: false)

        self._latentGateProj.wrappedValue = Self.makeLatentGateProj(config)
        self._vGateProj.wrappedValue = Self.makeValueGateProj(config)
        self._oNorm.wrappedValue = config.useOutputNorm ? RMSNorm(dimensions: config.vHeadDim, eps: config.rmsNormEps) : nil
        self._gProj.wrappedValue = Self.makeOutputGateProj(config)
        self._indexer.wrappedValue = config.indexer.map {
            DSALightningIndexer(hiddenSize: hidden, qInputDim: config.qLoraRank, config: $0)
        }
        self._dense.wrappedValue = Linear(config.numHeads * config.vHeadDim, hidden, bias: bias)
        self.rope = Self.makeRoPE(config)
    }

    package func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal,
        cache: DSALatentKVCache? = nil
    ) -> MLXArray {
        let prepared = prepareAttention(x, mask: mask, cache: cache)
        let attended = attend(prepared.context, mask: mask, qOffset: prepared.qOffset, forceExpanded: false)
        return finalizeOutput(attended, input: x, batch: x.dim(0), length: x.dim(1))
    }

    package func forwardExpandedReference(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal,
        cache: DSALatentKVCache? = nil
    ) -> MLXArray {
        let prepared = prepareAttention(x, mask: mask, cache: cache)
        let attended = attend(prepared.context, mask: mask, qOffset: prepared.qOffset, forceExpanded: true)
        return finalizeOutput(attended, input: x, batch: x.dim(0), length: x.dim(1))
    }

    private static func makeLatentGateProj(_ config: GatedMLAConfiguration) -> Linear? {
        config.valueGate == .latentGate ? Linear(config.hiddenSize, config.kvLoraRank, bias: false) : nil
    }

    private static func makeValueGateProj(_ config: GatedMLAConfiguration) -> Linear? {
        switch config.valueGate {
        case .none, .latentGate: nil
        case .valueHeadWise: Linear(config.hiddenSize, config.numHeads, bias: false)
        case .valueElementWise: Linear(config.hiddenSize, config.numHeads * config.vHeadDim, bias: false)
        }
    }

    private static func makeOutputGateProj(_ config: GatedMLAConfiguration) -> Linear? {
        switch config.outputGate {
        case .none: nil
        case .headWise: Linear(config.hiddenSize, config.numHeads, bias: false)
        case .elementWise: Linear(config.hiddenSize, config.numHeads * config.vHeadDim, bias: false)
        }
    }

    private static func makeRoPE(_ config: GatedMLAConfiguration) -> (any RoPELayer)? {
        guard config.qkRopeHeadDim > 0 else { return nil }
        return initializeRope(
            dims: config.qkRopeHeadDim,
            base: config.ropeTheta,
            traditional: config.ropeInterleave,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    private static func attentionScale(qkHeadDim: Int, ropeScaling: [String: StringOrNumber]?) -> Float {
        var value = pow(Float(qkHeadDim), -0.5)
        guard let ropeScaling,
              let mscaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat(), mscaleAllDim != 0,
              let factor = ropeScaling["factor"]?.asFloat(), factor > 1
        else {
            return value
        }
        let mscale = 0.1 * mscaleAllDim * log(factor) + 1.0
        value *= mscale * mscale
        return value
    }
}

extension GatedMLAAttention {
    struct AttendContext {
        let qNope: MLXArray?
        let qPe: MLXArray?
        let cachedLatent: MLXArray
        let cachedKPe: MLXArray?
        let cachedVGate: MLXArray?
        let topIndices: MLXArray?
    }

    func prepareAttention(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: DSALatentKVCache?
    ) -> (context: AttendContext, qOffset: Int) {
        let batch = x.dim(0)
        let length = x.dim(1)
        let prevOffset = cache?.offset ?? 0
        let ropeOffset = cache?.ropeOffset
        let (qNope, qPe, qLatent) = projectQuery(x, batch: batch, length: length, offset: ropeOffset)
        let (kvLatent, kPe) = projectLatentKV(x, batch: batch, length: length, offset: ropeOffset)
        let vGate = projectValueGate(x, batch: batch, length: length)
        let topIndices: MLXArray?
        if let indexer {
            let isCausal = Self.isCausalMask(mask)
            topIndices = indexer.selectTopK(
                x: x, qInput: qLatent, cache: cache, causal: isCausal, qOffset: prevOffset
            ).indices
        } else {
            topIndices = nil
        }
        let (cachedLatent, cachedKPe) = cache?.updateLatent(latent: kvLatent, kPe: kPe) ?? (kvLatent, kPe)
        let cachedVGate = vGate.map { gate in cache?.updateValueGate(gate) ?? gate }
        let context = AttendContext(
            qNope: qNope, qPe: qPe, cachedLatent: cachedLatent,
            cachedKPe: cachedKPe, cachedVGate: cachedVGate, topIndices: topIndices)
        return (context, prevOffset)
    }

    func projectQuery(
        _ x: MLXArray,
        batch: Int,
        length: Int,
        offset: RoPEOffset?
    ) -> (qNope: MLXArray?, qPe: MLXArray?, qLatent: MLXArray?) {
        let rawQ: MLXArray
        let qLatent: MLXArray?
        if let qAProj, let qALayerNorm, let qBProj {
            let normed = qALayerNorm(qAProj(x))
            qLatent = normed
            rawQ = qBProj(normed)
        } else if let qProj {
            qLatent = nil
            rawQ = qProj(x)
        } else {
            preconditionFailure("GatedMLAAttention query projection not initialized")
        }
        let shaped = rawQ.reshaped(batch, length, config.numHeads, config.qkHeadDim).transposed(0, 2, 1, 3)
        if config.qkRopeHeadDim == 0 {
            return (shaped, nil, qLatent)
        }
        guard let rope else { preconditionFailure("GatedMLAAttention RoPE missing for positive qkRopeHeadDim") }
        if config.qkNopeHeadDim == 0 {
            return (nil, applyRotaryPosition(rope, to: shaped, offset: offset), qLatent)
        }
        let parts = split(shaped, indices: [config.qkNopeHeadDim], axis: -1)
        return (parts[0], applyRotaryPosition(rope, to: parts[1], offset: offset), qLatent)
    }

    func projectLatentKV(
        _ x: MLXArray,
        batch: Int,
        length: Int,
        offset: RoPEOffset?
    ) -> (kvLatent: MLXArray, kPe: MLXArray?) {
        let compressed = kvAProjWithMqa(x)
        let rawLatent: MLXArray
        let kPe: MLXArray?
        if config.qkRopeHeadDim == 0 {
            rawLatent = compressed
            kPe = nil
        } else {
            guard let rope else { preconditionFailure("GatedMLAAttention RoPE missing for positive qkRopeHeadDim") }
            let parts = split(compressed, indices: [config.kvLoraRank], axis: -1)
            rawLatent = parts[0]
            let kPeRaw = parts[1].reshaped(batch, length, 1, config.qkRopeHeadDim).transposed(0, 2, 1, 3)
            kPe = applyRotaryPosition(rope, to: kPeRaw, offset: offset)
        }
        var kvLatent = kvALayerNorm(rawLatent).reshaped(batch, length, 1, config.kvLoraRank).transposed(0, 2, 1, 3)
        if let latentGateProj {
            let gate = sigmoid(latentGateProj(x).asType(.float32)).asType(kvLatent.dtype)
                .reshaped(batch, length, 1, config.kvLoraRank).transposed(0, 2, 1, 3)
            kvLatent = kvLatent * gate
        }
        return (kvLatent, kPe)
    }

    func projectValueGate(_ x: MLXArray, batch: Int, length: Int) -> MLXArray? {
        guard let vGateProj else { return nil }
        let raw = sigmoid(vGateProj(x).asType(.float32)).asType(x.dtype)
        switch config.valueGate {
        case .none, .latentGate:
            return nil
        case .valueHeadWise:
            return raw.reshaped(batch, length, config.numHeads, 1).transposed(0, 2, 1, 3)
        case .valueElementWise:
            return raw.reshaped(batch, length, config.numHeads, config.vHeadDim).transposed(0, 2, 1, 3)
        }
    }

    func finalizeOutput(_ attended: MLXArray, input x: MLXArray, batch: Int, length: Int) -> MLXArray {
        var merged = attended.transposed(0, 2, 1, 3)
        if let oNorm {
            merged = oNorm(merged)
        }
        if let gProj {
            let gate = sigmoid(gProj(x).asType(.float32)).asType(merged.dtype)
            switch config.outputGate {
            case .none: break
            case .headWise:
                merged = merged * gate.expandedDimensions(axis: -1)
            case .elementWise:
                merged = merged * gate.reshaped(batch, length, config.numHeads, config.vHeadDim)
            }
        }
        return dense(merged.reshaped(batch, length, -1))
    }
}
