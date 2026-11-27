import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

final class MLSTMLayer: Module {
    let numHeads: Int
    let qkDim: Int
    let vDim: Int
    let qkHeadDim: Int
    let vHeadDim: Int
    let chunkSize: Int
    let gateSoftCap: Float?
    let eps: Float
    let weightMode: XLSTMWeightMode

    @ModuleInfo(key: "q") var qProj: Linear?
    @ModuleInfo(key: "k") var kProj: Linear?
    @ModuleInfo(key: "v") var vProj: Linear?
    @ModuleInfo(key: "ogate_preact") var oGatePreact: Linear?
    @ModuleInfo(key: "igate_preact") var iGatePreact: Linear?
    @ModuleInfo(key: "fgate_preact") var fGatePreact: Linear?
    @ModuleInfo(key: "qkv_opreact") var qkvOPreact: Linear?
    @ModuleInfo(key: "ifgate_preact") var ifGatePreact: Linear?
    @ModuleInfo(key: "multihead_norm") var multiheadNorm: MultiHeadLayerNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: XLSTMConfiguration) {
        self.numHeads = config.numHeads
        self.qkDim = config.qkDim
        self.vDim = config.vDim
        self.qkHeadDim = config.qkHeadDim
        self.vHeadDim = config.vHeadDim
        self.chunkSize = config.chunkSize
        self.gateSoftCap = config.gateSoftCap
        self.eps = config.eps
        self.weightMode = config.weightMode
        let d = config.hiddenSize
        switch config.weightMode {
        case .single:
            _qProj.wrappedValue = Linear(d, config.qkDim, bias: config.useBias)
            _kProj.wrappedValue = Linear(d, config.qkDim, bias: config.useBias)
            _vProj.wrappedValue = Linear(d, config.vDim, bias: config.useBias)
            _oGatePreact.wrappedValue = Linear(d, config.vDim, bias: config.useBias)
            _iGatePreact.wrappedValue = Linear(d, config.numHeads, bias: true)
            _fGatePreact.wrappedValue = Linear(d, config.numHeads, bias: true)
        case .fused:
            _qkvOPreact.wrappedValue = Linear(d, 2 * config.qkDim + 2 * config.vDim, bias: config.useBias)
            _ifGatePreact.wrappedValue = Linear(d, 2 * config.numHeads, bias: true)
        }
        _multiheadNorm.wrappedValue = MultiHeadLayerNorm(
            numHeads: config.numHeads,
            headDim: config.vHeadDim,
            eps: config.normEps,
            useBias: config.useBias,
            forceFloat32: config.normReductionForceFloat32
        )
        _outProj.wrappedValue = Linear(config.vDim, d, bias: config.useBias)
    }

    func callAsFunction(_ x: MLXArray, cache: XLSTMLayerCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        guard l > 0 else { return MLXArray.zeros(like: x) }
        let proj = project(x)
        let q = proj.q.reshaped(b, l, numHeads, qkHeadDim).transposed(0, 2, 1, 3)
        let k = proj.k.reshaped(b, l, numHeads, qkHeadDim).transposed(0, 2, 1, 3)
        let v = proj.v.reshaped(b, l, numHeads, vHeadDim).transposed(0, 2, 1, 3)
        let iPreact = XLSTMOps.softCap(proj.iRaw.asType(.float32), cap: gateSoftCap).transposed(0, 2, 1)
        let fPreact = XLSTMOps.softCap(proj.fRaw.asType(.float32), cap: gateSoftCap).transposed(0, 2, 1)
        let initial: MLSTMState? = if let c = cache?.cState, let n = cache?.nState, let m = cache?.mState {
            MLSTMState(c: c, n: n, m: m)
        } else {
            nil
        }
        let inputs = MLSTMInputs(q: q, k: k, v: v, iGate: iPreact, fGate: fPreact)
        let (h, nextState) = XLSTMOps.mlstmForward(inputs, state: initial, chunkSize: chunkSize, eps: eps)
        if let cache {
            cache.cState = nextState.c
            cache.nState = nextState.n
            cache.mState = nextState.m
        }
        let hNorm = multiheadNorm(h.transposed(0, 2, 1, 3))
        let hOut = sigmoid(proj.oPreact.asType(.float32)).asType(hNorm.dtype) * hNorm
        return outProj(hOut)
    }

    private func project(_ x: MLXArray) -> (q: MLXArray, k: MLXArray, v: MLXArray, oPreact: MLXArray, iRaw: MLXArray, fRaw: MLXArray) {
        switch weightMode {
        case .single:
            guard let qProj, let kProj, let vProj, let oGatePreact, let iGatePreact, let fGatePreact else {
                preconditionFailure("MLSTMLayer single projections not initialized")
            }
            return (qProj(x), kProj(x), vProj(x), oGatePreact(x), iGatePreact(x), fGatePreact(x))
        case .fused:
            guard let qkvOPreact, let ifGatePreact else {
                preconditionFailure("MLSTMLayer fused projections not initialized")
            }
            let qkvo = qkvOPreact(x)
            let q = qkvo[0..., 0..., ..<qkDim]
            let k = qkvo[0..., 0..., qkDim ..< (2 * qkDim)]
            let v = qkvo[0..., 0..., (2 * qkDim) ..< (2 * qkDim + vDim)]
            let o = qkvo[0..., 0..., (2 * qkDim + vDim)...]
            let ifGates = ifGatePreact(x)
            return (q, k, v, o, ifGates[0..., 0..., ..<numHeads], ifGates[0..., 0..., numHeads...])
        }
    }
}

final class SLSTMLayer: Module {
    let config: XLSTMConfiguration
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "z_proj") var zProj: Linear
    @ModuleInfo(key: "i_proj") var iProj: Linear
    @ModuleInfo(key: "f_proj") var fProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ParameterInfo(key: "r_z") var rZ: MLXArray
    @ParameterInfo(key: "r_i") var rI: MLXArray
    @ParameterInfo(key: "r_f") var rF: MLXArray
    @ParameterInfo(key: "r_o") var rO: MLXArray
    @ModuleInfo(key: "multihead_norm") var multiheadNorm: MultiHeadLayerNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: XLSTMConfiguration) {
        self.config = config
        self.numHeads = config.numHeads
        self.headDim = config.slstmHeadDim
        let d = config.hiddenSize
        let scale = Float(1.0) / Float(config.slstmHeadDim).squareRoot()
        let rShape = [config.numHeads, config.slstmHeadDim, config.slstmHeadDim]
        _zProj.wrappedValue = Linear(d, d, bias: true)
        _iProj.wrappedValue = Linear(d, d, bias: true)
        _fProj.wrappedValue = Linear(d, d, bias: true)
        _oProj.wrappedValue = Linear(d, d, bias: true)
        _rZ.wrappedValue = MLXRandom.normal(rShape, dtype: .float32) * scale
        _rI.wrappedValue = MLXRandom.normal(rShape, dtype: .float32) * scale
        _rF.wrappedValue = MLXRandom.normal(rShape, dtype: .float32) * scale
        _rO.wrappedValue = MLXRandom.normal(rShape, dtype: .float32) * scale
        _multiheadNorm.wrappedValue = MultiHeadLayerNorm(
            numHeads: config.numHeads,
            headDim: config.slstmHeadDim,
            eps: config.normEps,
            useBias: config.useBias,
            forceFloat32: config.normReductionForceFloat32
        )
        _outProj.wrappedValue = Linear(d, d, bias: config.useBias)
    }

    func callAsFunction(_ x: MLXArray, cache: XLSTMLayerCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        guard l > 0 else { return MLXArray.zeros(like: x) }
        let inputs = SLSTMInputs(
            zIn: zProj(x).reshaped(b, l, numHeads, headDim),
            iIn: iProj(x).reshaped(b, l, numHeads, headDim),
            fIn: fProj(x).reshaped(b, l, numHeads, headDim),
            oIn: oProj(x).reshaped(b, l, numHeads, headDim)
        )
        let weights = SLSTMRecurrentWeights(rZ: rZ, rI: rI, rF: rF, rO: rO)
        let initial: SLSTMState? = if let c = cache?.cState, let n = cache?.nState,
                                      let m = cache?.mState, let h = cache?.hState {
            SLSTMState(c: c, n: n, m: m, h: h)
        } else {
            nil
        }
        let (hOut, nextState) = XLSTMOps.slstmRecurrence(inputs, weights: weights, state: initial, config: config)
        if let cache {
            cache.cState = nextState.c
            cache.nState = nextState.n
            cache.mState = nextState.m
            cache.hState = nextState.h
        }
        return outProj(multiheadNorm(hOut))
    }
}

final class XLSTMFFN: Module {
    let weightMode: XLSTMWeightMode
    let upProjDim: Int

    @ModuleInfo(key: "proj_up_gate") var projUpGate: Linear?
    @ModuleInfo(key: "proj_up") var projUp: Linear?
    @ModuleInfo(key: "proj_up_gate_z") var projUpGateZ: Linear?
    @ModuleInfo(key: "proj_down") var projDown: Linear

    init(_ config: XLSTMConfiguration) {
        self.weightMode = config.weightMode
        self.upProjDim = config.ffnHiddenDim
        switch config.weightMode {
        case .single:
            _projUpGate.wrappedValue = Linear(config.hiddenSize, config.ffnHiddenDim, bias: config.useBias)
            _projUp.wrappedValue = Linear(config.hiddenSize, config.ffnHiddenDim, bias: config.useBias)
        case .fused:
            _projUpGateZ.wrappedValue = Linear(config.hiddenSize, 2 * config.ffnHiddenDim, bias: config.useBias)
        }
        _projDown.wrappedValue = Linear(config.ffnHiddenDim, config.hiddenSize, bias: config.useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch weightMode {
        case .single:
            guard let projUpGate, let projUp else {
                preconditionFailure("XLSTMFFN single projections not initialized")
            }
            return projDown(silu(projUpGate(x)) * projUp(x))
        case .fused:
            guard let projUpGateZ else {
                preconditionFailure("XLSTMFFN fused projection not initialized")
            }
            let fused = projUpGateZ(x)
            let gate = fused[0..., 0..., ..<upProjDim]
            let z = fused[0..., 0..., upProjDim...]
            return projDown(silu(gate) * z)
        }
    }
}

final class XLSTMBlock: Module {
    let isSLSTM: Bool
    @ModuleInfo(key: "norm_mlstm") var normMLSTM: XLSTMRMSNorm?
    @ModuleInfo(key: "mlstm_layer") var mlstmLayer: MLSTMLayer?
    @ModuleInfo(key: "norm_slstm") var normSLSTM: XLSTMRMSNorm?
    @ModuleInfo(key: "slstm_layer") var slstmLayer: SLSTMLayer?
    @ModuleInfo(key: "norm_ffn") var normFFN: XLSTMRMSNorm
    @ModuleInfo(key: "ffn") var ffn: XLSTMFFN

    init(_ config: XLSTMConfiguration, layerIdx: Int) {
        self.isSLSTM = config.isSLSTM(layer: layerIdx)
        if isSLSTM {
            _normSLSTM.wrappedValue = XLSTMRMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps,
                useBias: config.useBias, forceFloat32: config.normReductionForceFloat32
            )
            _slstmLayer.wrappedValue = SLSTMLayer(config)
        } else {
            _normMLSTM.wrappedValue = XLSTMRMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps,
                useBias: config.useBias, forceFloat32: config.normReductionForceFloat32
            )
            _mlstmLayer.wrappedValue = MLSTMLayer(config)
        }
        _normFFN.wrappedValue = XLSTMRMSNorm(
            dimensions: config.hiddenSize, eps: config.normEps,
            useBias: config.useBias, forceFloat32: config.normReductionForceFloat32
        )
        _ffn.wrappedValue = XLSTMFFN(config)
    }

    func callAsFunction(_ x: MLXArray, cache: XLSTMLayerCache?) -> MLXArray {
        let recurrentOut: MLXArray
        if isSLSTM {
            guard let normSLSTM, let slstmLayer else { preconditionFailure("XLSTMBlock sLSTM layer not initialized") }
            recurrentOut = slstmLayer(normSLSTM(x), cache: cache)
        } else {
            guard let normMLSTM, let mlstmLayer else { preconditionFailure("XLSTMBlock mLSTM layer not initialized") }
            recurrentOut = mlstmLayer(normMLSTM(x), cache: cache)
        }
        let h = x + recurrentOut
        let out = h + ffn(normFFN(h))
        cache?.advance(by: x.dim(1))
        return out
    }
}

final class XLSTMBackbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "blocks") var blocks: [XLSTMBlock]
    @ModuleInfo(key: "out_norm") var outNorm: XLSTMRMSNorm?

    init(_ config: XLSTMConfiguration) {
        _embeddings.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _blocks.wrappedValue = (0 ..< config.numHiddenLayers).map { XLSTMBlock(config, layerIdx: $0) }
        if config.addOutNorm {
            _outNorm.wrappedValue = XLSTMRMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps,
                useBias: false, forceFloat32: config.normReductionForceFloat32
            )
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings(inputs)
        for (index, block) in blocks.enumerated() {
            let layerCache = cache.flatMap { index < $0.count ? ($0[index] as? XLSTMLayerCache) : nil }
            h = block(h, cache: layerCache)
        }
        return outNorm.map { $0(h) } ?? h
    }
}

public final class XLSTMModel: Module, LLMModel, KVCacheDimensionProvider {
    public static let modelType = "xlstm"
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: XLSTMConfiguration
    @ModuleInfo(key: "backbone") var backbone: XLSTMBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: XLSTMConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(repeating: config.numHeads, count: config.numHiddenLayers)
        _backbone.wrappedValue = XLSTMBackbone(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public convenience init(configurationData data: Data) throws {
        self.init(try JSONDecoder.json5().decode(XLSTMConfiguration.self, from: data))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = backbone(inputs, cache: cache)
        let raw = lmHead.map { $0(hidden) } ?? backbone.embeddings.asLinear(hidden)
        guard let cap = configuration.outputLogitSoftCap else { return raw }
        return XLSTMOps.softCap(raw.asType(.float32), cap: cap)
    }

    public var loraLayers: [Module] { backbone.blocks }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        (0 ..< configuration.numHiddenLayers).map {
            XLSTMLayerCache(kind: configuration.isSLSTM(layer: $0) ? .slstm : .mlstm)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        XLSTMCheckpoint.sanitize(weights, config: configuration)
    }
}

enum XLSTMCheckpoint {
    private static let prefixReplacements: [(String, String)] = [
        ("xlstm.backbone.", "backbone."),
        ("model.embed_tokens.", "backbone.embeddings."),
        ("model.embeddings.", "backbone.embeddings."),
        ("model.layers.", "backbone.blocks."),
        ("model.blocks.", "backbone.blocks."),
        ("model.out_norm.", "backbone.out_norm."),
        ("model.norm.", "backbone.out_norm."),
        ("blocks.", "backbone.blocks."),
        ("embeddings.", "backbone.embeddings."),
        ("out_norm.", "backbone.out_norm."),
    ]

    static func sanitize(_ weights: [String: MLXArray], config: XLSTMConfiguration) -> [String: MLXArray] {
        var normalized: [String: MLXArray] = [:]
        normalized.reserveCapacity(weights.count)
        for (rawKey, value) in weights {
            let key = canonicalKey(rawKey)
            guard !shouldDrop(key, config: config) else { continue }
            normalized[key] = value
        }
        switch config.weightMode {
        case .single:
            splitFusedWeights(&normalized, config: config)
        case .fused:
            fuseSingleWeights(&normalized, config: config)
        }
        return normalized
    }

    private static func canonicalKey(_ raw: String) -> String {
        for (from, to) in prefixReplacements where raw.hasPrefix(from) {
            return to + raw.dropFirst(from.count)
        }
        return raw
    }

    private static func shouldDrop(_ key: String, config: XLSTMConfiguration) -> Bool {
        if config.tieWordEmbeddings && key.hasPrefix("lm_head.") { return true }
        if !config.addOutNorm && key.hasPrefix("backbone.out_norm.") { return true }
        let parts = key.split(separator: ".", maxSplits: 3)
        return parts.count >= 3 && parts[0] == "backbone" && parts[1] == "blocks"
            && (Int(parts[2]).map { $0 >= config.numHiddenLayers } ?? false)
    }

    private static func splitFusedWeights(_ out: inout [String: MLXArray], config: XLSTMConfiguration) {
        for layer in 0 ..< config.numHiddenLayers {
            if !config.isSLSTM(layer: layer) {
                splitMLSTMFused(&out, prefix: "backbone.blocks.\(layer).mlstm_layer.", config: config)
            }
            let fPrefix = "backbone.blocks.\(layer).ffn."
            for suffix in ["weight", "bias"] {
                let key = fPrefix + "proj_up_gate_z." + suffix
                if let ffnFused = out[key], ffnFused.dim(0) == 2 * config.ffnHiddenDim {
                    out[key] = nil
                    out[fPrefix + "proj_up_gate." + suffix] = ffnFused[..<config.ffnHiddenDim]
                    out[fPrefix + "proj_up." + suffix] = ffnFused[config.ffnHiddenDim...]
                }
            }
        }
    }

    private static func splitMLSTMFused(_ out: inout [String: MLXArray], prefix: String, config: XLSTMConfiguration) {
        let (q, v, h) = (config.qkDim, config.vDim, config.numHeads)
        for suffix in ["weight", "bias"] {
            let qkvoKey = prefix + "qkv_opreact." + suffix
            let qkvKey = prefix + "qkv." + suffix
            if let qkvo = out[qkvoKey], qkvo.dim(0) == 2 * q + 2 * v {
                out[qkvoKey] = nil
                out[prefix + "q." + suffix] = qkvo[..<q]
                out[prefix + "k." + suffix] = qkvo[q ..< (2 * q)]
                out[prefix + "v." + suffix] = qkvo[(2 * q) ..< (2 * q + v)]
                out[prefix + "ogate_preact." + suffix] = qkvo[(2 * q + v)...]
            } else if let qkv = out[qkvKey], qkv.dim(0) == 2 * q + v {
                out[qkvKey] = nil
                out[prefix + "q." + suffix] = qkv[..<q]
                out[prefix + "k." + suffix] = qkv[q ..< (2 * q)]
                out[prefix + "v." + suffix] = qkv[(2 * q)...]
            }
            let ifKey = prefix + "ifgate_preact." + suffix
            if let ifGates = out[ifKey], ifGates.dim(0) == 2 * h {
                out[ifKey] = nil
                out[prefix + "igate_preact." + suffix] = ifGates[..<h]
                out[prefix + "fgate_preact." + suffix] = ifGates[h...]
            }
        }
    }

    private static func fuseSingleWeights(_ out: inout [String: MLXArray], config: XLSTMConfiguration) {
        for layer in 0 ..< config.numHiddenLayers {
            let mPrefix = "backbone.blocks.\(layer).mlstm_layer."
            let fPrefix = "backbone.blocks.\(layer).ffn."
            for suffix in ["weight", "bias"] {
                if !config.isSLSTM(layer: layer) {
                    fuseGroup(&out, keys: ["q", "k", "v", "ogate_preact"].map { mPrefix + $0 + "." + suffix }, target: mPrefix + "qkv_opreact." + suffix)
                    fuseGroup(&out, keys: ["qkv", "ogate_preact"].map { mPrefix + $0 + "." + suffix }, target: mPrefix + "qkv_opreact." + suffix)
                    fuseGroup(&out, keys: ["igate_preact", "fgate_preact"].map { mPrefix + $0 + "." + suffix }, target: mPrefix + "ifgate_preact." + suffix)
                }
                fuseGroup(&out, keys: ["proj_up_gate", "proj_up"].map { fPrefix + $0 + "." + suffix }, target: fPrefix + "proj_up_gate_z." + suffix)
            }
        }
    }

    private static func fuseGroup(_ out: inout [String: MLXArray], keys: [String], target: String) {
        guard out[target] == nil, keys.allSatisfy({ out[$0] != nil }) else { return }
        let parts = keys.compactMap { out.removeValue(forKey: $0) }
        out[target] = concatenated(parts, axis: 0)
    }
}
