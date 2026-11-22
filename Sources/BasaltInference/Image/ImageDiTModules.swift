import Foundation
import MLX
import MLXNN
import BasaltCore

package struct ImageDiTConfig: Sendable, Codable, Equatable {
    package var modelID: String
    package var hiddenSize: Int
    package var numHeads: Int
    package var numLayers: Int
    package var latentChannels: Int
    package var patchSize: Int
    package var vaeScale: Int
    package var decoderBaseChannels: Int
    package var textEmbedDim: Int
    package var quantizationBits: Int
    package var initSeed: UInt64

    package var headDim: Int { hiddenSize / numHeads }

    package init(
        modelID: String = "black-forest-labs/FLUX.2-klein-4B",
        hiddenSize: Int = 64,
        numHeads: Int = 4,
        numLayers: Int = 2,
        latentChannels: Int = 16,
        patchSize: Int = 2
    ) {
        let heads = max(1, numHeads)
        let rawHead = max(4, hiddenSize / heads)
        let alignedHead = ((rawHead + 3) / 4) * 4
        self.modelID = modelID
        self.numHeads = heads
        self.hiddenSize = heads * alignedHead
        self.numLayers = max(1, numLayers)
        self.latentChannels = max(4, latentChannels)
        self.patchSize = max(1, patchSize)
        self.vaeScale = 8
        self.decoderBaseChannels = 24
        self.textEmbedDim = 256
        self.quantizationBits = 8
        self.initSeed = 517
    }

    package static func tiny(modelID: String = "tiny-dit", hiddenSize: Int = 64) -> ImageDiTConfig {
        ImageDiTConfig(
            modelID: modelID,
            hiddenSize: hiddenSize,
            numHeads: 4,
            numLayers: 2,
            latentChannels: 16,
            patchSize: 2
        )
    }
}

package struct DiTStepContext {
    package let condVec: MLXArray
    package let ropeCos: MLXArray
    package let ropeSin: MLXArray
    package let timestep: Float
    package let guidance: Float

    package init(
        condVec: MLXArray,
        rope: (cos: MLXArray, sin: MLXArray),
        timestep: Float,
        guidance: Float
    ) {
        self.condVec = condVec
        self.ropeCos = rope.cos
        self.ropeSin = rope.sin
        self.timestep = timestep
        self.guidance = guidance
    }
}

package enum RoPE2D {
    package static func frequencies(
        gridH: Int,
        gridW: Int,
        headDim: Int,
        theta: Float = 10_000.0
    ) -> (cos: MLXArray, sin: MLXArray) {
        let quarter = max(1, headDim / 4)
        let exponents = MLXArray(0 ..< quarter).asType(.float32) / Float(quarter)
        let invFreq = 1.0 / pow(MLXArray(theta), exponents)
        let rows = MLXArray(0 ..< max(1, gridH)).asType(.float32)
        let cols = MLXArray(0 ..< max(1, gridW)).asType(.float32)
        let rowAngles = outer(rows, invFreq)
        let colAngles = outer(cols, invFreq)
        let rowGrid = broadcast(rowAngles.expandedDimensions(axis: 1), to: [gridH, gridW, quarter])
        let colGrid = broadcast(colAngles.expandedDimensions(axis: 0), to: [gridH, gridW, quarter])
        let paired = concatenated([rowGrid, colGrid, rowGrid, colGrid], axis: -1)
            .reshaped(1, 1, gridH * gridW, quarter * 4)
        return (MLX.cos(paired), MLX.sin(paired))
    }

    package static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let first = x[0..., 0..., 0..., ..<half]
        let second = x[0..., 0..., 0..., half...]
        let rotated = concatenated([-second, first], axis: -1)
        return x * cos + rotated * sin
    }
}

package struct AdaLNGates {
    package let shiftMSA: MLXArray
    package let scaleMSA: MLXArray
    package let gateMSA: MLXArray
    package let shiftMLP: MLXArray
    package let scaleMLP: MLXArray
    package let gateMLP: MLXArray
}

final class AdaLNZeroModulation: Module {
    let dimensions: Int
    @ModuleInfo(key: "time_in") var timeIn: Linear
    @ModuleInfo(key: "time_out") var timeOut: Linear
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    init(dimensions: Int) {
        let d = max(16, dimensions)
        self.dimensions = d
        self._timeIn.wrappedValue = Linear(32, d)
        self._timeOut.wrappedValue = Linear(d, d)
        self._condProj.wrappedValue = Linear(d, d)
        self._gateProj.wrappedValue = Linear(d, d * 6)
    }

    func gates(timestep: Float, guidance: Float, condVec: MLXArray) -> AdaLNGates {
        let tEmb = Self.sinusoidalPair(timestep: timestep, guidance: guidance)
        let h = MLXNN.silu(timeOut(MLXNN.silu(timeIn(tEmb))) + condProj(condVec))
        let raw = gateProj(h).expandedDimensions(axis: 1)
        let d = dimensions
        return AdaLNGates(
            shiftMSA: raw[0..., 0..., ..<d],
            scaleMSA: 0.25 * tanh(raw[0..., 0..., d ..< (2 * d)]),
            gateMSA: 0.5 + 0.5 * tanh(raw[0..., 0..., (2 * d) ..< (3 * d)]),
            shiftMLP: raw[0..., 0..., (3 * d) ..< (4 * d)],
            scaleMLP: 0.25 * tanh(raw[0..., 0..., (4 * d) ..< (5 * d)]),
            gateMLP: 0.5 + 0.5 * tanh(raw[0..., 0..., (5 * d)...])
        )
    }

    func modulatePerToken(
        _ normed: MLXArray,
        tokenTimesteps: MLXArray,
        guidance: Float,
        condVec: MLXArray
    ) -> MLXArray {
        let freqs = MLXArray((0 ..< 8).map { Float(pow(2.0, Double($0))) * Float.pi }).reshaped(1, 1, 8)
        let tExp = tokenTimesteps.expandedDimensions(axis: -1)
        let gArr = broadcast(MLXArray(guidance).reshaped(1, 1, 1), to: [tExp.dim(0), tExp.dim(1), 8])
        let b = tExp.dim(0)
        let n = tExp.dim(1)
        let tPair = stacked([MLX.sin(tExp * freqs), MLX.cos(tExp * freqs)], axis: -1).reshaped(b, n, 16)
        let gPair = stacked([MLX.sin(gArr * 0.25 * freqs), MLX.cos(gArr * 0.25 * freqs)], axis: -1).reshaped(b, n, 16)
        let emb = concatenated([tPair, gPair], axis: -1)
        let condExpanded = condProj(condVec).expandedDimensions(axis: 1)
        let h = MLXNN.silu(timeOut(MLXNN.silu(timeIn(emb))) + condExpanded)
        let raw = gateProj(h)
        let d = dimensions
        let shift = raw[0..., 0..., ..<d]
        let scale = 0.25 * tanh(raw[0..., 0..., d ..< (2 * d)])
        return normed * (1.0 + scale) + shift
    }

    static func sinusoidalPair(timestep: Float, guidance: Float) -> MLXArray {
        var vals: [Float] = []
        vals.reserveCapacity(32)
        for idx in 0 ..< 8 {
            let freq = Float(pow(2.0, Double(idx))) * Float.pi
            vals.append(sin(timestep * freq))
            vals.append(cos(timestep * freq))
        }
        for idx in 0 ..< 8 {
            let freq = Float(pow(2.0, Double(idx))) * 0.25 * Float.pi
            vals.append(sin(guidance * freq))
            vals.append(cos(guidance * freq))
        }
        return MLXArray(vals).reshaped(1, 32)
    }
}

final class JointDiTBlock: Module {
    let dimensions: Int
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "img_adaln") var imgAdaLN: AdaLNZeroModulation
    @ModuleInfo(key: "txt_adaln") var txtAdaLN: AdaLNZeroModulation
    @ModuleInfo(key: "img_norm1") var imgNorm1: LayerNorm
    @ModuleInfo(key: "img_norm2") var imgNorm2: LayerNorm
    @ModuleInfo(key: "txt_norm1") var txtNorm1: LayerNorm
    @ModuleInfo(key: "txt_norm2") var txtNorm2: LayerNorm
    @ModuleInfo(key: "img_qkv") var imgQKV: Linear
    @ModuleInfo(key: "txt_qkv") var txtQKV: Linear
    @ModuleInfo(key: "img_out") var imgOut: Linear
    @ModuleInfo(key: "txt_out") var txtOut: Linear
    @ModuleInfo(key: "img_mlp1") var imgMLP1: Linear
    @ModuleInfo(key: "img_mlp2") var imgMLP2: Linear
    @ModuleInfo(key: "txt_mlp1") var txtMLP1: Linear
    @ModuleInfo(key: "txt_mlp2") var txtMLP2: Linear

    init(dimensions: Int, heads: Int) {
        let d = max(16, dimensions)
        let h = max(1, heads)
        self.dimensions = d
        self.heads = h
        self.scale = 1.0 / Float(d / h).squareRoot()
        self._imgAdaLN.wrappedValue = AdaLNZeroModulation(dimensions: d)
        self._txtAdaLN.wrappedValue = AdaLNZeroModulation(dimensions: d)
        self._imgNorm1.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._imgNorm2.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._txtNorm1.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._txtNorm2.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._imgQKV.wrappedValue = Linear(d, d * 3)
        self._txtQKV.wrappedValue = Linear(d, d * 3)
        self._imgOut.wrappedValue = Linear(d, d)
        self._txtOut.wrappedValue = Linear(d, d)
        self._imgMLP1.wrappedValue = Linear(d, d * 2)
        self._imgMLP2.wrappedValue = Linear(d * 2, d)
        self._txtMLP1.wrappedValue = Linear(d, d * 2)
        self._txtMLP2.wrappedValue = Linear(d * 2, d)
    }

    func callAsFunction(
        img: MLXArray,
        txt: MLXArray,
        stepCtx: DiTStepContext,
        refKV: CachedReferenceLayerKV? = nil
    ) -> (img: MLXArray, txt: MLXArray) {
        let gImg = imgAdaLN.gates(timestep: stepCtx.timestep, guidance: stepCtx.guidance, condVec: stepCtx.condVec)
        let gTxt = txtAdaLN.gates(timestep: stepCtx.timestep, guidance: stepCtx.guidance, condVec: stepCtx.condVec)
        let mImg = imgNorm1(img) * (1.0 + gImg.scaleMSA) + gImg.shiftMSA
        let mTxt = txtNorm1(txt) * (1.0 + gTxt.scaleMSA) + gTxt.shiftMSA
        let (attnImg, attnTxt) = jointAttention(mImg: mImg, mTxt: mTxt, stepCtx: stepCtx, refKV: refKV)
        let hImg = img + gImg.gateMSA * imgOut(attnImg)
        let hTxt = txt + gTxt.gateMSA * txtOut(attnTxt)
        let fImg = imgNorm2(hImg) * (1.0 + gImg.scaleMLP) + gImg.shiftMLP
        let fTxt = txtNorm2(hTxt) * (1.0 + gTxt.scaleMLP) + gTxt.shiftMLP
        let nextImg = hImg + gImg.gateMLP * imgMLP2(geluApproximate(imgMLP1(fImg)))
        let nextTxt = hTxt + gTxt.gateMLP * txtMLP2(geluApproximate(txtMLP1(fTxt)))
        return (nextImg, nextTxt)
    }

    func projectReferenceTower(
        refImg: MLXArray,
        stepCtx: DiTStepContext
    ) -> (updatedRef: MLXArray, kv: CachedReferenceLayerKV) {
        let gImg = imgAdaLN.gates(timestep: stepCtx.timestep, guidance: stepCtx.guidance, condVec: stepCtx.condVec)
        let mRef = imgNorm1(refImg) * (1.0 + gImg.scaleMSA) + gImg.shiftMSA
        let b = mRef.dim(0)
        let nRef = mRef.dim(1)
        let headDim = dimensions / heads
        let parts = imgQKV(mRef).reshaped(b, nRef, 3, heads, headDim).transposed(2, 0, 3, 1, 4)
        let qRef = RoPE2D.apply(parts[0], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)
        let kRef = RoPE2D.apply(parts[1], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)
        let vRef = parts[2]
        let attended = MLXFast.scaledDotProductAttention(queries: qRef, keys: kRef, values: vRef, scale: scale, mask: .none)
            .transposed(0, 2, 1, 3)
            .reshaped(b, nRef, dimensions)
        let hRef = refImg + gImg.gateMSA * imgOut(attended)
        let fRef = imgNorm2(hRef) * (1.0 + gImg.scaleMLP) + gImg.shiftMLP
        let nextRef = hRef + gImg.gateMLP * imgMLP2(geluApproximate(imgMLP1(fRef)))
        return (nextRef, CachedReferenceLayerKV(keys: kRef, values: vRef))
    }

    private func jointAttention(
        mImg: MLXArray,
        mTxt: MLXArray,
        stepCtx: DiTStepContext,
        refKV: CachedReferenceLayerKV?
    ) -> (MLXArray, MLXArray) {
        let b = mImg.dim(0)
        let nImg = mImg.dim(1)
        let nTxt = mTxt.dim(1)
        let headDim = dimensions / heads
        let iParts = imgQKV(mImg).reshaped(b, nImg, 3, heads, headDim).transposed(2, 0, 3, 1, 4)
        let tParts = txtQKV(mTxt).reshaped(b, nTxt, 3, heads, headDim).transposed(2, 0, 3, 1, 4)
        let qImg = RoPE2D.apply(iParts[0], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)
        let kImg = RoPE2D.apply(iParts[1], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)
        let q = concatenated([tParts[0], qImg], axis: 2)
        let k = if let refKV {
            concatenated([refKV.keys, tParts[1], kImg], axis: 2)
        } else {
            concatenated([tParts[1], kImg], axis: 2)
        }
        let v = if let refKV {
            concatenated([refKV.values, tParts[2], iParts[2]], axis: 2)
        } else {
            concatenated([tParts[2], iParts[2]], axis: 2)
        }
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
            .transposed(0, 2, 1, 3)
            .reshaped(b, nTxt + nImg, dimensions)
        return (attended[0..., nTxt..., 0...], attended[0..., ..<nTxt, 0...])
    }
}

final class SingleStreamS3DiTBlock: Module {
    let dimensions: Int
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "adaln") var adaLN: AdaLNZeroModulation
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "mlp_gate") var mlpGate: Linear
    @ModuleInfo(key: "mlp_up") var mlpUp: Linear
    @ModuleInfo(key: "fused_out") var fusedOut: Linear

    init(dimensions: Int, heads: Int) {
        let d = max(16, dimensions)
        let h = max(1, heads)
        self.dimensions = d
        self.heads = h
        self.scale = 1.0 / Float(d / h).squareRoot()
        self._adaLN.wrappedValue = AdaLNZeroModulation(dimensions: d)
        self._norm.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._qkv.wrappedValue = Linear(d, d * 3)
        self._mlpGate.wrappedValue = Linear(d, d * 2)
        self._mlpUp.wrappedValue = Linear(d, d * 2)
        self._fusedOut.wrappedValue = Linear(d * 3, d)
    }

    func callAsFunction(
        condTokens: MLXArray,
        imgTokens: MLXArray,
        stepCtx: DiTStepContext,
        refKV: CachedReferenceLayerKV? = nil
    ) -> (cond: MLXArray, img: MLXArray) {
        let nCond = condTokens.dim(1)
        let nImg = imgTokens.dim(1)
        let unified = concatenated([condTokens, imgTokens], axis: 1)
        let gates = adaLN.gates(timestep: stepCtx.timestep, guidance: stepCtx.guidance, condVec: stepCtx.condVec)
        let mod = norm(unified) * (1.0 + gates.scaleMSA) + gates.shiftMSA
        let b = unified.dim(0)
        let total = nCond + nImg
        let headDim = dimensions / heads
        let parts = qkv(mod).reshaped(b, total, 3, heads, headDim).transposed(2, 0, 3, 1, 4)
        let qCond = parts[0][0..., 0..., ..<nCond, 0...]
        let kCond = parts[1][0..., 0..., ..<nCond, 0...]
        let qImg = RoPE2D.apply(parts[0][0..., 0..., nCond..., 0...], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)
        let kImg = RoPE2D.apply(parts[1][0..., 0..., nCond..., 0...], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)
        let q = concatenated([qCond, qImg], axis: 2)
        let k = if let refKV {
            concatenated([refKV.keys, kCond, kImg], axis: 2)
        } else {
            concatenated([kCond, kImg], axis: 2)
        }
        let v = if let refKV {
            concatenated([refKV.values, parts[2]], axis: 2)
        } else {
            parts[2]
        }
        let attn = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
            .transposed(0, 2, 1, 3)
            .reshaped(b, total, dimensions)
        let swiglu = MLXNN.silu(mlpGate(mod)) * mlpUp(mod)
        let updated = unified + gates.gateMSA * fusedOut(concatenated([attn, swiglu], axis: -1))
        return (updated[0..., ..<nCond, 0...], updated[0..., nCond..., 0...])
    }

    func projectReferenceTower(
        refTokens: MLXArray,
        stepCtx: DiTStepContext
    ) -> (updatedRef: MLXArray, kv: CachedReferenceLayerKV) {
        let b = refTokens.dim(0)
        let nRef = refTokens.dim(1)
        let gates = adaLN.gates(timestep: stepCtx.timestep, guidance: stepCtx.guidance, condVec: stepCtx.condVec)
        let mod = norm(refTokens) * (1.0 + gates.scaleMSA) + gates.shiftMSA
        let headDim = dimensions / heads
        let parts = qkv(mod).reshaped(b, nRef, 3, heads, headDim).transposed(2, 0, 3, 1, 4)
        let qRef = RoPE2D.apply(parts[0], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)
        let kRef = RoPE2D.apply(parts[1], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)
        let vRef = parts[2]
        let attn = MLXFast.scaledDotProductAttention(queries: qRef, keys: kRef, values: vRef, scale: scale, mask: .none)
            .transposed(0, 2, 1, 3)
            .reshaped(b, nRef, dimensions)
        let swiglu = MLXNN.silu(mlpGate(mod)) * mlpUp(mod)
        let updated = refTokens + gates.gateMSA * fusedOut(concatenated([attn, swiglu], axis: -1))
        return (updated, CachedReferenceLayerKV(keys: kRef, values: vRef))
    }
}

final class LinearAttentionSanaBlock: Module {
    let dimensions: Int
    let heads: Int
    @ModuleInfo(key: "adaln") var adaLN: AdaLNZeroModulation
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "cross_q") var crossQ: Linear
    @ModuleInfo(key: "cross_kv") var crossKV: Linear
    @ModuleInfo(key: "cross_out") var crossOut: Linear
    @ModuleInfo(key: "mix_conv") var mixConv: Conv1d
    @ModuleInfo(key: "mlp_out") var mlpOut: Linear

    init(dimensions: Int, heads: Int) {
        let d = max(16, dimensions)
        let h = max(1, heads)
        self.dimensions = d
        self.heads = h
        self._adaLN.wrappedValue = AdaLNZeroModulation(dimensions: d)
        self._norm1.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._norm2.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._qkv.wrappedValue = Linear(d, d * 3, bias: false)
        self._outProj.wrappedValue = Linear(d, d)
        self._crossQ.wrappedValue = Linear(d, d, bias: false)
        self._crossKV.wrappedValue = Linear(d, d * 2, bias: false)
        self._crossOut.wrappedValue = Linear(d, d)
        self._mixConv.wrappedValue = Conv1d(inputChannels: d, outputChannels: d * 2, kernelSize: 3, padding: 1)
        self._mlpOut.wrappedValue = Linear(d * 2, d)
    }

    func reluLinearAttention(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        let headDim = dimensions / heads
        let parts = qkv(x).reshaped(b, n, 3, heads, headDim).transposed(2, 0, 3, 1, 4)
        let phiQ = MLXNN.relu(parts[0]) + 1e-4
        let phiK = MLXNN.relu(parts[1]) + 1e-4
        let val = parts[2]
        let kv = matmul(phiK.transposed(0, 1, 3, 2), val)
        let kSum = sum(phiK, axis: 2, keepDims: true)
        let numer = matmul(phiQ, kv)
        let denom = maximum(sum(phiQ * kSum, axis: -1, keepDims: true), MLXArray(Float(1e-4)))
        let agg = (numer / denom).transposed(0, 2, 1, 3).reshaped(b, n, dimensions)
        return outProj(agg)
    }

    func callAsFunction(img: MLXArray, condTokens: MLXArray, stepCtx: DiTStepContext) -> MLXArray {
        let gates = adaLN.gates(timestep: stepCtx.timestep, guidance: stepCtx.guidance, condVec: stepCtx.condVec)
        let mod1 = norm1(img) * (1.0 + gates.scaleMSA) + gates.shiftMSA
        let h1 = img + gates.gateMSA * reluLinearAttention(mod1)
        let b = h1.dim(0)
        let nImg = h1.dim(1)
        let nCond = condTokens.dim(1)
        let headDim = dimensions / heads
        let q = crossQ(h1).reshaped(b, nImg, heads, headDim).transposed(0, 2, 1, 3)
        let kvParts = crossKV(condTokens).reshaped(b, nCond, 2, heads, headDim).transposed(2, 0, 3, 1, 4)
        let cScale = 1.0 / Float(headDim).squareRoot()
        let cAttn = MLXFast.scaledDotProductAttention(
            queries: q, keys: kvParts[0], values: kvParts[1], scale: cScale, mask: .none
        ).transposed(0, 2, 1, 3).reshaped(b, nImg, dimensions)
        let h2 = h1 + crossOut(cAttn)
        let mod2 = norm2(h2) * (1.0 + gates.scaleMLP) + gates.shiftMLP
        return h2 + gates.gateMLP * mlpOut(MLXNN.silu(mixConv(mod2)))
    }
}

final class TernaryBitNetLinear: Module {
    let inFeatures: Int
    let outFeatures: Int
    public let weight: MLXArray
    public let bias: MLXArray?
    @ModuleInfo(key: "act_norm") var actNorm: RMSNorm

    init(_ inFeatures: Int, _ outFeatures: Int, bias: Bool = true) {
        self.inFeatures = max(1, inFeatures)
        self.outFeatures = max(1, outFeatures)
        let bound = 1.0 / Float(max(1, inFeatures)).squareRoot()
        self.weight = MLXRandom.uniform(low: -bound, high: bound, [self.outFeatures, self.inFeatures])
        self.bias = bias ? MLXArray.zeros([self.outFeatures]) : nil
        self._actNorm.wrappedValue = RMSNorm(dimensions: self.inFeatures)
    }

    func quantizedTernaryWeights() -> (codes: MLXArray, gamma: MLXArray) {
        let gamma = maximum(mean(abs(weight)), MLXArray(Float(1e-6)))
        let codes = clip(round(weight / gamma), min: -1.0, max: 1.0)
        return (codes, gamma)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = actNorm(x)
        let absMax = maximum(max(abs(normed), axis: -1, keepDims: true), MLXArray(Float(1e-6)))
        let qScale = 127.0 / absMax
        let xQuant = clip(round(normed * qScale), min: -128.0, max: 127.0) / qScale
        let (codes, gamma) = quantizedTernaryWeights()
        var out = matmul(xQuant, codes.transposed(1, 0)) * gamma
        if let bias {
            out = out + bias
        }
        return out
    }
}

final class PatchVAECodec: Module {
    let latentChannels: Int
    let baseChannels: Int
    let decoderChannels: Int
    @ModuleInfo(key: "enc_in") var encIn: Conv2d
    @ModuleInfo(key: "enc_down1") var encDown1: Conv2d
    @ModuleInfo(key: "enc_down2") var encDown2: Conv2d
    @ModuleInfo(key: "enc_down3") var encDown3: Conv2d
    @ModuleInfo(key: "dec_in") var decIn: Conv2d
    @ModuleInfo(key: "dec_up1") var decUp1: Conv2d
    @ModuleInfo(key: "dec_up2") var decUp2: Conv2d
    @ModuleInfo(key: "dec_up3") var decUp3: Conv2d
    @ModuleInfo(key: "dec_rgba") var decRGBA: Conv2d

    init(latentChannels: Int = 16, baseChannels: Int = 24, decoderChannels: Int? = nil) {
        let lat = max(4, latentChannels)
        let ch = max(16, baseChannels)
        let decCh = max(12, decoderChannels ?? ch)
        self.latentChannels = lat
        self.baseChannels = ch
        self.decoderChannels = decCh
        self._encIn.wrappedValue = Conv2d(inputChannels: 4, outputChannels: ch, kernelSize: 3, padding: 1)
        self._encDown1.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 3, stride: 2, padding: 1)
        self._encDown2.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 3, stride: 2, padding: 1)
        self._encDown3.wrappedValue = Conv2d(inputChannels: ch, outputChannels: lat, kernelSize: 3, stride: 2, padding: 1)
        self._decIn.wrappedValue = Conv2d(inputChannels: lat, outputChannels: decCh, kernelSize: 3, padding: 1)
        self._decUp1.wrappedValue = Conv2d(inputChannels: decCh, outputChannels: decCh, kernelSize: 3, padding: 1)
        self._decUp2.wrappedValue = Conv2d(inputChannels: decCh, outputChannels: decCh, kernelSize: 3, padding: 1)
        self._decUp3.wrappedValue = Conv2d(inputChannels: decCh, outputChannels: decCh, kernelSize: 3, padding: 1)
        self._decRGBA.wrappedValue = Conv2d(inputChannels: decCh, outputChannels: 4, kernelSize: 3, padding: 1)
    }

    func encode(_ image: MLXArray) -> MLXArray {
        let nhwc = image.ndim == 3 ? image[.newAxis] : image
        let rgba: MLXArray
        if nhwc.dim(3) == 4 {
            rgba = nhwc
        } else {
            let alpha = MLXArray.ones([nhwc.dim(0), nhwc.dim(1), nhwc.dim(2), 1], dtype: nhwc.dtype)
            rgba = concatenated([nhwc[0..., 0..., 0..., ..<3], alpha], axis: 3)
        }
        let centered = rgba * 2.0 - 1.0
        let h0 = MLXNN.silu(encIn(centered))
        let h1 = MLXNN.silu(encDown1(h0))
        let h2 = MLXNN.silu(encDown2(h1))
        return encDown3(h2)
    }

    func decode(_ latents: MLXArray, includeAlpha: Bool, tileLatentSize: Int = 32) -> MLXArray {
        let nhwc = latents.ndim == 3 ? latents[.newAxis] : latents
        let rawRGBA: MLXArray
        if nhwc.dim(1) <= tileLatentSize && nhwc.dim(2) <= tileLatentSize {
            rawRGBA = decodeRawTile(nhwc)
        } else {
            rawRGBA = decodeTiled(nhwc, tileSize: max(8, tileLatentSize))
        }
        let rgb = sigmoid(rawRGBA[0..., 0..., 0..., ..<3])
        guard includeAlpha else { return rgb }
        let prior = Self.subjectMattePrior(height: rawRGBA.dim(1), width: rawRGBA.dim(2))
        let alpha = sigmoid(rawRGBA[0..., 0..., 0..., 3...] + prior)
        return concatenated([rgb, alpha], axis: 3)
    }

    private func decodeRawTile(_ z: MLXArray) -> MLXArray {
        let up = Upsample(scaleFactor: 2.0, mode: .nearest)
        let h0 = MLXNN.silu(decIn(z))
        let h1 = MLXNN.silu(decUp1(up(h0)))
        let h2 = MLXNN.silu(decUp2(up(h1)))
        let h3 = MLXNN.silu(decUp3(up(h2)))
        return decRGBA(h3)
    }

    private func decodeTiled(_ z: MLXArray, tileSize: Int) -> MLXArray {
        let hLat = z.dim(1)
        let wLat = z.dim(2)
        var rowStrips: [MLXArray] = []
        var r = 0
        while r < hLat {
            let rEnd = min(hLat, r + tileSize)
            var colTiles: [MLXArray] = []
            var c = 0
            while c < wLat {
                let cEnd = min(wLat, c + tileSize)
                colTiles.append(decodeRawTile(z[0..., r ..< rEnd, c ..< cEnd, 0...]))
                c = cEnd
            }
            rowStrips.append(concatenated(colTiles, axis: 2))
            r = rEnd
        }
        return concatenated(rowStrips, axis: 1)
    }

    static func subjectMattePrior(height: Int, width: Int) -> MLXArray {
        let ys = (MLXArray(0 ..< height).asType(.float32) + 0.5) / Float(max(1, height)) * 2.0 - 1.0
        let xs = (MLXArray(0 ..< width).asType(.float32) + 0.5) / Float(max(1, width)) * 2.0 - 1.0
        let r2 = ys.reshaped(1, height, 1, 1) * ys.reshaped(1, height, 1, 1)
            + xs.reshaped(1, 1, width, 1) * xs.reshaped(1, 1, width, 1)
        return 1.6 - 3.4 * r2
    }
}
