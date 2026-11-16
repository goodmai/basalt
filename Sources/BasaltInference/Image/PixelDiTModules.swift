import Foundation
import MLX
import MLXNN
import BasaltCore

final class PixelRefinementUnit: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(dimensions: Int) {
        let d = max(16, dimensions)
        self._norm.wrappedValue = LayerNorm(dimensions: d)
        self._fc1.wrappedValue = Linear(d, d * 2)
        self._fc2.wrappedValue = Linear(d * 2, d)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + 0.5 * fc2(MLXNN.silu(fc1(norm(x))))
    }
}

final class PixelHead16x16: Module {
    let dimensions: Int
    let patchSize: Int
    @ModuleInfo(key: "blocks") var blocks: [PixelRefinementUnit]
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pixel_proj") var pixelProj: Linear
    @ModuleInfo(key: "smooth_conv") var smoothConv: Conv2d

    init(dimensions: Int, patchSize: Int = 16) {
        let d = max(16, dimensions)
        let p = max(1, patchSize)
        self.dimensions = d
        self.patchSize = p
        self._blocks.wrappedValue = (0 ..< 4).map { _ in PixelRefinementUnit(dimensions: d) }
        self._norm.wrappedValue = LayerNorm(dimensions: d)
        self._pixelProj.wrappedValue = Linear(d, p * p * 4)
        self._smoothConv.wrappedValue = Conv2d(inputChannels: 4, outputChannels: 4, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ tokens: MLXArray, tokH: Int, tokW: Int) -> MLXArray {
        var h = tokens
        for unit in blocks {
            h = unit(h)
        }
        let p = patchSize
        let unpatched = pixelProj(norm(h))
            .reshaped(1, tokH, tokW, p, p, 4)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(1, tokH * p, tokW * p, 4)
        return unpatched + 0.25 * smoothConv(unpatched)
    }
}

final class DCAE32xCodec: Module {
    let latentChannels: Int
    let midChannels: Int
    @ModuleInfo(key: "enc_stem") var encStem: Conv2d
    @ModuleInfo(key: "enc_stages") var encStages: [Conv2d]
    @ModuleInfo(key: "dec_stem") var decStem: Conv2d
    @ModuleInfo(key: "dec_stages") var decStages: [Conv2d]
    @ModuleInfo(key: "dec_out") var decOut: Conv2d

    init(latentChannels: Int = 16, midChannels: Int = 24) {
        let lat = max(4, latentChannels)
        let ch = max(16, midChannels)
        self.latentChannels = lat
        self.midChannels = ch
        self._encStem.wrappedValue = Conv2d(inputChannels: 4, outputChannels: ch, kernelSize: 3, padding: 1)
        self._encStages.wrappedValue = (0 ..< 5).map { idx in
            let outCh = idx == 4 ? lat : ch
            return Conv2d(inputChannels: ch, outputChannels: outCh, kernelSize: 3, stride: 2, padding: 1)
        }
        self._decStem.wrappedValue = Conv2d(inputChannels: lat, outputChannels: ch, kernelSize: 3, padding: 1)
        self._decStages.wrappedValue = (0 ..< 5).map { _ in
            Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 3, padding: 1)
        }
        self._decOut.wrappedValue = Conv2d(inputChannels: ch, outputChannels: 4, kernelSize: 3, padding: 1)
    }

    static func paddedDimensions(height: Int, width: Int) -> (paddedH: Int, paddedW: Int, latH: Int, latW: Int) {
        let pH = ((max(32, height) + 31) / 32) * 32
        let pW = ((max(32, width) + 31) / 32) * 32
        return (pH, pW, pH / 32, pW / 32)
    }

    func encode(_ image: MLXArray) -> MLXArray {
        let nhwc = image.ndim == 3 ? image[.newAxis] : image
        let (pH, pW, _, _) = Self.paddedDimensions(height: nhwc.dim(1), width: nhwc.dim(2))
        var x = nhwc * 2.0 - 1.0
        if pH > nhwc.dim(1) || pW > nhwc.dim(2) {
            x = padded(
                x,
                widths: [
                    IntOrPair((0, 0)),
                    IntOrPair((0, pH - nhwc.dim(1))),
                    IntOrPair((0, pW - nhwc.dim(2))),
                    IntOrPair((0, 0)),
                ]
            )
        }
        var h = MLXNN.silu(encStem(x))
        for stage in encStages.dropLast() {
            h = MLXNN.silu(stage(h))
        }
        return encStages[encStages.count - 1](h)
    }

    func decode(_ latent: MLXArray, targetHeight: Int, targetWidth: Int, includeAlpha: Bool) -> MLXArray {
        let up = Upsample(scaleFactor: 2.0, mode: .nearest)
        var h = MLXNN.silu(decStem(latent))
        for stage in decStages {
            h = MLXNN.silu(stage(up(h)))
        }
        let raw = decOut(h)[0..., ..<targetHeight, ..<targetWidth, 0...]
        let rgb = sigmoid(raw[0..., 0..., 0..., ..<3])
        guard includeAlpha else { return rgb }
        let prior = PatchVAECodec.subjectMattePrior(height: targetHeight, width: targetWidth)
        let alpha = sigmoid(raw[0..., 0..., 0..., 3...] + prior)
        return concatenated([rgb, alpha], axis: 3)
    }
}

final class TernaryBitNetDiTBlock: Module {
    let dimensions: Int
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "adaln") var adaLN: AdaLNZeroModulation
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "qkv") var qkv: TernaryBitNetLinear
    @ModuleInfo(key: "out_proj") var outProj: TernaryBitNetLinear
    @ModuleInfo(key: "mlp_gate") var mlpGate: TernaryBitNetLinear
    @ModuleInfo(key: "mlp_up") var mlpUp: TernaryBitNetLinear
    @ModuleInfo(key: "mlp_down") var mlpDown: TernaryBitNetLinear

    init(dimensions: Int, heads: Int) {
        let d = max(16, dimensions)
        let h = max(1, heads)
        self.dimensions = d
        self.heads = h
        self.scale = 1.0 / Float(d / h).squareRoot()
        self._adaLN.wrappedValue = AdaLNZeroModulation(dimensions: d)
        self._norm1.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._norm2.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._qkv.wrappedValue = TernaryBitNetLinear(d, d * 3, bias: false)
        self._outProj.wrappedValue = TernaryBitNetLinear(d, d)
        self._mlpGate.wrappedValue = TernaryBitNetLinear(d, d * 2, bias: false)
        self._mlpUp.wrappedValue = TernaryBitNetLinear(d, d * 2, bias: false)
        self._mlpDown.wrappedValue = TernaryBitNetLinear(d * 2, d)
    }

    func callAsFunction(
        condTokens: MLXArray,
        imgTokens: MLXArray,
        stepCtx: DiTStepContext
    ) -> (cond: MLXArray, img: MLXArray) {
        let nC = condTokens.dim(1)
        let nI = imgTokens.dim(1)
        let unified = concatenated([condTokens, imgTokens], axis: 1)
        let gates = adaLN.gates(timestep: stepCtx.timestep, guidance: stepCtx.guidance, condVec: stepCtx.condVec)
        let mod1 = norm1(unified) * (1.0 + gates.scaleMSA) + gates.shiftMSA
        let headDim = dimensions / heads
        let p = qkv(mod1).reshaped(1, nC + nI, 3, heads, headDim).transposed(2, 0, 3, 1, 4)
        let q = concatenated([p[0][0..., 0..., ..<nC, 0...], RoPE2D.apply(p[0][0..., 0..., nC..., 0...], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)], axis: 2)
        let k = concatenated([p[1][0..., 0..., ..<nC, 0...], RoPE2D.apply(p[1][0..., 0..., nC..., 0...], cos: stepCtx.ropeCos, sin: stepCtx.ropeSin)], axis: 2)
        let attn = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: p[2], scale: scale, mask: .none)
            .transposed(0, 2, 1, 3)
            .reshaped(1, nC + nI, dimensions)
        let h1 = unified + gates.gateMSA * outProj(attn)
        let mod2 = norm2(h1) * (1.0 + gates.scaleMLP) + gates.shiftMLP
        let ffn = mlpDown(MLXNN.silu(mlpGate(mod2)) * mlpUp(mod2))
        let h2 = h1 + gates.gateMLP * ffn
        return (h2[0..., ..<nC, 0...], h2[0..., nC..., 0...])
    }
}

final class UnifiedMoTPixelShuffleDecoder: Module {
    let inChannels: Int
    let midChannels: Int
    @ModuleInfo(key: "stage1_conv") var stage1Conv: Conv2d
    @ModuleInfo(key: "stage2_conv") var stage2Conv: Conv2d
    @ModuleInfo(key: "out_conv") var outConv: Conv2d

    init(inChannels: Int = 16, midChannels: Int = 16) {
        let cIn = max(4, inChannels)
        let cMid = max(8, midChannels)
        self.inChannels = cIn
        self.midChannels = cMid
        self._stage1Conv.wrappedValue = Conv2d(inputChannels: cIn, outputChannels: cMid * 4, kernelSize: 3, padding: 1)
        self._stage2Conv.wrappedValue = Conv2d(inputChannels: cMid, outputChannels: cMid * 16, kernelSize: 3, padding: 1)
        self._outConv.wrappedValue = Conv2d(inputChannels: cMid, outputChannels: 4, kernelSize: 3, padding: 1)
    }

    func decode(_ featureGrid: MLXArray, includeAlpha: Bool) -> MLXArray {
        let up2 = Self.pixelShuffle(MLXNN.silu(stage1Conv(featureGrid)), factor: 2)
        let up8 = Self.pixelShuffle(MLXNN.silu(stage2Conv(up2)), factor: 4)
        let raw = outConv(up8)
        let rgb = sigmoid(raw[0..., 0..., 0..., ..<3])
        guard includeAlpha else { return rgb }
        let prior = PatchVAECodec.subjectMattePrior(height: raw.dim(1), width: raw.dim(2))
        let alpha = sigmoid(raw[0..., 0..., 0..., 3...] + prior)
        return concatenated([rgb, alpha], axis: 3)
    }

    static func pixelShuffle(_ x: MLXArray, factor: Int) -> MLXArray {
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)
        let outCh = x.dim(3) / (factor * factor)
        return x
            .reshaped(b, h, w, factor, factor, outCh)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(b, h * factor, w * factor, outCh)
    }
}

final class AudioFlowEditProjector: Module {
    let featureDim: Int
    let hiddenDim: Int
    let outChannels: Int
    @ModuleInfo(key: "token_proj") var tokenProj: Linear
    @ModuleInfo(key: "spatial_proj") var spatialProj: Linear

    init(featureDim: Int, hiddenDim: Int, outChannels: Int) {
        let f = max(16, featureDim)
        let h = max(16, hiddenDim)
        let c = max(4, outChannels)
        self.featureDim = f
        self.hiddenDim = h
        self.outChannels = c
        self._tokenProj.wrappedValue = Linear(f, h * 4)
        self._spatialProj.wrappedValue = Linear(f, c * 4)
    }

    func conditioningTokens(_ audioVector: [Float]) -> MLXArray {
        let vec = MLXArray(audioVector).reshaped(1, featureDim)
        return MLXNN.silu(tokenProj(vec)).reshaped(1, 4, hiddenDim)
    }

    func flowEditVelocity(audioVector: [Float], height: Int, width: Int, timestep: Float) -> MLXArray {
        let vec = MLXArray(audioVector).reshaped(1, featureDim)
        let quad = tanh(spatialProj(vec)).reshaped(1, 2, 2, outChannels)
        let field = TinyAutoencoderPreview.upsampleBilinearGrid(quad[0], height: height, width: width)[.newAxis]
        let envelope = Float(0.65 + 0.35 * cos(Double(timestep) * .pi))
        return field * envelope
    }
}
