import Foundation
import MLX
import MLXNN

final class NemotronMelFeatures: Module {
    let config: NemotronMelConfig
    @ParameterInfo(key: "window") var window: MLXArray
    @ParameterInfo(key: "fb") var fb: MLXArray

    init(_ config: NemotronMelConfig) {
        self.config = config
        self._window.wrappedValue = MLXArray.ones([config.winLength])
        self._fb.wrappedValue = MLXArray.zeros([1, config.features, config.nFFT / 2 + 1])
        super.init()
    }

    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let totalSamples = audio.dim(0)
        let validFrames = totalSamples / config.hopLength
        guard validFrames > 0 else {
            return MLXArray.zeros([1, config.features, 0], dtype: .float32)
        }
        let samples = audio.asType(.float32)
        let frameIds = MLXArray(Int32(0) ..< Int32(validFrames))
        let fftOffsets = MLXArray(Int32(0) ..< Int32(config.nFFT))
        let positions = frameIds.expandedDimensions(axis: 1) * Int32(config.hopLength)
            + fftOffsets.expandedDimensions(axis: 0) - Int32(config.nFFT / 2)

        func gather(_ indices: MLXArray) -> MLXArray {
            let clipped = clip(indices, min: 0, max: max(0, totalSamples - 1))
            let values = samples[clipped]
            let inBounds = (indices .>= 0) .&& (indices .< totalSamples)
            return which(inBounds, values, MLXArray(Float(0)))
        }

        let rawFrames = gather(positions) - config.preemphasis * gather(positions - 1)
        let validMask = (positions .>= 0) .&& (positions .< totalSamples)
        let frames = which(validMask, rawFrames, MLXArray(Float(0)))
        let padTotal = config.nFFT - config.winLength
        let padLeft = padTotal / 2
        let padRight = padTotal - padLeft
        let paddedWindow = padded(window.asType(.float32), widths: [IntOrPair((padLeft, padRight))])
        let spectrum = MLXFFT.rfft(frames * paddedWindow, axis: -1)
        let power = abs(spectrum).square()
        let filterbank = fb[0].asType(.float32).transposed()
        let floor = Float(pow(2.0, -24.0))
        let mel = log(matmul(power, filterbank) + floor)
        return contiguous(mel.transposed()).expandedDimensions(axis: 0)
    }
}

final class NemotronFeatureStacking: Module {
    let factor: Int
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ config: NemotronEncoderConfig) {
        self.factor = config.subsamplingFactor
        self._proj.wrappedValue = Linear(
            config.featIn * config.subsamplingFactor, config.dModel, bias: false
        )
        super.init()
    }

    var parameterDType: DType {
        if let quantized = proj as? QuantizedLinear {
            return quantized.scales.dtype
        }
        return proj.weight.dtype
    }

    func callAsFunction(_ features: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        var x = features.transposed(0, 2, 1)
        let b = x.dim(0)
        let t = x.dim(1)
        let c = x.dim(2)
        let rem = ((-t % factor) + factor) % factor
        if rem > 0 {
            x = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, rem)), IntOrPair((0, 0))])
        }
        let stacked = proj(x.reshaped(b, -1, c * factor))
        let outLengths = (lengths + Int32(factor - 1)) / Int32(factor)
        return (stacked, outLengths)
    }
}

final class NemotronEncoderAttention: Module {
    let nHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "w_qkv") var wQKV: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: LayerNorm?
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm?
    let rope: RoPE

    init(_ config: NemotronEncoderConfig) {
        self.nHeads = config.nHeads
        self.headDim = config.dModel / config.nHeads
        self.scale = 1.0 / Float(headDim).squareRoot()
        self._wQKV.wrappedValue = Linear(config.dModel, 3 * config.dModel, bias: config.qkvBias)
        self._outProj.wrappedValue = Linear(config.dModel, config.dModel, bias: true)
        if config.qkNorm {
            self._qNorm.wrappedValue = LayerNorm(dimensions: headDim)
            self._kNorm.wrappedValue = LayerNorm(dimensions: headDim)
        }
        let ropeDims = Int(Float(headDim) * config.rotaryFraction)
        self.rope = RoPE(dimensions: ropeDims, traditional: false, base: config.ropeBase)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let d = x.dim(2)
        let splits = split(wQKV(x), parts: 3, axis: -1)
        var q = splits[0].reshaped(b, t, nHeads, headDim).transposed(0, 2, 1, 3)
        var k = splits[1].reshaped(b, t, nHeads, headDim).transposed(0, 2, 1, 3)
        let v = splits[2].reshaped(b, t, nHeads, headDim).transposed(0, 2, 1, 3)
        if let qNorm, let kNorm {
            q = qNorm(q)
            k = kNorm(k)
        }
        q = rope(q)
        k = rope(k)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        return outProj(attended.transposed(0, 2, 1, 3).reshaped(b, t, d))
    }
}

final class NemotronEncoderFFN: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(_ config: NemotronEncoderConfig) {
        let hidden = Int(Float(config.dModel) * config.ffExpansion)
        self._linear1.wrappedValue = Linear(config.dModel, hidden, bias: true)
        self._linear2.wrappedValue = Linear(hidden, config.dModel, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}

final class NemotronEncoderBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NemotronEncoderAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: NemotronEncoderFFN

    init(_ config: NemotronEncoderConfig) {
        self._norm1.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._attn.wrappedValue = NemotronEncoderAttention(config)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._ffn.wrappedValue = NemotronEncoderFFN(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + attn(norm1(x), mask: mask)
        return h + ffn(norm2(h))
    }
}

final class NemotronConformerEncoder: Module {
    let inputScale: Float
    @ModuleInfo(key: "pre_encode") var preEncode: NemotronFeatureStacking
    @ModuleInfo(key: "embed_norm") var embedNorm: LayerNorm?
    @ModuleInfo(key: "layers") var layers: [NemotronEncoderBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    init(_ config: NemotronEncoderConfig) {
        self.inputScale = config.xscaling ? Float(config.dModel).squareRoot() : 1.0
        self._preEncode.wrappedValue = NemotronFeatureStacking(config)
        if config.preBlockNorm {
            self._embedNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        }
        self._layers.wrappedValue = (0 ..< config.nLayers).map { _ in
            NemotronEncoderBlock(config)
        }
        self._finalNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> MLXArray {
        let positions = MLXArray(Int32(0) ..< Int32(x.dim(1))).expandedDimensions(axis: 0)
        let valid = positions .< lengths.asType(.int32).expandedDimensions(axis: 1)
        let mask = valid.expandedDimensions(axes: [1, 2])
        var hidden = inputScale == 1.0 ? x : x * inputScale
        if let embedNorm {
            hidden = embedNorm(hidden)
        }
        for layer in layers {
            hidden = layer(hidden, mask: mask)
        }
        return finalNorm(hidden)
    }
}
