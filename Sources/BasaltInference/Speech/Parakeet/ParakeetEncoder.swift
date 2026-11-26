import Foundation
import MLX
import MLXNN

final class ParakeetFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(size: Int, hidden: Int, bias: Bool) {
        _linear1.wrappedValue = Linear(size, hidden, bias: bias)
        _linear2.wrappedValue = Linear(hidden, size, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

final class ParakeetConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwise1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwise: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwise2: Conv1d

    init(size: Int, kernel: Int, bias: Bool) {
        _pointwise1.wrappedValue = Conv1d(inputChannels: size, outputChannels: 2 * size, kernelSize: 1, bias: bias)
        _depthwise.wrappedValue = Conv1d(
            inputChannels: size, outputChannels: size, kernelSize: kernel, padding: (kernel - 1) / 2, groups: size, bias: bias)
        _batchNorm.wrappedValue = BatchNorm(featureCount: size)
        _pointwise2.wrappedValue = Conv1d(inputChannels: size, outputChannels: size, kernelSize: 1, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        pointwise2(silu(batchNorm(depthwise(glu(pointwise1(x), axis: -1)))))
    }
}

final class ParakeetAttention: Module {
    let heads: Int
    @ModuleInfo(key: "linear_q") var wq: Linear
    @ModuleInfo(key: "linear_k") var wk: Linear
    @ModuleInfo(key: "linear_v") var wv: Linear
    @ModuleInfo(key: "linear_out") var wo: Linear
    @ModuleInfo(key: "linear_pos") var wpos: Linear
    @ParameterInfo(key: "pos_bias_u") var biasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var biasV: MLXArray

    init(size: Int, heads: Int, bias: Bool) {
        self.heads = heads
        _wq.wrappedValue = Linear(size, size, bias: bias)
        _wk.wrappedValue = Linear(size, size, bias: bias)
        _wv.wrappedValue = Linear(size, size, bias: bias)
        _wo.wrappedValue = Linear(size, size, bias: bias)
        _wpos.wrappedValue = Linear(size, size, bias: false)
        _biasU.wrappedValue = MLXArray.zeros([heads, size / heads])
        _biasV.wrappedValue = MLXArray.zeros([heads, size / heads])
    }

    static func relativeShift(_ x: MLXArray) -> MLXArray {
        let (b, h, t, p) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let shifted = concatenated([MLXArray.zeros([b, h, t, 1], dtype: x.dtype), x], axis: -1).reshaped(b, h, p + 1, t)
        return shifted[0..., 0..., 1..., 0...].reshaped(b, h, t, p)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let q = wq(x).reshaped(b, t, heads, -1)
        let k = wk(x).reshaped(b, t, heads, -1).transposed(0, 2, 3, 1)
        let v = wv(x).reshaped(b, t, heads, -1).transposed(0, 2, 1, 3)
        let p = wpos(positions).reshaped(1, positions.dim(1), heads, -1).transposed(0, 2, 3, 1)
        let content = matmul((q + biasU).transposed(0, 2, 1, 3), k)
        let position = Self.relativeShift(matmul((q + biasV).transposed(0, 2, 1, 3), p))[.ellipsis, ..<t]
        let scale = 1 / Float(q.dim(-1)).squareRoot()
        let weights = softmax((content + position) * scale, axis: -1, precise: true)
        return wo(matmul(weights, v).transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }
}

final class ParakeetConformerBlock: Module {
    @ModuleInfo(key: "norm_feed_forward1") var feedForward1Norm: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: ParakeetFeedForward
    @ModuleInfo(key: "norm_self_att") var attentionNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: ParakeetAttention
    @ModuleInfo(key: "norm_conv") var convolutionNorm: LayerNorm
    @ModuleInfo(key: "conv") var convolution: ParakeetConvolution
    @ModuleInfo(key: "norm_feed_forward2") var feedForward2Norm: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: ParakeetFeedForward
    @ModuleInfo(key: "norm_out") var outputNorm: LayerNorm

    init(_ encoder: ParakeetConfiguration.Conformer) {
        let size = encoder.modelSize
        let hidden = size * encoder.feedForwardExpansion
        _feedForward1Norm.wrappedValue = LayerNorm(dimensions: size)
        _feedForward1.wrappedValue = ParakeetFeedForward(size: size, hidden: hidden, bias: encoder.bias)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: size)
        _attention.wrappedValue = ParakeetAttention(size: size, heads: encoder.heads, bias: encoder.bias)
        _convolutionNorm.wrappedValue = LayerNorm(dimensions: size)
        _convolution.wrappedValue = ParakeetConvolution(size: size, kernel: encoder.convKernelSize, bias: encoder.bias)
        _feedForward2Norm.wrappedValue = LayerNorm(dimensions: size)
        _feedForward2.wrappedValue = ParakeetFeedForward(size: size, hidden: hidden, bias: encoder.bias)
        _outputNorm.wrappedValue = LayerNorm(dimensions: size)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        var h = x + Float(0.5) * feedForward1(feedForward1Norm(x))
        h = h + attention(attentionNorm(h), positions: positions)
        h = h + convolution(convolutionNorm(h))
        h = h + Float(0.5) * feedForward2(feedForward2Norm(h))
        return outputNorm(h)
    }
}

final class ParakeetSubsampling: Module {
    @ModuleInfo(key: "conv") var conv: [Module]
    @ModuleInfo(key: "out") var out: Linear

    init(_ encoder: ParakeetConfiguration.Conformer) {
        let channels = encoder.subsamplingChannels
        var layers: [Module] = [Conv2d(inputChannels: 1, outputChannels: channels, kernelSize: 3, stride: 2, padding: 1), ReLU()]
        for _ in 1 ..< encoder.subsamplingStages {
            layers += [
                Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2, padding: 1, groups: channels),
                Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1), ReLU(),
            ]
        }
        let frequencies = (0 ..< encoder.subsamplingStages).reduce(encoder.featuresIn) { size, _ in (size - 1) / 2 + 1 }
        _conv.wrappedValue = layers
        _out.wrappedValue = Linear(channels * frequencies, encoder.modelSize)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = mel.expandedDimensions(axis: -1)
        for case let layer as any UnaryLayer in conv {
            x = layer(x)
        }
        return out(x.transposed(0, 1, 3, 2).reshaped(x.dim(0), x.dim(1), -1))
    }
}

final class ParakeetEncoder: Module {
    let xscale: Float
    @ModuleInfo(key: "pre_encode") var preEncode: ParakeetSubsampling
    @ModuleInfo(key: "layers") var layers: [ParakeetConformerBlock]

    init(_ encoder: ParakeetConfiguration.Conformer) {
        xscale = (encoder.xscaling ?? true) ? Float(encoder.modelSize).squareRoot() : 1
        _preEncode.wrappedValue = ParakeetSubsampling(encoder)
        _layers.wrappedValue = (0 ..< encoder.layers).map { _ in ParakeetConformerBlock(encoder) }
    }

    static func relativePositions(length: Int, size: Int) -> MLXArray {
        let positions = MLXArray((0 ..< (2 * length - 1)).map { Float(length - 1 - $0) }).expandedDimensions(axis: 1)
        let frequencies = MLXArray(stride(from: 0, to: size, by: 2).map { Float(Foundation.exp(-Double($0) * Foundation.log(10_000.0) / Double(size))) })
        let angles = positions * frequencies.expandedDimensions(axis: 0)
        return stacked([sin(angles), cos(angles)], axis: -1).reshaped(1, 2 * length - 1, size)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = preEncode(mel) * xscale
        let positions = Self.relativePositions(length: x.dim(1), size: x.dim(2)).asType(x.dtype)
        for layer in layers {
            x = layer(x, positions: positions)
        }
        return x
    }
}
