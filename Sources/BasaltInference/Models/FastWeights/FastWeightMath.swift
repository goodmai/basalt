import Foundation
import MLX
import MLXLMCommon
import MLXNN

package struct TitansTransition {
    package let pBeta: MLXArray
    package let pEta: MLXArray
    package let cS: MLXArray
    package let eLast: MLXArray
    package let uLast: MLXArray
}

package enum FastWeightMath {
    private static let geluCoeff: Float = 0.7978845608028654
    private static let geluCubic: Float = 0.044715
    private static let muonA: Float = 3.4445
    private static let muonB: Float = -4.7750
    private static let muonC: Float = 2.0315

    package static func causalLowerTriangular(length: Int) -> MLXArray {
        let rows = MLXArray(Int32(0) ..< Int32(length)).expandedDimensions(axis: 1)
        let cols = MLXArray(Int32(0) ..< Int32(length)).expandedDimensions(axis: 0)
        return (cols .<= rows).asType(.float32)
    }

    package static func l2Normalize(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let xf = x.asType(.float32)
        let norm = sqrt((xf * xf).sum(axis: -1, keepDims: true) + MLXArray(eps * eps, dtype: .float32))
        return xf / norm
    }

    package static func rowL2Norm(_ w: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let wf = w.asType(.float32)
        return sqrt((wf * wf).sum(axis: -1, keepDims: true)) + MLXArray(eps, dtype: .float32)
    }

    package static func preserveRowNorm(_ w: MLXArray, targetNorm: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let wf = w.asType(.float32)
        let denom = sqrt((wf * wf).sum(axis: -1, keepDims: true)) + MLXArray(eps, dtype: .float32)
        return wf * (targetNorm.asType(.float32) / denom)
    }

    package static func layerNormForward(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        eps: Float
    ) -> MLXArray {
        let xf = x.asType(.float32)
        let centered = xf - mean(xf, axis: -1, keepDims: true)
        let variance = mean(centered * centered, axis: -1, keepDims: true)
        let invStd = rsqrt(variance + MLXArray(eps, dtype: .float32))
        return centered * invStd * weight.asType(.float32) + bias.asType(.float32)
    }

    package static func lnFusedL2Backward(
        x: MLXArray,
        target: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        eps: Float
    ) -> MLXArray {
        let xf = x.asType(.float32)
        let tf = target.asType(.float32)
        let wf = weight.asType(.float32)
        let bf = bias.asType(.float32)
        let centered = xf - mean(xf, axis: -1, keepDims: true)
        let variance = mean(centered * centered, axis: -1, keepDims: true)
        let invStd = rsqrt(variance + MLXArray(eps, dtype: .float32))
        let xHat = centered * invStd
        let y = xHat * wf + bf
        let gradXHat = (y - tf) * wf
        let c1 = mean(gradXHat, axis: -1, keepDims: true)
        let c2 = mean(gradXHat * xHat, axis: -1, keepDims: true)
        return invStd * (gradXHat - c1 - xHat * c2)
    }

    package static func geluTanh(_ z: MLXArray) -> MLXArray {
        let zf = z.asType(.float32)
        let inner = MLXArray(geluCoeff, dtype: .float32) * (zf + MLXArray(geluCubic, dtype: .float32) * zf * zf * zf)
        return Float(0.5) * zf * (Float(1.0) + tanh(inner))
    }

    package static func geluTanhBackward(z: MLXArray, gradOut: MLXArray) -> MLXArray {
        let zf = z.asType(.float32)
        let z2 = zf * zf
        let inner = MLXArray(geluCoeff, dtype: .float32) * (zf + MLXArray(geluCubic, dtype: .float32) * z2 * zf)
        let t = tanh(inner)
        let dInner = MLXArray(geluCoeff, dtype: .float32) * (Float(1.0) + MLXArray(3 * geluCubic, dtype: .float32) * z2)
        let dGelu = Float(0.5) * (Float(1.0) + t) + Float(0.5) * zf * (Float(1.0) - t * t) * dInner
        return gradOut.asType(.float32) * dGelu
    }

    package static func siluBackward(z: MLXArray, gradOut: MLXArray) -> MLXArray {
        let zf = z.asType(.float32)
        let sig = sigmoid(zf)
        let deriv = sig * (Float(1.0) + zf * (Float(1.0) - sig))
        return gradOut.asType(.float32) * deriv
    }

    package static func titansTransition(beta: MLXArray, eta: MLXArray) -> TitansTransition {
        let bf = beta.asType(.float32)
        let ef = eta.asType(.float32)
        let (batch, heads, length) = (bf.dim(0), bf.dim(1), bf.dim(2))
        let one = MLXArray.ones([batch, heads], dtype: .float32)
        guard length > 0 else {
            let scalar = one.reshaped(batch, heads, 1, 1)
            let zero = MLXArray.zeros([batch, heads, 1, 1], dtype: .float32)
            let empty = MLXArray.zeros([batch, heads, 0, 1], dtype: .float32)
            return TitansTransition(pBeta: scalar, pEta: scalar, cS: zero, eLast: empty, uLast: empty)
        }
        var bRun = one
        var eRun = one
        var uRun = one
        var eReversed: [MLXArray] = []
        var uReversed: [MLXArray] = []
        eReversed.reserveCapacity(length)
        uReversed.reserveCapacity(length)
        eReversed.append(eRun)
        uReversed.append(uRun)
        if length > 1 {
            for idx in stride(from: length - 2, through: 0, by: -1) {
                let nextBeta = bf[0..., 0..., idx + 1]
                let nextEta = ef[0..., 0..., idx + 1]
                bRun = bRun * nextBeta
                eRun = eRun * nextEta
                uRun = bRun + nextEta * uRun
                eReversed.append(eRun)
                uReversed.append(uRun)
            }
        }
        let beta0 = bf[0..., 0..., 0]
        let eta0 = ef[0..., 0..., 0]
        let pBeta = (bRun * beta0).reshaped(batch, heads, 1, 1)
        let pEta = (eRun * eta0).reshaped(batch, heads, 1, 1)
        let cS = (eta0 * uRun).reshaped(batch, heads, 1, 1)
        let eLast = MLX.stacked(Array(eReversed.reversed()), axis: 2).expandedDimensions(axis: -1)
        let uLast = MLX.stacked(Array(uReversed.reversed()), axis: 2).expandedDimensions(axis: -1)
        return TitansTransition(pBeta: pBeta, pEta: pEta, cS: cS, eLast: eLast, uLast: uLast)
    }

    package static func newtonSchulz5(_ grad: MLXArray, steps: Int = 5, eps: Float = 1e-7) -> MLXArray {
        var x = grad.asType(.float32)
        let transposed = x.dim(-2) > x.dim(-1)
        if transposed {
            x = x.swappedAxes(-1, -2)
        }
        let frob = sqrt((x * x).sum(axes: [-2, -1], keepDims: true)) + MLXArray(eps, dtype: .float32)
        x = x / frob
        for _ in 0 ..< Swift.max(1, steps) {
            let gram = matmul(x, x.swappedAxes(-1, -2))
            let poly = MLXArray(muonB, dtype: .float32) * gram
                + MLXArray(muonC, dtype: .float32) * matmul(gram, gram)
            x = MLXArray(muonA, dtype: .float32) * x + matmul(poly, x)
        }
        return transposed ? x.swappedAxes(-1, -2) : x
    }

    package static func normalizeConvWeight(_ weight: MLXArray, kernelSize: Int) -> MLXArray {
        if weight.ndim == 2 {
            return weight.expandedDimensions(axis: -1)
        }
        if weight.ndim == 3, weight.dim(1) == 1, weight.dim(2) == kernelSize, kernelSize != 1 {
            return weight.swappedAxes(1, 2)
        }
        return weight
    }

    package static func prependPersistentMask(
        _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
        queryLength: Int,
        keyLength: Int,
        persistentCount: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard persistentCount > 0 else { return mask }
        if case .none = mask { return .none }
        if case .causal = mask {
            let prefix = MLXArray.ones([queryLength, persistentCount], dtype: .bool)
            let causal = createCausalMask(n: queryLength, offset: Swift.max(0, keyLength - queryLength), windowSize: nil)
            return .array(concatenated([prefix, causal], axis: -1))
        }
        if case .array(let arr) = mask {
            let prefixShape = Array(arr.shape.dropLast()) + [persistentCount]
            let prefix = arr.dtype == .bool
                ? MLXArray.ones(prefixShape, dtype: .bool)
                : MLXArray.zeros(prefixShape, dtype: arr.dtype)
            return .array(concatenated([prefix, arr], axis: -1))
        }
        return mask
    }
}

package final class CausalDepthwiseConv1d: Module {
    let kernelSize: Int
    let applySiLU: Bool
    @ModuleInfo(key: "conv") var conv: Conv1d

    package init(channels: Int, kernelSize: Int, bias: Bool = false, applySiLU: Bool = false) {
        self.kernelSize = kernelSize
        self.applySiLU = applySiLU
        _conv.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: channels,
            bias: bias
        )
    }

    package func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (output: MLXArray, state: MLXArray?) {
        let (batch, steps, channels) = (x.dim(0), x.dim(1), x.dim(2))
        let history = Swift.max(kernelSize - 1, 0)
        guard history > 0 else {
            let y = conv(x)
            return (applySiLU ? silu(y) : y, nil)
        }
        let previous = state?.asType(x.dtype) ?? MLXArray.zeros([batch, history, channels], dtype: x.dtype)
        guard steps > 0 else {
            return (MLXArray.zeros([batch, 0, channels], dtype: x.dtype), previous)
        }
        let convInput = concatenated([previous, x], axis: 1)
        let raw = conv(convInput)
        let out = applySiLU ? silu(raw) : raw
        let total = convInput.dim(1)
        let tail = FastWeightLayerCache.detach(convInput[0..., (total - history)..., 0...])
        return (out, tail)
    }
}

package final class FastWeightSwiGLUMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    package init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}
