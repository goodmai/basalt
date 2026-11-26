import BasaltCore
import Foundation
import MLX
import MLXNN

public struct HyperConnectionGates {
    public let hPre: MLXArray
    public let hPost: MLXArray
    public let hRes: MLXArray?

    public init(hPre: MLXArray, hPost: MLXArray, hRes: MLXArray?) {
        self.hPre = hPre
        self.hPost = hPost
        self.hRes = hRes
    }
}

public final class HyperConnectionSublayer: Module {
    public let hiddenSize: Int
    public let spec: HyperConnectionSpec
    public let rmsNormEps: Float
    public var forcePureMLX: Bool = false

    @ParameterInfo(key: "phi_weight") public var phiWeight: MLXArray?
    @ParameterInfo(key: "alpha_pre") public var alphaPre: MLXArray
    @ParameterInfo(key: "alpha_post") public var alphaPost: MLXArray
    @ParameterInfo(key: "alpha_res") public var alphaRes: MLXArray?
    @ParameterInfo(key: "bias_pre") public var biasPre: MLXArray
    @ParameterInfo(key: "bias_post") public var biasPost: MLXArray
    @ParameterInfo(key: "bias_res") public var biasRes: MLXArray?

    public init(
        hiddenSize: Int,
        spec: HyperConnectionSpec = HyperConnectionSpec(),
        rmsNormEps: Float = 1e-6
    ) {
        self.hiddenSize = hiddenSize
        self.spec = spec
        self.rmsNormEps = rmsNormEps
        let n = max(spec.expansionRate, 1)
        let usesRes = spec.variant != .ihc
        let projDim = usesRes ? (2 * n + n * n) : (2 * n)
        self._phiWeight.wrappedValue = spec.dynamic
            ? MLXArray.zeros([n * hiddenSize, projDim], dtype: .float32)
            : nil
        self._alphaPre.wrappedValue = MLXArray([spec.alphaInit])
        self._alphaPost.wrappedValue = MLXArray([spec.alphaInit])
        self._alphaRes.wrappedValue = usesRes ? MLXArray([spec.alphaInit]) : nil
        self._biasPre.wrappedValue = Self.initialBiasPre(variant: spec.variant, rate: n)
        self._biasPost.wrappedValue = Self.initialBiasPost(variant: spec.variant, rate: n)
        self._biasRes.wrappedValue = Self.initialBiasRes(variant: spec.variant, rate: n)
        super.init()
        train(false)
    }

    public func computeGates(_ streams: MLXArray) -> HyperConnectionGates {
        let x = streams.dtype == .float32 ? streams : streams.asType(.float32)
        let raw = rawGateLogits(x)
        switch spec.variant {
        case .hc:
            return HyperConnectionGates(hPre: raw.pre, hPost: raw.post, hRes: raw.res)
        case .mhc:
            let (hPre, hPost) = applyGateMode(pre: raw.pre, post: raw.post)
            let useMetal = !training && !forcePureMLX
            let hRes = raw.res.map {
                HyperConnectionSinkhornKernel.project(
                    logits: $0,
                    iterations: spec.sinkhornIterations,
                    eps: spec.sinkhornEps,
                    useMetal: useMetal
                )
            }
            return HyperConnectionGates(hPre: hPre, hPost: hPost, hRes: hRes)
        case .ihc:
            let (hPre, hPost) = applyGateMode(pre: raw.pre, post: raw.post)
            return HyperConnectionGates(hPre: hPre, hPost: hPost, hRes: nil)
        }
    }

    public func preRead(_ streams: MLXArray, gates: HyperConnectionGates) -> MLXArray {
        let x = streams.dtype == .float32 ? streams : streams.asType(.float32)
        let n = x.dim(-2)
        let preWeights = gates.hPre.ndim == 1
            ? gates.hPre.reshaped(1, 1, n, 1)
            : gates.hPre.expandedDimensions(axis: -1)
        return (x * preWeights).sum(axis: -2)
    }

    public func mix(
        _ streams: MLXArray,
        sublayerOutput: MLXArray,
        gates: HyperConnectionGates
    ) -> MLXArray {
        let x = streams.dtype == .float32 ? streams : streams.asType(.float32)
        let n = x.dim(-2)
        let postWeights = gates.hPost.ndim == 1
            ? gates.hPost.reshaped(1, 1, n, 1)
            : gates.hPost.expandedDimensions(axis: -1)
        let writeTerm = postWeights * sublayerOutput.asType(.float32).expandedDimensions(axis: -2)
        guard let hRes = gates.hRes else {
            return x + writeTerm
        }
        let hRes4D = hRes.ndim == 2 ? hRes.reshaped(1, 1, n, n) : hRes
        return matmul(hRes4D, x) + writeTerm
    }

    public func callAsFunction(
        _ streams: MLXArray,
        computeDType: DType,
        body: (MLXArray) -> MLXArray
    ) -> MLXArray {
        let x = streams.dtype == .float32 ? streams : streams.asType(.float32)
        let gates = computeGates(x)
        let hIn = preRead(x, gates: gates).asType(computeDType)
        let y = body(hIn)
        return mix(x, sublayerOutput: y, gates: gates)
    }

    public func resolvedResidualMatrix(for streams: MLXArray) -> MLXArray {
        let gates = computeGates(streams)
        if let hRes = gates.hRes { return hRes }
        return MLXArray.eye(spec.expansionRate, dtype: .float32)
    }

    private func rawGateLogits(
        _ x: MLXArray
    ) -> (pre: MLXArray, post: MLXArray, res: MLXArray?) {
        guard spec.dynamic, let phi = phiWeight else {
            let res = spec.variant == .ihc ? nil : biasRes?.asType(.float32)
            return (biasPre.asType(.float32), biasPost.asType(.float32), res)
        }
        let (b, l, n, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let flat = x.reshaped(b, l, n * d)
        let invRms = rsqrt((flat * flat).mean(axis: -1, keepDims: true) + rmsNormEps)
        let projected = matmul(flat, phi.asType(.float32)) * invRms
        let pre = projected[0..., 0..., 0 ..< n] * alphaPre.asType(.float32) + biasPre.asType(.float32)
        let post = projected[0..., 0..., n ..< (2 * n)] * alphaPost.asType(.float32) + biasPost.asType(.float32)
        let res: MLXArray?
        if spec.variant != .ihc, let aRes = alphaRes, let bRes = biasRes {
            let resSlice = projected[0..., 0..., (2 * n) ..< (2 * n + n * n)].reshaped(b, l, n, n)
            res = resSlice * aRes.asType(.float32) + bRes.asType(.float32)
        } else {
            res = nil
        }
        return (pre, post, res)
    }

    private func applyGateMode(pre: MLXArray, post: MLXArray) -> (MLXArray, MLXArray) {
        switch spec.gateMode {
        case .sigmoid:
            return (sigmoid(pre), sigmoid(post) * Float(2.0))
        case .monographSoftmaxSoftplus:
            return (softmax(pre, axis: -1), MLXNN.softplus(post))
        }
    }
}

extension HyperConnectionSublayer {
    public static func expandStreams(_ x: MLXArray, rate: Int) -> MLXArray {
        let fp32 = x.asType(.float32)
        let targetShape = Array(fp32.shape.dropLast()) + [max(rate, 1), fp32.dim(-1)]
        return broadcast(fp32.expandedDimensions(axis: -2), to: targetShape)
    }

    public static func collapseStreams(
        _ streams: MLXArray,
        mode: HyperConnectionCollapseMode
    ) -> MLXArray {
        let fp32 = streams.dtype == .float32 ? streams : streams.asType(.float32)
        switch mode {
        case .mean:
            return fp32.mean(axis: -2)
        case .sum:
            return fp32.sum(axis: -2)
        }
    }

    public static func distributeProbeDelta(
        streams: MLXArray,
        collapsed: MLXArray,
        edited: MLXArray,
        mode: HyperConnectionCollapseMode
    ) -> MLXArray {
        let fp32Streams = streams.dtype == .float32 ? streams : streams.asType(.float32)
        let delta = edited.asType(.float32) - collapsed.asType(.float32)
        let laneDelta: MLXArray
        switch mode {
        case .mean:
            laneDelta = delta
        case .sum:
            laneDelta = delta / Float(max(fp32Streams.dim(-2), 1))
        }
        return fp32Streams + laneDelta.expandedDimensions(axis: -2)
    }

    private static func initialBiasPre(variant: HyperConnectionVariant, rate: Int) -> MLXArray {
        switch variant {
        case .hc:
            return MLXArray.ones([rate], dtype: .float32) * (Float(1.0) / Float(rate))
        case .mhc, .ihc:
            return MLXArray.zeros([rate], dtype: .float32)
        }
    }

    private static func initialBiasPost(variant: HyperConnectionVariant, rate: Int) -> MLXArray {
        switch variant {
        case .hc:
            return MLXArray.ones([rate], dtype: .float32)
        case .mhc, .ihc:
            return MLXArray.zeros([rate], dtype: .float32)
        }
    }

    private static func initialBiasRes(variant: HyperConnectionVariant, rate: Int) -> MLXArray? {
        switch variant {
        case .hc:
            return MLXArray.eye(rate, dtype: .float32)
        case .mhc:
            return MLXArray.eye(rate, dtype: .float32) * Float(2.0)
        case .ihc:
            return nil
        }
    }
}
