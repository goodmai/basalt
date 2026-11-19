import Foundation
import MLX
import MLXNN
import BasaltCore

public enum CausalPaddingMode: String, Sendable, Codable, Equatable, CaseIterable {
    case replicateFirstFrame = "replicate_first_frame"
    case zero = "zero"
}

public struct CausalConv3dSpec: Sendable, Equatable {
    public let inputChannels: Int
    public let outputChannels: Int
    public let kernelT: Int
    public let kernelH: Int
    public let kernelW: Int
    public let strideT: Int
    public let strideH: Int
    public let strideW: Int
    public let dilationT: Int
    public let dilationH: Int
    public let dilationW: Int
    public let groups: Int
    public let paddingMode: CausalPaddingMode
    public let bias: Bool

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: IntOrTriple = 3,
        stride: IntOrTriple = 1,
        paddingMode: CausalPaddingMode = .replicateFirstFrame,
        bias: Bool = true
    ) {
        self.inputChannels = Swift.max(1, inputChannels)
        self.outputChannels = Swift.max(1, outputChannels)
        self.kernelT = Swift.max(1, kernelSize.first)
        self.kernelH = Swift.max(1, kernelSize.second)
        self.kernelW = Swift.max(1, kernelSize.third)
        self.strideT = Swift.max(1, stride.first)
        self.strideH = Swift.max(1, stride.second)
        self.strideW = Swift.max(1, stride.third)
        self.dilationT = 1
        self.dilationH = 1
        self.dilationW = 1
        self.groups = 1
        self.paddingMode = paddingMode
        self.bias = bias
    }

    private init(base: CausalConv3dSpec, dilation: IntOrTriple, groups: Int) {
        self.inputChannels = base.inputChannels
        self.outputChannels = base.outputChannels
        self.kernelT = base.kernelT
        self.kernelH = base.kernelH
        self.kernelW = base.kernelW
        self.strideT = base.strideT
        self.strideH = base.strideH
        self.strideW = base.strideW
        self.dilationT = Swift.max(1, dilation.first)
        self.dilationH = Swift.max(1, dilation.second)
        self.dilationW = Swift.max(1, dilation.third)
        self.groups = Swift.max(1, groups)
        self.paddingMode = base.paddingMode
        self.bias = base.bias
    }

    public func withDilation(_ dilation: IntOrTriple, groups: Int = 1) -> CausalConv3dSpec {
        CausalConv3dSpec(base: self, dilation: dilation, groups: groups)
    }

    public var causalPadT: Int {
        Swift.max(0, (kernelT - 1) * dilationT)
    }

    public var effectiveKernelT: Int {
        causalPadT + 1
    }

    public var spatialPadH: Int {
        ((kernelH - 1) * dilationH) / 2
    }

    public var spatialPadW: Int {
        ((kernelW - 1) * dilationW) / 2
    }

    public func trailingCacheRange(totalInputFrames: Int) -> Range<Int> {
        guard causalPadT > 0, totalInputFrames > 0 else {
            return totalInputFrames ..< totalInputFrames
        }
        if totalInputFrames < effectiveKernelT {
            return 0 ..< totalInputFrames
        }
        let nextStart = ((totalInputFrames - effectiveKernelT) / strideT + 1) * strideT
        let clampedStart = Swift.min(Swift.max(0, nextStart), totalInputFrames)
        return clampedStart ..< totalInputFrames
    }

    public func outputSpatialSize(inputH: Int, inputW: Int) -> (Int, Int) {
        let effKH = (kernelH - 1) * dilationH + 1
        let effKW = (kernelW - 1) * dilationW + 1
        let outH = Swift.max(0, (inputH + 2 * spatialPadH - effKH) / strideH + 1)
        let outW = Swift.max(0, (inputW + 2 * spatialPadW - effKW) / strideW + 1)
        return (outH, outW)
    }
}

public struct TemporalConvCache {
    public let maxFrames: Int
    public private(set) var bufferedFrames: MLXArray?
    public private(set) var chunksProcessed: Int

    public init(maxFrames: Int) {
        self.maxFrames = Swift.max(0, maxFrames)
        self.bufferedFrames = nil
        self.chunksProcessed = 0
    }

    public var isFirstChunk: Bool {
        chunksProcessed == 0
    }

    public var cachedFrameCount: Int {
        bufferedFrames?.dim(1) ?? 0
    }

    public mutating func advance(storing tail: MLXArray?) {
        if maxFrames == 0 {
            bufferedFrames = nil
        } else if let tail, tail.dim(1) > 0 {
            if tail.dim(1) > maxFrames {
                let start = tail.dim(1) - maxFrames
                bufferedFrames = tail[0..., start ..< tail.dim(1), 0..., 0..., 0...]
            } else {
                bufferedFrames = tail
            }
        } else {
            bufferedFrames = nil
        }
        chunksProcessed += 1
    }

    public mutating func reset() {
        bufferedFrames = nil
        chunksProcessed = 0
    }
}

public struct TemporalDecodeSession: ~Copyable {
    private var caches: [TemporalConvCache] = []
    private var cursor: Int = 0
    public private(set) var chunkIndex: Int = 0

    public init() {}

    public var layerCount: Int {
        caches.count
    }

    public mutating func beginChunk() {
        cursor = 0
    }

    public mutating func endChunk() {
        chunkIndex += 1
    }

    public mutating func withNextCache<R>(
        maxFrames: Int,
        _ body: (inout TemporalConvCache) -> R
    ) -> R {
        if cursor == caches.count {
            caches.append(TemporalConvCache(maxFrames: maxFrames))
        }
        let idx = cursor
        cursor += 1
        var local = caches[idx]
        let output = body(&local)
        caches[idx] = local
        return output
    }

    public mutating func reset() {
        for idx in caches.indices {
            caches[idx].reset()
        }
        cursor = 0
        chunkIndex = 0
    }

    public consuming func release() {
        caches.removeAll()
    }
}

public final class CausalConv3d: Module {
    public let spec: CausalConv3dSpec
    @ModuleInfo(key: "conv") public var conv: Conv3d

    public init(spec: CausalConv3dSpec) {
        self.spec = spec
        self._conv.wrappedValue = Conv3d(
            inputChannels: spec.inputChannels,
            outputChannels: spec.outputChannels,
            kernelSize: IntOrTriple([spec.kernelT, spec.kernelH, spec.kernelW]),
            stride: IntOrTriple([spec.strideT, spec.strideH, spec.strideW]),
            padding: IntOrTriple([0, spec.spatialPadH, spec.spatialPadW]),
            dilation: IntOrTriple([spec.dilationT, spec.dilationH, spec.dilationW]),
            groups: spec.groups,
            bias: spec.bias
        )
    }

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: IntOrTriple = 3,
        stride: IntOrTriple = 1,
        paddingMode: CausalPaddingMode = .replicateFirstFrame,
        bias: Bool = true
    ) {
        let resolved = CausalConv3dSpec(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            paddingMode: paddingMode,
            bias: bias
        )
        self.spec = resolved
        self._conv.wrappedValue = Conv3d(
            inputChannels: resolved.inputChannels,
            outputChannels: resolved.outputChannels,
            kernelSize: IntOrTriple([resolved.kernelT, resolved.kernelH, resolved.kernelW]),
            stride: IntOrTriple([resolved.strideT, resolved.strideH, resolved.strideW]),
            padding: IntOrTriple([0, resolved.spatialPadH, resolved.spatialPadW]),
            dilation: IntOrTriple([resolved.dilationT, resolved.dilationH, resolved.dilationW]),
            groups: resolved.groups,
            bias: resolved.bias
        )
    }

    public func makeCache() -> TemporalConvCache {
        TemporalConvCache(maxFrames: spec.causalPadT)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard spec.causalPadT > 0, x.dim(1) > 0 else {
            return conv(x)
        }
        return conv(applyInitialPadding(x))
    }

    public func callAsFunction(_ x: MLXArray, cache: inout TemporalConvCache) -> MLXArray {
        guard x.dim(1) > 0 else { return x }
        let fullInput: MLXArray
        if spec.causalPadT == 0 {
            fullInput = x
        } else if let history = cache.bufferedFrames {
            fullInput = concatenated([history, x], axis: 1)
        } else {
            fullInput = applyInitialPadding(x)
        }

        let tailRange = spec.trailingCacheRange(totalInputFrames: fullInput.dim(1))
        let tailSlice = tailRange.isEmpty ? nil : fullInput[0..., tailRange, 0..., 0..., 0...]
        cache.advance(storing: tailSlice)

        if fullInput.dim(1) < spec.effectiveKernelT {
            let (outH, outW) = spec.outputSpatialSize(inputH: x.dim(2), inputW: x.dim(3))
            return MLXArray.zeros([x.dim(0), 0, outH, outW, spec.outputChannels], dtype: x.dtype)
        }
        return conv(fullInput)
    }

    public func forwardCached(_ x: MLXArray, session: inout TemporalDecodeSession) -> MLXArray {
        session.withNextCache(maxFrames: spec.causalPadT) { cache in
            self(x, cache: &cache)
        }
    }

    private func applyInitialPadding(_ x: MLXArray) -> MLXArray {
        switch spec.paddingMode {
        case .replicateFirstFrame:
            let firstFrame = x[0..., 0 ..< 1, 0..., 0..., 0...]
            let prefix = MLX.repeated(firstFrame, count: spec.causalPadT, axis: 1)
            return concatenated([prefix, x], axis: 1)
        case .zero:
            return padded(
                x,
                widths: [
                    IntOrPair((0, 0)),
                    IntOrPair((spec.causalPadT, 0)),
                    IntOrPair((0, 0)),
                    IntOrPair((0, 0)),
                    IntOrPair((0, 0))
                ]
            )
        }
    }
}
