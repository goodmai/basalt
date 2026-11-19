import BasaltCore
import Foundation
import MLX

package enum SpatialCoordinateMode: Sendable, Equatable {
    case centeredSymmetric
    case unitInterval
    case patchIndex

    package func coordinate(index: Int, extent: Int) -> Float {
        let clampedExtent = max(1, extent)
        switch self {
        case .centeredSymmetric:
            return ((Float(index) + 0.5) / Float(clampedExtent)) * 2.0 - 1.0
        case .unitInterval:
            return (Float(index) + 0.5) / Float(clampedExtent)
        case .patchIndex:
            return Float(index)
        }
    }
}

package struct Fractional3DRoPEAxes: Sendable, Equatable {
    package let temporalDim: Int
    package let heightDim: Int
    package let widthDim: Int
    package let temporalTheta: Float
    package let spatialTheta: Float

    package var headDim: Int {
        temporalDim + heightDim + widthDim
    }

    package init(
        temporalDim: Int,
        heightDim: Int,
        widthDim: Int,
        temporalTheta: Float = 10_000.0,
        spatialTheta: Float = 10_000.0
    ) throws(BasaltError) {
        guard temporalDim >= 2, temporalDim.isMultiple(of: 2),
              heightDim >= 0, heightDim.isMultiple(of: 2),
              widthDim >= 0, widthDim.isMultiple(of: 2)
        else {
            throw .invalidRequestStructure(
                details: "Fractional3DRoPEAxes requires even (d_t >= 2, d_y >= 0, d_x >= 0)"
            )
        }
        guard temporalTheta > 0, temporalTheta.isFinite,
              spatialTheta > 0, spatialTheta.isFinite
        else {
            throw .invalidRequestStructure(details: "Fractional3DRoPEAxes requires positive finite theta")
        }
        self.temporalDim = temporalDim
        self.heightDim = heightDim
        self.widthDim = widthDim
        self.temporalTheta = temporalTheta
        self.spatialTheta = spatialTheta
    }

    package static func balanced(
        headDim: Int,
        temporalTheta: Float = 10_000.0,
        spatialTheta: Float = 10_000.0
    ) throws(BasaltError) -> Fractional3DRoPEAxes {
        guard headDim >= 2, headDim.isMultiple(of: 2) else {
            throw .invalidRequestStructure(details: "Fractional3DRoPEAxes.balanced requires even headDim >= 2")
        }
        let totalPairs = headDim / 2
        let spatialPairs = totalPairs / 3
        let temporalPairs = totalPairs - 2 * spatialPairs
        return try Fractional3DRoPEAxes(
            temporalDim: temporalPairs * 2,
            heightDim: spatialPairs * 2,
            widthDim: spatialPairs * 2,
            temporalTheta: temporalTheta,
            spatialTheta: spatialTheta
        )
    }
}

package struct VideoPatchTimeline: Sendable, Equatable {
    package let frames: Int
    package let height: Int
    package let width: Int
    package let temporalStride: Int
    package let fps: Double
    package let spatialMode: SpatialCoordinateMode

    package var tokenCount: Int {
        frames * height * width
    }

    package init(
        frames: Int,
        height: Int,
        width: Int,
        temporalStride: Int = 1,
        fps: Double = 24.0,
        spatialMode: SpatialCoordinateMode = .centeredSymmetric
    ) throws(BasaltError) {
        guard frames >= 1, height >= 1, width >= 1, temporalStride >= 1, fps > 0, fps.isFinite else {
            throw .invalidRequestStructure(details: "Invalid VideoPatchTimeline dimensions or fps")
        }
        self.frames = frames
        self.height = height
        self.width = width
        self.temporalStride = temporalStride
        self.fps = fps
        self.spatialMode = spatialMode
    }

    package func timeSeconds(frame: Int) -> Double {
        Double(frame * temporalStride) / fps
    }
}

package struct AudioLatentTimeline: Sendable, Equatable {
    package let frameCount: Int
    package let hopSamples: Int
    package let sampleRate: Double

    package init(
        frameCount: Int,
        hopSamples: Int,
        sampleRate: Double
    ) throws(BasaltError) {
        guard frameCount >= 1, hopSamples >= 1, sampleRate > 0, sampleRate.isFinite else {
            throw .invalidRequestStructure(details: "Invalid AudioLatentTimeline parameters")
        }
        self.frameCount = frameCount
        self.hopSamples = hopSamples
        self.sampleRate = sampleRate
    }

    package func timeSeconds(frame: Int) -> Double {
        Double(frame * hopSamples) / sampleRate
    }

    package func nearestFrame(atSeconds targetSeconds: Double) -> Int {
        let exact = targetSeconds * sampleRate / Double(hopSamples)
        return max(0, min(frameCount - 1, Int(exact.rounded())))
    }
}

package struct MultimodalCoordinate3D: Sendable, Equatable {
    package let timeSeconds: Double
    package let yNorm: Float
    package let xNorm: Float

    package init(timeSeconds: Double, yNorm: Float, xNorm: Float) {
        self.timeSeconds = timeSeconds
        self.yNorm = yNorm
        self.xNorm = xNorm
    }
}

package struct CPU3DRotaryShape: Sendable, Equatable {
    package let batch: Int
    package let heads: Int
    package let tokens: Int
    package let headDim: Int

    package init(batch: Int, heads: Int, tokens: Int, headDim: Int) {
        self.batch = batch
        self.heads = heads
        self.tokens = tokens
        self.headDim = headDim
    }
}

package enum Fractional3DMultimodalRoPE {
    package static func inverseFrequencies(dim: Int, theta: Float) -> [Double] {
        let pairs = dim / 2
        guard pairs > 0 else { return [] }
        let base = Double(theta)
        return (0 ..< pairs).map { idx in
            Foundation.pow(base, -Double(idx) / Double(pairs))
        }
    }

    package static func videoCoordinates(_ timeline: VideoPatchTimeline) -> [MultimodalCoordinate3D] {
        var coords: [MultimodalCoordinate3D] = []
        coords.reserveCapacity(timeline.tokenCount)
        let ys = (0 ..< timeline.height).map { timeline.spatialMode.coordinate(index: $0, extent: timeline.height) }
        let xs = (0 ..< timeline.width).map { timeline.spatialMode.coordinate(index: $0, extent: timeline.width) }
        for f in 0 ..< timeline.frames {
            let tSec = timeline.timeSeconds(frame: f)
            for y in ys {
                for x in xs {
                    coords.append(MultimodalCoordinate3D(timeSeconds: tSec, yNorm: y, xNorm: x))
                }
            }
        }
        return coords
    }

    package static func audioCoordinates(_ timeline: AudioLatentTimeline) -> [MultimodalCoordinate3D] {
        (0 ..< timeline.frameCount).map { m in
            MultimodalCoordinate3D(timeSeconds: timeline.timeSeconds(frame: m), yNorm: 0.0, xNorm: 0.0)
        }
    }

    package static func videoFrequencies(
        axes: Fractional3DRoPEAxes,
        timeline: VideoPatchTimeline
    ) -> (cos: MLXArray, sin: MLXArray) {
        let tTimes = (0 ..< timeline.frames).map { Float(timeline.timeSeconds(frame: $0)) }
        let yVals = (0 ..< timeline.height).map { timeline.spatialMode.coordinate(index: $0, extent: timeline.height) }
        let xVals = (0 ..< timeline.width).map { timeline.spatialMode.coordinate(index: $0, extent: timeline.width) }

        var parts: [MLXArray] = [
            broadcastAxisAngles(
                positions: tTimes,
                dim: axes.temporalDim,
                theta: axes.temporalTheta,
                axis: 0,
                gridShape: [timeline.frames, timeline.height, timeline.width]
            )
        ]
        if axes.heightDim > 0 {
            parts.append(
                broadcastAxisAngles(
                    positions: yVals,
                    dim: axes.heightDim,
                    theta: axes.spatialTheta,
                    axis: 1,
                    gridShape: [timeline.frames, timeline.height, timeline.width]
                )
            )
        }
        if axes.widthDim > 0 {
            parts.append(
                broadcastAxisAngles(
                    positions: xVals,
                    dim: axes.widthDim,
                    theta: axes.spatialTheta,
                    axis: 2,
                    gridShape: [timeline.frames, timeline.height, timeline.width]
                )
            )
        }
        let combined = (parts.count == 1 ? parts[0] : concatenated(parts, axis: -1))
            .reshaped(1, 1, timeline.tokenCount, axes.headDim)
        return (MLX.cos(combined), MLX.sin(combined))
    }

    package static func audioFrequencies(
        axes: Fractional3DRoPEAxes,
        timeline: AudioLatentTimeline
    ) -> (cos: MLXArray, sin: MLXArray) {
        let tTimes = (0 ..< timeline.frameCount).map { Float(timeline.timeSeconds(frame: $0)) }
        let invT = inverseFrequencies(dim: axes.temporalDim, theta: axes.temporalTheta).map { Float($0) }
        let pairAngles = outer(MLXArray(tTimes), MLXArray(invT))
        let tAngles = stacked([pairAngles, pairAngles], axis: -1)
            .reshaped(timeline.frameCount, axes.temporalDim)
        let cosT = MLX.cos(tAngles)
        let sinT = MLX.sin(tAngles)
        let spatialDim = axes.heightDim + axes.widthDim
        guard spatialDim > 0 else {
            return (
                cosT.reshaped(1, 1, timeline.frameCount, axes.headDim),
                sinT.reshaped(1, 1, timeline.frameCount, axes.headDim)
            )
        }
        let cosSpatial = MLXArray.ones([timeline.frameCount, spatialDim], dtype: .float32)
        let sinSpatial = MLXArray.zeros([timeline.frameCount, spatialDim], dtype: .float32)
        return (
            concatenated([cosT, cosSpatial], axis: -1).reshaped(1, 1, timeline.frameCount, axes.headDim),
            concatenated([sinT, sinSpatial], axis: -1).reshaped(1, 1, timeline.frameCount, axes.headDim)
        )
    }

    package static func jointAudioVideoFrequencies(
        axes: Fractional3DRoPEAxes,
        video: VideoPatchTimeline,
        audio: AudioLatentTimeline,
        audioFirst: Bool = false
    ) -> (cos: MLXArray, sin: MLXArray) {
        let vFreq = videoFrequencies(axes: axes, timeline: video)
        let aFreq = audioFrequencies(axes: axes, timeline: audio)
        if audioFirst {
            return (
                concatenated([aFreq.cos, vFreq.cos], axis: 2),
                concatenated([aFreq.sin, vFreq.sin], axis: 2)
            )
        }
        return (
            concatenated([vFreq.cos, aFreq.cos], axis: 2),
            concatenated([vFreq.sin, aFreq.sin], axis: 2)
        )
    }

    package static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        if x.ndim == 2 {
            return apply(x.expandedDimensions(axes: [0, 1]), cos: cos, sin: sin).squeezed(axes: [0, 1])
        }
        if x.ndim == 3 {
            return apply(x.expandedDimensions(axis: 1), cos: cos, sin: sin).squeezed(axis: 1)
        }
        let b = x.dim(0)
        let h = x.dim(1)
        let n = x.dim(2)
        let d = x.dim(3)
        let x32 = x.asType(.float32)
        let pairs = x32.reshaped(b, h, n, d / 2, 2)
        let xEven = pairs[0..., 0..., 0..., 0..., 0]
        let xOdd = pairs[0..., 0..., 0..., 0..., 1]
        let rotated = stacked([-xOdd, xEven], axis: -1).reshaped(b, h, n, d)
        return (x32 * cos + rotated * sin).asType(x.dtype)
    }
}

extension Fractional3DMultimodalRoPE {
    package static func temporalPhaseDifferenceRadians(
        videoTimeSeconds: Double,
        audioTimeSeconds: Double,
        axes: Fractional3DRoPEAxes
    ) -> Float {
        let delta = abs(videoTimeSeconds - audioTimeSeconds)
        let invT = inverseFrequencies(dim: axes.temporalDim, theta: axes.temporalTheta)
        var peak: Double = 0.0
        for omega in invT {
            peak = max(peak, delta * omega)
        }
        return Float(peak)
    }

    package static func cpuReferenceApply(
        _ x: [Float],
        shape: CPU3DRotaryShape,
        coordinates: [MultimodalCoordinate3D],
        axes: Fractional3DRoPEAxes
    ) -> [Float] {
        let total = shape.batch * shape.heads * shape.tokens * shape.headDim
        guard x.count == total, coordinates.count == shape.tokens, shape.headDim == axes.headDim else {
            return x
        }
        let tables = cpuRotaryTables(coordinates: coordinates, axes: axes)
        var out = [Float](repeating: 0, count: total)
        let pairCount = shape.headDim / 2
        for bh in 0 ..< (shape.batch * shape.heads) {
            let headBase = bh * shape.tokens * shape.headDim
            for tok in 0 ..< shape.tokens {
                let tokBase = headBase + tok * shape.headDim
                let tabBase = tok * pairCount
                for p in 0 ..< pairCount {
                    let u0 = x[tokBase + 2 * p]
                    let u1 = x[tokBase + 2 * p + 1]
                    let c = tables.cos[tabBase + p]
                    let s = tables.sin[tabBase + p]
                    out[tokBase + 2 * p] = u0 * c - u1 * s
                    out[tokBase + 2 * p + 1] = u0 * s + u1 * c
                }
            }
        }
        return out
    }

    private static func broadcastAxisAngles(
        positions: [Float],
        dim: Int,
        theta: Float,
        axis: Int,
        gridShape: [Int]
    ) -> MLXArray {
        let inv = inverseFrequencies(dim: dim, theta: theta).map { Float($0) }
        let pairAngles = outer(MLXArray(positions), MLXArray(inv))
        let repeatedAngles = stacked([pairAngles, pairAngles], axis: -1)
            .reshaped(positions.count, dim)
        var viewShape = [1, 1, 1, dim]
        viewShape[axis] = positions.count
        return broadcast(repeatedAngles.reshaped(viewShape), to: [gridShape[0], gridShape[1], gridShape[2], dim])
    }

    private static func cpuRotaryTables(
        coordinates: [MultimodalCoordinate3D],
        axes: Fractional3DRoPEAxes
    ) -> (cos: [Float], sin: [Float]) {
        let invT = inverseFrequencies(dim: axes.temporalDim, theta: axes.temporalTheta)
        let invY = inverseFrequencies(dim: axes.heightDim, theta: axes.spatialTheta)
        let invX = inverseFrequencies(dim: axes.widthDim, theta: axes.spatialTheta)
        let pairsPerToken = axes.headDim / 2
        var cosVals: [Float] = []
        var sinVals: [Float] = []
        cosVals.reserveCapacity(coordinates.count * pairsPerToken)
        sinVals.reserveCapacity(coordinates.count * pairsPerToken)
        for coord in coordinates {
            for omega in invT {
                let angle = coord.timeSeconds * omega
                cosVals.append(Float(Foundation.cos(angle)))
                sinVals.append(Float(Foundation.sin(angle)))
            }
            for omega in invY {
                let angle = Double(coord.yNorm) * omega
                cosVals.append(Float(Foundation.cos(angle)))
                sinVals.append(Float(Foundation.sin(angle)))
            }
            for omega in invX {
                let angle = Double(coord.xNorm) * omega
                cosVals.append(Float(Foundation.cos(angle)))
                sinVals.append(Float(Foundation.sin(angle)))
            }
        }
        return (cosVals, sinVals)
    }
}
