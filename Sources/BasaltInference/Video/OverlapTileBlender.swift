import Foundation
import MLX
import BasaltCore

public struct SpatialTileConfig: Sendable, Codable, Equatable {
    public var tileLatentHeight: Int
    public var tileLatentWidth: Int
    public var overlapLatentHeight: Int
    public var overlapLatentWidth: Int
    public var haloLatents: Int?

    public static let uma24GB = SpatialTileConfig(
        tileLatentHeight: 64,
        tileLatentWidth: 64,
        overlapLatentHeight: 16,
        overlapLatentWidth: 16,
        haloLatents: nil
    )

    public init(
        tileLatentHeight: Int = 64,
        tileLatentWidth: Int = 64,
        overlapLatentHeight: Int = 16,
        overlapLatentWidth: Int = 16,
        haloLatents: Int? = nil
    ) {
        let clampedH = Swift.max(2, tileLatentHeight)
        let clampedW = Swift.max(2, tileLatentWidth)
        self.tileLatentHeight = clampedH
        self.tileLatentWidth = clampedW
        self.overlapLatentHeight = Swift.min(Swift.max(0, overlapLatentHeight), clampedH / 2)
        self.overlapLatentWidth = Swift.min(Swift.max(0, overlapLatentWidth), clampedW / 2)
        self.haloLatents = haloLatents.map { Swift.max(0, $0) }
    }

    public func resolvedHalo(defaultHalo: Int) -> Int {
        haloLatents ?? Swift.max(0, defaultHalo)
    }
}

public struct SpatialTileDescriptor: Sendable, Equatable {
    public let row: Int
    public let col: Int
    public let latentY: Range<Int>
    public let latentX: Range<Int>
    public let fetchLatentY: Range<Int>
    public let fetchLatentX: Range<Int>
    public let pixelY: Range<Int>
    public let pixelX: Range<Int>
    public let cropPixelY: Range<Int>
    public let cropPixelX: Range<Int>
    public let topOverlapPixels: Int
    public let bottomOverlapPixels: Int
    public let leftOverlapPixels: Int
    public let rightOverlapPixels: Int
}

public struct SpatialTileGrid: Sendable, Equatable {
    public let latentHeight: Int
    public let latentWidth: Int
    public let spatialScale: Int
    public let rowCount: Int
    public let colCount: Int
    public let tiles: [SpatialTileDescriptor]

    public var pixelHeight: Int { latentHeight * spatialScale }
    public var pixelWidth: Int { latentWidth * spatialScale }

    public init(
        latentHeight: Int,
        latentWidth: Int,
        spatialScale: Int,
        config: SpatialTileConfig,
        defaultHaloLatents: Int = 8
    ) {
        let totalH = Swift.max(1, latentHeight)
        let totalW = Swift.max(1, latentWidth)
        let scale = Swift.max(1, spatialScale)
        let halo = config.resolvedHalo(defaultHalo: defaultHaloLatents)
        let ySpans = Self.axisSpans(total: totalH, tileSize: config.tileLatentHeight, overlap: config.overlapLatentHeight)
        let xSpans = Self.axisSpans(total: totalW, tileSize: config.tileLatentWidth, overlap: config.overlapLatentWidth)

        self.latentHeight = totalH
        self.latentWidth = totalW
        self.spatialScale = scale
        self.rowCount = ySpans.count
        self.colCount = xSpans.count
        self.tiles = Self.buildDescriptors(
            ySpans: ySpans,
            xSpans: xSpans,
            totalH: totalH,
            totalW: totalW,
            scale: scale,
            halo: halo
        )
    }

    static func axisSpans(total: Int, tileSize: Int, overlap: Int) -> [Range<Int>] {
        guard total > 0 else { return [] }
        let clampedTile = Swift.max(1, Swift.min(total, tileSize))
        if clampedTile >= total {
            return [0 ..< total]
        }
        let effOverlap = Swift.min(Swift.max(0, overlap), clampedTile / 2)
        let step = Swift.max(1, clampedTile - effOverlap)
        var spans: [Range<Int>] = []
        var start = 0
        while start < total {
            let end = Swift.min(total, start + clampedTile)
            spans.append(start ..< end)
            if end == total { break }
            start += step
        }
        return spans
    }

    private static func buildDescriptors(
        ySpans: [Range<Int>],
        xSpans: [Range<Int>],
        totalH: Int,
        totalW: Int,
        scale: Int,
        halo: Int
    ) -> [SpatialTileDescriptor] {
        var list: [SpatialTileDescriptor] = []
        list.reserveCapacity(ySpans.count * xSpans.count)
        for (r, ySpan) in ySpans.enumerated() {
            let fetchY = Swift.max(0, ySpan.lowerBound - halo) ..< Swift.min(totalH, ySpan.upperBound + halo)
            let topOv = r == 0 ? 0 : (ySpans[r - 1].upperBound - ySpan.lowerBound) * scale
            let botOv = r == ySpans.count - 1 ? 0 : (ySpan.upperBound - ySpans[r + 1].lowerBound) * scale
            let cropY = ((ySpan.lowerBound - fetchY.lowerBound) * scale) ..< ((ySpan.upperBound - fetchY.lowerBound) * scale)
            for (c, xSpan) in xSpans.enumerated() {
                let fetchX = Swift.max(0, xSpan.lowerBound - halo) ..< Swift.min(totalW, xSpan.upperBound + halo)
                let leftOv = c == 0 ? 0 : (xSpans[c - 1].upperBound - xSpan.lowerBound) * scale
                let rightOv = c == xSpans.count - 1 ? 0 : (xSpan.upperBound - xSpans[c + 1].lowerBound) * scale
                let cropX = ((xSpan.lowerBound - fetchX.lowerBound) * scale) ..< ((xSpan.upperBound - fetchX.lowerBound) * scale)
                list.append(
                    SpatialTileDescriptor(
                        row: r,
                        col: c,
                        latentY: ySpan,
                        latentX: xSpan,
                        fetchLatentY: fetchY,
                        fetchLatentX: fetchX,
                        pixelY: (ySpan.lowerBound * scale) ..< (ySpan.upperBound * scale),
                        pixelX: (xSpan.lowerBound * scale) ..< (xSpan.upperBound * scale),
                        cropPixelY: cropY,
                        cropPixelX: cropX,
                        topOverlapPixels: topOv,
                        bottomOverlapPixels: botOv,
                        leftOverlapPixels: leftOv,
                        rightOverlapPixels: rightOv
                    )
                )
            }
        }
        return list
    }

    public func weightSumMatrix() -> [[Float]] {
        var sums = Array(repeating: Array(repeating: Float(0), count: pixelWidth), count: pixelHeight)
        for tile in tiles {
            let window = TileWindow(for: tile)
            for localY in 0 ..< window.height {
                let globalY = tile.pixelY.lowerBound + localY
                let wy = window.rowWeights[localY]
                for localX in 0 ..< window.width {
                    let globalX = tile.pixelX.lowerBound + localX
                    sums[globalY][globalX] += wy * window.colWeights[localX]
                }
            }
        }
        return sums
    }

    public func accumulatedWeightExtrema() -> (min: Float, max: Float) {
        var rowSums = [Float](repeating: 0, count: pixelHeight)
        var colSums = [Float](repeating: 0, count: pixelWidth)
        for r in 0 ..< rowCount {
            let tile = tiles[r * colCount]
            let wy = TileWindow.raisedCosineAxis(
                length: tile.pixelY.count,
                leadingOverlap: tile.topOverlapPixels,
                trailingOverlap: tile.bottomOverlapPixels
            )
            for idx in 0 ..< wy.count {
                rowSums[tile.pixelY.lowerBound + idx] += wy[idx]
            }
        }
        for c in 0 ..< colCount {
            let tile = tiles[c]
            let wx = TileWindow.raisedCosineAxis(
                length: tile.pixelX.count,
                leadingOverlap: tile.leftOverlapPixels,
                trailingOverlap: tile.rightOverlapPixels
            )
            for idx in 0 ..< wx.count {
                colSums[tile.pixelX.lowerBound + idx] += wx[idx]
            }
        }
        let minRow = rowSums.min() ?? 0
        let maxRow = rowSums.max() ?? 0
        let minCol = colSums.min() ?? 0
        let maxCol = colSums.max() ?? 0
        return (minRow * minCol, maxRow * maxCol)
    }
}

public struct TileWindow: Sendable, Equatable {
    public let height: Int
    public let width: Int
    public let rowWeights: [Float]
    public let colWeights: [Float]

    public init(
        height: Int,
        width: Int,
        topOverlap: Int,
        bottomOverlap: Int,
        leftOverlap: Int,
        rightOverlap: Int
    ) {
        self.height = Swift.max(1, height)
        self.width = Swift.max(1, width)
        self.rowWeights = Self.raisedCosineAxis(
            length: self.height,
            leadingOverlap: topOverlap,
            trailingOverlap: bottomOverlap
        )
        self.colWeights = Self.raisedCosineAxis(
            length: self.width,
            leadingOverlap: leftOverlap,
            trailingOverlap: rightOverlap
        )
    }

    public init(for tile: SpatialTileDescriptor) {
        let h = Swift.max(1, tile.pixelY.count)
        let w = Swift.max(1, tile.pixelX.count)
        self.height = h
        self.width = w
        self.rowWeights = Self.raisedCosineAxis(
            length: h,
            leadingOverlap: tile.topOverlapPixels,
            trailingOverlap: tile.bottomOverlapPixels
        )
        self.colWeights = Self.raisedCosineAxis(
            length: w,
            leadingOverlap: tile.leftOverlapPixels,
            trailingOverlap: tile.rightOverlapPixels
        )
    }

    public static func raisedCosineAxis(
        length: Int,
        leadingOverlap: Int,
        trailingOverlap: Int
    ) -> [Float] {
        guard length > 0 else { return [] }
        var weights = [Float](repeating: 1.0, count: length)
        let lead = Swift.min(Swift.max(0, leadingOverlap), length)
        let trail = Swift.min(Swift.max(0, trailingOverlap), length - lead)
        if lead > 0 {
            for idx in 0 ..< lead {
                let u = (Double(idx) + 0.5) / Double(lead)
                let s = Foundation.sin(0.5 * Double.pi * u)
                weights[idx] = Float(s * s)
            }
        }
        if trail > 0 {
            let offset = length - trail
            for idx in 0 ..< trail {
                let u = (Double(idx) + 0.5) / Double(trail)
                let c = Foundation.cos(0.5 * Double.pi * u)
                weights[offset + idx] = Float(c * c)
            }
        }
        return weights
    }

    public func weight(y: Int, x: Int) -> Float {
        rowWeights[y] * colWeights[x]
    }

    public func asMLXArray(dtype: DType = .float32) -> MLXArray {
        let wy = MLXArray(rowWeights).reshaped(1, 1, height, 1, 1)
        let wx = MLXArray(colWeights).reshaped(1, 1, 1, width, 1)
        return (wy * wx).asType(dtype)
    }
}

public enum OverlapTileBlender {
    public static func decodeTiled(
        latents: MLXArray,
        grid: SpatialTileGrid,
        outputChannels: Int,
        decodeTile: (MLXArray) throws(BasaltError) -> MLXArray
    ) throws(BasaltError) -> MLXArray {
        guard latents.ndim == 5 else {
            throw .invalidRequestStructure(details: "OverlapTileBlender expects 5D [B, T, H, W, C] latents, got \(latents.ndim)D")
        }
        guard latents.dim(2) == grid.latentHeight, latents.dim(3) == grid.latentWidth else {
            throw .invalidRequestStructure(details: "Latent spatial dimensions (\(latents.dim(2))x\(latents.dim(3))) do not match SpatialTileGrid (\(grid.latentHeight)x\(grid.latentWidth))")
        }
        guard !grid.tiles.isEmpty else {
            throw .invalidRequestStructure(details: "SpatialTileGrid has no tiles")
        }
        if grid.tiles.count == 1 {
            return try decodeTile(latents)
        }

        let firstTile = grid.tiles[0]
        let firstDecoded = try decodeTile(latents[0..., 0..., firstTile.fetchLatentY, firstTile.fetchLatentX, 0...])
        let outFrames = firstDecoded.dim(1)
        let outDType = firstDecoded.dtype
        var accumulator = MLXArray.zeros(
            [latents.dim(0), outFrames, grid.pixelHeight, grid.pixelWidth, outputChannels],
            dtype: .float32
        )
        blendTileChunk(
            into: &accumulator,
            rawDecoded: firstDecoded,
            tile: firstTile,
            window: TileWindow(for: firstTile).asMLXArray(dtype: .float32),
            frameRange: 0 ..< outFrames
        )
        eval(accumulator)

        for idx in 1 ..< grid.tiles.count {
            let tile = grid.tiles[idx]
            let tileInput = latents[0..., 0..., tile.fetchLatentY, tile.fetchLatentX, 0...]
            let decoded = try decodeTile(tileInput)
            let window = TileWindow(for: tile).asMLXArray(dtype: .float32)
            blendTileChunk(
                into: &accumulator,
                rawDecoded: decoded,
                tile: tile,
                window: window,
                frameRange: 0 ..< outFrames
            )
            eval(accumulator)
        }
        return accumulator.asType(outDType)
    }

    public static func blendTileChunk(
        into accumulator: inout MLXArray,
        rawDecoded: MLXArray,
        tile: SpatialTileDescriptor,
        window: MLXArray,
        frameRange: Range<Int>
    ) {
        let cropped = rawDecoded[0..., 0..., tile.cropPixelY, tile.cropPixelX, 0...].asType(.float32)
        let weighted = cropped * window
        accumulator = accumulator.at[0..., frameRange, tile.pixelY, tile.pixelX, 0...].add(weighted)
    }
}
