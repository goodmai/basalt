import Foundation
import MLX
import MLXNN
import BasaltCore

public struct SeedVR2Config: Sendable, Codable, Equatable {
    public var modelID: String
    public var hiddenSize: Int
    public var numHeads: Int
    public var numLayers: Int
    public var maxScaleFactor: Int
    public var defaultTileSize: Int
    public var defaultOverlap: Int
    public var quantizationBits: Int
    public var initSeed: UInt64

    public init(
        modelID: String = "ByteDance-Seed/SeedVR2-3B",
        hiddenSize: Int = 48,
        numHeads: Int = 4,
        numLayers: Int = 2,
        defaultTileSize: Int = 128,
        defaultOverlap: Int = 24
    ) {
        let h = max(1, numHeads)
        let aligned = max(4, ((hiddenSize / h + 3) / 4) * 4)
        self.modelID = modelID
        self.numHeads = h
        self.hiddenSize = h * aligned
        self.numLayers = max(1, numLayers)
        self.maxScaleFactor = 4
        self.defaultTileSize = max(32, defaultTileSize)
        self.defaultOverlap = max(4, min(defaultTileSize / 2, defaultOverlap))
        self.quantizationBits = 8
        self.initSeed = 913
    }
}

package enum SeedVR2TileBlender {
    package struct TileSpan: Sendable, Equatable {
        package let start: Int
        package let length: Int
    }

    package static func computeAxisTiles(total: Int, tileSize: Int, overlap: Int) -> [TileSpan] {
        let clampedTile = max(8, min(total, tileSize))
        guard clampedTile < total else {
            return [TileSpan(start: 0, length: total)]
        }
        let clampedOverlap = max(2, min(clampedTile / 2, overlap))
        let strideStep = max(1, clampedTile - clampedOverlap)
        var spans: [TileSpan] = []
        var pos = 0
        while pos + clampedTile < total {
            spans.append(TileSpan(start: pos, length: clampedTile))
            pos += strideStep
        }
        let lastStart = max(0, total - clampedTile)
        if spans.last?.start != lastStart {
            spans.append(TileSpan(start: lastStart, length: clampedTile))
        }
        return spans
    }

    package static func featherWindow1D(
        length: Int,
        overlap: Int,
        fadeStart: Bool,
        fadeEnd: Bool
    ) -> [Float] {
        guard length > 0 else { return [] }
        let ramp = min(overlap, length / 2)
        guard ramp > 0 else { return [Float](repeating: 1.0, count: length) }
        var weights = [Float](repeating: 1.0, count: length)
        if fadeStart {
            for idx in 0 ..< ramp {
                let phase = Double(idx + 1) / Double(ramp + 1)
                weights[idx] = Float(0.5 - 0.5 * cos(.pi * phase))
            }
        }
        if fadeEnd {
            for idx in 0 ..< ramp {
                let phase = Double(idx + 1) / Double(ramp + 1)
                weights[length - 1 - idx] *= Float(0.5 - 0.5 * cos(.pi * phase))
            }
        }
        return weights
    }

    package struct TileEdgeMask: Sendable, Equatable {
        package let top: Bool
        package let bottom: Bool
        package let left: Bool
        package let right: Bool
    }

    package static func featherWindow2D(
        height: Int,
        width: Int,
        overlapH: Int,
        overlapW: Int,
        edges: TileEdgeMask
    ) -> MLXArray {
        let wy = featherWindow1D(length: height, overlap: overlapH, fadeStart: edges.top, fadeEnd: edges.bottom)
        let wx = featherWindow1D(length: width, overlap: overlapW, fadeStart: edges.left, fadeEnd: edges.right)
        let yArr = MLXArray(wy).reshaped(1, height, 1, 1)
        let xArr = MLXArray(wx).reshaped(1, 1, width, 1)
        return yArr * xArr
    }

    package static func upsampleBilinearNHWC(_ input: MLXArray, scaleFactor: Int) -> MLXArray {
        guard scaleFactor > 1 else { return input }
        let h = input.dim(1)
        let w = input.dim(2)
        let outH = h * scaleFactor
        let outW = w * scaleFactor
        let grid = TinyAutoencoderPreview.upsampleBilinearGrid(input[0], height: outH, width: outW)
        return grid[.newAxis]
    }
}

final class SeedVR2Module: Module {
    let config: SeedVR2Config
    @ModuleInfo(key: "feat_in") var featIn: Conv2d
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "blocks") var blocks: [LinearAttentionSanaBlock]
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm
    @ModuleInfo(key: "up2_head") var up2Head: Conv2d
    @ModuleInfo(key: "up4_head") var up4Head: Conv2d

    init(config: SeedVR2Config) {
        self.config = config
        let d = config.hiddenSize
        self._featIn.wrappedValue = Conv2d(inputChannels: 4, outputChannels: d, kernelSize: 3, padding: 1)
        self._condProj.wrappedValue = Linear(4, d)
        self._blocks.wrappedValue = (0 ..< config.numLayers).map { _ in
            LinearAttentionSanaBlock(dimensions: d, heads: config.numHeads)
        }
        self._normOut.wrappedValue = LayerNorm(dimensions: d)
        self._up2Head.wrappedValue = Conv2d(inputChannels: d, outputChannels: 4 * 4, kernelSize: 3, padding: 1)
        self._up4Head.wrappedValue = Conv2d(inputChannels: d, outputChannels: 4 * 16, kernelSize: 3, padding: 1)
    }

    func upscaleTile(_ tile: MLXArray, scaleFactor: Int) -> MLXArray {
        let channels = tile.dim(3)
        let rgba: MLXArray
        if channels == 4 {
            rgba = tile
        } else {
            let alpha = MLXArray.ones([tile.dim(0), tile.dim(1), tile.dim(2), 1], dtype: tile.dtype)
            rgba = concatenated([tile[0..., 0..., 0..., ..<3], alpha], axis: 3)
        }
        let baseUp = SeedVR2TileBlender.upsampleBilinearNHWC(rgba, scaleFactor: scaleFactor)
        guard scaleFactor == 2 || scaleFactor == 4 else {
            return baseUp[0..., 0..., 0..., ..<channels]
        }
        let h = rgba.dim(1)
        let w = rgba.dim(2)
        let centered = rgba * 2.0 - 1.0
        var tokens = featIn(centered).reshaped(1, h * w, config.hiddenSize)
        let stats = mean(centered, axes: [1, 2])
        let condVec = condProj(stats)
        let condTokens = condVec.expandedDimensions(axis: 1)
        let rope = RoPE2D.frequencies(gridH: h, gridW: w, headDim: config.hiddenSize / config.numHeads)
        let stepCtx = DiTStepContext(condVec: condVec, rope: rope, timestep: 1.0, guidance: 1.0)
        for block in blocks {
            tokens = block(img: tokens, condTokens: condTokens, stepCtx: stepCtx)
        }
        let featGrid = normOut(tokens).reshaped(1, h, w, config.hiddenSize)
        let rawSubpixel = scaleFactor == 4 ? up4Head(featGrid) : up2Head(featGrid)
        let shuffled = rawSubpixel
            .reshaped(1, h, w, scaleFactor, scaleFactor, 4)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(1, h * scaleFactor, w * scaleFactor, 4)
        let detail = 0.06 * tanh(shuffled)
        let restored = clip(baseUp + detail, min: 0.0, max: 1.0)
        return restored[0..., 0..., 0..., ..<channels]
    }
}

public actor SeedVR2OneStepUpscaler: ImageSynthesizer {
    public nonisolated let modelIdentifier: String
    public nonisolated let descriptor: ImageModelDescriptor
    public nonisolated let config: SeedVR2Config
    private var module: SeedVR2Module?

    public init(config: SeedVR2Config = SeedVR2Config()) {
        self.modelIdentifier = config.modelID
        self.descriptor = .seedVR2_3B(id: config.modelID)
        self.config = config
        let built = withRandomState(MLXRandom.RandomState(seed: config.initSeed)) {
            SeedVR2Module(config: config)
        }
        eval(built)
        self.module = built
    }

    public init(directory: URL, identifier: String) throws(BasaltError) {
        let (desc, ditCfg) = try ImageRunnerCore.parseDirectoryConfig(directory: directory, identifier: identifier)
        var cfg = SeedVR2Config(
            modelID: identifier,
            hiddenSize: min(96, ditCfg.hiddenSize),
            numHeads: ditCfg.numHeads,
            numLayers: min(4, ditCfg.numLayers)
        )
        cfg.initSeed = ditCfg.initSeed
        self.modelIdentifier = identifier
        self.descriptor = desc.family == .seedVR2Upscaler ? desc : .seedVR2_3B(id: identifier)
        self.config = cfg
        let built = withRandomState(MLXRandom.RandomState(seed: cfg.initSeed)) {
            SeedVR2Module(config: cfg)
        }
        try ImageRunnerCore.loadSafetensorsIfPresent(in: directory, into: built)
        eval(built)
        self.module = built
    }

    public func unload() async {
        module = nil
        Memory.clearCache()
    }

    public func upscalePixels(
        _ pixels: [Float],
        shape: [Int],
        scaleFactor: Int = 2,
        tileSize: Int? = nil,
        overlap: Int? = nil
    ) throws(BasaltError) -> (pixels: [Float], shape: [Int]) {
        let out = try upscaleTensor(
            MLXArray(pixels, shape),
            scaleFactor: scaleFactor,
            tileSize: tileSize,
            overlap: overlap
        )
        let f32 = out.asType(.float32)
        eval(f32)
        return (f32.reshaped(-1).asArray(Float.self), f32.shape)
    }

    private func upscaleTensor(
        _ image: MLXArray,
        scaleFactor: Int = 2,
        tileSize: Int? = nil,
        overlap: Int? = nil
    ) throws(BasaltError) -> MLXArray {
        guard let module else { throw .modelNotLoaded }
        guard [1, 2, 4].contains(scaleFactor) else {
            throw .invalidRequestStructure(details: "seedvr2: scaleFactor must be 1, 2 or 4")
        }
        let squeezeOut = image.ndim == 3
        let nhwc = squeezeOut ? image[.newAxis] : image
        guard nhwc.ndim == 4, nhwc.dim(0) == 1, nhwc.dim(3) == 3 || nhwc.dim(3) == 4 else {
            throw .invalidRequestStructure(details: "seedvr2: expected [H,W,C] or [1,H,W,C] with C in {3,4}")
        }
        if scaleFactor == 1 {
            return squeezeOut ? nhwc[0] : nhwc
        }
        let resolvedTile = max(16, tileSize ?? config.defaultTileSize)
        let resolvedOverlap = max(4, min(resolvedTile / 2, overlap ?? config.defaultOverlap))
        let stitched = Self.runTiledUpscale(
            module: module,
            nhwc: nhwc,
            scaleFactor: scaleFactor,
            tileSize: resolvedTile,
            overlap: resolvedOverlap
        )
        eval(stitched)
        return squeezeOut ? stitched[0] : stitched
    }

    public func upscaleEncoded(
        _ data: Data,
        scaleFactor: Int = 2,
        format: ImageOutputFormat = .png,
        forceOpaque: Bool = false,
        tileSize: Int? = nil,
        overlap: Int? = nil
    ) throws(BasaltError) -> (encoded: EncodedImageBuffer, dimensions: ImageDimensions) {
        let decoded = try ImagePNGCodec.decodePixels(data)
        let channels = decoded.hasTransparency && !forceOpaque && format != .jpeg ? 4 : 3
        let tensorRGBA = MLXArray(decoded.pixels, [1, decoded.height, decoded.width, 4])
        let inputTensor = channels == 4 ? tensorRGBA : tensorRGBA[0..., 0..., 0..., ..<3]
        let upscaled = try upscaleTensor(
            inputTensor,
            scaleFactor: scaleFactor,
            tileSize: tileSize,
            overlap: overlap
        )
        let outDims = ImageDimensions(width: upscaled.dim(2), height: upscaled.dim(1))
        let encoded = try ImagePNGCodec.encode(upscaled, format: format, forceOpaque: forceOpaque)
        return (encoded, outDims)
    }

    public func generate(_ request: ImageGenerationRequest) async throws -> AsyncThrowingStream<ImageGenerationEvent, any Error> {
        try request.validate()
        guard let firstRef = request.config.referenceImages.first else {
            throw BasaltError.invalidRequestStructure(details: "seedvr2: upscaler requires at least one reference image")
        }
        let scale = request.config.upscaleFactor ?? 2
        let wantsAlpha = ImageRunnerCore.wantsNativeAlpha(request: request, descriptor: descriptor)
        let started = ContinuousClock.now
        let (encoded, outDims) = try upscaleEncoded(
            firstRef,
            scaleFactor: scale,
            format: request.outputFormat,
            forceOpaque: !wantsAlpha
        )
        let elapsed = ContinuousClock.now - started
        let ms = max(1, Int(elapsed.components.seconds * 1_000 + elapsed.components.attoseconds / 1_000_000_000_000_000))
        var artifact = ImageCompletedArtifact(
            imageData: encoded.data,
            format: encoded.format,
            dimensions: outDims,
            hasAlpha: encoded.hasAlpha,
            seed: request.config.seed ?? 0,
            model: modelIdentifier
        )
        artifact.stepsExecuted = 1
        artifact.upscaledBy = scale
        artifact.elapsedMs = ms
        return AsyncThrowingStream { continuation in
            continuation.yield(.completed(artifact))
            continuation.finish()
        }
    }
}

extension SeedVR2OneStepUpscaler {
    static func runTiledUpscale(
        module: SeedVR2Module,
        nhwc: MLXArray,
        scaleFactor: Int,
        tileSize: Int,
        overlap: Int
    ) -> MLXArray {
        let inH = nhwc.dim(1)
        let inW = nhwc.dim(2)
        let ch = nhwc.dim(3)
        let yTiles = SeedVR2TileBlender.computeAxisTiles(total: inH, tileSize: tileSize, overlap: overlap)
        let xTiles = SeedVR2TileBlender.computeAxisTiles(total: inW, tileSize: tileSize, overlap: overlap)
        if yTiles.count == 1 && xTiles.count == 1 {
            return module.upscaleTile(nhwc, scaleFactor: scaleFactor)
        }
        let outH = inH * scaleFactor
        let outW = inW * scaleFactor
        let outOverlap = overlap * scaleFactor
        let accum = MLXArray.zeros([1, outH, outW, ch], dtype: .float32)
        let weights = MLXArray.zeros([1, outH, outW, 1], dtype: .float32)
        for (yi, ySpan) in yTiles.enumerated() {
            for (xi, xSpan) in xTiles.enumerated() {
                let sub = nhwc[0..., ySpan.start ..< (ySpan.start + ySpan.length), xSpan.start ..< (xSpan.start + xSpan.length), 0...]
                let upTile = module.upscaleTile(sub, scaleFactor: scaleFactor).asType(.float32)
                let win = SeedVR2TileBlender.featherWindow2D(
                    height: ySpan.length * scaleFactor,
                    width: xSpan.length * scaleFactor,
                    overlapH: outOverlap,
                    overlapW: outOverlap,
                    edges: SeedVR2TileBlender.TileEdgeMask(
                        top: yi > 0,
                        bottom: yi + 1 < yTiles.count,
                        left: xi > 0,
                        right: xi + 1 < xTiles.count
                    )
                )
                let oy0 = ySpan.start * scaleFactor
                let oy1 = oy0 + ySpan.length * scaleFactor
                let ox0 = xSpan.start * scaleFactor
                let ox1 = ox0 + xSpan.length * scaleFactor
                accum[0..., oy0 ..< oy1, ox0 ..< ox1, 0...] = accum[0..., oy0 ..< oy1, ox0 ..< ox1, 0...] + upTile * win
                weights[0..., oy0 ..< oy1, ox0 ..< ox1, 0...] = weights[0..., oy0 ..< oy1, ox0 ..< ox1, 0...] + win
                eval(accum, weights)
            }
        }
        return clip(accum / maximum(weights, MLXArray(Float(1e-6))), min: 0.0, max: 1.0)
    }
}
