import Foundation
import MLX
import MLXNN
import BasaltCore

public enum VideoVAEVariant: String, Sendable, Codable, Equatable, CaseIterable {
    case ltx25DiffVAE = "ltx_2_5_diff_vae"
    case wan22VAE = "wan_2_2_vae"
    case kandinsky6VAE = "kandinsky_6_vae"

    public var defaultLatentChannels: Int {
        switch self {
        case .ltx25DiffVAE: 32
        case .wan22VAE, .kandinsky6VAE: 16
        }
    }

    public var defaultBaseChannels: Int {
        switch self {
        case .ltx25DiffVAE, .kandinsky6VAE: 128
        case .wan22VAE: 96
        }
    }

    public var usesTimestepConditioning: Bool {
        switch self {
        case .ltx25DiffVAE: true
        case .wan22VAE, .kandinsky6VAE: false
        }
    }

    public var supportsJointAudio: Bool {
        switch self {
        case .ltx25DiffVAE, .kandinsky6VAE: true
        case .wan22VAE: false
        }
    }

    public var defaultAudioSampleRate: Int {
        switch self {
        case .ltx25DiffVAE: 24_000
        case .wan22VAE: 0
        case .kandinsky6VAE: 44_100
        }
    }

    public var defaultMelBins: Int {
        switch self {
        case .ltx25DiffVAE: 64
        case .wan22VAE: 0
        case .kandinsky6VAE: 128
        }
    }
}

public struct Causal3DVAEConfig: Sendable, Codable, Equatable {
    public var variant: VideoVAEVariant
    public var latentChannels: Int
    public var outputChannels: Int
    public var baseChannels: Int
    public var channelMultipliers: [Int]
    public var resBlocksPerStage: Int
    public var temporalScale: Int
    public var spatialScale: Int
    public var causalPadding: CausalPaddingMode
    public var normEps: Float
    public var defaultDecodeTimestep: Float
    public var audioSampleRate: Int
    public var melBins: Int

    public init(
        variant: VideoVAEVariant = .wan22VAE,
        latentChannels: Int? = nil,
        baseChannels: Int? = nil,
        temporalScale: Int = 4,
        spatialScale: Int = 8,
        causalPadding: CausalPaddingMode = .replicateFirstFrame
    ) {
        self.variant = variant
        self.latentChannels = Swift.max(1, latentChannels ?? variant.defaultLatentChannels)
        self.outputChannels = 3
        self.baseChannels = Swift.max(4, baseChannels ?? variant.defaultBaseChannels)
        self.channelMultipliers = [4, 2, 2, 1]
        self.resBlocksPerStage = 2
        self.temporalScale = Swift.max(1, temporalScale)
        self.spatialScale = Swift.max(1, spatialScale)
        self.causalPadding = causalPadding
        self.normEps = 1e-6
        self.defaultDecodeTimestep = variant.usesTimestepConditioning ? 0.05 : 0.0
        self.audioSampleRate = variant.defaultAudioSampleRate
        self.melBins = variant.defaultMelBins
    }

    public static func preset(_ variant: VideoVAEVariant) -> Causal3DVAEConfig {
        Causal3DVAEConfig(variant: variant)
    }

    public static func tiny(_ variant: VideoVAEVariant = .wan22VAE) -> Causal3DVAEConfig {
        var cfg = Causal3DVAEConfig(
            variant: variant, latentChannels: 8, baseChannels: 16,
            temporalScale: 4, spatialScale: 8, causalPadding: .replicateFirstFrame
        )
        cfg.channelMultipliers = [2, 2, 1, 1]
        cfg.resBlocksPerStage = 1
        return cfg
    }

    public var spatialUpsampleStages: Int { spatialScale.trailingZeroBitCount }
    public var temporalUpsampleStages: Int { temporalScale.trailingZeroBitCount }

    public func expandedFrameCount(latentFrames: Int) -> Int {
        latentFrames > 0 ? 1 + (latentFrames - 1) * temporalScale : 0
    }

    public var recommendedHaloLatents: Int {
        var radius = 1 + 2 * resBlocksPerStage
        for _ in 0 ..< spatialUpsampleStages {
            radius = (radius + 2) / 2 + 2 * resBlocksPerStage
        }
        return radius + 2 * resBlocksPerStage + 1
    }

    public func validate() throws(BasaltError) {
        let validT = temporalScale > 0 && (temporalScale & (temporalScale - 1)) == 0
        let validS = spatialScale > 0 && (spatialScale & (spatialScale - 1)) == 0
        guard validT, validS else {
            throw .invalidRequestStructure(details: "temporalScale (\(temporalScale)) and spatialScale (\(spatialScale)) must be powers of two")
        }
        guard temporalUpsampleStages <= spatialUpsampleStages else {
            throw .invalidRequestStructure(details: "temporalUpsampleStages (\(temporalUpsampleStages)) cannot exceed spatialUpsampleStages (\(spatialUpsampleStages))")
        }
        guard channelMultipliers.count == spatialUpsampleStages + 1 else {
            throw .invalidRequestStructure(details: "channelMultipliers count (\(channelMultipliers.count)) must equal spatialUpsampleStages + 1 (\(spatialUpsampleStages + 1))")
        }
    }
}

final class CausalTimestepEmbedder: Module {
    let frequencyDim: Int
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(frequencyDim: Int, embedDim: Int) {
        let evenDim = Swift.max(4, frequencyDim - (frequencyDim % 2))
        self.frequencyDim = evenDim
        self._fc1.wrappedValue = Linear(evenDim, embedDim)
        self._fc2.wrappedValue = Linear(embedDim, embedDim)
    }

    func callAsFunction(timestep: Float, dtype: DType) -> MLXArray {
        let half = frequencyDim / 2
        var values: [Float] = []
        values.reserveCapacity(frequencyDim)
        let logMax = Foundation.log(10_000.0)
        for idx in 0 ..< half {
            let exponent = -Double(idx) * logMax / Double(Swift.max(1, half - 1))
            let angle = Double(timestep) * Foundation.exp(exponent) * 1_000.0
            values.append(Float(Foundation.cos(angle)))
            values.append(Float(Foundation.sin(angle)))
        }
        let emb = MLXArray(values).reshaped(1, frequencyDim).asType(dtype)
        return fc2(MLXNN.silu(fc1(emb)))
    }
}

final class CausalResBlock3d: Module {
    let outChannels: Int
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "conv1") var conv1: CausalConv3d
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "conv2") var conv2: CausalConv3d
    @ModuleInfo(key: "shortcut") var shortcut: Conv3d?
    @ModuleInfo(key: "time_proj") var timeProj: Linear?

    init(inChannels: Int, outChannels: Int, paddingMode: CausalPaddingMode, normEps: Float, timeEmbedDim: Int?) {
        self.outChannels = outChannels
        self._norm1.wrappedValue = RMSNorm(dimensions: inChannels, eps: normEps)
        self._conv1.wrappedValue = CausalConv3d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, paddingMode: paddingMode)
        self._norm2.wrappedValue = RMSNorm(dimensions: outChannels, eps: normEps)
        let secondConv = CausalConv3d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, paddingMode: paddingMode)
        secondConv.update(parameters: secondConv.parameters().mapValues { $0 * Float(0.25) })
        self._conv2.wrappedValue = secondConv
        if inChannels != outChannels {
            self._shortcut.wrappedValue = Conv3d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
        }
        if let timeEmbedDim {
            self._timeProj.wrappedValue = Linear(timeEmbedDim, 2 * outChannels)
        }
    }

    func callAsFunction(_ x: MLXArray, timeEmbed: MLXArray?) -> MLXArray {
        let residual = shortcut?(x) ?? x
        let h1 = conv1(MLXNN.silu(norm1(x)))
        let h2 = modulate(norm2(h1), timeEmbed: timeEmbed)
        return residual + conv2(MLXNN.silu(h2))
    }

    func forwardCached(_ x: MLXArray, timeEmbed: MLXArray?, session: inout TemporalDecodeSession) -> MLXArray {
        let residual = shortcut?(x) ?? x
        let h1 = conv1.forwardCached(MLXNN.silu(norm1(x)), session: &session)
        let h2 = modulate(norm2(h1), timeEmbed: timeEmbed)
        return residual + conv2.forwardCached(MLXNN.silu(h2), session: &session)
    }

    private func modulate(_ normed: MLXArray, timeEmbed: MLXArray?) -> MLXArray {
        guard let timeProj, let timeEmbed else { return normed }
        let raw = timeProj(MLXNN.silu(timeEmbed))
        let scale = 0.25 * MLX.tanh(raw[0..., ..<outChannels]).reshaped(1, 1, 1, 1, outChannels)
        let shift = raw[0..., outChannels...].reshaped(1, 1, 1, 1, outChannels)
        return normed * (1.0 + scale) + shift
    }
}

final class CausalTemporalUpsample3d: Module {
    let channels: Int
    @ModuleInfo(key: "time_conv") var timeConv: CausalConv3d

    init(channels: Int, paddingMode: CausalPaddingMode) {
        self.channels = channels
        self._timeConv.wrappedValue = CausalConv3d(
            inputChannels: channels,
            outputChannels: 2 * channels,
            kernelSize: IntOrTriple([3, 1, 1]),
            paddingMode: paddingMode
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), h = x.dim(2), w = x.dim(3)
        let expanded = timeConv(x).reshaped(b, t, h, w, 2, channels).transposed(0, 1, 4, 2, 3, 5).reshaped(b, 2 * t, h, w, channels)
        return expanded[0..., 1..., 0..., 0..., 0...]
    }

    func forwardCached(_ x: MLXArray, session: inout TemporalDecodeSession) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), h = x.dim(2), w = x.dim(3)
        return session.withNextCache(maxFrames: timeConv.spec.causalPadT) { cache in
            let dropInitial = cache.isFirstChunk
            let expanded = timeConv(x, cache: &cache).reshaped(b, t, h, w, 2, channels).transposed(0, 1, 4, 2, 3, 5).reshaped(b, 2 * t, h, w, channels)
            return dropInitial ? expanded[0..., 1..., 0..., 0..., 0...] : expanded
        }
    }
}

final class CausalSpatialUpsample3d: Module {
    @ModuleInfo(key: "upsample") var upsample: Upsample
    @ModuleInfo(key: "spatial_conv") var spatialConv: CausalConv3d

    init(inChannels: Int, outChannels: Int, paddingMode: CausalPaddingMode) {
        self._upsample.wrappedValue = Upsample(scaleFactor: [1.0, 2.0, 2.0], mode: .nearest)
        self._spatialConv.wrappedValue = CausalConv3d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrTriple([1, 3, 3]),
            paddingMode: paddingMode
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { spatialConv(upsample(x)) }

    func forwardCached(_ x: MLXArray, session: inout TemporalDecodeSession) -> MLXArray {
        spatialConv.forwardCached(upsample(x), session: &session)
    }
}

final class CausalUpStage3d: Module {
    @ModuleInfo(key: "blocks") var blocks: [CausalResBlock3d]
    @ModuleInfo(key: "temporal_up") var temporalUp: CausalTemporalUpsample3d?
    @ModuleInfo(key: "spatial_up") var spatialUp: CausalSpatialUpsample3d?

    init(inChannels: Int, outChannels: Int, upsampleTemporal: Bool, upsampleSpatial: Bool, timeEmbedDim: Int?, config: Causal3DVAEConfig) {
        self._blocks.wrappedValue = (0 ..< config.resBlocksPerStage).map { idx in
            CausalResBlock3d(
                inChannels: idx == 0 ? inChannels : outChannels,
                outChannels: outChannels,
                paddingMode: config.causalPadding,
                normEps: config.normEps,
                timeEmbedDim: timeEmbedDim
            )
        }
        if upsampleTemporal {
            self._temporalUp.wrappedValue = CausalTemporalUpsample3d(channels: outChannels, paddingMode: config.causalPadding)
        }
        if upsampleSpatial {
            self._spatialUp.wrappedValue = CausalSpatialUpsample3d(inChannels: outChannels, outChannels: outChannels, paddingMode: config.causalPadding)
        }
    }

    func callAsFunction(_ x: MLXArray, timeEmbed: MLXArray?) -> MLXArray {
        var h = x
        for block in blocks { h = block(h, timeEmbed: timeEmbed) }
        if let temporalUp { h = temporalUp(h) }
        if let spatialUp { h = spatialUp(h) }
        return h
    }

    func forwardCached(_ x: MLXArray, timeEmbed: MLXArray?, session: inout TemporalDecodeSession) -> MLXArray {
        var h = x
        for block in blocks { h = block.forwardCached(h, timeEmbed: timeEmbed, session: &session) }
        if let temporalUp { h = temporalUp.forwardCached(h, session: &session) }
        if let spatialUp { h = spatialUp.forwardCached(h, session: &session) }
        return h
    }
}

public protocol AudioVocoderBridge {
    var sampleRate: Int { get }
    var melBins: Int { get }
    var hopLength: Int { get }
    func synthesizeWaveform(fromMel melSpectrogram: MLXArray) throws(BasaltError) -> MLXArray
}

public struct JointAudioVideoOutput {
    public let videoFrames: MLXArray
    public let audioWaveform: MLXArray
    public let audioSampleRate: Int

    public init(videoFrames: MLXArray, audioWaveform: MLXArray, audioSampleRate: Int) {
        self.videoFrames = videoFrames
        self.audioWaveform = audioWaveform
        self.audioSampleRate = audioSampleRate
    }
}

public final class CausalMelVocoderAdapter: Module, AudioVocoderBridge {
    public let sampleRate: Int
    public let melBins: Int
    public let hopLength: Int
    @ModuleInfo(key: "in_conv") var inConv: Conv1d
    @ModuleInfo(key: "up_convs") var upConvs: [ConvTransposed1d]
    @ModuleInfo(key: "res_convs") var resConvs: [Conv1d]
    @ModuleInfo(key: "out_conv") var outConv: Conv1d

    public init(sampleRate: Int = 44_100, melBins: Int = 128, hiddenChannels: Int = 64, upsampleRates: [Int] = [8, 8, 4]) {
        let validRates = upsampleRates.isEmpty ? [4, 4] : upsampleRates.map { Swift.max(2, $0 - ($0 % 2)) }
        var prod = 1
        for rate in validRates { prod *= rate }
        self.sampleRate = Swift.max(8_000, sampleRate)
        self.melBins = Swift.max(8, melBins)
        self.hopLength = prod

        var ch = Swift.max(16, hiddenChannels)
        self._inConv.wrappedValue = Conv1d(inputChannels: self.melBins, outputChannels: ch, kernelSize: 3, padding: 0)
        var ups: [ConvTransposed1d] = []
        var res: [Conv1d] = []
        for rate in validRates {
            let nextCh = Swift.max(8, ch / 2)
            ups.append(ConvTransposed1d(inputChannels: ch, outputChannels: nextCh, kernelSize: 2 * rate, stride: rate, padding: rate / 2))
            res.append(Conv1d(inputChannels: nextCh, outputChannels: nextCh, kernelSize: 3, padding: 0))
            ch = nextCh
        }
        self._upConvs.wrappedValue = ups
        self._resConvs.wrappedValue = res
        self._outConv.wrappedValue = Conv1d(inputChannels: ch, outputChannels: 1, kernelSize: 3, padding: 0)
    }

    public static func tiny(for variant: VideoVAEVariant) -> CausalMelVocoderAdapter {
        CausalMelVocoderAdapter(
            sampleRate: Swift.max(24_000, variant.defaultAudioSampleRate),
            melBins: Swift.max(16, variant.defaultMelBins),
            hiddenChannels: 16,
            upsampleRates: [4, 4]
        )
    }

    public func synthesizeWaveform(fromMel melSpectrogram: MLXArray) throws(BasaltError) -> MLXArray {
        guard melSpectrogram.ndim == 3 else {
            throw .invalidRequestStructure(details: "CausalMelVocoderAdapter expects 3D [B, T_mel, melBins], got \(melSpectrogram.ndim)D")
        }
        guard melSpectrogram.dim(2) == melBins else {
            throw .invalidRequestStructure(details: "Expected \(melBins) mel bins, got \(melSpectrogram.dim(2))")
        }
        var h = inConv(Self.causalPad1d(melSpectrogram, padLeft: 2))
        for idx in upConvs.indices {
            h = upConvs[idx](MLXNN.silu(h))
            h = h + resConvs[idx](MLXNN.silu(Self.causalPad1d(h, padLeft: 2)))
        }
        return MLX.tanh(outConv(MLXNN.silu(Self.causalPad1d(h, padLeft: 2))).squeezed(axis: -1))
    }

    private static func causalPad1d(_ x: MLXArray, padLeft: Int) -> MLXArray {
        padded(x, widths: [IntOrPair((0, 0)), IntOrPair((padLeft, 0)), IntOrPair((0, 0))])
    }
}

public final class Causal3DVAE: Module {
    public let config: Causal3DVAEConfig
    @ModuleInfo(key: "conv_in") var convIn: CausalConv3d
    @ModuleInfo(key: "noise_proj") var noiseProj: Conv3d?
    @ParameterInfo(key: "noise_gate") var noiseGate: MLXArray?
    @ModuleInfo(key: "time_embedder") var timeEmbedder: CausalTimestepEmbedder?
    @ModuleInfo(key: "mid_blocks") var midBlocks: [CausalResBlock3d]
    @ModuleInfo(key: "up_stages") var upStages: [CausalUpStage3d]
    @ModuleInfo(key: "norm_out") var normOut: RMSNorm
    @ModuleInfo(key: "conv_out") var convOut: CausalConv3d

    public init(_ config: Causal3DVAEConfig = .preset(.wan22VAE)) {
        self.config = config
        let widths = config.channelMultipliers.map { Swift.max(4, config.baseChannels * $0) }
        let tEmbedDim: Int? = config.variant.usesTimestepConditioning ? widths[0] : nil

        self._convIn.wrappedValue = CausalConv3d(
            inputChannels: config.latentChannels, outputChannels: widths[0],
            kernelSize: 3, paddingMode: config.causalPadding
        )
        if let tEmbedDim {
            self._timeEmbedder.wrappedValue = CausalTimestepEmbedder(frequencyDim: Swift.max(16, config.baseChannels), embedDim: tEmbedDim)
            self._noiseProj.wrappedValue = Conv3d(inputChannels: config.latentChannels, outputChannels: config.latentChannels, kernelSize: 1)
            self._noiseGate.wrappedValue = MLXArray.ones([config.latentChannels], dtype: .float32) * 0.2
        }
        self._midBlocks.wrappedValue = (0 ..< config.resBlocksPerStage).map { _ in
            CausalResBlock3d(inChannels: widths[0], outChannels: widths[0], paddingMode: config.causalPadding, normEps: config.normEps, timeEmbedDim: tEmbedDim)
        }

        let numSpatial = config.spatialUpsampleStages
        let numTemporal = config.temporalUpsampleStages
        self._upStages.wrappedValue = (0 ... numSpatial).map { idx in
            CausalUpStage3d(
                inChannels: idx == 0 ? widths[0] : widths[idx - 1],
                outChannels: widths[idx],
                upsampleTemporal: idx < numTemporal,
                upsampleSpatial: idx < numSpatial,
                timeEmbedDim: tEmbedDim,
                config: config
            )
        }

        let finalCh = widths[numSpatial]
        self._normOut.wrappedValue = RMSNorm(dimensions: finalCh, eps: config.normEps)
        let finalConv = CausalConv3d(inputChannels: finalCh, outputChannels: config.outputChannels, kernelSize: 3, paddingMode: config.causalPadding)
        finalConv.update(parameters: finalConv.parameters().mapValues { $0 * Float(0.25) })
        self._convOut.wrappedValue = finalConv
    }

    public func decodeMonolithic(
        _ latents: MLXArray,
        timestep: Float? = nil,
        residualNoise: MLXArray? = nil
    ) throws(BasaltError) -> MLXArray {
        try validateLatents(latents, residualNoise: residualNoise)
        let stepValue = timestep ?? config.defaultDecodeTimestep
        let tEmbed = timeEmbedder?(timestep: stepValue, dtype: latents.dtype)
        var h = convIn(stemInput(latents, timestep: stepValue, residualNoise: residualNoise))
        for block in midBlocks { h = block(h, timeEmbed: tEmbed) }
        for stage in upStages { h = stage(h, timeEmbed: tEmbed) }
        return convOut(MLXNN.silu(normOut(h)))
    }

    public func decodeStreaming(
        _ latents: MLXArray,
        temporalChunkLatents: Int = 1,
        tileConfig: SpatialTileConfig? = nil,
        timestep: Float? = nil,
        residualNoise: MLXArray? = nil
    ) throws(BasaltError) -> MLXArray {
        try validateLatents(latents, residualNoise: residualNoise)
        let chunkSize = Swift.max(1, temporalChunkLatents)
        let stepValue = timestep ?? config.defaultDecodeTimestep
        let conditioning = DecodeConditioning(
            timestep: stepValue,
            timeEmbed: timeEmbedder?(timestep: stepValue, dtype: latents.dtype)
        )
        guard let tileConfig else {
            return decodeTileTemporalStream(latents: latents, residualNoise: residualNoise, chunkSize: chunkSize, conditioning: conditioning)
        }
        let grid = SpatialTileGrid(
            latentHeight: latents.dim(2), latentWidth: latents.dim(3),
            spatialScale: config.spatialScale, config: tileConfig,
            defaultHaloLatents: config.recommendedHaloLatents
        )
        if grid.tiles.count == 1 {
            return decodeTileTemporalStream(latents: latents, residualNoise: residualNoise, chunkSize: chunkSize, conditioning: conditioning)
        }
        return decodeTiledAndChunked(latents: latents, residualNoise: residualNoise, grid: grid, chunkSize: chunkSize, conditioning: conditioning)
    }

    public func decodeJointAudioVideo<V: AudioVocoderBridge>(
        videoLatents: MLXArray,
        melSpectrogram: MLXArray,
        vocoder: V,
        temporalChunkLatents: Int = 1,
        tileConfig: SpatialTileConfig? = nil
    ) throws(BasaltError) -> JointAudioVideoOutput {
        guard config.variant.supportsJointAudio else {
            throw .invalidRequestStructure(details: "Variant \(config.variant.rawValue) does not support joint audio-video synthesis")
        }
        let frames = try decodeStreaming(videoLatents, temporalChunkLatents: temporalChunkLatents, tileConfig: tileConfig)
        let waveform = try vocoder.synthesizeWaveform(fromMel: melSpectrogram)
        return JointAudioVideoOutput(videoFrames: frames, audioWaveform: waveform, audioSampleRate: vocoder.sampleRate)
    }
}

private struct DecodeConditioning {
    let timestep: Float
    let timeEmbed: MLXArray?
}

extension Causal3DVAE {
    private func validateLatents(_ latents: MLXArray, residualNoise: MLXArray?) throws(BasaltError) {
        try config.validate()
        guard latents.ndim == 5 else {
            throw .invalidRequestStructure(details: "Causal3DVAE expects 5D [B, T, H, W, C] latents, got \(latents.ndim)D")
        }
        guard latents.dim(1) > 0, latents.dim(2) > 0, latents.dim(3) > 0 else {
            throw .invalidRequestStructure(details: "Latent spatio-temporal dimensions must be positive, got \(latents.shape)")
        }
        guard latents.dim(4) == config.latentChannels else {
            throw .invalidRequestStructure(details: "Expected \(config.latentChannels) latent channels, got \(latents.dim(4))")
        }
        if let residualNoise, residualNoise.shape != latents.shape {
            throw .invalidRequestStructure(details: "residualNoise shape \(residualNoise.shape) must match latents shape \(latents.shape)")
        }
    }

    private func stemInput(_ latents: MLXArray, timestep: Float, residualNoise: MLXArray?) -> MLXArray {
        guard let noiseProj, let noiseGate, let residualNoise else { return latents }
        let clampedT = Swift.min(Swift.max(0.0, Double(timestep)), 1.0)
        let gateScale = Float(Foundation.sin(0.5 * Double.pi * clampedT))
        let gate = noiseGate.asType(latents.dtype).reshaped(1, 1, 1, 1, config.latentChannels) * gateScale
        return latents + gate * noiseProj(residualNoise.asType(latents.dtype))
    }

    private func forwardSingleChunk(
        _ chunkLatents: MLXArray,
        chunkNoise: MLXArray?,
        conditioning: DecodeConditioning,
        session: inout TemporalDecodeSession
    ) -> MLXArray {
        session.beginChunk()
        var h = convIn.forwardCached(stemInput(chunkLatents, timestep: conditioning.timestep, residualNoise: chunkNoise), session: &session)
        for block in midBlocks { h = block.forwardCached(h, timeEmbed: conditioning.timeEmbed, session: &session) }
        for stage in upStages { h = stage.forwardCached(h, timeEmbed: conditioning.timeEmbed, session: &session) }
        let out = convOut.forwardCached(MLXNN.silu(normOut(h)), session: &session)
        session.endChunk()
        return out
    }

    private func decodeTileTemporalStream(
        latents: MLXArray,
        residualNoise: MLXArray?,
        chunkSize: Int,
        conditioning: DecodeConditioning
    ) -> MLXArray {
        let totalT = latents.dim(1)
        var session = TemporalDecodeSession()
        var chunks: [MLXArray] = []
        var start = 0
        while start < totalT {
            let end = Swift.min(totalT, start + chunkSize)
            let latSlice = latents[0..., start ..< end, 0..., 0..., 0...]
            let noiseSlice = residualNoise?[0..., start ..< end, 0..., 0..., 0...]
            let decoded = forwardSingleChunk(latSlice, chunkNoise: noiseSlice, conditioning: conditioning, session: &session)
            eval(decoded)
            chunks.append(decoded)
            start = end
        }
        session.release()
        return chunks.count == 1 ? chunks[0] : concatenated(chunks, axis: 1)
    }

    private func decodeTiledAndChunked(
        latents: MLXArray,
        residualNoise: MLXArray?,
        grid: SpatialTileGrid,
        chunkSize: Int,
        conditioning: DecodeConditioning
    ) -> MLXArray {
        let totalT = latents.dim(1)
        let outFrames = config.expandedFrameCount(latentFrames: totalT)
        var accumulator = MLXArray.zeros(
            [latents.dim(0), outFrames, grid.pixelHeight, grid.pixelWidth, config.outputChannels],
            dtype: .float32
        )
        for tile in grid.tiles {
            let window = TileWindow(for: tile).asMLXArray(dtype: .float32)
            var session = TemporalDecodeSession()
            var start = 0
            var outCursor = 0
            while start < totalT {
                let end = Swift.min(totalT, start + chunkSize)
                let latSlice = latents[0..., start ..< end, tile.fetchLatentY, tile.fetchLatentX, 0...]
                let noiseSlice = residualNoise?[0..., start ..< end, tile.fetchLatentY, tile.fetchLatentX, 0...]
                let rawChunk = forwardSingleChunk(latSlice, chunkNoise: noiseSlice, conditioning: conditioning, session: &session)
                let emitted = rawChunk.dim(1)
                OverlapTileBlender.blendTileChunk(
                    into: &accumulator, rawDecoded: rawChunk, tile: tile,
                    window: window, frameRange: outCursor ..< (outCursor + emitted)
                )
                eval(accumulator)
                outCursor += emitted
                start = end
            }
            session.release()
        }
        return accumulator.asType(latents.dtype)
    }
}
