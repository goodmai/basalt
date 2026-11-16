import Foundation
import MLX
import MLXNN
import BasaltCore

public struct KokoroMLXConfig: Sendable, Codable, Equatable {
    public var modelID: String = "Kokoro-82M"
    public var vocabSize: Int = 178
    public var hiddenDim: Int = 256
    public var styleDim: Int = 128
    public var numVoices: Int = 54
    public var upsampleStride: Int = 48
    public var nFFT: Int = 20
    public var hopLength: Int = 5
    public var sampleRate: Int = SpeechRequest.defaultSampleRate

    public static var tiny: KokoroMLXConfig {
        var cfg = KokoroMLXConfig(modelID: "Kokoro-82M-Tiny", hiddenDim: 32, styleDim: 32)
        cfg.upsampleStride = 16
        cfg.nFFT = 16
        cfg.hopLength = 4
        return cfg
    }

    public init(
        modelID: String = "Kokoro-82M",
        vocabSize: Int = 178,
        hiddenDim: Int = 256,
        styleDim: Int = 128,
        numVoices: Int = 54,
        sampleRate: Int = SpeechRequest.defaultSampleRate
    ) {
        self.modelID = modelID
        self.vocabSize = max(16, vocabSize)
        self.hiddenDim = max(8, hiddenDim)
        self.styleDim = max(8, styleDim)
        self.numVoices = max(1, numVoices)
        self.sampleRate = max(8_000, sampleRate)
    }

    public static func from(directory: URL, fallbackID: String = "Kokoro-82M") -> KokoroMLXConfig {
        let url = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: url),
              let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return KokoroMLXConfig(modelID: fallbackID)
        }
        let istft = raw["istftnet"] as? [String: Any]
        var cfg = KokoroMLXConfig(
            modelID: raw["model_id"] as? String ?? fallbackID,
            vocabSize: raw["n_token"] as? Int ?? raw["vocab_size"] as? Int ?? 178,
            hiddenDim: raw["hidden_dim"] as? Int ?? 256,
            styleDim: raw["style_dim"] as? Int ?? 128,
            numVoices: raw["num_voices"] as? Int ?? 54,
            sampleRate: raw["sample_rate"] as? Int ?? SpeechRequest.defaultSampleRate
        )
        let rawNFFT = istft?["gen_istft_n_fft"] as? Int ?? 20
        cfg.upsampleStride = max(2, istft?["upsample_stride"] as? Int ?? 48)
        cfg.nFFT = max(4, rawNFFT.isMultiple(of: 2) ? rawNFFT : rawNFFT + 1)
        cfg.hopLength = max(1, istft?["gen_istft_hop_size"] as? Int ?? 5)
        return cfg
    }
}

final class KokoroAdaINBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "norm") var norm: InstanceNorm
    @ModuleInfo(key: "style_proj") var styleProj: Linear
    let channels: Int

    init(channels: Int, styleDim: Int) {
        self.channels = channels
        self._conv.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        self._norm.wrappedValue = InstanceNorm(dimensions: channels)
        self._styleProj.wrappedValue = Linear(styleDim, channels * 2)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, style: MLXArray) -> MLXArray {
        let normalized = norm(conv(x))
        let gammaBeta = styleProj(style).expandedDimensions(axis: 1)
        let gamma = gammaBeta[0..., 0..., ..<channels]
        let beta = gammaBeta[0..., 0..., channels...]
        return x + leakyRelu((1.0 + gamma) * normalized + beta, negativeSlope: 0.2)
    }
}

final class KokoroMLXModule: Module {
    let config: KokoroMLXConfig
    @ModuleInfo(key: "token_emb") var tokenEmbedding: Embedding
    @ModuleInfo(key: "voice_style") var voiceStyle: Embedding
    @ModuleInfo(key: "prosody_block") var prosodyBlock: KokoroAdaINBlock
    @ModuleInfo(key: "duration_proj") var durationProj: Linear
    @ModuleInfo(key: "f0_proj") var f0Proj: Linear
    @ModuleInfo(key: "upsample") var upsample: ConvTransposed1d
    @ModuleInfo(key: "decoder_block") var decoderBlock: KokoroAdaINBlock
    @ModuleInfo(key: "harmonic_proj") var harmonicProj: Linear
    @ModuleInfo(key: "stft_head") var stftHead: Linear

    init(_ config: KokoroMLXConfig) {
        self.config = config
        let halfStyle = max(4, config.styleDim / 2)
        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenDim)
        self._voiceStyle.wrappedValue = Embedding(embeddingCount: config.numVoices, dimensions: halfStyle * 2)
        self._prosodyBlock.wrappedValue = KokoroAdaINBlock(channels: config.hiddenDim, styleDim: halfStyle)
        self._durationProj.wrappedValue = Linear(config.hiddenDim, 1)
        self._f0Proj.wrappedValue = Linear(config.hiddenDim, 1)
        self._upsample.wrappedValue = ConvTransposed1d(
            inputChannels: config.hiddenDim,
            outputChannels: config.hiddenDim,
            kernelSize: config.upsampleStride,
            stride: config.upsampleStride
        )
        self._decoderBlock.wrappedValue = KokoroAdaINBlock(channels: config.hiddenDim, styleDim: halfStyle)
        self._harmonicProj.wrappedValue = Linear(4, config.hiddenDim)
        self._stftHead.wrappedValue = Linear(config.hiddenDim, (config.nFFT / 2 + 1) * 2)
        super.init()
    }

    func forwardSTFT(tokens: [Int32], voiceIndex: Int, speed: Double) -> (magnitude: MLXArray, phase: MLXArray) {
        let halfStyle = max(4, config.styleDim / 2)
        let ids = MLXArray(tokens).reshaped(1, tokens.count)
        let styleFull = voiceStyle(MLXArray([Int32(voiceIndex % config.numVoices)]))
        let styleProsody = styleFull[0..., ..<halfStyle]
        let styleAcoustic = styleFull[0..., halfStyle...]
        let encoded = prosodyBlock(tokenEmbedding(ids), style: styleProsody)
        let durScores = sigmoid(durationProj(encoded)).reshaped(-1).asArray(Float.self)
        let repeatCounts = durScores.map { score in
            max(2, Int(((2.5 + Double(score) * 3.5) / max(0.25, speed)).rounded()))
        }
        var gatherIdx: [Int32] = []
        for (idx, count) in repeatCounts.enumerated() {
            gatherIdx.append(contentsOf: repeatElement(Int32(idx), count: count))
        }
        let aligned = encoded.take(MLXArray(gatherIdx), axis: 1)
        let f0Aligned = 140.0 + 100.0 * sigmoid(f0Proj(aligned))
        let upsampled = upsample(aligned)
        let totalFrames = upsampled.dim(1)
        let frameF0 = stacked(Array(repeating: f0Aligned, count: config.upsampleStride), axis: 2)
            .reshaped(1, totalFrames, 1)
        let phaseStep = frameF0 * Float(config.hopLength) / Float(config.sampleRate)
        let basePhase = (2.0 * Float.pi) * phaseStep.cumsum(axis: 1)
        let harmonics = concatenated(
            (1 ... 4).map { h in sin(Float(h) * basePhase) / Float(h) },
            axis: -1
        )
        let decoded = decoderBlock(upsampled + harmonicProj(harmonics), style: styleAcoustic)
        let bins = config.nFFT / 2 + 1
        let rawSTFT = stftHead(decoded)[0]
        let logMag = clip(rawSTFT[0..., ..<bins], min: -3.5, max: 1.2)
        let carrier = MLXArray((0 ..< bins).map { b in b == 1 || b == 2 ? Float(0.45) : Float(0.05) }).reshaped(1, bins)
        let magnitude = exp(logMag) + carrier
        let phase = Float.pi * tanh(rawSTFT[0..., bins...]) + basePhase[0, 0..., 0 ..< 1]
        return (magnitude, phase)
    }
}

public actor KokoroMLXSynthesizer: SpeechSynthesizer {
    public struct STFTParityResult: Sendable {
        public let tokenIDs: [Int32]
        public let mlxWaveform: [Float]
        public let cpuWaveform: [Float]
    }

    public nonisolated let config: KokoroMLXConfig
    private var module: KokoroMLXModule?
    private var epoch = 0

    public init(config: KokoroMLXConfig = KokoroMLXConfig(), seed: UInt64 = 493) {
        self.config = config
        let built = withRandomState(MLXRandom.RandomState(seed: seed)) {
            KokoroMLXModule(config)
        }
        eval(built)
        self.module = built
    }

    public init(directory: URL, modelID: String? = nil) throws(BasaltError) {
        let resolved = KokoroMLXConfig.from(directory: directory, fallbackID: modelID ?? directory.lastPathComponent)
        self.config = resolved
        let built = withRandomState(MLXRandom.RandomState(seed: 493)) {
            KokoroMLXModule(resolved)
        }
        try ISTFTNetSynthesizer.loadSafetensorsIfPresent(in: directory, into: built)
        self.module = built
    }

    public var voices: [VoiceSpec] {
        get async { TTSVoiceCatalog.kokoroVoices(model: config.modelID, compute: .mlx) }
    }

    public func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        SpeechRouteDecision(
            selected: SpeechCandidate(
                id: config.modelID,
                model: config.modelID,
                compute: .mlx,
                isAvailable: module != nil,
                estimatedTTFAMs: 55
            )
        )
    }

    public func unload() async {
        epoch &+= 1
        module = nil
        Memory.clearCache()
    }

    public func synthesize(_ request: SpeechRequest) async throws(BasaltError) -> AsyncThrowingStream<AudioChunk, any Error> {
        try request.validate()
        guard module != nil else { throw .modelNotLoaded }
        guard let voiceIndex = TTSVoiceCatalog.kokoroVoiceIndex(for: request.voice, language: request.language) else {
            throw .invalidRequestStructure(details: "speech: unknown Kokoro voice '\(request.voice)'")
        }
        let tokens = Self.encodePhonemeTokens(request.input, vocabSize: config.vocabSize)
        return Self.makeStream(owner: self, tokens: tokens, voiceIndex: voiceIndex, speed: request.speed, expectedEpoch: epoch)
    }

    nonisolated private static func makeStream(
        owner: KokoroMLXSynthesizer,
        tokens: [Int32],
        voiceIndex: Int,
        speed: Double,
        expectedEpoch: Int
    ) -> AsyncThrowingStream<AudioChunk, any Error> {
        let rate = owner.config.sampleRate
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try Task.checkCancellation()
                    let samples = try await owner.runSynthesis(
                        tokens: tokens,
                        voiceIndex: voiceIndex,
                        speed: speed,
                        expectedEpoch: expectedEpoch
                    )
                    let chunkLen = max(1_200, rate / 8)
                    var offset = 0
                    while offset < samples.count {
                        try Task.checkCancellation()
                        let next = min(samples.count, offset + chunkLen)
                        continuation.yield(
                            AudioChunk(samples: Array(samples[offset ..< next]), sampleRate: rate, isFinal: next == samples.count)
                        )
                        offset = next
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

extension KokoroMLXSynthesizer {
    public func predictSTFTParity(
        text: String,
        voice: String = "af_heart",
        speed: Double = 1.0
    ) throws(BasaltError) -> STFTParityResult {
        guard let module else { throw .modelNotLoaded }
        guard let voiceIndex = TTSVoiceCatalog.kokoroVoiceIndex(for: voice, language: nil) else {
            throw .invalidRequestStructure(details: "speech: unknown Kokoro voice '\(voice)'")
        }
        let tokens = Self.encodePhonemeTokens(text, vocabSize: config.vocabSize)
        let (mag, ph) = module.forwardSTFT(tokens: tokens, voiceIndex: voiceIndex, speed: speed)
        let mlxOut = ISTFTNetSynthesizer.mlxInverseSTFT(
            magnitude: mag,
            phase: ph,
            nFFT: config.nFFT,
            hopLength: config.hopLength
        )
        eval(mag, ph, mlxOut)
        let bins = config.nFFT / 2 + 1
        let frames = mag.dim(0)
        let flatMag = mag.asType(.float32).reshaped(-1).asArray(Float.self)
        let flatPh = ph.asType(.float32).reshaped(-1).asArray(Float.self)
        let magRows = (0 ..< frames).map { t in Array(flatMag[(t * bins) ..< ((t + 1) * bins)]) }
        let phRows = (0 ..< frames).map { t in Array(flatPh[(t * bins) ..< ((t + 1) * bins)]) }
        let cpuSamples = ISTFTNetSynthesizer.cpuInverseSTFT(
            magnitude: magRows,
            phase: phRows,
            nFFT: config.nFFT,
            hopLength: config.hopLength
        )
        return STFTParityResult(
            tokenIDs: tokens,
            mlxWaveform: mlxOut.asArray(Float.self),
            cpuWaveform: cpuSamples
        )
    }

    private func runSynthesis(
        tokens: [Int32],
        voiceIndex: Int,
        speed: Double,
        expectedEpoch: Int
    ) throws(BasaltError) -> [Float] {
        guard let module, epoch == expectedEpoch else { throw .modelNotLoaded }
        let (mag, ph) = module.forwardSTFT(tokens: tokens, voiceIndex: voiceIndex, speed: speed)
        let wave = ISTFTNetSynthesizer.mlxInverseSTFT(
            magnitude: mag,
            phase: ph,
            nFFT: config.nFFT,
            hopLength: config.hopLength
        )
        eval(wave)
        return TTSWaveformMetrics.normalizeSoft(wave.asArray(Float.self))
    }

    static func encodePhonemeTokens(_ text: String, vocabSize: Int) -> [Int32] {
        let cap = max(4, vocabSize - 1)
        var tokens: [Int32] = []
        tokens.reserveCapacity(min(512, max(1, text.unicodeScalars.count)))
        for scalar in text.unicodeScalars.prefix(512) {
            tokens.append(Int32((Int(scalar.value) % cap) + 1))
        }
        return tokens.isEmpty ? [1] : tokens
    }
}
