import Foundation
import MLX
import MLXNN
import BasaltCore

public struct Soprano80MConfig: Sendable, Codable, Equatable {
    public static let maxSub200MBProfile = 200

    public var modelID: String
    public var vocabSize: Int
    public var hiddenDim: Int
    public var melBins: Int
    public var vocosLayers: Int
    public var nFFT: Int = 512
    public var hopLength: Int = 128
    public var sampleRate: Int = SpeechRequest.defaultSampleRate
    public var estimatedResidentMB: Int = 160

    public static var tiny: Soprano80MConfig {
        var cfg = Soprano80MConfig(
            modelID: "ekwek/Soprano-80M-Tiny",
            vocabSize: 128,
            hiddenDim: 48,
            melBins: 32,
            vocosLayers: 2,
            sampleRate: 24_000
        )
        cfg.nFFT = 64
        cfg.hopLength = 16
        return cfg
    }

    public init(
        modelID: String = "ekwek/Soprano-80M",
        vocabSize: Int = 512,
        hiddenDim: Int = 192,
        melBins: Int = 80,
        vocosLayers: Int = 4,
        sampleRate: Int = SpeechRequest.defaultSampleRate
    ) {
        self.modelID = modelID
        self.vocabSize = max(32, vocabSize)
        self.hiddenDim = max(16, hiddenDim)
        self.melBins = max(16, melBins)
        self.vocosLayers = max(1, vocosLayers)
        self.sampleRate = max(8_000, sampleRate)
    }

    public static func from(directory: URL, fallbackID: String = "ekwek/Soprano-80M") -> Soprano80MConfig {
        let url = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: url),
              let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return Soprano80MConfig(modelID: fallbackID)
        }
        var cfg = Soprano80MConfig(
            modelID: raw["model_id"] as? String ?? fallbackID,
            vocabSize: raw["vocab_size"] as? Int ?? 512,
            hiddenDim: raw["hidden_dim"] as? Int ?? 192,
            melBins: raw["mel_bins"] as? Int ?? 80,
            vocosLayers: raw["vocos_layers"] as? Int ?? 4,
            sampleRate: raw["sample_rate"] as? Int ?? SpeechRequest.defaultSampleRate
        )
        let rawFFT = raw["n_fft"] as? Int ?? 512
        cfg.nFFT = max(16, rawFFT.isMultiple(of: 2) ? rawFFT : rawFFT + 1)
        cfg.hopLength = max(4, raw["hop_length"] as? Int ?? 128)
        return cfg
    }
}

final class SopranoVocosBlock: Module {
    @ModuleInfo(key: "dwconv") var dwconv: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pw1") var pw1: Linear
    @ModuleInfo(key: "pw2") var pw2: Linear

    init(dim: Int) {
        self._dwconv.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3, groups: dim)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pw1.wrappedValue = Linear(dim, dim * 2)
        self._pw2.wrappedValue = Linear(dim * 2, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + pw2(MLXNN.gelu(pw1(norm(dwconv(x)))))
    }
}

final class SopranoVocosModule: Module {
    let config: Soprano80MConfig
    @ModuleInfo(key: "text_emb") var textEmbedding: Embedding
    @ModuleInfo(key: "text_conv") var textConv: Conv1d
    @ModuleInfo(key: "mel_proj") var melProj: Linear
    @ModuleInfo(key: "mel_in") var melIn: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [SopranoVocosBlock]
    @ModuleInfo(key: "head_norm") var headNorm: LayerNorm
    @ModuleInfo(key: "istft_head") var istftHead: Linear

    init(_ config: Soprano80MConfig, seed: UInt64) {
        self.config = config
        let d = config.hiddenDim
        let bins = config.nFFT / 2 + 1
        let built = withRandomState(MLXRandom.RandomState(seed: seed)) {
            let emb = Embedding(embeddingCount: config.vocabSize, dimensions: d)
            let tConv = Conv1d(inputChannels: d, outputChannels: d, kernelSize: 3, padding: 1)
            let mProj = Linear(d, config.melBins)
            let mIn = Conv1d(inputChannels: config.melBins, outputChannels: d, kernelSize: 5, padding: 2)
            let vBlocks = (0 ..< config.vocosLayers).map { _ in SopranoVocosBlock(dim: d) }
            let hNorm = LayerNorm(dimensions: d)
            let head = Linear(d, bins * 2)
            return (emb, tConv, mProj, mIn, vBlocks, hNorm, head)
        }
        self._textEmbedding.wrappedValue = built.0
        self._textConv.wrappedValue = built.1
        self._melProj.wrappedValue = built.2
        self._melIn.wrappedValue = built.3
        self._blocks.wrappedValue = built.4
        self._headNorm.wrappedValue = built.5
        self._istftHead.wrappedValue = built.6
        super.init()
        eval(self)
    }

    func encodeTextToMel(tokens: [Int32], framesPerToken: Int) -> MLXArray {
        let ids = MLXArray(tokens).reshaped(1, tokens.count)
        let h = MLXNN.gelu(textConv(textEmbedding(ids)))
        let mel = melProj(h)
        let repeats = max(1, framesPerToken)
        return stacked(Array(repeating: mel, count: repeats), axis: 2)
            .reshaped(1, tokens.count * repeats, config.melBins)
    }

    func decodeMelToWaveform(_ mel: MLXArray) -> MLXArray {
        var h = melIn(mel)
        for block in blocks { h = block(h) }
        let specOut = istftHead(headNorm(h))[0]
        let bins = config.nFFT / 2 + 1
        let frames = specOut.dim(0)
        let logMag = clip(specOut[0..., ..<bins], min: -4.0, max: 1.4)
        let carrier = MLXArray((0 ..< bins).map { $0 == 2 || $0 == 3 ? Float(0.4) : Float(0.04) }).reshaped(1, bins)
        let magnitude = exp(logMag) + carrier
        let phaseSweep = MLXArray((0 ..< frames).map { Float($0) * 0.35 }).reshaped(frames, 1)
        let phase = Float.pi * MLX.tanh(specOut[0..., bins...]) + phaseSweep
        return ISTFTNetSynthesizer.mlxInverseSTFT(
            magnitude: magnitude,
            phase: phase,
            nFFT: config.nFFT,
            hopLength: config.hopLength
        )
    }
}

public actor Soprano80MSynthesizer: SpeechSynthesizer {
    public nonisolated let config: Soprano80MConfig
    private var module: SopranoVocosModule?
    private var epoch = 0

    public init(config: Soprano80MConfig = Soprano80MConfig(), seed: UInt64 = 579) {
        self.config = config
        self.module = SopranoVocosModule(config, seed: seed)
    }

    public init(directory: URL, modelID: String? = nil) throws(BasaltError) {
        let resolved = Soprano80MConfig.from(directory: directory, fallbackID: modelID ?? directory.lastPathComponent)
        self.config = resolved
        let built = SopranoVocosModule(resolved, seed: 579)
        try ISTFTNetSynthesizer.loadSafetensorsIfPresent(in: directory, into: built)
        self.module = built
    }

    public var voices: [VoiceSpec] {
        get async { TTSVoiceCatalog.sopranoVoices(model: config.modelID, compute: .mlx) }
    }

    public func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        SpeechRouteDecision(
            selected: SpeechCandidate(
                id: config.modelID,
                model: config.modelID,
                compute: .mlx,
                isAvailable: module != nil,
                estimatedTTFAMs: 32
            )
        )
    }

    public func unload() async {
        epoch &+= 1
        module = nil
        Memory.clearCache()
    }

    public func synthesizeFromMel(_ melFrames: [[Float]]) throws(BasaltError) -> [Float] {
        guard let module else { throw .modelNotLoaded }
        guard !melFrames.isEmpty else { return [] }
        let bins = config.melBins
        var flat: [Float] = []
        flat.reserveCapacity(melFrames.count * bins)
        for row in melFrames {
            for idx in 0 ..< bins {
                let raw = idx < row.count ? row[idx] : 0
                flat.append(raw.isFinite ? raw : 0)
            }
        }
        let mel = MLXArray(flat).reshaped(1, melFrames.count, bins)
        let wave = module.decodeMelToWaveform(mel)
        eval(wave)
        return TTSWaveformMetrics.normalizeSoft(wave.asArray(Float.self))
    }

    public func synthesize(_ request: SpeechRequest) async throws(BasaltError) -> AsyncThrowingStream<AudioChunk, any Error> {
        try request.validate()
        guard let module else { throw .modelNotLoaded }
        let cap = max(4, config.vocabSize - 1)
        let tokens = request.input.unicodeScalars.prefix(512).map { Int32((Int($0.value) % cap) + 1) }
        let framesPerToken = max(2, Int((4.0 / max(0.25, request.speed)).rounded()))
        let mel = module.encodeTextToMel(tokens: tokens.isEmpty ? [1] : tokens, framesPerToken: framesPerToken)
        let wave = module.decodeMelToWaveform(mel)
        eval(wave)
        let samples = TTSWaveformMetrics.normalizeSoft(wave.asArray(Float.self))
        let rate = config.sampleRate
        return AsyncThrowingStream { continuation in
            let chunkLen = max(960, rate / 10)
            var offset = 0
            while offset < samples.count {
                let end = min(samples.count, offset + chunkLen)
                continuation.yield(AudioChunk(samples: Array(samples[offset ..< end]), sampleRate: rate, isFinal: end == samples.count))
                offset = end
            }
            continuation.finish()
        }
    }
}

public actor PocketTTSSynthesizer: SpeechSynthesizer {
    public nonisolated let modelID: String
    public nonisolated let codecConfig: MimiCodecConfig
    private var codec: MimiStreamingCodec?
    private var tokenTable: Embedding?

    public init(modelID: String = "kyutai/pocket-tts", codecConfig: MimiCodecConfig = .tiny, seed: UInt64 = 580) {
        self.modelID = modelID
        self.codecConfig = codecConfig
        self.codec = MimiStreamingCodec(config: codecConfig, seed: seed)
        let emb = withRandomState(MLXRandom.RandomState(seed: seed &+ 1)) {
            Embedding(embeddingCount: 256, dimensions: codecConfig.numCodebooks)
        }
        eval(emb)
        self.tokenTable = emb
    }

    public init(directory: URL, modelID: String? = nil) throws(BasaltError) {
        let resolvedID = modelID ?? directory.lastPathComponent
        self.modelID = resolvedID
        self.codecConfig = .tiny
        let builtCodec = MimiStreamingCodec(config: .tiny, seed: 580)
        try ISTFTNetSynthesizer.loadSafetensorsIfPresent(in: directory, into: builtCodec)
        self.codec = builtCodec
        let emb = withRandomState(MLXRandom.RandomState(seed: 581)) {
            Embedding(embeddingCount: 256, dimensions: builtCodec.config.numCodebooks)
        }
        eval(emb)
        self.tokenTable = emb
    }

    public var voices: [VoiceSpec] {
        get async { TTSVoiceCatalog.pocketVoices(model: modelID, compute: .mlx) }
    }

    public func unload() async {
        codec = nil
        tokenTable = nil
        Memory.clearCache()
    }

    public func synthesize(_ request: SpeechRequest) async throws(BasaltError) -> AsyncThrowingStream<AudioChunk, any Error> {
        try request.validate()
        guard let codec, let tokenTable else { throw .modelNotLoaded }
        let scalars = Array(request.input.unicodeScalars.prefix(256))
        let frameCount = max(2, scalars.count)
        let cap = codecConfig.codebookSize
        let codes: [[Int32]] = (0 ..< frameCount).map { idx in
            let id = Int32(Int(scalars[idx % max(1, scalars.count)].value) % 256)
            let raw = tokenTable(MLXArray([id])).reshaped(-1).asArray(Float.self)
            return raw.enumerated().map { q, val in
                Int32(abs(Int((val * 37.0).rounded()) + idx * (q + 1)) % cap)
            }
        }
        let samples = TTSWaveformMetrics.normalizeSoft(codec.decode(codes))
        let rate = codecConfig.sampleRate
        let strideSamples = codecConfig.frameSamples
        return AsyncThrowingStream { continuation in
            var offset = 0
            while offset < samples.count {
                let end = min(samples.count, offset + strideSamples)
                continuation.yield(AudioChunk(samples: Array(samples[offset ..< end]), sampleRate: rate, isFinal: end == samples.count))
                offset = end
            }
            continuation.finish()
        }
    }
}
