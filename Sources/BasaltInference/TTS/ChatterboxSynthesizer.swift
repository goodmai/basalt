import Foundation
import MLX
import MLXNN
import BasaltCore

public struct ChatterboxConfig: Sendable, Codable, Equatable {
    public var modelID: String = "Chatterbox-Turbo"
    public var isMultilingual: Bool = true
    public var vocabSize: Int = 512
    public var hiddenSize: Int = 384
    public var melBins: Int = 80
    public var speakerDim: Int = 192
    public var hopLength: Int = 240
    public var sampleRate: Int = SpeechRequest.defaultSampleRate
    public var defaultExaggeration: Double = 0.5

    public static var tiny: ChatterboxConfig {
        var cfg = ChatterboxConfig(modelID: "Chatterbox-Turbo-Tiny", isMultilingual: true, vocabSize: 256, hiddenSize: 32)
        cfg.melBins = 24
        cfg.speakerDim = 32
        cfg.hopLength = 60
        return cfg
    }

    public init(
        modelID: String = "Chatterbox-Turbo",
        isMultilingual: Bool = true,
        vocabSize: Int = 512,
        hiddenSize: Int = 384,
        sampleRate: Int = SpeechRequest.defaultSampleRate,
        defaultExaggeration: Double = 0.5
    ) {
        self.modelID = modelID
        self.isMultilingual = isMultilingual
        self.vocabSize = max(32, vocabSize)
        self.hiddenSize = max(16, hiddenSize)
        self.sampleRate = max(8_000, sampleRate)
        self.defaultExaggeration = min(2.0, max(0.0, defaultExaggeration))
    }

    public static func from(directory: URL, fallbackID: String = "Chatterbox-Turbo") -> ChatterboxConfig {
        let url = directory.appending(component: "config.json")
        let lower = fallbackID.lowercased()
        let multiByID = lower.contains("multilingual") || !lower.contains("turbo")
        guard let data = try? Data(contentsOf: url),
              let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return ChatterboxConfig(modelID: fallbackID, isMultilingual: multiByID)
        }
        var cfg = ChatterboxConfig(
            modelID: raw["model_id"] as? String ?? fallbackID,
            isMultilingual: raw["multilingual"] as? Bool ?? multiByID,
            vocabSize: raw["vocab_size"] as? Int ?? 512,
            hiddenSize: raw["hidden_size"] as? Int ?? 384,
            sampleRate: raw["sample_rate"] as? Int ?? SpeechRequest.defaultSampleRate,
            defaultExaggeration: raw["exaggeration_default"] as? Double ?? 0.5
        )
        cfg.melBins = max(8, raw["n_mels"] as? Int ?? raw["mel_bins"] as? Int ?? 80)
        cfg.speakerDim = max(8, raw["speaker_dim"] as? Int ?? 192)
        cfg.hopLength = max(16, raw["hop_length"] as? Int ?? 240)
        return cfg
    }
}

final class ChatterboxS3GenModule: Module {
    let config: ChatterboxConfig
    @ModuleInfo(key: "token_emb") var tokenEmbedding: Embedding
    @ModuleInfo(key: "tag_emb") var tagEmbedding: Embedding
    @ModuleInfo(key: "speaker_proj") var speakerProj: Linear
    @ModuleInfo(key: "s3gen_prior") var priorProj: Linear
    @ModuleInfo(key: "s3gen_velocity") var velocityConv: Conv1d
    @ModuleInfo(key: "mel_proj") var melProj: Linear
    @ModuleInfo(key: "vocoder_up") var vocoderUp: ConvTransposed1d
    @ModuleInfo(key: "vocoder_out") var vocoderOut: Conv1d

    init(_ config: ChatterboxConfig) {
        self.config = config
        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._tagEmbedding.wrappedValue = Embedding(
            embeddingCount: ChatterboxTagParser.ParalinguisticTag.allCases.count + 1,
            dimensions: config.hiddenSize
        )
        self._speakerProj.wrappedValue = Linear(config.speakerDim, config.hiddenSize)
        self._priorProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        self._velocityConv.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: 3,
            padding: 1
        )
        self._melProj.wrappedValue = Linear(config.hiddenSize, config.melBins)
        self._vocoderUp.wrappedValue = ConvTransposed1d(
            inputChannels: config.melBins,
            outputChannels: max(8, config.melBins / 2),
            kernelSize: config.hopLength,
            stride: config.hopLength
        )
        self._vocoderOut.wrappedValue = Conv1d(
            inputChannels: max(8, config.melBins / 2),
            outputChannels: 1,
            kernelSize: 3,
            padding: 1
        )
        super.init()
    }

    func synthesizeMelAndWave(
        tokenIDs: [Int32],
        tagIDs: [Int32],
        burstGain: [Float],
        speakerVec: [Float],
        exaggeration: Float
    ) -> (waveform: MLXArray, mel: MLXArray) {
        let count = tokenIDs.count
        let tokH = tokenEmbedding(MLXArray(tokenIDs).reshaped(1, count))
        let tagH = tagEmbedding(MLXArray(tagIDs).reshaped(1, count))
        let spkH = speakerProj(MLXArray(speakerVec).reshaped(1, 1, config.speakerDim))
        let prosodyScale = 0.75 + 0.75 * exaggeration
        let cond = (tokH + prosodyScale * tagH) * prosodyScale + spkH
        let x0 = tanh(priorProj(cond))
        let x1 = x0 + velocityConv(MLXNN.silu(x0 + cond)) * prosodyScale
        let mel = melProj(x1)
        let neuralWave = vocoderOut(MLXNN.silu(vocoderUp(mel))).reshaped(-1)
        let totalSamples = neuralWave.dim(0)
        let burstPerFrame = MLXArray(burstGain).reshaped(count, 1)
        let sampleEnvelope = broadcast(burstPerFrame, to: [count, config.hopLength]).reshaped(-1)
        let envTrimmed = sampleEnvelope.dim(0) >= totalSamples
            ? sampleEnvelope[..<totalSamples]
            : padded(sampleEnvelope, widths: [IntOrPair((0, totalSamples - sampleEnvelope.dim(0)))])
        let baseFreq = 165.0 + 45.0 * exaggeration
        let t = MLXArray((0 ..< totalSamples).map { Float($0) / Float(config.sampleRate) })
        let vibrato = (0.04 + 0.08 * exaggeration) * sin(11.0 * Float.pi * t)
        let excitation = (0.28 + 0.16 * exaggeration) * envTrimmed * sin((2.0 * Float.pi * baseFreq) * t + vibrato)
        let gain = 0.65 + 0.45 * exaggeration
        return (tanh(gain * neuralWave + excitation), mel[0])
    }
}

public actor ChatterboxSynthesizer: SpeechSynthesizer {
    private struct S3GenPlan: Sendable {
        let tokens: [Int32]
        let tags: [Int32]
        let bursts: [Float]
        let speakerVec: [Float]
        let exaggeration: Float
        let expectedEpoch: Int
    }

    public nonisolated let config: ChatterboxConfig
    public nonisolated let aneVocoder: ANEAcousticVocoder?
    private var module: ChatterboxS3GenModule?
    private var epoch = 0

    public init(
        config: ChatterboxConfig = ChatterboxConfig(),
        aneVocoder: ANEAcousticVocoder? = nil,
        seed: UInt64 = 350
    ) {
        self.config = config
        self.aneVocoder = aneVocoder
        let built = withRandomState(MLXRandom.RandomState(seed: seed)) {
            ChatterboxS3GenModule(config)
        }
        eval(built)
        self.module = built
    }

    public init(
        directory: URL,
        modelID: String? = nil,
        aneVocoder: ANEAcousticVocoder? = nil
    ) throws(BasaltError) {
        let resolved = ChatterboxConfig.from(directory: directory, fallbackID: modelID ?? directory.lastPathComponent)
        self.config = resolved
        self.aneVocoder = aneVocoder
        let built = withRandomState(MLXRandom.RandomState(seed: 350)) {
            ChatterboxS3GenModule(resolved)
        }
        try ISTFTNetSynthesizer.loadSafetensorsIfPresent(in: directory, into: built)
        self.module = built
    }

    public var voices: [VoiceSpec] {
        get async { TTSVoiceCatalog.chatterboxVoices(model: config.modelID, compute: .mlx) }
    }

    public func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        SpeechRouteDecision(
            selected: SpeechCandidate(
                id: config.modelID,
                model: config.modelID,
                compute: .mlx,
                isAvailable: module != nil,
                estimatedTTFAMs: 130
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
        try validateLanguage(request.language)
        let voiceSeed = request.voice.unicodeScalars.reduce(7) { ($0 &* 31 &+ Int($1.value)) & 0x7fff }
        let speakerVec = try ReferenceAudioConditioner.speakerEmbedding(
            from: request.referenceAudio,
            voiceSeed: voiceSeed,
            dimension: config.speakerDim,
            maxSeconds: 8.0
        )
        let (tokens, tags, bursts) = Self.encodeSegments(
            ChatterboxTagParser.parse(request.input),
            vocabSize: config.vocabSize,
            speed: request.speed
        )
        let plan = S3GenPlan(
            tokens: tokens,
            tags: tags,
            bursts: bursts,
            speakerVec: speakerVec,
            exaggeration: Float(request.exaggeration ?? config.defaultExaggeration),
            expectedEpoch: epoch
        )
        return Self.makeStream(owner: self, plan: plan)
    }

    nonisolated private static func makeStream(
        owner: ChatterboxSynthesizer,
        plan: S3GenPlan
    ) -> AsyncThrowingStream<AudioChunk, any Error> {
        let vocoder = owner.aneVocoder
        let rate = vocoder?.sampleRate ?? owner.config.sampleRate
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try Task.checkCancellation()
                    let (mlxSamples, melRows) = try await owner.runS3Gen(plan)
                    let samples = if let vocoder {
                        try await vocoder.decodeMel(melRows)
                    } else {
                        mlxSamples
                    }
                    let step = max(1_200, rate / 6)
                    var pos = 0
                    while pos < samples.count {
                        try Task.checkCancellation()
                        let end = min(samples.count, pos + step)
                        continuation.yield(
                            AudioChunk(samples: Array(samples[pos ..< end]), sampleRate: rate, isFinal: end == samples.count)
                        )
                        pos = end
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

extension ChatterboxSynthesizer {
    private func validateLanguage(_ language: String?) throws(BasaltError) {
        guard let language, !language.isEmpty else { return }
        let code = String(language.lowercased().prefix { $0 != "-" && $0 != "_" })
        if !config.isMultilingual, code != "en" {
            throw .invalidRequestStructure(details: "speech: \(config.modelID) supports English only; use chatterbox-multilingual for '\(language)'")
        }
        guard TTSVoiceCatalog.chatterboxLanguages.contains(code) else {
            throw .invalidRequestStructure(details: "speech: Chatterbox supports 23 languages, got '\(language)'")
        }
    }

    private func runS3Gen(_ plan: S3GenPlan) throws(BasaltError) -> ([Float], [[Float]]) {
        guard let module, epoch == plan.expectedEpoch else { throw .modelNotLoaded }
        let (wave, mel) = module.synthesizeMelAndWave(
            tokenIDs: plan.tokens,
            tagIDs: plan.tags,
            burstGain: plan.bursts,
            speakerVec: plan.speakerVec,
            exaggeration: plan.exaggeration
        )
        eval(wave, mel)
        let bins = config.melBins
        let frames = mel.dim(0)
        let flatMel = mel.asType(.float32).reshaped(-1).asArray(Float.self)
        let melRows = (0 ..< frames).map { idx in Array(flatMel[(idx * bins) ..< ((idx + 1) * bins)]) }
        return (wave.asType(.float32).asArray(Float.self), melRows)
    }

    static func encodeSegments(
        _ segments: [ChatterboxTagParser.Segment],
        vocabSize: Int,
        speed: Double
    ) -> (tokens: [Int32], tags: [Int32], bursts: [Float]) {
        let cap = max(4, vocabSize - 1)
        let allTags = ChatterboxTagParser.ParalinguisticTag.allCases
        var tokens: [Int32] = []
        var tags: [Int32] = []
        var bursts: [Float] = []
        let repeatText = max(1, Int((2.0 / max(0.25, speed)).rounded()))
        for segment in segments {
            switch segment {
            case .text(let str):
                for scalar in str.unicodeScalars where tokens.count < 512 {
                    let tok = Int32((Int(scalar.value) % cap) + 1)
                    for _ in 0 ..< repeatText {
                        tokens.append(tok)
                        tags.append(0)
                        bursts.append(1.0)
                    }
                }
            case .tag(let tag):
                let tagIdx = Int32((allTags.firstIndex(of: tag) ?? 0) + 1)
                let burstFrames = (tag == .laugh || tag == .chuckle) ? 6 : 4
                for pulse in 0 ..< burstFrames {
                    tokens.append(Int32((Int(tagIdx) * 17 + pulse) % cap + 1))
                    tags.append(tagIdx)
                    bursts.append(pulse.isMultiple(of: 2) ? 1.85 : 0.65)
                }
            }
        }
        if tokens.isEmpty {
            return ([1, 2], [0, 0], [1.0, 1.0])
        }
        return (tokens, tags, bursts)
    }
}
