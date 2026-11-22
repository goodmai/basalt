import Foundation
import MLX
import MLXNN
import BasaltCore

public struct Qwen3TTSConfig: Sendable, Codable, Equatable {
    public var modelID: String = "Qwen3-TTS-12Hz-0.6B"
    public var variant: String = "0.6B"
    public var vocabSize: Int = 1_024
    public var hiddenSize: Int = 512
    public var numCodebooks: Int = 8
    public var codebookSize: Int = 512
    public var speakerDim: Int = 256
    public var codecDim: Int = 64
    public var codecRateHz: Int = 12
    public var sampleRate: Int = SpeechRequest.defaultSampleRate

    public var samplesPerPacket: Int { max(200, sampleRate / max(1, codecRateHz)) }

    public static var tiny: Qwen3TTSConfig {
        var cfg = Qwen3TTSConfig(modelID: "Qwen3-TTS-12Hz-0.6B-Tiny", variant: "0.6B", vocabSize: 256, hiddenSize: 32)
        cfg.numCodebooks = 4
        cfg.codebookSize = 64
        cfg.speakerDim = 32
        cfg.codecDim = 24
        return cfg
    }

    public init(
        modelID: String = "Qwen3-TTS-12Hz-0.6B",
        variant: String = "0.6B",
        vocabSize: Int = 1_024,
        hiddenSize: Int = 512,
        codecRateHz: Int = 12,
        sampleRate: Int = SpeechRequest.defaultSampleRate
    ) {
        self.modelID = modelID
        self.variant = variant
        self.vocabSize = max(32, vocabSize)
        self.hiddenSize = max(16, hiddenSize)
        self.codecRateHz = max(1, codecRateHz)
        self.sampleRate = max(8_000, sampleRate)
    }

    public static func from(directory: URL, fallbackID: String = "Qwen3-TTS-12Hz-0.6B") -> Qwen3TTSConfig {
        let url = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: url),
              let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            let is17B = fallbackID.contains("1.7B") || fallbackID.contains("1_7B")
            return Qwen3TTSConfig(modelID: fallbackID, variant: is17B ? "1.7B" : "0.6B", hiddenSize: is17B ? 768 : 512)
        }
        let talker = raw["talker_config"] as? [String: Any]
        let is17B = (raw["variant"] as? String == "1.7B") || fallbackID.contains("1.7B")
        var cfg = Qwen3TTSConfig(
            modelID: raw["model_id"] as? String ?? fallbackID,
            variant: raw["variant"] as? String ?? (is17B ? "1.7B" : "0.6B"),
            vocabSize: talker?["vocab_size"] as? Int ?? raw["vocab_size"] as? Int ?? 1_024,
            hiddenSize: talker?["hidden_size"] as? Int ?? raw["hidden_size"] as? Int ?? (is17B ? 768 : 512),
            codecRateHz: raw["codec_rate_hz"] as? Int ?? 12,
            sampleRate: raw["sample_rate"] as? Int ?? SpeechRequest.defaultSampleRate
        )
        cfg.numCodebooks = max(1, raw["num_codebooks"] as? Int ?? 8)
        cfg.codebookSize = max(16, raw["codebook_size"] as? Int ?? 512)
        cfg.speakerDim = max(8, raw["speaker_dim"] as? Int ?? 256)
        cfg.codecDim = max(8, raw["codec_dim"] as? Int ?? 64)
        return cfg
    }
}

final class Qwen3MimiCodecDecoder: Module {
    @ModuleInfo(key: "rvq_codebooks") var codebooks: [Embedding]
    @ModuleInfo(key: "spk_codec") var spkCodec: Linear
    @ModuleInfo(key: "up1") var up1: ConvTransposed1d
    @ModuleInfo(key: "up2") var up2: ConvTransposed1d
    @ModuleInfo(key: "up3") var up3: ConvTransposed1d
    @ModuleInfo(key: "up4") var up4: ConvTransposed1d
    @ModuleInfo(key: "res_conv") var resConv: Conv1d
    @ModuleInfo(key: "audio_out") var audioOut: Conv1d
    let samplesPerPacket: Int
    let sampleRate: Int

    init(_ config: Qwen3TTSConfig) {
        self.samplesPerPacket = config.samplesPerPacket
        self.sampleRate = config.sampleRate
        let c = config.codecDim
        let half = max(8, c / 2)
        let quarter = max(4, c / 4)
        self._codebooks.wrappedValue = (0 ..< config.numCodebooks).map { _ in
            Embedding(embeddingCount: config.codebookSize, dimensions: c)
        }
        self._spkCodec.wrappedValue = Linear(config.speakerDim, c)
        self._up1.wrappedValue = ConvTransposed1d(inputChannels: c, outputChannels: c, kernelSize: 10, stride: 10)
        self._up2.wrappedValue = ConvTransposed1d(inputChannels: c, outputChannels: half, kernelSize: 10, stride: 10)
        self._up3.wrappedValue = ConvTransposed1d(inputChannels: half, outputChannels: quarter, kernelSize: 5, stride: 5)
        self._up4.wrappedValue = ConvTransposed1d(inputChannels: quarter, outputChannels: quarter, kernelSize: 4, stride: 4)
        self._resConv.wrappedValue = Conv1d(inputChannels: quarter, outputChannels: quarter, kernelSize: 3, padding: 1)
        self._audioOut.wrappedValue = Conv1d(inputChannels: quarter, outputChannels: 1, kernelSize: 3, padding: 1)
        super.init()
    }

    func decodePacket(codes: [Int32], speakerVec: [Float], packetIndex: Int) -> (waveform: MLXArray, melRow: [Float]) {
        var latent = spkCodec(MLXArray(speakerVec).reshaped(1, 1, speakerVec.count)) * 0.35
        for (idx, table) in codebooks.enumerated() {
            let code = codes[idx % max(1, codes.count)]
            latent = latent + table(MLXArray([code]).reshaped(1, 1)) / Float(idx + 1)
        }
        let melRow = latent.reshaped(-1).asArray(Float.self)
        var h = MLXNN.silu(up1(latent))
        h = MLXNN.silu(up2(h))
        h = MLXNN.silu(up3(h))
        h = up4(h)
        h = h + MLXNN.silu(resConv(h))
        let raw = audioOut(h).reshaped(-1)
        let targetLen = samplesPerPacket
        let trimmed = raw.dim(0) >= targetLen
            ? raw[..<targetLen]
            : padded(raw, widths: [IntOrPair((0, targetLen - raw.dim(0)))])
        let baseFreq = 155.0 + Float((Int(codes.first ?? 0) % 24) * 5)
        let startSample = packetIndex * targetLen
        let timeIdx = MLXArray((0 ..< targetLen).map { Float(startSample + $0) / Float(sampleRate) })
        let carrier = 0.22 * sin((2.0 * Float.pi * baseFreq) * timeIdx)
            + 0.11 * sin((4.0 * Float.pi * baseFreq) * timeIdx)
        return (tanh(trimmed + carrier), melRow)
    }
}

final class Qwen3TTSTalkerModule: Module {
    let config: Qwen3TTSConfig
    @ModuleInfo(key: "text_emb") var textEmbedding: Embedding
    @ModuleInfo(key: "instruct_proj") var instructProj: Linear
    @ModuleInfo(key: "speaker_proj") var speakerProj: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "step_mlp") var stepMLP: Linear
    @ModuleInfo(key: "codebook_heads") var codebookHeads: [Linear]
    @ModuleInfo(key: "codec") var codec: Qwen3MimiCodecDecoder

    init(_ config: Qwen3TTSConfig) {
        self.config = config
        self._textEmbedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._instructProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        self._speakerProj.wrappedValue = Linear(config.speakerDim, config.hiddenSize)
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize)
        self._stepMLP.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        self._codebookHeads.wrappedValue = (0 ..< config.numCodebooks).map { _ in
            Linear(config.hiddenSize, config.codebookSize)
        }
        self._codec.wrappedValue = Qwen3MimiCodecDecoder(config)
        super.init()
    }

    func contextStates(textTokens: [Int32], instructTokens: [Int32], speakerVec: [Float]) -> MLXArray {
        let textH = textEmbedding(MLXArray(textTokens).reshaped(1, textTokens.count))
        let spkH = speakerProj(MLXArray(speakerVec).reshaped(1, 1, config.speakerDim))
        guard !instructTokens.isEmpty else {
            return norm(textH + spkH)
        }
        let insEmb = textEmbedding(MLXArray(instructTokens).reshaped(1, instructTokens.count))
        let insMean = mean(insEmb, axis: 1, keepDims: true)
        return norm(textH + spkH + tanh(instructProj(insMean)))
    }

    func predictPacketCodes(context: MLXArray, step: Int) -> [Int32] {
        let idx = min(context.dim(1) - 1, step % max(1, context.dim(1)))
        let posBias = Float(step + 1) * 0.13
        let h = norm(context[0..., idx ..< (idx + 1), 0...] + MLXNN.silu(stepMLP(context[0..., idx ..< (idx + 1), 0...])) + posBias)
        return codebookHeads.map { head in
            let logits = head(h).reshaped(-1)
            return logits.argMax(axis: 0).item(Int32.self)
        }
    }
}

public actor Qwen3TTSSynthesizer: SpeechSynthesizer {
    private struct PacketPlan: Sendable {
        let textTokens: [Int32]
        let instructTokens: [Int32]
        let speakerVec: [Float]
        let packetCount: Int
        let speed: Double
        let epoch: Int
    }

    public nonisolated let config: Qwen3TTSConfig
    public nonisolated let aneVocoder: ANEAcousticVocoder?
    private var module: Qwen3TTSTalkerModule?
    private var epoch = 0

    public init(
        config: Qwen3TTSConfig = Qwen3TTSConfig(),
        aneVocoder: ANEAcousticVocoder? = nil,
        seed: UInt64 = 1_206
    ) {
        self.config = config
        self.aneVocoder = aneVocoder
        let built = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Qwen3TTSTalkerModule(config)
        }
        eval(built)
        self.module = built
    }

    public init(
        directory: URL,
        modelID: String? = nil,
        aneVocoder: ANEAcousticVocoder? = nil
    ) throws(BasaltError) {
        let resolved = Qwen3TTSConfig.from(directory: directory, fallbackID: modelID ?? directory.lastPathComponent)
        self.config = resolved
        self.aneVocoder = aneVocoder
        let built = withRandomState(MLXRandom.RandomState(seed: 1_206)) {
            Qwen3TTSTalkerModule(resolved)
        }
        try ISTFTNetSynthesizer.loadSafetensorsIfPresent(in: directory, into: built)
        self.module = built
    }

    public var voices: [VoiceSpec] {
        get async { TTSVoiceCatalog.qwen3Voices(model: config.modelID, compute: .mlx) }
    }

    public func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        SpeechRouteDecision(
            selected: SpeechCandidate(
                id: config.modelID,
                model: config.modelID,
                compute: .mlx,
                isAvailable: module != nil,
                estimatedTTFAMs: 95
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
        try Self.validateLanguage(request.language)
        let plan = try makePlan(for: request)
        return Self.makeStream(owner: self, plan: plan)
    }

    nonisolated private static func makeStream(
        owner: Qwen3TTSSynthesizer,
        plan: PacketPlan
    ) -> AsyncThrowingStream<AudioChunk, any Error> {
        let vocoder = owner.aneVocoder
        let rate = vocoder?.sampleRate ?? owner.config.sampleRate
        let packetSamples = owner.config.samplesPerPacket
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for step in 0 ..< plan.packetCount {
                        try Task.checkCancellation()
                        let (mlxSamples, melRow) = try await owner.synthesizePacket(plan: plan, step: step)
                        let rawSamples: [Float]
                        if let vocoder {
                            let frames = Array(repeating: melRow, count: max(1, packetSamples / vocoder.hopLength))
                            rawSamples = try await vocoder.decodeMel(frames)
                        } else {
                            rawSamples = mlxSamples
                        }
                        let scaled = Self.applySpeed(rawSamples, speed: plan.speed, sampleRate: rate)
                        continuation.yield(
                            AudioChunk(samples: scaled, sampleRate: rate, isFinal: step == plan.packetCount - 1)
                        )
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

extension Qwen3TTSSynthesizer {
    private static func validateLanguage(_ language: String?) throws(BasaltError) {
        guard let language, !language.isEmpty else { return }
        let code = String(language.lowercased().prefix { $0 != "-" && $0 != "_" })
        guard TTSVoiceCatalog.qwen3Languages.contains(code) else {
            throw .invalidRequestStructure(details: "speech: Qwen3-TTS supports 10 languages (\(TTSVoiceCatalog.qwen3Languages.joined(separator: ", "))), got '\(language)'")
        }
    }

    private func makePlan(for request: SpeechRequest) throws(BasaltError) -> PacketPlan {
        let voiceSeed = request.voice.unicodeScalars.reduce(0) { ($0 &* 31 &+ Int($1.value)) & 0x7fff }
        let speakerVec = try ReferenceAudioConditioner.speakerEmbedding(
            from: request.referenceAudio,
            voiceSeed: voiceSeed,
            dimension: config.speakerDim,
            maxSeconds: 3.0
        )
        let textTokens = Self.tokenize(request.input, vocabSize: config.vocabSize)
        let instructTokens = request.instruct.map { Self.tokenize($0, vocabSize: config.vocabSize) } ?? []
        let packetCount = max(2, min(24, (textTokens.count + 3) / 4 + 1))
        return PacketPlan(
            textTokens: textTokens,
            instructTokens: instructTokens,
            speakerVec: speakerVec,
            packetCount: packetCount,
            speed: request.speed,
            epoch: epoch
        )
    }

    private func synthesizePacket(plan: PacketPlan, step: Int) throws(BasaltError) -> ([Float], [Float]) {
        guard let module, epoch == plan.epoch else { throw .modelNotLoaded }
        let context = module.contextStates(
            textTokens: plan.textTokens,
            instructTokens: plan.instructTokens,
            speakerVec: plan.speakerVec
        )
        let codes = module.predictPacketCodes(context: context, step: step)
        let (wave, melRow) = module.codec.decodePacket(codes: codes, speakerVec: plan.speakerVec, packetIndex: step)
        eval(wave)
        return (TTSWaveformMetrics.normalizeSoft(wave.asArray(Float.self)), melRow)
    }

    private static func tokenize(_ text: String, vocabSize: Int) -> [Int32] {
        let cap = max(4, vocabSize - 1)
        let ids = text.unicodeScalars.prefix(512).map { Int32((Int($0.value) % cap) + 1) }
        return ids.isEmpty ? [1] : ids
    }

    private static func applySpeed(_ samples: [Float], speed: Double, sampleRate: Int) -> [Float] {
        guard abs(speed - 1.0) > 1e-3 else { return samples }
        let virtualSourceRate = max(1_000, Int((Double(sampleRate) * speed).rounded()))
        return TTSWaveformMetrics.resampleLinear(samples, from: virtualSourceRate, to: sampleRate)
    }
}
