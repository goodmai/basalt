import Foundation
import MLX
import MLXNN
import BasaltCore

final class CosyVoice3PipelineModule: Module {
    let config: CosyVoice3Config
    @ModuleInfo(key: "talker") var talker: CosyVoice3TalkerModule
    @ModuleInfo(key: "dit") var dit: CosyVoice3CausalDiTModule
    @ModuleInfo(key: "hift") var hift: CosyVoice3CausalHiFTModule

    init(_ config: CosyVoice3Config) {
        self.config = config
        self._talker.wrappedValue = CosyVoice3TalkerModule(config)
        self._dit.wrappedValue = CosyVoice3CausalDiTModule(config)
        self._hift.wrappedValue = CosyVoice3CausalHiFTModule(config)
        super.init()
    }
}

public actor CosyVoice3Synthesizer: SpeechSynthesizer {
    private struct ChunkPlan: Sendable {
        let promptTokens: [Int32]
        let speakerVec: [Float]
        let totalFSQSteps: Int
        let speed: Double
        let epoch: Int
    }

    public nonisolated let config: CosyVoice3Config
    public nonisolated let aneVocoder: ANEAcousticVocoder?
    private var module: CosyVoice3PipelineModule?
    private var epoch = 0

    public init(
        config: CosyVoice3Config = CosyVoice3Config(),
        aneVocoder: ANEAcousticVocoder? = nil,
        seed: UInt64 = 2_512
    ) {
        self.config = config
        self.aneVocoder = aneVocoder
        let built = withRandomState(MLXRandom.RandomState(seed: seed)) {
            CosyVoice3PipelineModule(config)
        }
        eval(built)
        self.module = built
    }

    public init(
        directory: URL,
        modelID: String? = nil,
        aneVocoder: ANEAcousticVocoder? = nil
    ) throws(BasaltError) {
        let resolved = try CosyVoice3Config.from(
            directory: directory,
            fallbackID: modelID ?? directory.lastPathComponent
        )
        self.config = resolved
        self.aneVocoder = aneVocoder
        let built = withRandomState(MLXRandom.RandomState(seed: 2_512)) {
            CosyVoice3PipelineModule(resolved)
        }
        try ISTFTNetSynthesizer.loadSafetensorsIfPresent(in: directory, into: built)
        self.module = built
    }

    public var voices: [VoiceSpec] {
        get async { TTSVoiceCatalog.cosyVoice3Voices(model: config.modelID, compute: .mlx) }
    }

    public func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        SpeechRouteDecision(
            selected: SpeechCandidate(
                id: config.modelID,
                model: config.modelID,
                compute: .mlx,
                isAvailable: module != nil,
                estimatedTTFAMs: 110
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
        let plan = try buildPlan(for: request)
        return Self.makeStream(owner: self, plan: plan)
    }

    nonisolated private static func makeStream(
        owner: CosyVoice3Synthesizer,
        plan: ChunkPlan
    ) -> AsyncThrowingStream<AudioChunk, any Error> {
        let vocoder = owner.aneVocoder
        let rate = vocoder?.sampleRate ?? owner.config.sampleRate
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try Task.checkCancellation()
                    let fsqTokens = try await owner.runStage0Talker(plan: plan)
                    let chunkSize = max(4, min(8, fsqTokens.count))
                    var start = 0
                    while start < fsqTokens.count {
                        try Task.checkCancellation()
                        let end = min(fsqTokens.count, start + chunkSize)
                        let slice = Array(fsqTokens[start ..< end])
                        let (mlxWave, melRows) = try await owner.runStage1And2(
                            fsqSlice: slice,
                            speakerVec: plan.speakerVec,
                            expectedEpoch: plan.epoch
                        )
                        let rawSamples = if let vocoder {
                            try await vocoder.decodeMel(melRows)
                        } else {
                            mlxWave
                        }
                        let adjusted = Self.scaleSpeed(rawSamples, speed: plan.speed, sampleRate: rate)
                        continuation.yield(
                            AudioChunk(samples: adjusted, sampleRate: rate, isFinal: end == fsqTokens.count)
                        )
                        start = end
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

extension CosyVoice3Synthesizer {
    private static func validateLanguage(_ language: String?) throws(BasaltError) {
        guard let language, !language.isEmpty else { return }
        let code = String(language.lowercased().prefix { $0 != "-" && $0 != "_" })
        guard TTSVoiceCatalog.cosyVoice3Languages.contains(code) else {
            throw .invalidRequestStructure(details: "speech: Fun-CosyVoice3 does not support language '\(language)'")
        }
    }

    private func buildPlan(for request: SpeechRequest) throws(BasaltError) -> ChunkPlan {
        let formattedPrompt = CosyVoice3PromptBuilder.build(
            input: request.input,
            instruct: request.instruct
        )
        let inlineSpans = CosyVoice3PromptBuilder.extractInlineControlSpans(request.input)
        let promptTokens = Self.encodePrompt(formattedPrompt, vocabSize: config.textVocabSize)
        let voiceSeed = request.voice.unicodeScalars.reduce(19) { ($0 &* 31 &+ Int($1.value)) & 0x7fff }
        let speakerVec = try ReferenceAudioConditioner.speakerEmbedding(
            from: request.referenceAudio,
            voiceSeed: voiceSeed,
            dimension: config.speakerDim,
            maxSeconds: 6.0
        )
        let baseSteps = max(6, min(24, request.input.count / 3 + 4 + inlineSpans.count * 2))
        return ChunkPlan(
            promptTokens: promptTokens,
            speakerVec: speakerVec,
            totalFSQSteps: baseSteps,
            speed: request.speed,
            epoch: epoch
        )
    }

    private func runStage0Talker(plan: ChunkPlan) throws(BasaltError) -> [Int32] {
        guard let module, epoch == plan.epoch else { throw .modelNotLoaded }
        let tokens = module.talker.generateFSQTokens(
            promptTokens: plan.promptTokens,
            targetSteps: plan.totalFSQSteps
        )
        return tokens
    }

    private func runStage1And2(
        fsqSlice: [Int32],
        speakerVec: [Float],
        expectedEpoch: Int
    ) throws(BasaltError) -> ([Float], [[Float]]) {
        guard let module, epoch == expectedEpoch else { throw .modelNotLoaded }
        let mel = module.dit.solveCausalFlowMel(fsqTokens: fsqSlice, speakerVec: speakerVec)
        let wave = module.hift.synthesizeHiFT(mel: mel)
        eval(mel, wave)
        let bins = config.melBins
        let frames = mel.dim(1)
        let flatMel = mel[0].asType(.float32).reshaped(-1).asArray(Float.self)
        let melRows = (0 ..< frames).map { t in Array(flatMel[(t * bins) ..< ((t + 1) * bins)]) }
        let normalized = TTSWaveformMetrics.normalizeSoft(wave.asType(.float32).asArray(Float.self))
        return (normalized, melRows)
    }

    static func encodePrompt(_ prompt: String, vocabSize: Int) -> [Int32] {
        let cap = max(8, vocabSize - 1)
        let tokens = prompt.unicodeScalars.prefix(512).map { Int32((Int($0.value) % cap) + 1) }
        return tokens.isEmpty ? [1] : tokens
    }

    private static func scaleSpeed(_ samples: [Float], speed: Double, sampleRate: Int) -> [Float] {
        guard abs(speed - 1.0) > 1e-3 else { return samples }
        let srcRate = max(1_000, Int((Double(sampleRate) * speed).rounded()))
        return TTSWaveformMetrics.resampleLinear(samples, from: srcRate, to: sampleRate)
    }
}
