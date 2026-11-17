import Foundation
import BasaltCore

public enum SpeechSynthesizerLoader {
    public struct LoadedSynthesizer: Sendable {
        public let synthesizer: any SpeechSynthesizer
        public let candidate: SpeechCandidate
        public let family: TTSModelFamily

        public init(synthesizer: any SpeechSynthesizer, candidate: SpeechCandidate, family: TTSModelFamily) {
            (self.synthesizer, self.candidate, self.family) = (synthesizer, candidate, family)
        }
    }

    public static func load(
        identifier: String,
        directory: URL,
        aneVocoder: ANEAcousticVocoder? = nil
    ) async throws(BasaltError) -> LoadedSynthesizer {
        let family = try TTSModelFamily.validateSupported(identifier: identifier, directory: directory)
        if CoreAISpeechSynthesizer.isBundle(directory) {
            let ane = try await CoreAISpeechSynthesizer(directory: directory)
            let candidate = SpeechCandidate(
                id: identifier,
                model: ane.bundle.modelID,
                compute: ane.bundle.compute,
                isAvailable: true,
                estimatedTTFAMs: CoreAISpeechSynthesizer.defaultEstimatedTTFAMs
            )
            return LoadedSynthesizer(synthesizer: ane, candidate: candidate, family: family)
        }
        let resolvedVocoder: ANEAcousticVocoder? = if let aneVocoder { aneVocoder } else { try? await loadBundledVocoderIfPresent(in: directory) }
        return try await loadMLXSynthesizer(
            family: family, identifier: identifier, directory: directory, aneVocoder: resolvedVocoder
        )
    }

    private static func loadMLXSynthesizer(
        family: TTSModelFamily,
        identifier: String,
        directory: URL,
        aneVocoder: ANEAcousticVocoder?
    ) async throws(BasaltError) -> LoadedSynthesizer {
        switch family {
        case .kokoro82M:
            let synth = try KokoroMLXSynthesizer(directory: directory, modelID: identifier)
            return LoadedSynthesizer(
                synthesizer: synth,
                candidate: SpeechCandidate(id: identifier, model: synth.config.modelID, compute: .mlx, isAvailable: true, estimatedTTFAMs: 55),
                family: .kokoro82M
            )
        case .qwen3TTS:
            let synth = try Qwen3TTSSynthesizer(directory: directory, modelID: identifier, aneVocoder: aneVocoder)
            return LoadedSynthesizer(
                synthesizer: synth,
                candidate: SpeechCandidate(id: identifier, model: synth.config.modelID, compute: .mlx, isAvailable: true, estimatedTTFAMs: 95),
                family: .qwen3TTS
            )
        case .chatterbox:
            let synth = try ChatterboxSynthesizer(directory: directory, modelID: identifier, aneVocoder: aneVocoder)
            return LoadedSynthesizer(
                synthesizer: synth,
                candidate: SpeechCandidate(id: identifier, model: synth.config.modelID, compute: .mlx, isAvailable: true, estimatedTTFAMs: 130),
                family: .chatterbox
            )
        case .cosyVoice3:
            let synth = try CosyVoice3Synthesizer(directory: directory, modelID: identifier, aneVocoder: aneVocoder)
            return LoadedSynthesizer(
                synthesizer: synth,
                candidate: SpeechCandidate(id: identifier, model: synth.config.modelID, compute: .mlx, isAvailable: true, estimatedTTFAMs: 110),
                family: .cosyVoice3
            )
        case .zonos3Mini:
            let synth = try await Zonos3MiniSynthesizer(directory: directory, modelID: identifier, aneVocoder: aneVocoder)
            return LoadedSynthesizer(
                synthesizer: synth,
                candidate: SpeechCandidate(id: identifier, model: synth.config.modelID, compute: .mlx, isAvailable: true, estimatedTTFAMs: 85),
                family: .zonos3Mini
            )
        case .appleSpeech:
            throw .modelConfigurationInvalid(
                path: directory.path,
                reason: "apple-speech is a system synthesizer and cannot be loaded from a model directory"
            )
        }
    }

    public static func inspectOnDemand(
        identifier: String,
        directory: URL
    ) throws(BasaltError) -> (candidate: SpeechCandidate, voices: [VoiceSpec]) {
        let family = try TTSModelFamily.validateSupported(identifier: identifier, directory: directory)
        if CoreAISpeechSynthesizer.isBundle(directory) {
            let bundle = try CoreAISpeechBundle(directory: directory)
            let candidate = SpeechCandidate(
                id: identifier,
                model: bundle.modelID,
                compute: bundle.compute,
                isAvailable: true,
                estimatedTTFAMs: CoreAISpeechSynthesizer.defaultEstimatedTTFAMs
            )
            return (candidate, bundle.voiceSpecs)
        }
        let candidate = SpeechCandidate(
            id: identifier,
            model: identifier,
            compute: .mlx,
            isAvailable: true,
            estimatedTTFAMs: family.estimatedTTFAMs
        )
        let voices: [VoiceSpec] = switch family {
        case .kokoro82M: TTSVoiceCatalog.kokoroVoices(model: identifier, compute: .mlx)
        case .qwen3TTS: TTSVoiceCatalog.qwen3Voices(model: identifier, compute: .mlx)
        case .chatterbox: TTSVoiceCatalog.chatterboxVoices(model: identifier, compute: .mlx)
        case .cosyVoice3: TTSVoiceCatalog.cosyVoice3Voices(model: identifier, compute: .mlx)
        case .zonos3Mini: TTSVoiceCatalog.zonos3Voices(model: identifier, compute: .mlx)
        case .appleSpeech: []
        }
        return (candidate, voices)
    }

    @discardableResult
    public static func probeSpeechSynthesizer(
        _ synthesizer: any SpeechSynthesizer,
        voice: String? = nil
    ) async throws(BasaltError) -> Float {
        let availableVoices = await synthesizer.voices
        let chosenVoice = voice ?? availableVoices.first?.id ?? "alloy"
        let modelName = availableVoices.first?.model ?? "tts-probe"
        let request = try SpeechRequest(model: modelName, input: "Basalt ready.", voice: chosenVoice)
        let stream = try await synthesizer.synthesize(request)
        var collected: [Float] = []
        var sampleRate = 0
        do {
            for try await chunk in stream {
                if chunk.sampleRate > 0 { sampleRate = chunk.sampleRate }
                collected.append(contentsOf: chunk.samples)
            }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .modelProbeFailed(path: modelName, check: "speech_synthesis", reason: error.localizedDescription)
        }
        guard sampleRate > 0, !collected.isEmpty else {
            throw .modelProbeFailed(path: modelName, check: "speech_synthesis", reason: "synthesizer produced zero audio samples")
        }
        guard TTSWaveformMetrics.allFinite(collected) else {
            throw .modelProbeFailed(path: modelName, check: "speech_finite", reason: "synthesizer emitted NaN or Inf audio samples")
        }
        let rms = TTSWaveformMetrics.rms(collected)
        guard rms >= 1e-4 else {
            throw .modelProbeFailed(path: modelName, check: "speech_rms", reason: "synthesizer output is silent (rms=\(rms))")
        }
        return rms
    }

    private static func loadBundledVocoderIfPresent(in directory: URL) async throws(BasaltError) -> ANEAcousticVocoder? {
        let fm = FileManager.default
        let hasAI = fm.fileExists(atPath: directory.appending(component: "vocoder.aimodel").path)
        let hasML = fm.fileExists(atPath: directory.appending(component: "vocoder.mlmodelc").path)
        guard hasAI || hasML else { return nil }
        return try await ANEAcousticVocoder(directory: directory)
    }
}
