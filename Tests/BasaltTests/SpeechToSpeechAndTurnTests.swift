import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltAppCore
@testable import BasaltApp

@Suite("Semantic VoiceTurnCore, SileroVAD, SmartTurn EOT gating, and 4-speaker Sortformer diarization (#575)")
struct VoiceTurnCoreTests {
    @Test("VoiceTurnCore holds turn during low-EOT thinking pauses and commits immediately when semantic EOT reaches threshold")
    func holdsThinkingPauseAndCommitsOnSemanticEOT() {
        let config = VoiceTurnConfig(
            vadSpeechThreshold: 0.5,
            semanticTurnThreshold: 0.70,
            minSpeechSeconds: 0.16,
            minSilenceSeconds: 0.02,
            maxThinkingPauseSeconds: 2.0,
            diarizationThreshold: 0.45
        )
        var core = VoiceTurnCore(config: config)

        for _ in 0 ..< 4 {
            let decision = core.step(
                vadProbability: 0.92,
                semanticEOTProbability: 0.12,
                speakerPosteriors: [0.1, 0.88, 0.05, 0.02],
                frameDurationSeconds: 0.08
            )
            #expect(decision == .listening)
        }

        for _ in 0 ..< 5 {
            let pauseDecision = core.step(
                vadProbability: 0.08,
                semanticEOTProbability: 0.24,
                speakerPosteriors: [0.05, 0.20, 0.05, 0.05],
                frameDurationSeconds: 0.08
            )
            #expect(pauseDecision == .holdThinkingPause)
        }

        let committed = core.step(
            vadProbability: 0.05,
            semanticEOTProbability: 0.82,
            speakerPosteriors: [0.05, 0.15, 0.05, 0.05],
            frameDurationSeconds: 0.08
        )
        #expect(committed == .commitTurn(speakerId: 1))
        #expect(core.speechSecondsInTurn == 0)
        #expect(core.trailingSilenceSeconds == 0)
    }

    @Test("VoiceTurnCore tracks 4 Sortformer speakers, merges contiguous speaker segments, and scores SileroVAD and SmartTurn cues")
    func fourSpeakerSortformerAndHeuristicScorers() {
        var core = VoiceTurnCore()
        let posteriorsBySpeaker: [[Float]] = [
            [0.91, 0.05, 0.02, 0.02],
            [0.87, 0.06, 0.04, 0.03],
            [0.04, 0.89, 0.04, 0.03],
            [0.03, 0.05, 0.93, 0.02],
            [0.02, 0.03, 0.04, 0.94],
            [0.02, 0.04, 0.03, 0.90],
        ]
        for row in posteriorsBySpeaker {
            _ = core.step(vadProbability: 0.88, semanticEOTProbability: 0.15, speakerPosteriors: row, frameDurationSeconds: 0.10)
        }
        #expect(core.segments.count == 4)
        #expect(core.segments.map { $0.speakerId } == [0, 1, 2, 3])
        #expect(abs(core.segments[0].durationSeconds - 0.20) < 1e-6)
        #expect(abs(core.segments[3].durationSeconds - 0.20) < 1e-6)
        #expect(core.dominantSpeakerId == 3)

        let commit = core.step(vadProbability: 0.02, semanticEOTProbability: 0.91, frameDurationSeconds: 0.08)
        #expect(commit == .commitTurn(speakerId: 3))

        let silentPCM = Array(repeating: Float(0), count: 1_280)
        let voicedPCM = (0 ..< 1_280).map { Float(0.35 * sin(2.0 * .pi * 220.0 * Double($0) / 16_000.0)) }
        #expect(SileroVADScorer.score(silentPCM) < 0.1)
        #expect(SileroVADScorer.score(voicedPCM) > 0.7)
        #expect(SmartTurnScorer.score(transcript: "I think we should wait because") < 0.35)
        #expect(SmartTurnScorer.score(transcript: "Deploy the release now.", trailingEnergyDrop: 0.8) > 0.80)
    }
}

@Suite("Streaming 24 kHz Mimi & SNAC codecs and LFM2.5-Audio duplex runner (#575)", .serialized)
struct StreamingAudioCodecsAndDuplexTests {
    @Test("MimiStreamingCodec and SNACMultiScaleCodec maintain sub-80 ms frame stride and exact round-trip waveform shapes")
    func mimiAndSNACRoundTripShapes() throws {
        let mimi = MimiStreamingCodec(config: .tiny, seed: 101)
        #expect(mimi.config.sampleRate == 24_000)
        #expect(mimi.config.frameSamples == 960)
        #expect(mimi.config.frameStrideMs < 80.0)

        let threeFramesPCM = (0 ..< 2_880).map { idx in
            Float(0.4 * sin(2.0 * .pi * 330.0 * Double(idx) / 24_000.0))
        }
        let mimiCodes = mimi.encode(threeFramesPCM)
        #expect(mimiCodes.count == 3)
        #expect(mimiCodes.allSatisfy { $0.count == mimi.config.numCodebooks })
        #expect(mimiCodes.joined().allSatisfy { $0 >= 0 && $0 < Int32(mimi.config.codebookSize) })

        let mimiDecoded = mimi.decode(mimiCodes)
        #expect(mimiDecoded.count == 2_880)
        #expect(mimiDecoded.allSatisfy { $0.isFinite && abs($0) <= 1.0 })

        let unalignedTail = mimi.decode(mimi.encode(Array(threeFramesPCM.prefix(2_150))), expectedSamples: 2_150)
        #expect(unalignedTail.count == 2_150)

        let snac = SNACMultiScaleCodec(config: .tiny, seed: 202)
        #expect(snac.config.sampleRate == 24_000)
        #expect(snac.config.frameSamples == 960)
        #expect(snac.config.frameStrideMs < 80.0)

        let packet = snac.encode(threeFramesPCM)
        #expect(packet.frameCount == 3)
        #expect(packet.scale0.count == 3)
        #expect(packet.scale1.count == 6)
        #expect(packet.scale2.count == 12)
        let flattened = packet.flattenedFrames
        #expect(flattened.count == 3 && flattened.allSatisfy { $0.count == SNACMultiScalePacket.tokensPerFrame })
        #expect(try SNACMultiScalePacket.fromFlattened(flattened) == packet)

        let snacDecoded = snac.decode(packet)
        #expect(snacDecoded.count == 2_880)
        #expect(snacDecoded.allSatisfy { $0.isFinite && abs($0) <= 1.0 })
    }

    @Test("LFM25AudioSessionRunner holds thinking pauses, emits interleaved text and audio tokens on commit, and handles barge-in")
    func lfm25DuplexSessionStepsAndBargeIn() async throws {
        let runner = LFM25AudioSessionRunner(config: .tiny, seed: 303)
        let voicedFrame = (0 ..< 960).map { Float(0.35 * sin(2.0 * .pi * 300.0 * Double($0) / 24_000.0)) }
        let silentFrame = Array(repeating: Float(0), count: 960)

        for _ in 0 ..< 5 {
            let step = try await runner.step(pcmFrame: voicedFrame, vadOverride: 0.9, eotOverride: 0.15, speakerPosteriors: [0.0, 0.0, 0.9, 0.0])
            #expect(step.turnDecision == .listening)
            #expect(step.synthesizedSamples.isEmpty)
        }

        let holdStep = try await runner.step(pcmFrame: silentFrame, vadOverride: 0.05, eotOverride: 0.25)
        #expect(holdStep.turnDecision == .holdThinkingPause)
        #expect(holdStep.synthesizedSamples.isEmpty)

        let commitStep = try await runner.step(pcmFrame: silentFrame, vadOverride: 0.05, eotOverride: 0.88)
        #expect(commitStep.turnDecision == .commitTurn(speakerId: 2))
        #expect(commitStep.textToken != nil)
        #expect(commitStep.outputAudioCodes.count == 8)
        #expect(commitStep.synthesizedSamples.count == 960)

        let bargeStep = try await runner.step(pcmFrame: voicedFrame, vadOverride: 0.92, eotOverride: 0.10)
        #expect(bargeStep.turnDecision == .listening)
        let state = await runner.currentState()
        #expect(state.bargeInCount == 1)
    }
}

@Suite("MossFormer2 denoiser SNR, Soprano-80M Vocos synthesis, Qwen3ForcedAligner CTC timestamps, and VoiceEnhancementOptions (#575)", .serialized)
struct SpeechEnhancementAlignerAndOptionsTests {
    @Test("MossFormer2Enhancer improves SNR on synthetic 440 Hz sine plus noise while preserving exact sample counts")
    func mossFormer2ImprovesSNRAndPreservesLength() async throws {
        let enhancer = MossFormer2Enhancer(config: MossFormer2Config(), seed: 404)
        #expect(try await enhancer.enhance([]).isEmpty)
        #expect(try await enhancer.enhance([0.1, -0.2, 0.3]).count == 3)

        let rate = 16_000
        let count = 4_096
        var rng = SplitMix64(seed: 575)
        var clean = [Float](repeating: 0, count: count)
        var noisy = [Float](repeating: 0, count: count)
        for i in 0 ..< count {
            let t = Double(i) / Double(rate)
            let s = Float(0.45 * sin(2.0 * .pi * 440.0 * t))
            let white = Float(Double(Int64(bitPattern: rng.next())) / Double(Int64.max)) * 0.08
            let outBand = Float(0.28 * sin(2.0 * .pi * 5_600.0 * t) + 0.22 * cos(2.0 * .pi * 6_800.0 * t))
            clean[i] = s
            noisy[i] = s + outBand + white
        }

        let enhanced = try await enhancer.enhance(noisy, sampleRate: rate)
        #expect(enhanced.count == count)
        #expect(enhanced.allSatisfy { $0.isFinite })

        let snrBefore = Self.snrDecibels(reference: clean, estimate: noisy)
        let snrAfter = Self.snrDecibels(reference: clean, estimate: enhanced)
        #expect(snrAfter > snrBefore + 8.0)

        let primeSamples = Array(noisy.prefix(1_003))
        let primeEnhanced = try await enhancer.enhance(primeSamples, sampleRate: rate)
        #expect(primeEnhanced.count == 1_003)
    }

    @Test("Qwen3ForcedAligner guarantees monotonic 0 <= start <= end word timestamps and populates verbose_json words")
    func qwen3ForcedAlignerMonotonicTimestamps() async throws {
        let dt = 0.02
        let words = ["native", "duplex", "basalt", "speech"]
        let wordTokens = [[1], [2], [3], [4]]
        var logProbs = Array(repeating: [Float]([-0.1, -4.0, -4.0, -4.0, -4.0]), count: 40)
        for t in 4 ..< 10 { logProbs[t] = [-4.0, -0.05, -4.0, -4.0, -4.0] }
        for t in 13 ..< 20 { logProbs[t] = [-4.0, -4.0, -0.05, -4.0, -4.0] }
        for t in 22 ..< 29 { logProbs[t] = [-4.0, -4.0, -4.0, -0.05, -4.0] }
        for t in 31 ..< 38 { logProbs[t] = [-4.0, -4.0, -4.0, -4.0, -0.05] }

        let viterbiAligned = Qwen3CTCAlignerCore.align(
            logProbabilities: logProbs,
            wordTokenIDs: wordTokens,
            words: words,
            frameDurationSeconds: dt
        )
        #expect(viterbiAligned.map { $0.word } == words)
        Self.assertStrictMonotonic(viterbiAligned, maxDuration: 40.0 * dt)
        #expect(abs(viterbiAligned[0].start - 0.08) < 1e-6 && abs(viterbiAligned[0].end - 0.20) < 1e-6)
        #expect(abs(viterbiAligned[3].start - 0.62) < 1e-6 && abs(viterbiAligned[3].end - 0.76) < 1e-6)

        let shortFallback = Qwen3CTCAlignerCore.align(
            logProbabilities: [[-1.0, -1.0], [-1.0, -1.0]],
            wordTokenIDs: [[1], [1], [1], [1], [1]],
            words: ["a", "b", "c", "d", "e"],
            frameDurationSeconds: 0.02
        )
        #expect(shortFallback.count == 5)
        Self.assertStrictMonotonic(shortFallback, maxDuration: 0.04 + 1e-9)

        let aligner = Qwen3ForcedAligner(vocabSize: 64, hiddenDim: 32, frameDurationSeconds: 0.02, seed: 505)
        let burstPCM = (0 ..< 16_000).map { idx -> Float in
            let inBurst = (1_600 ..< 4_800).contains(idx) || (6_400 ..< 9_600).contains(idx) || (11_200 ..< 14_400).contains(idx)
            return inBurst ? Float(0.4 * sin(2.0 * .pi * 350.0 * Double(idx) / 16_000.0)) : 0
        }
        let clip = try AudioClip(samples: burstPCM)
        let enriched = try await aligner.alignTranscription(Transcription(text: "hello local voice", language: "en"), audio: clip)
        #expect(enriched.words.count == 3)
        Self.assertStrictMonotonic(enriched.words, maxDuration: clip.seconds + 1e-6)

        let upload = TranscriptionsController.Upload(request: try TranscriptionRequest(audio: clip, language: "en"), format: .verboseJSON)
        let response = try TranscriptionsController.response(enriched, for: upload)
        #expect(response.status == .ok)
    }

    @Test("Soprano80MSynthesizer stays under the 200 MB profile, synthesizes 24 kHz audio, and VoiceEnhancementOptions persist with EN/RU parity")
    func soprano80MAndVoiceEnhancementOptions() async throws {
        #expect(TTSModelFamily.detect(identifier: "ekwek/Soprano-80M") == .soprano80M)
        #expect(TTSModelFamily.detect(identifier: "kyutai/pocket-tts") == .pocketTTS)
        let soprano = Soprano80MSynthesizer(config: .tiny, seed: 606)
        #expect(soprano.config.estimatedResidentMB < Soprano80MConfig.maxSub200MBProfile)

        var mel: [[Float]] = []
        for f in 0 ..< 12 {
            var row: [Float] = []
            for b in 0 ..< soprano.config.melBins {
                row.append(Float(sin(Double(f + b) * 0.2)))
            }
            mel.append(row)
        }
        let melWave = try await soprano.synthesizeFromMel(mel)
        #expect(!melWave.isEmpty && melWave.allSatisfy { $0.isFinite })
        let probeRMS = try await SpeechSynthesizerLoader.probeSpeechSynthesizer(soprano)
        #expect(probeRMS >= 1e-4)

        let suiteName = "basalt-voice-enhancement-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        #expect(AppSettings.voiceEnhancementOptions(in: defaults) == VoiceEnhancementOptions())
        let custom = VoiceEnhancementOptions(noiseSuppressionEnabled: false, semanticTurnTakingEnabled: true, duplexSpeechToSpeechEnabled: true)
        AppSettings.setVoiceEnhancementOptions(custom, in: defaults)
        #expect(AppSettings.voiceEnhancementOptions(in: defaults) == custom)

        var voiceState = VoiceSessionState()
        _ = VoiceReducer.reduce(&voiceState, .setEnhancement(custom))
        _ = VoiceReducer.reduce(&voiceState, .pushToTalkPressed)
        #expect(VoiceReducer.reduce(&voiceState, .turnEvaluated(.holdThinkingPause)).isEmpty && voiceState.holdingThinkingPause)
        #expect(VoiceReducer.reduce(&voiceState, .turnEvaluated(.commitTurn(speakerId: 0))) == [.stopCapture])
        #expect(voiceState.capture == .transcribing && !voiceState.holdingThinkingPause)

        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let en = try #require(NSDictionary(contentsOf: root.appending(path: "Localization/en.lproj/Localizable.strings")) as? [String: String])
        let ru = try #require(NSDictionary(contentsOf: root.appending(path: "Localization/ru.lproj/Localizable.strings")) as? [String: String])
        #expect(Set(en.keys) == Set(ru.keys))
        for key in ["Noise Suppression", "Semantic Turn-Taking", "Duplex Speech-to-Speech"] {
            #expect(en[key] != nil && ru[key] != nil)
        }
    }

    private static func snrDecibels(reference: [Float], estimate: [Float]) -> Double {
        var sigPow = 0.0
        var errPow = 0.0
        for (r, e) in zip(reference, estimate) {
            sigPow += Double(r) * Double(r)
            let diff = Double(e) - Double(r)
            errPow += diff * diff
        }
        return 10.0 * log10(max(1e-12, sigPow) / max(1e-12, errPow))
    }

    private static func assertStrictMonotonic(_ timestamps: [WordTimestamp], maxDuration: Double) {
        var previousEnd = 0.0
        for item in timestamps {
            #expect(item.start >= 0.0)
            #expect(item.start <= item.end)
            #expect(item.start >= previousEnd - 1e-9)
            #expect(item.end <= maxDuration)
            previousEnd = item.end
        }
    }
}
