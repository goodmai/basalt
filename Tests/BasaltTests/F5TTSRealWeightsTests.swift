import Foundation
import Testing
import MLX
import BasaltCore
@testable import BasaltInference

@Suite("F5-TTS + Vocos-24kHz real-weight strict loading and synthesis (#821)")
struct F5TTSRealWeightsTests {
    @Test("F5TTSWeightLoader.loadStrict refuses random initialization when weights are missing")
    func refusesMissingWeights() throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "f5-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = F5DiTModel()
        #expect(throws: BasaltError.self) {
            _ = try F5TTSWeightLoader.loadStrict(from: dir, into: model)
        }
    }

    @Test("VocosModel.loadStrict refuses random initialization when weights are missing")
    func vocosRefusesMissingWeights() throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "vocos-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: BasaltError.self) {
            _ = try VocosModel.loadStrict(from: dir)
        }
    }

    @Test("F5DiTModel real checkpoint forward pass produces finite outputs",
          .enabled(if: ProcessInfo.processInfo.environment["BASALT_RUN_REAL_MODELS"] == "1"))
    func ditForwardFiniteCheck() async throws {
        let f5Path = ProcessInfo.processInfo.environment["BASALT_F5_DIR"]
            ?? "/Users/alexeiboklag/.cache/huggingface/hub/models--lucasnewman--f5-tts-mlx/snapshots/2d719cadec8fd3887c8599475a5f65924d523654"
        let model = F5DiTModel()
        _ = try F5TTSWeightLoader.loadStrict(from: URL(fileURLWithPath: f5Path), into: model)
        let x = MLXArray.ones([1, 8, 100], dtype: .float32) * Float(0.5)
        let cond = MLXArray.zeros([1, 8, 100], dtype: .float32)
        let text = MLXArray([Int32(39), 325, 615]).reshaped(1, 3)
        let t = MLXArray([Float(0.5)])
        let out = model(x: x, cond: cond, textIDs: text, time: t, dropAudioCond: false, dropText: false)
        eval(out)
        let first = out.reshaped(-1)[0 ..< 4].asArray(Float.self)
        #expect(TTSWaveformMetrics.allFinite(first))
    }

    @Test("F5-TTS v1 real checkpoint loads all 364 DiT tensors + 83 Vocos tensors strictly and synthesizes 20 EN sentences",
          .enabled(if: ProcessInfo.processInfo.environment["BASALT_RUN_REAL_MODELS"] == "1"))
    func realCheckpointStrictLoadAndRoundTripWER() async throws {
        let env = ProcessInfo.processInfo.environment
        let f5Path = env["BASALT_F5_DIR"] ?? "/Users/alexeiboklag/.cache/huggingface/hub/models--lucasnewman--f5-tts-mlx/snapshots/2d719cadec8fd3887c8599475a5f65924d523654"
        let vocosPath = env["BASALT_VOCOS_DIR"] ?? "/Users/alexeiboklag/.cache/huggingface/hub/models--lucasnewman--vocos-mel-24khz/snapshots/ee89d00fac94b61e1235e0869f3e8d49711b7484"
        let kokoroPath = env["BASALT_KOKORO_DIR"] ?? "/Users/alexeiboklag/.cache/huggingface/hub/models--mlx-community--Kokoro-82M-bf16/snapshots/a71e4d38b236d968966a2002c4c895dbd12b1c3c"
        let whisperPath = env["BASALT_WHISPER_DIR"] ?? "/Users/alexeiboklag/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"

        let kokoro = try KokoroMLXSynthesizer(directory: URL(fileURLWithPath: kokoroPath), modelID: "Kokoro-82M")
        let refText = "Hello everyone, this is a clear reference voice recording."
        var refAudio: [Float] = []
        for try await chunk in try await kokoro.synthesize(try SpeechRequest(model: "Kokoro-82M", input: refText, voice: "af_heart")) {
            refAudio.append(contentsOf: chunk.samples)
        }
        await kokoro.unload()

        let f5 = try F5TTSSynthesizer(modelDirectory: URL(fileURLWithPath: f5Path), vocosDirectory: URL(fileURLWithPath: vocosPath))
        #expect(f5.strictLoadReport.checkpointKeyCount == 364 && f5.strictLoadReport.modelParameterCount == 364)
        let items = TTSRoundTripEval.englishSentences.map { ($0.text, $0.text) }
        let config = F5EvalConfig(refAudio: refAudio, refText: refText, language: "en", items: items)
        let wer = try await Self.evaluateF5(f5: f5, whisperPath: whisperPath, config: config)
        #expect(wer < 0.35)
    }

    @Test("F5-TTS v1 Russian checkpoint synthesizes 20 Russian sentences with round-trip Whisper RU WER (#821)",
          .enabled(if: ProcessInfo.processInfo.environment["BASALT_RUN_REAL_MODELS"] == "1"))
    func russianRoundTripWER() async throws {
        let env = ProcessInfo.processInfo.environment
        let f5Path = env["BASALT_F5_RU_DIR"] ?? "/Users/alexeiboklag/.cache/huggingface/hub/models--Misha24-10--F5-TTS_RUSSIAN/snapshots/ea166adeae4c80ec5ee423a671e2bdb83906cf84/F5TTS_v1_Base_v2"
        let vocosPath = env["BASALT_VOCOS_DIR"] ?? "/Users/alexeiboklag/.cache/huggingface/hub/models--lucasnewman--vocos-mel-24khz/snapshots/ee89d00fac94b61e1235e0869f3e8d49711b7484"
        let whisperPath = env["BASALT_WHISPER_DIR"] ?? "/Users/alexeiboklag/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
        let refWav = try WAVEncoder.decode(try Data(contentsOf: URL(fileURLWithPath: env["BASALT_RU_REF_WAV"] ?? "/tmp/ref_ru_clean.wav")))
        let refText = "Здр+авствуйте, +это ч+истая з+апись этал+онного р+усского г+олоса."

        let f5 = try F5TTSSynthesizer(modelDirectory: URL(fileURLWithPath: f5Path), vocosDirectory: URL(fileURLWithPath: vocosPath))
        #expect(f5.strictLoadReport.checkpointKeyCount == 364 && f5.strictLoadReport.modelParameterCount == 364)
        let items = TTSRoundTripEval.russianSentences.map { ($0.stressed, $0.plain) }
        let config = F5EvalConfig(refAudio: refWav.samples, refText: refText, language: "ru", items: items)
        let wer = try await Self.evaluateF5(f5: f5, whisperPath: whisperPath, config: config)
        #expect(wer < 0.20)
    }

    private struct F5EvalConfig {
        let refAudio: [Float]
        let refText: String
        let language: String
        let items: [(String, String)]
    }

    private static func evaluateF5(f5: F5TTSSynthesizer, whisperPath: String, config: F5EvalConfig) async throws -> Double {
        let asr = MLXInferenceEngine(.standard)
        try await asr.load(modelPath: whisperPath)
        defer { Task { await asr.unload() } }

        var totalAudioSec = 0.0
        var totalSynthSec = 0.0
        var totalEdits = 0
        var totalRefWords = 0
        Memory.peakMemory = 0

        for (index, item) in config.items.enumerated() {
            let started = ContinuousClock.now
            let samples = await f5.synthesize(
                text: item.0,
                referenceAudio: config.refAudio,
                referenceText: config.refText,
                options: F5SynthesisOptions(steps: 8, cfgStrength: 2.0, seed: UInt64(42 + index))
            )
            totalSynthSec += started.duration(to: .now).inSeconds
            let audioSec = Double(samples.count) / 24_000.0
            totalAudioSec += audioSec
            #expect(audioSec > 0.5 && TTSWaveformMetrics.allFinite(samples))
            let clip = try AudioClip(samples: TTSRoundTripEval.resampleLinear(samples, from: 24_000, to: 16_000))
            let hyp = try await asr.transcribe(try TranscriptionRequest(audio: clip, language: config.language, temperature: 0)).text
            let (edits, words) = TTSRoundTripEval.wordEdits(reference: item.1, hypothesis: hyp)
            totalEdits += edits
            totalRefWords += words
            print("F5_\(config.language.uppercased())[\(index)]: edits=\(edits)/\(words)")
        }

        let wer = Double(totalEdits) / Double(max(1, totalRefWords))
        let rtf = totalSynthSec / max(0.001, totalAudioSec)
        let peakMB = Double(Memory.peakMemory) / 1_048_576.0
        print("F5_\(config.language.uppercased())_SUMMARY: RTF=\(String(format: "%.3f", rtf)) WER=\(String(format: "%.2f%%", wer * 100)) (\(totalEdits)/\(totalRefWords)) peakMB=\(String(format: "%.1f", peakMB))")
        return wer
    }
}
