import Foundation
import Testing
import MLX
import BasaltCore
@testable import BasaltInference

enum TTSRoundTripEval {
    static let englishSentences: [(voice: String, text: String)] = [
        ("af_heart", "The quick brown fox jumps over the lazy dog."),
        ("af_heart", "Artificial intelligence is transforming modern software engineering."),
        ("af_heart", "Please remember to check the weather forecast before leaving home."),
        ("af_heart", "A gentle breeze moved across the quiet mountain lake at sunrise."),
        ("af_heart", "Scientists discovered a new species of frog in the tropical rainforest."),
        ("af_bella", "She walked along the sandy beach listening to the ocean waves."),
        ("af_bella", "Good morning everyone, thank you for joining our meeting today."),
        ("af_bella", "The library closes at nine o clock every Friday evening."),
        ("af_bella", "Fresh coffee and warm bread filled the kitchen with a pleasant aroma."),
        ("af_bella", "Always verify your measurements before publishing benchmark results."),
        ("bf_emma", "London bridges and historic parks attract millions of visitors each year."),
        ("bf_emma", "The train to Edinburgh departs from platform four in ten minutes."),
        ("bf_emma", "Careful planning helps engineering teams deliver reliable software on schedule."),
        ("bf_emma", "Autumn leaves covered the narrow stone path through the garden."),
        ("bf_emma", "Music can quickly change the mood of a crowded room."),
        ("am_adam", "Mountain climbers prepared their ropes and checked the oxygen tanks."),
        ("am_adam", "Clear communication between developers prevents costly integration mistakes."),
        ("am_adam", "The old clock tower chimed twelve times in the middle of the night."),
        ("am_adam", "Solar panels on the roof generated enough power for the entire house."),
        ("am_adam", "Reading books every evening improves both focus and vocabulary."),
    ]

    static let russianSentences: [(stressed: String, plain: String)] = [
        ("Сег+одня на +улице т+ёплая и с+олнечная пог+ода.", "Сегодня на улице тёплая и солнечная погода."),
        ("Иск+усственный интелл+ект б+ыстро мен+яет разраб+отку прогр+амм.", "Искусственный интеллект быстро меняет разработку программ."),
        ("Пож+алуйста, пров+ерьте прогн+оз пог+оды п+еред в+ыходом из д+ома.", "Пожалуйста, проверьте прогноз погоды перед выходом из дома."),
        ("Т+ихий в+етер д+ует над г+орным +озером на рассв+ете.", "Тихий ветер дует над горным озером на рассвете."),
        ("Св+ежий к+офе и т+ёплый хл+еб нап+олнили к+ухню при+ятным аром+атом.", "Свежий кофе и тёплый хлеб наполнили кухню приятным ароматом."),
        ("Всегд+а провер+яйте измер+ения п+еред публик+ацией результ+атов.", "Всегда проверяйте измерения перед публикацией результатов."),
        ("П+оезд в С+анкт-Петерб+ург отправл+яется с четв+ёртой платф+ормы.", "Поезд в Санкт-Петербург отправляется с четвёртой платформы."),
        ("Ос+енние л+истья покр+ыли +узкую к+аменную дор+ожку в сад+у.", "Осенние листья покрыли узкую каменную дорожку в саду."),
        ("Чт+ение кн+иг к+аждый в+ечер улучш+ает вним+ание и п+амять.", "Чтение книг каждый вечер улучшает внимание и память."),
        ("+Ясное общ+ение в ком+анде помог+ает избеж+ать ош+ибок.", "Ясное общение в команде помогает избежать ошибок."),
        ("Уч+ёные откр+ыли н+овый в+ид б+абочек в троп+ическом лес+у.", "Учёные открыли новый вид бабочек в тропическом лесу."),
        ("Д+оброе +утро, спас+ибо что подключ+ились к н+ашей встр+ече.", "Доброе утро, спасибо что подключились к нашей встрече."),
        ("Библиот+ека закрыв+ается в д+евять час+ов в+ечера по п+ятницам.", "Библиотека закрывается в девять часов вечера по пятницам."),
        ("М+узыка спос+обна б+ыстро измен+ить настро+ение в к+омнате.", "Музыка способна быстро изменить настроение в комнате."),
        ("Альпин+исты пров+ерили снаряж+ение п+еред в+ыходом на маршр+ут.", "Альпинисты проверили снаряжение перед выходом на маршрут."),
        ("С+олнечные пан+ели на кр+ыше обесп+ечили д+ом электр+ичеством.", "Солнечные панели на крыше обеспечили дом электричеством."),
        ("Ст+арые час+ы на б+ашне проб+или р+овно п+олночь.", "Старые часы на башне пробили ровно полночь."),
        ("Тщ+ательное план+ирование помог+ает сдав+ать про+екты в+овремя.", "Тщательное планирование помогает сдавать проекты вовремя."),
        ("На берег+у м+оря шум+ели спок+ойные веч+ерние в+олны.", "На берегу моря шумели спокойные вечерние волны."),
        ("Н+овый рел+из включ+ает подд+ержку с+интеза р+ечи на ре+альных вес+ах.", "Новый релиз включает поддержку синтеза речи на реальных весах."),
    ]

    static func resampleLinear(_ samples: [Float], from srcRate: Int, to dstRate: Int) -> [Float] {
        guard !samples.isEmpty, srcRate > 0, dstRate > 0 else { return [] }
        let ratio = Double(srcRate) / Double(dstRate)
        let outCount = max(1, Int((Double(samples.count) / ratio).rounded()))
        var out = [Float](repeating: 0, count: outCount)
        for i in 0 ..< outCount {
            let pos = Double(i) * ratio
            let idx = min(samples.count - 1, Int(pos))
            let next = min(samples.count - 1, idx + 1)
            let frac = Float(pos - Double(idx))
            out[i] = samples[idx] * (1 - frac) + samples[next] * frac
        }
        return out
    }

    static func normalizeWords(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }

    static func wordEdits(reference: String, hypothesis: String) -> (edits: Int, refWords: Int) {
        let r = normalizeWords(reference)
        let h = normalizeWords(hypothesis)
        var dp = Array(repeating: Array(repeating: 0, count: h.count + 1), count: r.count + 1)
        for i in 0 ... r.count { dp[i][0] = i }
        for j in 0 ... h.count { dp[0][j] = j }
        for i in 1 ... max(1, r.count) where i <= r.count {
            for j in 1 ... max(1, h.count) where j <= h.count {
                let cost = r[i - 1] == h[j - 1] ? 0 : 1
                dp[i][j] = min(dp[i - 1][j] + 1, min(dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost))
            }
        }
        return (dp[r.count][h.count], r.count)
    }
}

@Suite("Kokoro-82M real-weight strict loading and synthesis (#821)")
struct KokoroRealWeightsTests {
    @Test("KokoroMLXSynthesizer(directory:) refuses random initialization when weights are missing")
    func refusesMissingWeights() throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "kokoro-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: BasaltError.self) {
            _ = try KokoroMLXSynthesizer(directory: dir, modelID: "Kokoro-82M")
        }
    }

    @Test("KokoroWeightLoader.loadStrict rejects partial checkpoint leaving randomly initialized parameters")
    func refusesPartialCheckpoint() throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "kokoro-partial-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appending(component: "kokoro-v1_0.safetensors")
        try MLX.save(arrays: ["bert_encoder.weight": MLXArray.zeros([512, 768], dtype: .float32)], url: file)
        let model = KokoroStyleTTS2Model()
        #expect(throws: BasaltError.self) {
            _ = try KokoroWeightLoader.loadStrict(from: file, into: model)
        }
    }

    @Test("KokoroPhonemizer.tokenize wraps IPA tokens with BOS/EOS zeros and rejects >510 tokens")
    func tokenizerWrapsWithZerosAndEnforcesLimit() throws {
        let vocab = ["h": 50, "ə": 83, "l": 54, "o": 57]
        let tokens = try KokoroPhonemizer.tokenize("həlo", vocab: vocab)
        #expect(tokens == [0, 50, 83, 54, 57, 0])
        let tooLong = String(repeating: "h", count: KokoroPhonemizer.maxPhonemeTokens + 1)
        #expect(throws: BasaltError.self) {
            _ = try KokoroPhonemizer.tokenize(tooLong, vocab: vocab)
        }
    }

    @Test("Kokoro-82M real checkpoint loads all 548 tensors strictly and synthesizes 24 kHz speech",
          .enabled(if: ProcessInfo.processInfo.environment["BASALT_RUN_REAL_MODELS"] == "1"))
    func realCheckpointStrictLoadAndSynthesis() async throws {
        let path = ProcessInfo.processInfo.environment["BASALT_KOKORO_DIR"]
            ?? "/Users/alexeiboklag/.cache/huggingface/hub/models--mlx-community--Kokoro-82M-bf16/snapshots/a71e4d38b236d968966a2002c4c895dbd12b1c3c"
        let synth = try KokoroMLXSynthesizer(directory: URL(fileURLWithPath: path), modelID: "mlx-community/Kokoro-82M-bf16")
        let report = try #require(synth.strictLoadReport)
        #expect(report.checkpointKeyCount == 548)
        #expect(report.modelParameterCount > 300)

        let whisperPath = ProcessInfo.processInfo.environment["BASALT_WHISPER_DIR"]
            ?? "/Users/alexeiboklag/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
        let asr = MLXInferenceEngine(.standard)
        try await asr.load(modelPath: whisperPath)
        defer { Task { await asr.unload() } }

        var totalAudioSec = 0.0
        var totalSynthSec = 0.0
        var totalEdits = 0
        var totalRefWords = 0
        Memory.peakMemory = 0

        for (index, item) in TTSRoundTripEval.englishSentences.enumerated() {
            let req = try SpeechRequest(model: "mlx-community/Kokoro-82M-bf16", input: item.text, voice: item.voice)
            let started = ContinuousClock.now
            var samples: [Float] = []
            for try await chunk in try await synth.synthesize(req) { samples.append(contentsOf: chunk.samples) }
            totalSynthSec += started.duration(to: .now).inSeconds
            let audioSec = Double(samples.count) / 24_000.0
            totalAudioSec += audioSec
            #expect(audioSec > 0.5 && TTSWaveformMetrics.allFinite(samples))
            let clip = try AudioClip(samples: TTSRoundTripEval.resampleLinear(samples, from: 24_000, to: 16_000))
            let hyp = try await asr.transcribe(try TranscriptionRequest(audio: clip, language: "en", temperature: 0)).text
            let (edits, words) = TTSRoundTripEval.wordEdits(reference: item.text, hypothesis: hyp)
            totalEdits += edits
            totalRefWords += words
            print("KOKORO_UTT[\(index)] (\(item.voice)): edits=\(edits)/\(words)")
        }

        let wer = Double(totalEdits) / Double(max(1, totalRefWords))
        let rtf = totalSynthSec / max(0.001, totalAudioSec)
        let peakMB = Double(Memory.peakMemory) / 1_048_576.0
        print("KOKORO_SUMMARY: RTF=\(String(format: "%.3f", rtf)) WER=\(String(format: "%.2f%%", wer * 100)) (\(totalEdits)/\(totalRefWords)) peakMB=\(String(format: "%.1f", peakMB))")
        #expect(wer < 0.25)
    }
}
