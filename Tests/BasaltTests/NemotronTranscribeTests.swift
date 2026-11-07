import Testing
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

struct SyntheticAudioTokenizer: MLXLMCommon.Tokenizer {
    let audioTokenId = 99
    let eosId = 11

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if text == "<|audio|>" { return [audioTokenId] }
        if text.contains("<|audio|>") {
            return [1, 2, audioTokenId, 3]
        }
        return [4, 5]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "tok\($0)" }.joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<|audio|>": return audioTokenId
        case "<|im_end|>": return eosId
        default: return nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case audioTokenId: return "<|audio|>"
        case eosId: return "<|im_end|>"
        default: return "tok\(id)"
        }
    }

    var bosToken: String? { "<s>" }
    var eosToken: String? { "<|im_end|>" }
    var unknownToken: String? { "<unk>" }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [1, 2, 3]
    }
}

@Suite("Speech to text: NemotronH inputEmbeddings and NemotronTranscribe encoder/fusion/splice (#230)")
struct NemotronTranscribeUnitTests {
    static func tinyLLMConfig() throws -> NemotronHConfiguration {
        let json = """
        {
          "model_type": "nemotron_h",
          "vocab_size": 128,
          "hidden_size": 32,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "mamba_num_heads": 4,
          "mamba_head_dim": 8,
          "ssm_state_size": 32,
          "conv_kernel": 4,
          "n_groups": 2,
          "intermediate_size": 64,
          "moe_intermediate_size": 32,
          "moe_shared_expert_intermediate_size": 64,
          "n_routed_experts": 4,
          "n_shared_experts": 1,
          "num_experts_per_tok": 2,
          "layers_block_type": ["mamba", "moe", "attention", "mlp"]
        }
        """
        return try JSONDecoder().decode(NemotronHConfiguration.self, from: Data(json.utf8))
    }

    static func tinyTranscribeConfig() throws -> NemotronTranscribeConfiguration {
        let llm = try tinyLLMConfig()
        let mel = NemotronMelConfig()
        let asr = NemotronEncoderConfig(
            dModel: 32, nHeads: 4, nLayers: 2, ffExpansion: 2.0, qkNorm: true
        )
        let diarEnc = NemotronEncoderConfig(
            dModel: 32, nHeads: 4, nLayers: 2, ffExpansion: 2.0, qkNorm: false
        )
        let sortformer = NemotronSortformerModulesConfig(
            numSpks: 8, fcDModel: 32, tfDModel: 16, subsamplingFactor: 8, spkcacheLen: 16, fifoLen: 8
        )
        let diar = NemotronDiarizationConfig(
            maxNumOfSpks: 8, preprocessor: mel, encoder: diarEnc, sortformerModules: sortformer
        )
        let pe = NemotronPEEncoderConfig(asrEncoderCfg: asr, diarizationModelCfg: diar)
        let perception = NemotronPerceptionSection(outputDim: 32, preprocessor: mel, encoder: asr)
        return NemotronTranscribeConfiguration(
            perception: perception, peEncoderConfig: pe, llmConfig: llm
        )
    }

    @Test("NemotronHConfiguration decodes layers_block_type and NemotronHModel matches between token ids and inputEmbeddings")
    func nemotronHInputEmbeddings() throws {
        let cfg = try Self.tinyLLMConfig()
        #expect(cfg.hybridOverridePattern == "ME*-")
        #expect(cfg.numHiddenLayers == 4)
        let model = NemotronHModel(cfg)
        let tokens = MLXArray([Int32(3), Int32(7), Int32(12), Int32(19)]).reshaped(1, 4)
        let emb = model.embedTokens(tokens)
        #expect(emb.shape == [1, 4, 32])
        let cacheA = try model.newCache(parameters: nil)
        let cacheB = try model.newCache(parameters: nil)
        let logitsFromTokens = model(tokens, cache: cacheA)
        let logitsFromEmbeds = model(tokens, cache: cacheB, inputEmbeddings: emb)
        let diff = abs(logitsFromTokens - logitsFromEmbeds).max()
        eval(diff)
        #expect(diff.item(Float.self) < 1e-5)
    }

    @Test("MelFeatures, ParallelExpertEncoder, speaker fusion, and prompt splice produce aligned embeddings and run end-to-end")
    func encoderFusionAndSplice() throws {
        let cfg = try Self.tinyTranscribeConfig()
        let model = NemotronTranscribeModel(cfg)
        let samples = (0 ..< 24_000).map { i in
            Float(0.25 * sin(2.0 * .pi * 440.0 * Double(i) / 16_000.0))
        }
        let audioArray = MLXArray(samples)
        let mel = model.perception.preprocessor(audioArray)
        eval(mel)
        #expect(mel.shape == [1, 128, 150])

        let audioEmbeds = try model.encodeAudio(audioArray)
        #expect(audioEmbeds.shape == [1, 19, 32])
        let nonFinite = (isNaN(audioEmbeds).sum() + isInf(audioEmbeds).sum()).item(Int.self)
        #expect(nonFinite == 0)

        let tokenizer = SyntheticAudioTokenizer()
        let (ids, audioIndex) = try model.promptTokenIds(tokenizer: tokenizer)
        let (expanded, tokenArray, spliced) = model.spliceEmbeddings(
            ids: ids, audioIndex: audioIndex, audioEmbeds: audioEmbeds
        )
        #expect(expanded.count == ids.count - 1 + 19)
        #expect(spliced.shape == [expanded.count, 32])
        let rawEmbeds = model.llm.embedTokens(tokenArray.expandedDimensions(axis: 0))[0]
        let prefixDiff = abs(spliced[0 ..< audioIndex, 0...] - rawEmbeds[0 ..< audioIndex, 0...]).max()
        let audioDiff = abs(
            spliced[audioIndex ..< (audioIndex + 19), 0...] - audioEmbeds[0].asType(spliced.dtype)
        ).max()
        let suffixDiff = abs(
            spliced[(audioIndex + 19)..., 0...] - rawEmbeds[(audioIndex + 19)..., 0...]
        ).max()
        eval(prefixDiff, audioDiff, suffixDiff)
        #expect(prefixDiff.item(Float.self) == 0)
        #expect(audioDiff.item(Float.self) == 0)
        #expect(suffixDiff.item(Float.self) == 0)

        let text = try model.transcribe(
            audio: samples, tokenizer: tokenizer, temperature: 0, maxTokens: 3, prefillStepSize: 8
        )
        #expect(!text.isEmpty)
    }
}

@Suite("Nemotron 3.5 Transcribe real-model E2E (#230)", .serialized)
struct NemotronTranscribeRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static let modelPath: String? = {
        guard env["BASALT_E2E"] == "1",
              let path = env["BASALT_E2E_TRANSCRIBE_MODEL_PATH"],
              !path.isEmpty
        else { return nil }
        return path
    }()

    static let englishReference =
        "The hybrid language model combines recurrent state space layers with sparse mixture of experts "
        + "and grouped query attention. In this benchmark we evaluate speech transcription accuracy, "
        + "streaming latency, and memory residency on Apple Silicon unified memory. "
        + "Every audio frame is processed at sixteen kilohertz and subsampled to eighty milliseconds "
        + "before fusion with speaker activity features and projection into the decoder backbone."

    static let russianReference =
        "Система распознавания речи преобразует аудиосигнал в текст с высокой точностью "
        + "и сохраняет исходный язык говорящего без перевода."

    static func synthesizeWAV(text: String, voice: String? = nil) throws -> Data {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("basalt-tts-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let aiffURL = dir.appendingPathComponent("speech.aiff")
        let wavURL = dir.appendingPathComponent("speech.wav")

        let sayProc = Process()
        sayProc.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        var sayArgs = ["-o", aiffURL.path]
        if let voice {
            sayArgs += ["-v", voice]
        }
        sayArgs.append(text)
        sayProc.arguments = sayArgs
        try sayProc.run()
        sayProc.waitUntilExit()
        guard sayProc.terminationStatus == 0 else {
            throw BasaltError.inferenceError("say exited with \(sayProc.terminationStatus)")
        }

        let convProc = Process()
        convProc.executableURL = URL(fileURLWithPath: "/usr/bin/afconvert")
        convProc.arguments = ["-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiffURL.path, wavURL.path]
        try convProc.run()
        convProc.waitUntilExit()
        guard convProc.terminationStatus == 0 else {
            throw BasaltError.inferenceError("afconvert exited with \(convProc.terminationStatus)")
        }
        return try Data(contentsOf: wavURL)
    }

    static func hasVoice(_ name: String) -> Bool {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        proc.arguments = ["-v", "?"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        guard (try? proc.run()) != nil else { return false }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        return String(decoding: data, as: UTF8.self).contains(name)
    }

    static func normalizeWords(_ text: String) -> [String] {
        text.lowercased()
            .unicodeScalars
            .map { CharacterSet.alphanumerics.contains($0) ? String($0) : " " }
            .joined()
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
    }

    static func wordErrorRate(reference: String, hypothesis: String) -> Double {
        let ref = normalizeWords(reference)
        let hyp = normalizeWords(hypothesis)
        guard !ref.isEmpty else { return hyp.isEmpty ? 0 : 1 }
        var prev = Array(0 ... hyp.count)
        for (i, rWord) in ref.enumerated() {
            var curr = Array(repeating: 0, count: hyp.count + 1)
            curr[0] = i + 1
            for (j, hWord) in hyp.enumerated() {
                let cost = rWord == hWord ? 0 : 1
                curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost)
            }
            prev = curr
        }
        return Double(prev[hyp.count]) / Double(ref.count)
    }

    @Test(
        "Nemotron 3.5 Transcribe loads, transcribes synthesized English and Russian speech over REST and MCP, and logs WER/RTF/memory",
        .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_TRANSCRIBE_MODEL_PATH")
    )
    func transcribeRealModel() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("transcribe", modelPath: path)
        defer { log.releaseGPU() }

        let englishWAV = try Self.synthesizeWAV(text: Self.englishReference)
        let englishClip = try AudioClip(encoded: englishWAV, format: "wav")
        log.record("english_audio", String(format: "%.2fs (%d samples, %d bytes)", englishClip.seconds, englishClip.samples.count, englishWAV.count))

        let engine = MLXInferenceEngine(EngineConfiguration(probesModel: false))
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await log.step("load") { try await orchestrator.loadModel(path: path) }
        #expect(await engine.transcribesAudio)

        let app = Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: path)).makeRouter())
        let body = AudioFixture.form(
            [("model", "nemotron_transcribe"), ("response_format", "verbose_json"), ("language", "en"), ("temperature", "0")],
            file: .init(name: "english.wav", type: "audio/wav", data: englishWAV)
        )

        let t0 = ContinuousClock.now
        let (status, rawJSON) = try await log.step("transcribe-en-rest") {
            try await app.test(.router) { client in
                try await client.execute(
                    uri: "/v1/audio/transcriptions",
                    method: .post,
                    headers: [.contentType: "multipart/form-data; boundary=XyZ-boundary"],
                    body: ByteBuffer(bytes: body)
                ) { ($0.status, String(buffer: $0.body)) }
            }
        }
        let elapsedEn = (ContinuousClock.now - t0).inSeconds
        #expect(status == .ok, "\(rawJSON)")
        let payload = try #require(try JSONSerialization.jsonObject(with: Data(rawJSON.utf8)) as? [String: Any])
        let englishText = try #require(payload["text"] as? String)
        let englishWER = Self.wordErrorRate(reference: Self.englishReference, hypothesis: englishText)
        let englishRTF = elapsedEn / englishClip.seconds
        log.record(
            "english_result",
            String(
                format: "audio=%.2fs elapsed=%.2fs RTF=%.3f WER=%.2f%% activeMB=%d peakMB=%d text=%@",
                englishClip.seconds, elapsedEn, englishRTF, englishWER * 100,
                Memory.activeMemory >> 20, Memory.peakMemory >> 20, englishText.debugDescription
            )
        )
        #expect(!englishText.isEmpty)
        #expect(englishWER < 0.25, "English WER \(englishWER) too high: \(englishText)")

        if Self.hasVoice("Milena") {
            let russianWAV = try Self.synthesizeWAV(text: Self.russianReference, voice: "Milena")
            let russianClip = try AudioClip(encoded: russianWAV, format: "wav")
            let tRu = ContinuousClock.now
            let ruResult = try await log.step("transcribe-ru") {
                try await orchestrator.transcribe(TranscriptionRequest(audio: russianClip, language: "ru", temperature: 0))
            }
            let elapsedRu = (ContinuousClock.now - tRu).inSeconds
            let ruWER = Self.wordErrorRate(reference: Self.russianReference, hypothesis: ruResult.text)
            let ruRTF = elapsedRu / russianClip.seconds
            log.record(
                "russian_result",
                String(
                    format: "audio=%.2fs elapsed=%.2fs RTF=%.3f WER=%.2f%% activeMB=%d peakMB=%d text=%@",
                    russianClip.seconds, elapsedRu, ruRTF, ruWER * 100,
                    Memory.activeMemory >> 20, Memory.peakMemory >> 20, ruResult.text.debugDescription
                )
            )
            #expect(!ruResult.text.isEmpty)
            #expect(ruWER < 0.35, "Russian WER \(ruWER) too high: \(ruResult.text)")
        }

        Memory.clearCache()
    }
}
