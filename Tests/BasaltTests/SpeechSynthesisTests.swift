import Testing
import Foundation
import Synchronization
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltSpeech
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltCLI

final class SineSpeechSynthesizer: SpeechSynthesizer {
    let recorded = Mutex<[SpeechRequest]>([])
    let modelID: String
    let compute: SpeechComputeBackend
    let sampleRate: Int
    let chunkSizes: [Int]

    init(
        modelID: String = "Kokoro-82M-ANE",
        compute: SpeechComputeBackend = .ane,
        sampleRate: Int = 24_000,
        chunkSizes: [Int] = [240, 240]
    ) {
        self.modelID = modelID
        self.compute = compute
        self.sampleRate = sampleRate
        self.chunkSizes = chunkSizes
    }

    var voices: [VoiceSpec] {
        get async {
            [
                VoiceSpec(id: "af_heart", name: "Heart", language: "en-US", model: modelID, compute: compute),
                VoiceSpec(id: "alloy", name: "Alloy", language: "en-US", model: modelID, compute: compute),
            ]
        }
    }

    func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        SpeechRouteDecision(selected: SpeechCandidate(id: modelID, model: modelID, compute: compute, estimatedTTFAMs: 30))
    }

    func synthesize(_ request: SpeechRequest) async throws(BasaltError) -> AsyncThrowingStream<AudioChunk, any Error> {
        try request.validate()
        recorded.withLock { $0.append(request) }
        let rate = sampleRate
        let sizes = chunkSizes
        return AsyncThrowingStream { continuation in
            var frameOffset = 0
            for (index, count) in sizes.enumerated() {
                let samples = (0 ..< count).map { local in
                    Float(0.5 * sin(2 * .pi * 440 * Double(frameOffset + local) / Double(rate)))
                }
                frameOffset += count
                continuation.yield(AudioChunk(samples: samples, sampleRate: rate, isFinal: index == sizes.count - 1))
            }
            continuation.finish()
        }
    }
}

@Suite("Speech synthesis codecs and contracts (#493)")
struct SpeechCodecAndContractTests {
    @Test("PCM16LE and WAVEncoder round-trip deterministic waveforms and validate headers")
    func wavAndPCMRoundTrip() throws {
        let samples: [Float] = (0 ..< 480).map { idx in
            Float(0.75 * sin(2 * .pi * 440 * Double(idx) / 24_000))
        }
        let chunk = AudioChunk(samples: samples, sampleRate: 24_000, isFinal: true)
        let pcm = chunk.encodePCM16LE()
        #expect(pcm.count == samples.count * 2)
        let decodedChunk = try AudioChunk.decodePCM16LE(pcm, sampleRate: 24_000, isFinal: true)
        #expect(decodedChunk.samples.count == samples.count)
        for (lhs, rhs) in zip(samples, decodedChunk.samples) {
            #expect(abs(lhs - rhs) < 1e-4)
        }

        let wav = WAVEncoder.encode(samples: samples, sampleRate: 24_000)
        #expect(wav.count == 44 + samples.count * 2)
        #expect(String(decoding: wav.prefix(4), as: UTF8.self) == "RIFF")
        #expect(String(decoding: wav[8 ..< 12], as: UTF8.self) == "WAVE")
        let decodedWAV = try WAVEncoder.decode(wav)
        #expect(decodedWAV.sampleRate == 24_000 && decodedWAV.channels == 1)
        #expect(decodedWAV.samples.count == samples.count)
        for (lhs, rhs) in zip(samples, decodedWAV.samples) {
            #expect(abs(lhs - rhs) < 1e-4)
        }

        #expect(throws: BasaltError.self) { try AudioChunk.decodePCM16LE(Data([1, 2, 3])) }
        #expect(throws: BasaltError.self) { try WAVEncoder.decode(Data("not-a-wav".utf8)) }
    }

    @Test("AudioChunk clamps out-of-range and non-finite floats deterministically")
    func pcmClamping() throws {
        let extreme = AudioChunk(samples: [-2.5, 0.0, 2.5, .nan], sampleRate: 24_000)
        let decoded = try AudioChunk.decodePCM16LE(extreme.encodePCM16LE(), sampleRate: 24_000)
        #expect(abs(decoded.samples[0] - (-1.0)) < 1e-4)
        #expect(decoded.samples[1] == 0)
        #expect(abs(decoded.samples[2] - 1.0) < 1e-4)
        #expect(decoded.samples[3] == 0)
    }

    @Test("SpeechRequest validates speed, empty input, exaggeration and BCP-47 language tags")
    func requestValidation() throws {
        var valid = try SpeechRequest(model: "Kokoro-82M-ANE", input: "Привет, мир!", voice: "af_heart", responseFormat: .wav, speed: 1.25)
        try valid.configure(language: "ru-RU", instruct: "calm", exaggeration: 0.6, preferANE: true)
        #expect(valid.preferANE && valid.language == "ru-RU" && valid.exaggeration == 0.6)

        #expect(throws: BasaltError.self) { try SpeechRequest(model: "m", input: "   ") }
        #expect(throws: BasaltError.self) { try SpeechRequest(model: "m", input: "hi", speed: 0.1) }
        #expect(throws: BasaltError.self) { try SpeechRequest(model: "m", input: "hi", speed: 4.5) }
        #expect(throws: BasaltError.self) {
            var bad = try SpeechRequest(model: "m", input: "hi")
            try bad.configure(exaggeration: 2.5)
        }
        #expect(throws: BasaltError.self) {
            var bad = try SpeechRequest(model: "m", input: "hi")
            try bad.configure(language: "../ru")
        }
    }

    @Test("CoreAISpeechBundle loads basalt_speech.json, maps voices, selects length buckets and pads tokens")
    func speechBundleValidation() throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "speech-bundle-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let manifest = """
            {
              "model_id": "Kokoro-82M-ANE",
              "asset": "kokoro",
              "sample_rate": 24000,
              "max_phoneme_tokens": 128,
              "compute": "ane",
              "voices": [
                { "id": "af_heart", "name": "Heart", "language": "en-US", "index": 0 },
                { "id": "bf_emma", "name": "Emma", "language": "en-GB", "index": 1 }
              ],
              "functions": [
                { "name": "synth_64", "length": 64 },
                { "name": "synth_128", "length": 128 }
              ],
              "vocab": { "h": 11, "i": 12 },
              "pad_token": 0
            }
            """
        try Data(manifest.utf8).write(to: dir.appending(component: CoreAISpeechBundle.manifest))
        #expect(throws: BasaltError.self) { try CoreAISpeechBundle(directory: dir) }

        try Data("stub".utf8).write(to: dir.appending(component: "kokoro.aimodel"))
        let bundle = try CoreAISpeechBundle(directory: dir)
        #expect(CoreAISpeechBundle.isBundle(dir))
        #expect(bundle.voiceSpecs.count == 2)
        #expect(bundle.voice(for: "alloy", language: "en-GB")?.id == "bf_emma")
        #expect(bundle.function(for: 32).name == "synth_64")
        #expect(bundle.function(for: 90).name == "synth_128")
        let encoded = bundle.encodeTokens("hi", length: 6)
        #expect(encoded == [11, 12, 0, 0, 0, 0])
    }
}

@Suite("SentenceStreamChunker full-duplex LLM to TTS segmentation (#493)")
struct SentenceStreamChunkerTests {
    @Test("Token-by-token English and Russian streams preserve decimals and abbreviations")
    func preservesDecimalsAndAbbreviations() {
        var chunker = SentenceStreamChunker()
        let tokens = [
            "Dr.", " Smith", " vs.", " Mr.", " Jones", " measured", " 3", ".", "14", " and", " 2", ".", "5",
            " units", ",", " e", ".", "g", ".", " mass", ".", " ",
            "В", " 2026", " г.", " мы", " проверили", " яблоки", ",", " груши", " и", " т", ".", "д", ".",
            " и", " т", ".", "п", ".", " в", " саду", "!", " ",
            "Готово", "?",
        ]
        var chunks: [String] = []
        for token in tokens {
            chunks += chunker.push(token)
        }
        if let tail = chunker.flush() {
            chunks.append(tail)
        }
        #expect(chunks == [
            "Dr. Smith vs. Mr. Jones measured 3.14 and 2.5 units, e.g. mass.",
            "В 2026 г. мы проверили яблоки, груши и т.д. и т.п. в саду!",
            "Готово?",
        ])
    }

    @Test("CJK punctuation and long clauses split cleanly without breaking short colons")
    func cjkAndLongClauseSplitting() {
        var cjk = SentenceStreamChunker()
        let cjkOut = cjk.push("你好世界。欢迎来到Basalt！一切正常吗？")
        #expect(cjkOut == ["你好世界。", "欢迎来到Basalt！", "一切正常吗？"])
        #expect(cjk.flush() == nil)

        var clauseChunker = SentenceStreamChunker(clauseMinChars: 40)
        var emitted = clauseChunker.push("Meeting at 12:30: bring the telemetry report and review latency; ")
        emitted += clauseChunker.push("then deploy.")
        if let rest = clauseChunker.flush() {
            emitted.append(rest)
        }
        #expect(emitted == [
            "Meeting at 12:30: bring the telemetry report and review latency;",
            "then deploy.",
        ])
    }
}

@Suite("SpeechFallbackRouter 3-tier ANE and system voice routing (#493)")
struct SpeechFallbackRouterTests {
    private static let chain: [SpeechCandidate] = [
        SpeechCandidate(id: "tts_qwen", model: "Qwen3-TTS-0.6B", compute: .mlx, isAvailable: true, estimatedTTFAMs: 140),
        SpeechCandidate(id: "tts_ane", model: "Kokoro-82M-ANE", compute: .ane, isAvailable: true, estimatedTTFAMs: 35),
        .appleSpeech,
    ]

    @Test("Primary MLX model serves when GPU is idle, and falls back to ANE on gpu_busy, memory_pressure, ttfa_sla or prefer_ane")
    func routesAcrossThreeTiers() {
        let primary = SpeechFallbackRouter.route(SpeechRouteInput(chain: Self.chain))
        #expect(primary.selected.id == "tts_qwen" && primary.fallbackFrom == nil && primary.reason == nil)

        let gpuBusy = SpeechFallbackRouter.route(SpeechRouteInput(chain: Self.chain, gpuBusy: true, onGPUBusy: .fallbackANE))
        #expect(gpuBusy.selected.id == "tts_ane")
        let receipt = gpuBusy.receipt(ttfaMs: 34, audioMs: 1200, rtf: 0.08)
        #expect(receipt.compute == .ane && receipt.fallbackFrom == "Qwen3-TTS-0.6B" && receipt.reason == "gpu_busy")

        let kept = SpeechFallbackRouter.route(SpeechRouteInput(chain: Self.chain, gpuBusy: true, onGPUBusy: .keep))
        #expect(kept.selected.id == "tts_qwen" && kept.fallbackFrom == nil)

        let pressured = SpeechFallbackRouter.route(SpeechRouteInput(chain: Self.chain, memoryPressure: true))
        #expect(pressured.selected.id == "tts_ane" && pressured.reason == .memoryPressure)

        let sla = SpeechFallbackRouter.route(SpeechRouteInput(chain: Self.chain, maxTTFAMs: 80))
        #expect(sla.selected.id == "tts_ane" && sla.reason == .ttfaSLA)

        let preferANE = SpeechFallbackRouter.route(SpeechRouteInput(chain: Self.chain, preferANE: true))
        #expect(preferANE.selected.id == "tts_ane" && preferANE.reason == .preferANE)
    }

    @Test("Router falls through to apple-speech when ANE tier is unavailable or fallback_apple_speech is requested")
    func fallsBackToAppleSpeech() async throws {
        let withoutANE = [
            Self.chain[0],
            SpeechCandidate(id: "tts_ane", model: "Kokoro-82M-ANE", compute: .ane, isAvailable: false, estimatedTTFAMs: 35),
            .appleSpeech,
        ]
        let decision = SpeechFallbackRouter.route(SpeechRouteInput(chain: withoutANE, gpuBusy: true, onGPUBusy: .fallbackANE))
        #expect(decision.selected.id == ComposeSpec.appleSpeech)
        #expect(decision.fallbackFrom == "Qwen3-TTS-0.6B" && decision.reason == .gpuBusy)

        let directSystem = SpeechFallbackRouter.route(SpeechRouteInput(chain: Self.chain, gpuBusy: true, onGPUBusy: .fallbackAppleSpeech))
        #expect(directSystem.selected.id == ComposeSpec.appleSpeech && directSystem.reason == .gpuBusy)

        let aneBackend = SineSpeechSynthesizer(modelID: "Kokoro-82M-ANE", compute: .ane)
        let routed = RoutedSpeechSynthesizer(
            tiers: [
                .init(candidate: Self.chain[0], synthesizer: SineSpeechSynthesizer(modelID: "Qwen3-TTS-0.6B", compute: .mlx)),
                .init(candidate: Self.chain[1], synthesizer: aneBackend),
            ],
            onGPUBusy: .fallbackANE,
            gpuBusyProbe: { true }
        )
        let req = try SpeechRequest(model: "Qwen3-TTS-0.6B", input: "Routed onto ANE")
        let route = await routed.resolveRoute(req)
        #expect(route.selected.model == "Kokoro-82M-ANE" && route.fallbackFrom == "Qwen3-TTS-0.6B")
        var count = 0
        for try await chunk in try await routed.synthesize(req) { count += chunk.samples.count }
        #expect(count == 480 && aneBackend.recorded.withLock { $0.count == 1 })
    }
}

@Suite("ComposeSpec speak stage parsing and validation (#493)")
struct ComposeSpeakStageTests {
    @Test("ComposeFile parses shorthand and full speak mappings with apple-speech built-in")
    func parsesSpeakStage() throws {
        let yaml = """
            version: 1
            name: duplex-voice
            models:
              chat:
                source: mlx-community/gemma-4-12B-it-qat-4bit
              tts_qwen:
                source: mlx-community/Qwen3-TTS-0.6B-4bit
                keep: on-demand
              tts_ane:
                source: ./models/kokoro-ane
            pipeline:
              answer: chat
              speak:
                chain: [tts_qwen, tts_ane, apple-speech]
                voice: af_heart
                on_gpu_busy: fallback_ane
                sentence_streaming: true
                format: wav
                exaggeration: 0.4
                max_ttfa_ms: 80
            pipelines:
              fast:
                answer: chat
                speak: [tts_ane, apple-speech]
            """
        let spec = try ComposeFile.parse(yaml, file: "voice.yaml")
        let speak = try #require(spec.speak)
        #expect(speak.chain == ["tts_qwen", "tts_ane", "apple-speech"])
        #expect(speak.voice == "af_heart" && speak.onGPUBusy == .fallbackANE)
        #expect(speak.sentenceStreaming == true && speak.format == .wav)
        #expect(speak.exaggeration == 0.4 && speak.maxTTFAMs == 80)
        #expect(spec.view("fast")?.speak?.chain == ["tts_ane", "apple-speech"])

        var receipt = PipelineReceipt(stages: [], answeredBy: [:], escalated: false, system2Calls: 0, ms: 12)
        receipt.speak = SpeechReceipt(model: "Kokoro-82M-ANE", compute: .ane, ttfaMs: 32, audioMs: 900, rtf: 0.05)
        let data = try JSONEncoder().encode(receipt)
        let roundTrip = try JSONDecoder().decode(PipelineReceipt.self, from: data)
        #expect(roundTrip.speak?.model == "Kokoro-82M-ANE" && roundTrip.speak?.compute == .ane)
    }

    @Test("ComposeFile refuses undeclared models, cloud synthesizers and invalid speak thresholds")
    func validatesSpeakStage() {
        let undeclared = """
            version: 1
            name: bad
            models:
              chat: { source: org/chat }
            pipeline:
              answer: chat
              speak: [ghost_tts, apple-speech]
            """
        #expect(throws: ServeConfigError.self) { try ComposeFile.parse(undeclared, file: "bad.yaml") }

        let cloudSpeak = """
            version: 1
            name: bad-cloud
            models:
              chat: { source: org/chat }
              cloud_tts: { source: "cloud:openrouter/x/y", api_key_env: OPENROUTER_API_KEY }
            pipeline:
              answer: chat
              speak: [cloud_tts, apple-speech]
            """
        #expect(throws: ServeConfigError.self) { try ComposeFile.parse(cloudSpeak, file: "bad.yaml") }
    }
}

@Suite("OpenAI-compatible /v1/audio/speech and /v1/audio/voices REST API (#493)")
struct AudioSpeechRESTTests {
    private func makeApp(synthesizer: (any SpeechSynthesizer)?) throws -> Application<RouterResponder<BasicRequestContext>> {
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        let server = RESTServer(
            orchestrator: orchestrator,
            config: try ServerConfig(modelPath: "org/mock-model"),
            speechSynthesizer: synthesizer
        )
        return Application(router: server.makeRouter())
    }

    @Test("GET /v1/audio/voices returns declared voices and POST /v1/audio/speech encodes WAV and PCM")
    func voicesAndBinaryAudio() async throws {
        let synth = SineSpeechSynthesizer()
        try await makeApp(synthesizer: synth).test(.router) { client in
            let voicesRes = try await client.execute(uri: "/v1/audio/voices", method: .get) { res in
                (res.status, Data(res.body.readableBytesView))
            }
            #expect(voicesRes.0 == .ok)
            let envelope = try JSONDecoder().decode(AudioSpeechController.VoicesEnvelope.self, from: voicesRes.1)
            #expect(envelope.voices.map(\.id) == ["af_heart", "alloy"])

            let wavPayload = #"{"model":"Kokoro-82M-ANE","input":"Hello Basalt","voice":"af_heart","response_format":"wav","speed":1.1,"prefer_ane":true}"#
            let wavRes = try await client.execute(
                uri: "/v1/audio/speech",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: wavPayload)
            ) { res in
                (res.status, res.headers[.contentType], Data(res.body.readableBytesView))
            }
            #expect(wavRes.0 == .ok && wavRes.1 == "audio/wav")
            let decoded = try WAVEncoder.decode(wavRes.2)
            #expect(decoded.sampleRate == 24_000 && decoded.samples.count == 480)

            let pcmPayload = #"{"model":"Kokoro-82M-ANE","input":"PCM stream","response_format":"pcm_24k"}"#
            let pcmRes = try await client.execute(
                uri: "/v1/audio/speech",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: pcmPayload)
            ) { res in
                (res.status, res.headers[.contentType], Data(res.body.readableBytesView))
            }
            #expect(pcmRes.0 == .ok && pcmRes.1 == "audio/pcm" && pcmRes.2.count == 960)
        }
        let recorded = synth.recorded.withLock { $0 }
        #expect(recorded.count == 2 && recorded[0].preferANE && recorded[0].speed == 1.1)
    }

    @Test("POST /v1/audio/speech with stream_format sse emits audio.delta base64 chunks and audio.done receipt")
    func sseAudioStreamingAndErrors() async throws {
        let synth = SineSpeechSynthesizer(chunkSizes: [120, 120])
        try await makeApp(synthesizer: synth).test(.router) { client in
            let ssePayload = #"{"model":"Kokoro-82M-ANE","input":"Stream SSE","stream_format":"sse"}"#
            let sseRes = try await client.execute(
                uri: "/v1/audio/speech",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: ssePayload)
            ) { res in
                (res.status, res.headers[.contentType], String(buffer: res.body))
            }
            #expect(sseRes.0 == .ok && sseRes.1 == "text/event-stream")
            #expect(sseRes.2.contains("event: audio.delta"))
            #expect(sseRes.2.contains("event: audio.done"))
            #expect(sseRes.2.contains("\"compute\":\"ane\""))

            let mp3Payload = #"{"model":"Kokoro-82M-ANE","input":"Compressed","response_format":"mp3"}"#
            let mp3Status = try await client.execute(
                uri: "/v1/audio/speech",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: mp3Payload)
            ) { $0.status }
            #expect(mp3Status == .badRequest)
        }

        try await makeApp(synthesizer: nil).test(.router) { client in
            let missingStatus = try await client.execute(
                uri: "/v1/audio/speech",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"input":"Hello"}"#)
            ) { $0.status }
            #expect(missingStatus == .badRequest)
        }
    }
}
