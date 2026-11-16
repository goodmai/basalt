import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltSpeech
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltCLI
@testable import BasaltCloud

@Suite("TTSModelFamily detection, CosyVoice3 zero-legacy guard, and tag/prompt parsers (#493)")
struct TTSDescriptorAndGuardTests {
    @Test("TTSModelFamily detects Kokoro-82M, Qwen3-TTS, Chatterbox, Fun-CosyVoice3, kokoro_ane.json and apple-speech")
    func detectsSupportedFamilies() throws {
        #expect(TTSModelFamily.detect(identifier: "hexgrad/Kokoro-82M") == .kokoro82M)
        #expect(TTSModelFamily.detect(identifier: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice") == .qwen3TTS)
        #expect(TTSModelFamily.detect(identifier: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign") == .qwen3TTS)
        #expect(TTSModelFamily.detect(identifier: "ResembleAI/chatterbox-multilingual") == .chatterbox)
        #expect(TTSModelFamily.detect(identifier: "FunAudioLLM/Fun-CosyVoice3-0.5B-2512") == .cosyVoice3)
        #expect(TTSModelFamily.detect(identifier: "mlx-community/Fun-CosyVoice3-0.5B-2512-4bit") == .cosyVoice3)
        #expect(TTSModelFamily.detect(identifier: "apple-speech") == .appleSpeech)

        let dir = FileManager.default.temporaryDirectory.appending(path: "kokoro-ane-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_id":"Kokoro-82M-ANE","synthetic":true}"#.utf8)
            .write(to: dir.appending(component: CoreAISpeechBundle.kokoroANEManifest))
        #expect(try TTSModelFamily.validateSupported(identifier: "local-tts", directory: dir) == .kokoro82M)
        #expect(TTSVoiceCatalog.kokoroVoiceIDs.count == 54)
        #expect(TTSVoiceCatalog.qwen3Languages.contains("en") && TTSVoiceCatalog.qwen3Languages.contains("ru"))
        #expect(TTSVoiceCatalog.chatterboxLanguages.count == 23 && TTSVoiceCatalog.chatterboxLanguages.contains("ru"))
    }

    @Test("Legacy CosyVoice v1 and v2 repos, configs and manifests are explicitly rejected")
    func rejectsLegacyCosyVoice() async throws {
        #expect(throws: BasaltError.self) {
            _ = try TTSModelFamily.validateSupported(identifier: "FunAudioLLM/CosyVoice-300M")
        }
        #expect(throws: BasaltError.self) {
            _ = try TTSModelFamily.validateSupported(identifier: "FunAudioLLM/CosyVoice2-0.5B")
        }
        #expect(throws: BasaltError.self) {
            _ = try TTSModelFamily.validateSupported(identifier: "custom-tts", config: ["model_type": "cosyvoice2"])
        }
        #expect(throws: BasaltError.self) {
            _ = try TTSModelFamily.validateSupported(identifier: "custom-tts", config: ["flow_estimator": "unet"])
        }
        #expect(throws: BasaltError.self) {
            _ = try TTSModelFamily.validateSupported(identifier: "custom-tts", config: ["speech_tokenizer": "speech_tokenizer_v1.onnx"])
        }

        let legacyDir = FileManager.default.temporaryDirectory.appending(path: "legacy-cosy2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: legacyDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: legacyDir) }
        try Data("legacy: true\n".utf8).write(to: legacyDir.appending(component: "cosyvoice2.yaml"))
        await #expect(throws: BasaltError.self) {
            _ = try await SpeechSynthesizerLoader.load(identifier: "local-model", directory: legacyDir)
        }
    }

    @Test("SpeechRequest validates reference_audio extensions and rejects path traversal")
    func validatesReferenceAudio() throws {
        var req = try SpeechRequest(model: "Qwen3-TTS-0.6B", input: "Clone this voice.")
        try req.configure(language: "ru", instruct: "Calm tone", referenceAudio: "/tmp/speaker_sample.wav")
        let encoded = try JSONEncoder().encode(req)
        let decoded = try JSONDecoder().decode(SpeechRequest.self, from: encoded)
        #expect(decoded.referenceAudio == "/tmp/speaker_sample.wav")

        for invalid in ["../secret.wav", "/tmp/../etc/passwd.wav", "..\\escape.wav", "https://example.com/ref.wav", "/tmp/ref.mp3", "   "] {
            #expect(throws: BasaltError.self) {
                var bad = try SpeechRequest(model: "Qwen3-TTS-0.6B", input: "Test")
                try bad.configure(referenceAudio: invalid)
            }
        }
    }

    @Test("ChatterboxTagParser and CosyVoice3PromptBuilder preserve prosody tags and unified endofprompt contract")
    func parsesTagsAndCosyVoice3Prompts() {
        let segments = ChatterboxTagParser.parse("Hello [laugh] note [1] and [chuckle] done [cough]!")
        #expect(segments == [
            .text("Hello "),
            .tag(.laugh),
            .text(" note [1] and "),
            .tag(.chuckle),
            .text(" done "),
            .tag(.cough),
            .text("!"),
        ])

        let instructed = CosyVoice3PromptBuilder.build(
            input: "Привет [breath] <strong>мир</strong> [AA1 R]!",
            instruct: "Speak with warm Russian intonation."
        )
        #expect(instructed == "You are a helpful assistant. Speak with warm Russian intonation.<|endofprompt|>Привет [breath] <strong>мир</strong> [AA1 R]!")
        let zeroShot = CosyVoice3PromptBuilder.build(input: "Target speech.", promptTranscript: "Reference transcript. ")
        #expect(zeroShot == "You are a helpful assistant.<|endofprompt|>Reference transcript. Target speech.")
        let spans = CosyVoice3PromptBuilder.extractInlineControlSpans("Привет [breath] <strong>мир</strong> [AA1 R]!")
        #expect(spans == ["[breath]", "<strong>", "</strong>", "[AA1 R]"])
    }
}

@Suite("Tiny-fixture #228 MLX synthesis and parity for Kokoro-82M, Qwen3-TTS, Chatterbox and CosyVoice3 (#493)", .serialized)
struct NeuralTTSBackendParityTests {
    private static func writeReferenceWAV(freq: Double, directory: URL, name: String) throws -> String {
        let samples = (0 ..< 4_800).map { idx in
            Float(0.6 * sin(2.0 * .pi * freq * Double(idx) / 24_000.0))
        }
        let url = directory.appending(component: name)
        try WAVEncoder.encode(samples: samples, sampleRate: 24_000).write(to: url)
        return url.path
    }

    private static func collectAudio(
        _ synthesizer: any SpeechSynthesizer,
        _ request: SpeechRequest
    ) async throws -> (samples: [Float], chunkLengths: [Int], rate: Int) {
        var all: [Float] = []
        var lengths: [Int] = []
        var rate = 0
        for try await chunk in try await synthesizer.synthesize(request) {
            rate = chunk.sampleRate
            lengths.append(chunk.samples.count)
            all.append(contentsOf: chunk.samples)
        }
        return (all, lengths, rate)
    }

    @Test("KokoroMLXSynthesizer tiny fixture matches CPU ISTFT reference (#228) and synthesizes 24 kHz voices")
    func kokoroMLXParityAndSynthesis() async throws {
        let kokoro = KokoroMLXSynthesizer(config: .tiny, seed: 493)
        let parity = try await kokoro.predictSTFTParity(text: "Basalt Kokoro parity", voice: "af_heart", speed: 1.0)
        #expect(!parity.tokenIDs.isEmpty)
        #expect(parity.mlxWaveform.count == parity.cpuWaveform.count && !parity.mlxWaveform.isEmpty)
        var maxDiff: Float = 0
        for (lhs, rhs) in zip(parity.mlxWaveform, parity.cpuWaveform) {
            maxDiff = max(maxDiff, abs(lhs - rhs))
        }
        #expect(maxDiff < 1e-3)

        let req = try SpeechRequest(model: "Kokoro-82M", input: "Hello from Kokoro MLX.", voice: "bf_emma")
        let out = try await Self.collectAudio(kokoro, req)
        #expect(out.rate == 24_000 && out.samples.count > 200)
        #expect(TTSWaveformMetrics.allFinite(out.samples) && TTSWaveformMetrics.rms(out.samples) > 1e-3)
        let probeRMS = try await SpeechSynthesizerLoader.probeSpeechSynthesizer(kokoro, voice: "af_heart")
        #expect(probeRMS > 1e-3)
    }

    @Test("Qwen3TTSSynthesizer streams 12 Hz 2000-sample packets with instruct, Russian and referenceAudio conditioning")
    func qwen3StreamingAndConditioning() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "qwen3-tts-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let refA = try Self.writeReferenceWAV(freq: 180.0, directory: dir, name: "ref_a.wav")
        let refB = try Self.writeReferenceWAV(freq: 390.0, directory: dir, name: "ref_b.wav")

        let qwen = Qwen3TTSSynthesizer(config: .tiny, seed: 1_206)
        var reqA = try SpeechRequest(model: "Qwen3-TTS-12Hz-0.6B-Tiny", input: "Привет, мир! Streaming check.", voice: "aiden")
        try reqA.configure(language: "ru", instruct: "Speak clearly and calmly", referenceAudio: refA)
        let outA = try await Self.collectAudio(qwen, reqA)
        #expect(outA.rate == 24_000 && outA.chunkLengths.count >= 2)
        #expect(outA.chunkLengths.allSatisfy { $0 == 2_000 })
        #expect(TTSWaveformMetrics.allFinite(outA.samples) && TTSWaveformMetrics.rms(outA.samples) > 1e-3)

        var reqB = reqA
        try reqB.configure(language: "ru", instruct: "Speak clearly and calmly", referenceAudio: refB)
        let outB = try await Self.collectAudio(qwen, reqB)
        let diffEnergy = zip(outA.samples, outB.samples).reduce(Float(0)) { $0 + abs($1.0 - $1.1) }
        #expect(diffEnergy > 0)
        #expect(try await SpeechSynthesizerLoader.probeSpeechSynthesizer(qwen) > 1e-3)
    }

    @Test("ChatterboxSynthesizer scales prosody with exaggeration and expands paralinguistic [laugh] bursts")
    func chatterboxExaggerationAndTags() async throws {
        let chatterbox = ChatterboxSynthesizer(config: .tiny, seed: 350)
        var calm = try SpeechRequest(model: "Chatterbox-Turbo-Tiny", input: "Short line.", voice: "default")
        try calm.configure(language: "en", exaggeration: 0.1)
        let calmOut = try await Self.collectAudio(chatterbox, calm)

        var expressive = try SpeechRequest(model: "Chatterbox-Turbo-Tiny", input: "Short line [laugh]!", voice: "expressive")
        try expressive.configure(language: "en", exaggeration: 1.8)
        let expOut = try await Self.collectAudio(chatterbox, expressive)

        #expect(calmOut.rate == 24_000 && expOut.rate == 24_000)
        #expect(expOut.samples.count > calmOut.samples.count)
        #expect(TTSWaveformMetrics.allFinite(expOut.samples) && TTSWaveformMetrics.rms(expOut.samples) > 1e-3)
        #expect(TTSWaveformMetrics.rms(expOut.samples) > TTSWaveformMetrics.rms(calmOut.samples) * 0.8)
        #expect(try await SpeechSynthesizerLoader.probeSpeechSynthesizer(chatterbox) > 1e-3)
    }

    @Test("CosyVoice3Synthesizer runs 3-stage FSQ Talker + Causal DiT + HiFT pipeline for English and Russian")
    func cosyVoice3ThreeStageSynthesis() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "cosy3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let refPath = try Self.writeReferenceWAV(freq: 220.0, directory: dir, name: "prompt.wav")

        let cfg = TinyCosyVoice3Config.make()
        #expect(cfg.fsqSpeechTokens == 6_561 && cfg.sosToken == 6_561 && cfg.eosToken == 6_562)
        let cosy = CosyVoice3Synthesizer(config: cfg, seed: 2_512)
        var req = try SpeechRequest(
            model: cfg.modelID,
            input: "Добрый день [breath] <strong>Basalt</strong>!",
            voice: "cosy_ru"
        )
        try req.configure(language: "ru", instruct: "Warm conversational voice", referenceAudio: refPath)
        let out = try await Self.collectAudio(cosy, req)
        #expect(out.rate == 24_000 && !out.chunkLengths.isEmpty && out.samples.count > 200)
        #expect(TTSWaveformMetrics.allFinite(out.samples) && TTSWaveformMetrics.rms(out.samples) > 1e-3)
        #expect(try await SpeechSynthesizerLoader.probeSpeechSynthesizer(cosy) > 1e-3)
    }
}

@Suite("RoutedSpeechSynthesizer 3-tier ANE/MLX/System routing and ComposeHost speak wiring (#493)", .serialized)
struct RoutedAndComposedSpeechTests {
    @Test("RoutedSpeechSynthesizer routes between Qwen3-TTS (.mlx), Kokoro-ANE (.ane) and apple-speech on live probes")
    func routesNeuralMLXToANEAndSystem() async throws {
        let qwen = Qwen3TTSSynthesizer(config: .tiny, seed: 42)
        let kokoroANE = try CoreAISpeechSynthesizer.makeFixture(modelID: "Kokoro-82M-ANE")
        let aneRoute = await kokoroANE.resolveRoute(try SpeechRequest(model: "Kokoro-82M-ANE", input: "Check"))
        #expect(aneRoute.selected.compute == .ane && aneRoute.selected.estimatedTTFAMs == 45)

        let tiers: [RoutedSpeechSynthesizer.Tier] = [
            .init(candidate: SpeechCandidate(id: "qwen_tts", model: qwen.config.modelID, compute: .mlx, estimatedTTFAMs: 95), synthesizer: qwen),
            .init(candidate: SpeechCandidate(id: "kokoro_ane", model: "Kokoro-82M-ANE", compute: .ane, estimatedTTFAMs: 45), synthesizer: kokoroANE),
            .init(candidate: .appleSpeech, synthesizer: SineSpeechSynthesizer(modelID: ComposeSpec.appleSpeech, compute: .appleSpeech)),
        ]

        let busyRouted = RoutedSpeechSynthesizer(tiers: tiers, onGPUBusy: .fallbackANE, gpuBusyProbe: { true }, memoryPressureProbe: { false })
        let req = try SpeechRequest(model: qwen.config.modelID, input: "Fallback on GPU busy", voice: "af_heart")
        let busyDecision = await busyRouted.resolveRoute(req)
        let busyReceipt = busyDecision.receipt(ttfaMs: 42, audioMs: 800, rtf: 0.05)
        #expect(busyDecision.selected.id == "kokoro_ane" && busyReceipt.fallbackFrom == qwen.config.modelID && busyReceipt.reason == "gpu_busy")

        var aneSamples: [Float] = []
        for try await chunk in try await busyRouted.synthesize(req) { aneSamples.append(contentsOf: chunk.samples) }
        #expect(TTSWaveformMetrics.rms(aneSamples) > 1e-3)

        let pressuredRouted = RoutedSpeechSynthesizer(tiers: tiers, gpuBusyProbe: { false }, memoryPressureProbe: { true })
        let memDecision = await pressuredRouted.resolveRoute(req)
        #expect(memDecision.selected.id == "kokoro_ane" && memDecision.reason == .memoryPressure)

        var preferReq = req
        try preferReq.configure(preferANE: true)
        let idleRouted = RoutedSpeechSynthesizer(tiers: tiers, gpuBusyProbe: { false }, memoryPressureProbe: { false })
        let preferDecision = await idleRouted.resolveRoute(preferReq)
        #expect(preferDecision.selected.id == "kokoro_ane" && preferDecision.reason == .preferANE)
    }

    private static func makeComposedSpeechHost(
        aneDir: URL
    ) async throws -> (ComposeHost, ModelOrchestratorActor<MockInferenceEngine>) {
        let yaml = """
            version: 1
            name: voice-compose
            models:
              chat:
                source: org/mock-chat
              tts_qwen:
                source: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
              tts_ane:
                source: \(aneDir.path)
            pipeline:
              answer: chat
              speak:
                chain: [tts_qwen, tts_ane, apple-speech]
                on_gpu_busy: fallback_ane
            """
        let spec = try ComposeFile.parse(yaml, file: "voice.yaml")
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "org/mock-chat")
        let qwen = Qwen3TTSSynthesizer(config: .tiny, seed: 77)
        let services = ComposeServices(
            resolve: { $0 },
            load: { _, _, _, _ in nil },
            decider: { _ in StubDecisionEngine() },
            decisions: { _, _, _, _, _ in nil },
            key: { _ in "" },
            cloudLimits: { _ in [:] },
            speaker: { source, path in
                if source.contains("Qwen3-TTS") {
                    return SpeechSynthesizerLoader.LoadedSynthesizer(
                        synthesizer: qwen,
                        candidate: SpeechCandidate(id: source, model: qwen.config.modelID, compute: .mlx, estimatedTTFAMs: 95),
                        family: .qwen3TTS
                    )
                }
                return try await SpeechSynthesizerLoader.load(identifier: source, directory: URL(fileURLWithPath: path))
            }
        )
        var start = ComposeStart(chat: orchestrator, decider: nil, decisions: nil, cloudGate: CloudGate(limits: [:], storage: nil), startup: PipelineStages(), besideDecode: true)
        start.admission = ComposeAdmission(sizes: ComposeSpec.Sizes(bytes: { _ in 1_000_000 }, onNeuralEngine: { $0.contains("kokoro-ane") })) { (16_000_000_000, .desktop) }
        let host = ComposeHost(services: services, spec: spec, start: start)
        try await host.start(defaultDecisions: nil)
        return (host, orchestrator)
    }

    @Test("ComposeHost wires speak chain into PipelineRegistry and falls back to ANE while chat GenerationGate is busy")
    func composeHostWiresSpeakChain() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "compose-tts-\(UUID().uuidString)")
        let aneDir = dir.appending(component: "kokoro-ane")
        try FileManager.default.createDirectory(at: aneDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_id":"Kokoro-82M-ANE","synthetic":true}"#.utf8)
            .write(to: aneDir.appending(component: CoreAISpeechBundle.kokoroANEManifest))

        let (host, orchestrator) = try await Self.makeComposedSpeechHost(aneDir: aneDir)
        let speech = try #require(await host.registry.system(ComposeSpec.defaultPipeline)?.speech)
        let idleReq = try SpeechRequest(model: "tts_qwen", input: "Idle check", voice: "af_heart")
        let idleDecision = await speech.resolveRoute(idleReq)
        #expect(idleDecision.selected.id == "tts_qwen" && idleDecision.selected.compute == .mlx)

        let heldSlot = try await orchestrator.gate.admit()
        let busyDecision = await speech.resolveRoute(idleReq)
        await orchestrator.gate.release(heldSlot)
        #expect(busyDecision.selected.id == "tts_ane" && busyDecision.selected.compute == .ane && busyDecision.reason == .gpuBusy)

        let registry = await host.registry
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock-chat"))
            .withPipelines(registry, compose: host.control)
        try await Application(router: server.makeRouter()).test(.router) { client in
            let status = try await client.execute(
                uri: "/v1/audio/speech",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"tts_qwen","input":"Hello composed TTS","voice":"af_heart","prefer_ane":true}"#)
            ) { res in
                (res.status, res.headers[HTTPField.Name("X-Basalt-Speech-Compute")!])
            }
            #expect(status.0 == .ok && status.1 == "ane")
        }
    }
}
