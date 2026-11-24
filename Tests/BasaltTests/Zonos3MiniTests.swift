import Testing
import Foundation
import MLX
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

@Suite("Zyphra ZONOS3-Mini descriptor, 43-language catalog, 27 voices, and Compose speak: schema (#525)")
struct Zonos3DescriptorAndComposeTests {
    @Test("TTSModelFamily detects ZONOS3-Mini repos, configs and voice tables while rejecting legacy Zonos v0.1 and v2")
    func detectsZonos3AndRejectsLegacy() throws {
        #expect(TTSModelFamily.detect(identifier: "Zyphra/ZONOS3-Mini") == .zonos3Mini)
        #expect(TTSModelFamily.detect(identifier: "Zyphra/ZONOS3-Mini-hf") == .zonos3Mini)
        #expect(TTSModelFamily.detect(identifier: "mlx-community/ZONOS3-Mini-bf16") == .zonos3Mini)
        #expect(TTSModelFamily.detect(identifier: "custom", config: ["model_type": "zonos3_mini"]) == .zonos3Mini)
        #expect(TTSModelFamily.detect(identifier: "custom", config: ["architectures": ["Zonos3MiniForTextToSpeech"]]) == .zonos3Mini)
        #expect(TTSModelFamily.zonos3Mini.defaultSampleRate == 44_100)
        #expect(TTSModelFamily.zonos3Mini.licenseSPDX == "Apache-2.0")

        let dir = FileManager.default.temporaryDirectory.appending(path: "zonos3-dir-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data().write(to: dir.appending(component: "voices.safetensors"))
        #expect(try TTSModelFamily.validateSupported(identifier: "local-zonos", directory: dir) == .zonos3Mini)

        for legacy in ["Zyphra/Zonos-v0.1-hybrid", "Zyphra/Zonos-v0.1-transformer", "Zyphra/ZONOS2"] {
            #expect(throws: BasaltError.self) {
                _ = try TTSModelFamily.validateSupported(identifier: legacy)
            }
        }
        #expect(throws: BasaltError.self) {
            _ = try TTSModelFamily.validateSupported(identifier: "custom-tts", config: ["model_type": "zonos"])
        }
    }

    @Test("Zonos3Catalog normalizes all 44 locale codes across 43 languages, collapses line breaks, lists 27 default voices, and resamples chunks")
    func catalogLanguagesVoicesAndResample() {
        #expect(Zonos3Catalog.supportedLanguages.count == 44)
        #expect(Zonos3Catalog.defaultVoices.count == 27)
        #expect(Zonos3Catalog.normalizeLanguage(nil) == "en_us")
        #expect(Zonos3Catalog.normalizeLanguage("en") == "en_us")
        #expect(Zonos3Catalog.normalizeLanguage("en-GB") == "en_gb")
        #expect(Zonos3Catalog.normalizeLanguage("ru-RU") == "ru")
        #expect(Zonos3Catalog.normalizeLanguage("zh") == "cmn")
        #expect(Zonos3Catalog.normalizeLanguage("pt-BR") == "pt_br")
        #expect(Zonos3Catalog.normalizeLanguage("tlh") == nil)
        #expect(Zonos3Catalog.collapseLineBreaks("First line\n\nSecond\r\nThird") == "First line Second Third")

        let voices = TTSVoiceCatalog.zonos3Voices(model: "Zyphra/ZONOS3-Mini-hf", compute: .mlx)
        #expect(voices.count == 27 && voices.first?.id == "Savannah" && voices.first?.language == "en-US")

        let chunk44k = AudioChunk(
            samples: (0 ..< 4_410).map { Float(sin(2.0 * .pi * 440.0 * Double($0) / 44_100.0)) },
            sampleRate: 44_100,
            isFinal: true
        )
        let chunk24k = chunk44k.resampled(to: 24_000)
        #expect(chunk24k.sampleRate == 24_000 && chunk24k.samples.count == 2_400 && chunk24k.isFinal)
    }

    @Test("Bundled zonos-voice compose profile and speak language/reference_audio schema validate cleanly")
    func composeBundledAndSpeakValidation() throws {
        let bundled = try ComposeFile.load("zonos-voice")
        #expect(bundled.name == "zonos-voice")
        #expect(bundled.speak?.chain == ["tts", "apple-speech"])
        #expect(bundled.speak?.voice == "Savannah")
        #expect(bundled.speak?.language == "en_us")
        #expect(bundled.speak?.onGPUBusy == .fallbackAppleSpeech)

        let yaml = """
            version: 2
            name: zonos-ru
            models:
              chat:
                source: mlx-community/Qwen3-4B-4bit
              tts:
                source: Zyphra/ZONOS3-Mini-hf
            pipeline:
              answer: chat
              speak:
                chain: [tts, apple-speech]
                voice: Seraphina
                language: ru
                reference_audio: /tmp/ref_voice.wav
                exaggeration: 0.7
            """
        let parsed = try ComposeFile.parse(yaml, file: "zonos-ru.yaml")
        #expect(parsed.speak?.language == "ru")
        #expect(parsed.speak?.referenceAudio == "/tmp/ref_voice.wav")
        #expect(parsed.speak?.exaggeration == 0.7)

        var bad = parsed
        bad.speak?.referenceAudio = "../escape.wav"
        #expect(bad.issues.contains { $0.path == "pipeline.speak.reference_audio" })
    }
}

@Suite("ZONOS3-Mini #228 CPU-vs-MLX parity for delay schedule, Gated DeltaNet, FullAttnRes, and 44.1 kHz codec (#525)", .serialized)
struct Zonos3MiniKernelParityTests {
    @Test("Zonos3DelaySchedule shearRows, undelay, takeCodes and outputStage round-trip multi-codebook frames")
    func delayScheduleRoundTripAndOutputStage() {
        let frames: [[Int32]] = (0 ..< 8).map { f in (0 ..< 4).map { c in Int32(f * 10 + c + 1) } }
        let sheared = Zonos3DelaySchedule.shearRows(codes: frames, pad: 99, length: frames.count + 3)
        let undelayed = Zonos3DelaySchedule.undelay(buffer: sheared, start: 0, end: frames.count, n: 4, pad: 99)
        #expect(undelayed == frames)

        var lm = Zonos3MiniConfig.LM()
        lm.nCodebooks = 4
        lm.audioPadId = 99
        lm.eoaId = 98
        var withEOA = sheared
        withEOA.append([98, 99, 99, 99])
        let (taken, start) = Zonos3DelaySchedule.takeCodes(
            rows: withEOA, eosFrame: 6, prefillRows: nil, lm: lm, overlapFrames: 2
        )
        #expect(start == 0 && taken.count == 6 && taken == Array(frames.prefix(6)))

        let loud = [Float](repeating: 0.95, count: 882)
        let staged = Zonos3DelaySchedule.outputStage(loud, headroomDB: 3.0, outputLimitDBFS: -1.0, fadeOutMs: 5.0, sampleRate: 44_100)
        #expect(staged.count == loud.count && staged.last == 0 && (staged.max() ?? 1.0) < 0.9)
    }

    @Test("Gated DeltaNet recurrence, FullAttnRes routing, and chunked LM continuation match CPU float32 reference (#228)")
    func gdnAndAttnResMatchCPUReference() {
        withRandomState(MLXRandom.RandomState(seed: 525)) {
            let t = 5, h = 2, d = 4
            let q = MLXRandom.normal([t, h, d]) * 0.2
            let k = MLXRandom.normal([t, h, d]) * 0.2
            let v = MLXRandom.normal([t, h, d]) * 0.2
            let g = -abs(MLXRandom.normal([t, h])) * 0.1
            let beta = sigmoid(MLXRandom.normal([t, h]))
            let (mlxOut, mlxState) = Zonos3GDNMixer.recurrentGatedDeltaRule(
                q: q, k: k, v: v, g: g, beta: beta, initialState: nil
            )
            eval(mlxOut, mlxState)
            let cpuOut = Self.cpuRecurrentGDN(GDNRefInput(
                q: q.asArray(Float.self), k: k.asArray(Float.self), v: v.asArray(Float.self),
                g: g.asArray(Float.self), beta: beta.asArray(Float.self), t: t, h: h, d: d
            ))
            let gdnDiff = zip(mlxOut.asArray(Float.self), cpuOut).map { abs($0 - $1) }.max() ?? 1.0
            #expect(gdnDiff < 1e-4)

            let s0 = MLXRandom.normal([3, 8])
            let s1 = MLXRandom.normal([3, 8])
            let query = MLXRandom.normal([8])
            let routed = Zonos3MiniLMModule.fullAttnResRoute(sources: [s0, s1], query: query, eps: 1e-5)
            eval(routed)
            let cpuRouted = Self.cpuFullAttnRes(
                sources: [s0.asArray(Float.self), s1.asArray(Float.self)],
                query: query.asArray(Float.self), rows: 3, dim: 8, eps: 1e-5
            )
            let routeDiff = zip(routed.asArray(Float.self), cpuRouted).map { abs($0 - $1) }.max() ?? 1.0
            #expect(routeDiff < 1e-4)

            let lm = Zonos3MiniLMModule(Zonos3MiniConfig.tiny.lm)
            eval(lm)
            let ids = MLXArray((0 ..< (6 * 11)).map { Int32(($0 * 3 + 1) % 32) }).reshaped(6, 11)
            let fullCache = Zonos3MiniCache()
            let fullLogits = lm.forward(ids: ids, cache: fullCache, logitsToKeep: 1)
            let splitCache = Zonos3MiniCache()
            _ = lm.forward(ids: ids[..<3], cache: splitCache, logitsToKeep: 1)
            let contLogits = lm.forward(ids: ids[3...], cache: splitCache, logitsToKeep: 1)
            eval(fullLogits, contLogits)
            let lmDiff = zip(fullLogits.asArray(Float.self), contLogits.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 1.0
            #expect(lmDiff < 1e-3)
        }
    }

    @Test("Multi-band iSTFT, 44.16 kHz -> 44.1 kHz sinc resampler, and weight key remapper match CPU reference (#228)")
    func codecISTFTAndResamplerMatchCPUReference() {
        withRandomState(MLXRandom.RandomState(seed: 526)) {
            let nFFT = 64, hop = 16, frames = 6
            let bins = nFFT / 2 + 1
            let mag = exp(MLXRandom.normal([frames, bins]) * 0.3)
            let phase = MLXRandom.normal([frames, bins])
            let mlxWave = Zonos3CodecMath.mlxCenterInverseSTFT(
                magnitude: mag, phase: phase, nFFT: nFFT, hopLength: hop, targetLength: frames * hop
            )
            eval(mlxWave)
            let flatMag = mag.asArray(Float.self)
            let flatPhase = phase.asArray(Float.self)
            let magRows = (0 ..< frames).map { f in Array(flatMag[(f * bins) ..< ((f + 1) * bins)]) }
            let phaseRows = (0 ..< frames).map { f in Array(flatPhase[(f * bins) ..< ((f + 1) * bins)]) }
            let cpuWave = Zonos3CodecMath.cpuCenterInverseSTFT(
                magnitude: magRows, phase: phaseRows, nFFT: nFFT, hopLength: hop, targetLength: frames * hop
            )
            let stftDiff = zip(mlxWave.asArray(Float.self), cpuWave).map { abs($0 - $1) }.max() ?? 1.0
            #expect(stftDiff < 1e-4)

            let codec = Zonos3MiniCodecModule(Zonos3MiniConfig.tiny.codec)
            eval(codec)
            let codes = MLXArray((0 ..< 30).map { Int32($0 % 32) }).reshaped(3, 10)
            let decoded = codec.decode(codes: codes)
            eval(decoded)
            #expect(decoded.shape == [3 * Zonos3Catalog.frameHopSamples])

            let sig = MLXRandom.normal([1, 2_944]) * 0.25
            let mlxResampled = codec.resampleToExternal(sig)
            eval(mlxResampled, codec.toExternalKernel)
            let cpuResampled = Zonos3CodecMath.cpuSincResample(
                sig.reshaped(-1).asArray(Float.self),
                kernel: codec.toExternalKernel.reshaped(-1).asArray(Float.self),
                orig: 736, new: 735, width: 17
            )
            let resDiff = zip(mlxResampled.reshaped(-1).asArray(Float.self), cpuResampled).map { abs($0 - $1) }.max() ?? 1.0
            #expect(mlxResampled.dim(1) == 2_940 && resDiff < 1e-3)

            #expect(Zonos3WeightLoader.remapKey("speaker_encoder.conv.weight", semanticDepth: 4) == nil)
            #expect(Zonos3WeightLoader.remapKey("codec.semantic_head.2.dwconv.weight", semanticDepth: 4) == "codec.semantic_blocks.2.dwconv.weight")
            #expect(Zonos3WeightLoader.remapKey("codec.semantic_head.4.weight", semanticDepth: 4) == "codec.semantic_proj.weight")
            #expect(Zonos3WeightLoader.remapKey("codec.stages.6.layers.1.ffn.0.weight", semanticDepth: 4) == "codec.moss_stages.3.layers.1.ffn_in.weight")
            #expect(Zonos3WeightLoader.needsConv1dTranspose("lm.layers.0.attention.conv.weight", ndim: 3))
        }
    }

    private struct GDNRefInput {
        let q: [Float]
        let k: [Float]
        let v: [Float]
        let g: [Float]
        let beta: [Float]
        let t: Int
        let h: Int
        let d: Int
    }

    private static func cpuRecurrentGDN(_ input: GDNRefInput) -> [Float] {
        let q = input.q, k = input.k, v = input.v, g = input.g, beta = input.beta
        let t = input.t, h = input.h, d = input.d
        let scale = 1.0 / Float(d).squareRoot()
        var state = [Float](repeating: 0, count: h * d * d)
        var out = [Float](repeating: 0, count: t * h * d)
        for step in 0 ..< t {
            for head in 0 ..< h {
                let qOff = (step * h + head) * d
                let decay = exp(g[step * h + head])
                let bVal = beta[step * h + head]
                let sOff = head * d * d
                for i in 0 ..< (d * d) { state[sOff + i] *= decay }
                for valIdx in 0 ..< d {
                    var kvMem: Float = 0
                    for keyIdx in 0 ..< d {
                        kvMem += state[sOff + keyIdx * d + valIdx] * k[qOff + keyIdx]
                    }
                    let delta = (v[qOff + valIdx] - kvMem) * bVal
                    var oVal: Float = 0
                    for keyIdx in 0 ..< d {
                        state[sOff + keyIdx * d + valIdx] += k[qOff + keyIdx] * delta
                        oVal += state[sOff + keyIdx * d + valIdx] * (q[qOff + keyIdx] * scale)
                    }
                    out[qOff + valIdx] = oVal
                }
            }
        }
        return out
    }

    private static func cpuFullAttnRes(
        sources: [[Float]], query: [Float], rows: Int, dim: Int, eps: Float
    ) -> [Float] {
        var out = [Float](repeating: 0, count: rows * dim)
        for r in 0 ..< rows {
            var logits = [Double]()
            for src in sources {
                var meanSq: Float = 0
                for c in 0 ..< dim { meanSq += src[r * dim + c] * src[r * dim + c] }
                let inv = 1.0 / (meanSq / Float(dim) + eps).squareRoot()
                var dot: Double = 0
                for c in 0 ..< dim { dot += Double(src[r * dim + c] * inv * query[c]) }
                logits.append(dot)
            }
            let maxL = logits.max() ?? 0
            let exps = logits.map { exp($0 - maxL) }
            let sumE = exps.reduce(0, +)
            for (sIdx, src) in sources.enumerated() {
                let w = Float(exps[sIdx] / sumE)
                for c in 0 ..< dim { out[r * dim + c] += w * src[r * dim + c] }
            }
        }
        return out
    }
}

@Suite("ZONOS3-Mini tiny-fixture synthesis, REST 44.1 kHz / 24 kHz endpoints, and Compose speak: routing (#525)", .serialized)
struct Zonos3MiniSynthesisAndComposeE2ETests {
    private static func collectAudio(
        _ synthesizer: any SpeechSynthesizer,
        _ request: SpeechRequest
    ) async throws -> (samples: [Float], rate: Int) {
        var all: [Float] = []
        var rate = 0
        for try await chunk in try await synthesizer.synthesize(request) {
            rate = chunk.sampleRate
            all.append(contentsOf: chunk.samples)
        }
        return (all, rate)
    }

    @Test("Zonos3MiniSynthesizer synthesizes 44.1 kHz English, Russian, nonverbal markup and reference_audio cloning")
    func tinySynthesizerMultilingualAndCloning() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "zonos3-tiny-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let refSamples = (0 ..< 8_820).map { Float(0.5 * sin(2.0 * .pi * 220.0 * Double($0) / 44_100.0)) }
        let refPath = dir.appending(component: "speaker.wav")
        try WAVEncoder.encode(samples: refSamples, sampleRate: 44_100).write(to: refPath)

        let zonos = Zonos3MiniSynthesizer(config: .tiny, seed: 525)
        var enReq = try SpeechRequest(model: "Zyphra/ZONOS3-Mini-hf", input: "Hello [sigh] from Zonos three!", voice: "Savannah")
        try enReq.configure(language: "en_us", exaggeration: 0.65)
        let enOut = try await Self.collectAudio(zonos, enReq)
        #expect(enOut.rate == 44_100 && enOut.samples.count >= Zonos3Catalog.frameHopSamples)
        #expect(TTSWaveformMetrics.allFinite(enOut.samples) && TTSWaveformMetrics.rms(enOut.samples) > 1e-4)

        var ruCloneReq = try SpeechRequest(model: "Zyphra/ZONOS3-Mini-hf", input: "Привет [laugh] это проверка синтеза.", voice: "Seraphina")
        try ruCloneReq.configure(language: "ru", referenceAudio: refPath.path)
        let ruOut = try await Self.collectAudio(zonos, ruCloneReq)
        #expect(ruOut.rate == 44_100 && TTSWaveformMetrics.allFinite(ruOut.samples) && TTSWaveformMetrics.rms(ruOut.samples) > 1e-4)

        var badLang = try SpeechRequest(model: "Zyphra/ZONOS3-Mini-hf", input: "Unsupported")
        try badLang.configure(language: "tlh")
        await #expect(throws: BasaltError.self) {
            _ = try await zonos.synthesize(badLang)
        }
    }

    private static func makeZonosComposeHost(aneDir: URL) async throws -> (ComposeHost, ModelOrchestratorActor<MockInferenceEngine>) {
        let yaml = """
            version: 2
            name: zonos-compose
            models:
              chat:
                source: org/mock-chat
              tts_zonos:
                source: Zyphra/ZONOS3-Mini-hf
              tts_ane:
                source: \(aneDir.path)
            pipeline:
              answer: chat
              speak:
                chain: [tts_zonos, tts_ane, apple-speech]
                voice: Savannah
                language: ru
                exaggeration: 0.6
                on_gpu_busy: fallback_ane
            """
        let spec = try ComposeFile.parse(yaml, file: "zonos.yaml")
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "org/mock-chat")
        let zonos = Zonos3MiniSynthesizer(config: .tiny, seed: 99)
        let services = ComposeServices(
            resolve: { $0 }, load: { _, _, _, _ in nil }, decider: { _ in StubDecisionEngine() },
            decisions: { _, _, _, _, _ in nil }, key: { _ in "" }, cloudLimits: { _ in [:] },
            speaker: { source, path in
                if source.contains("ZONOS3") {
                    return SpeechSynthesizerLoader.LoadedSynthesizer(
                        synthesizer: zonos,
                        candidate: SpeechCandidate(id: source, model: zonos.config.modelID, compute: .mlx, estimatedTTFAMs: 85),
                        family: .zonos3Mini
                    )
                }
                return try await SpeechSynthesizerLoader.load(identifier: source, directory: URL(fileURLWithPath: path))
            },
            memoryPressure: { false }
        )
        var start = ComposeStart(
            chat: orchestrator, decider: nil, decisions: nil,
            cloudGate: CloudGate(limits: [:], storage: nil), startup: PipelineStages(), besideDecode: true
        )
        start.admission = ComposeAdmission(
            sizes: ComposeSpec.Sizes(bytes: { _ in 1_000_000 }, onNeuralEngine: { $0.contains("kokoro-ane") })
        ) { (16_000_000_000, .desktop) }
        let host = ComposeHost(services: services, spec: spec, start: start)
        try await host.start(defaultDecisions: nil)
        return (host, orchestrator)
    }

    @Test("ComposeHost wires ZONOS3-Mini speak defaults and /v1/audio/speech serves 44.1 kHz PCM and 24 kHz resampled PCM")
    func composeHostSpeakDefaultsAndRESTRates() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "compose-zonos-\(UUID().uuidString)")
        let aneDir = dir.appending(component: "kokoro-ane")
        try FileManager.default.createDirectory(at: aneDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_id":"Kokoro-82M-ANE","synthetic":true}"#.utf8)
            .write(to: aneDir.appending(component: CoreAISpeechBundle.kokoroANEManifest))

        let (host, orchestrator) = try await Self.makeZonosComposeHost(aneDir: aneDir)
        let registry = await host.registry
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock-chat"))
            .withPipelines(registry, compose: host.control)
        try await Application(router: server.makeRouter()).test(.router) { client in
            let nativePCM = try await client.execute(
                uri: "/v1/audio/speech",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"tts_zonos","input":"Привет из композ","response_format":"pcm"}"#)
            ) { res in
                (res.status, res.headers[HTTPField.Name("X-Basalt-Speech-Sample-Rate")!], res.body.readableBytes)
            }
            #expect(nativePCM.0 == .ok && nativePCM.1 == "44100" && nativePCM.2 > 0)

            let pcm24k = try await client.execute(
                uri: "/v1/audio/speech",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"tts_zonos","input":"Привет из композ","response_format":"pcm_24k"}"#)
            ) { res in
                (res.status, res.headers[HTTPField.Name("X-Basalt-Speech-Sample-Rate")!], res.body.readableBytes)
            }
            #expect(pcm24k.0 == .ok && pcm24k.1 == "24000" && pcm24k.2 > 0 && pcm24k.2 < nativePCM.2)
        }
    }
}

@Suite("Zyphra ZONOS3-Mini real-model 44.1 kHz synthesis verification on Metal (#525)", .serialized)
struct Zonos3MiniRealModelTests {
    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_ZONOS3_MODEL_PATH"]
    }

    @Test("Real Zyphra/ZONOS3-Mini-hf checkpoint loads all 422 LM+Codec tensors and 27 voices and synthesizes 44.1 kHz English and Russian speech",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_ZONOS3_MODEL_PATH"))
    func realZonos3EndToEnd() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("zonos3-mini", modelPath: path)
        defer { log.releaseGPU() }

        let dir = URL(fileURLWithPath: path)
        let loadStart = ContinuousClock.now
        let synth = try await log.step("load") {
            try await Zonos3MiniSynthesizer(directory: dir, modelID: "Zyphra/ZONOS3-Mini-hf")
        }
        let loadSec = (ContinuousClock.now - loadStart) / .seconds(1)
        let activeMB = Memory.activeMemory / 1_048_576
        let peakMB = Memory.peakMemory / 1_048_576
        log.record("weights", String(
            format: "load %.2fs tensors %d voices %d tokenizer %@ activeMB %d peakMB %d",
            loadSec, synth.loadedTensorCount, synth.loadedVoiceCount, "\(synth.hasTokenizer)", activeMB, peakMB
        ))
        #expect(synth.loadedTensorCount == 422)
        #expect(synth.loadedVoiceCount == 27)
        #expect(synth.hasTokenizer)

        for (tag, voice, lang, text) in [
            ("en-savannah", "Savannah", "en_us", "Basalt now runs Zyphra Zonos three Mini natively on Apple Silicon."),
            ("de-liesel", "Liesel", "de", "Basalt erzeugt klare Sprache direkt auf Apple Silicon."),
            ("ru-savannah", "Savannah", "ru", "Базальт синтезирует чистую речь на частоте сорок четыре килогерца."),
        ] {
            var req = try SpeechRequest(model: "Zyphra/ZONOS3-Mini-hf", input: text, voice: voice)
            try req.configure(language: lang)
            let t0 = ContinuousClock.now
            var firstChunkMs: Double = 0
            var samples: [Float] = []
            var sampleRate = 0
            for try await chunk in try await synth.synthesize(req) {
                if samples.isEmpty {
                    firstChunkMs = ((ContinuousClock.now - t0) / .seconds(1)) * 1_000.0
                }
                sampleRate = chunk.sampleRate
                samples.append(contentsOf: chunk.samples)
            }
            let wallSec = (ContinuousClock.now - t0) / .seconds(1)
            let audioSec = Double(samples.count) / Double(max(1, sampleRate))
            let rtf = wallSec / max(1e-6, audioSec)
            let rms = TTSWaveformMetrics.rms(samples)
            let peak = samples.map(abs).max() ?? 0
            log.record(tag, String(
                format: "voice %@ lang %@ rate %d audio %.2fs wall %.2fs rtf %.3f ttfa %.0fms rms %.4f peak %.4f activeMB %d peakMB %d",
                voice, lang, sampleRate, audioSec, wallSec, rtf, firstChunkMs, rms, peak,
                Memory.activeMemory / 1_048_576, Memory.peakMemory / 1_048_576
            ))
            #expect(sampleRate == 44_100)
            #expect(audioSec > 0.4)
            #expect(TTSWaveformMetrics.allFinite(samples))
            #expect(rms > 1e-3)
        }
    }
}
