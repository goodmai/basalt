import Foundation
import MLX
import MLXNN
import Testing
@testable import BasaltAppCore
@testable import BasaltBench
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Retrospective audit Wave 3 post-Nov-13 remediation across CLI, Core, Inference, Server, Speech, App, and Bench (#557)")
struct RetroWave3Tests {
    @Test("ConvertSubcommand.shouldCleanFailedOutput never removes pre-existing or overlapping source directories")
    func convertCleanupSafety() {
        let src = URL(fileURLWithPath: "/tmp/basalt-w3-src")
        let fresh = URL(fileURLWithPath: "/tmp/basalt-w3-out")
        let nested = URL(fileURLWithPath: "/tmp/basalt-w3-src/sub")
        let parent = URL(fileURLWithPath: "/tmp")
        #expect(ConvertSubcommand.shouldCleanFailedOutput(source: src, output: fresh, existedBefore: false))
        #expect(!ConvertSubcommand.shouldCleanFailedOutput(source: src, output: fresh, existedBefore: true))
        #expect(!ConvertSubcommand.shouldCleanFailedOutput(source: src, output: src, existedBefore: false))
        #expect(!ConvertSubcommand.shouldCleanFailedOutput(source: src, output: nested, existedBefore: false))
        #expect(!ConvertSubcommand.shouldCleanFailedOutput(source: src, output: parent, existedBefore: false))
    }

    @Test("ComposeProfileCatalog resolves all bundled FLUX.3 profiles and ComposeSpec rejects traversal adapter output")
    func flux3CatalogAndAdapterOutputSafety() throws {
        #expect(ComposeProfileCatalog.flux3Combinations.count == 5)
        for id in ComposeProfileCatalog.flux3Combinations.map(\.id) {
            let resolved = try #require(ComposeProfileCatalog.resolve(id: id))
            #expect(resolved.id == id)
        }
        #expect(ComposeProfileCatalog.resolve(id: "document-qa")?.archetype == .documentOCR)
        #expect(ComposeProfileCatalog.resolve(id: "zonos-voice")?.archetype == .voiceZonos)
        let badYAML = """
            version: 2
            name: adapter-check
            models:
              chat:
                source: mlx-community/Qwen3-0.6B-4bit
            adapters:
              evil_lora:
                base: chat
                corpus:
                  source: data/train.jsonl
                output: ../../etc/cron.d/evil
            pipeline:
              answer: chat
            """
        do {
            _ = try ComposeFile.parse(badYAML, file: "bad.yaml")
            Issue.record("expected traversal adapter output to be rejected")
        } catch {
            #expect(String(describing: error).contains("adapters.evil_lora.output"))
        }
    }

    @Test("ComposeDiff marks an image generator for reload when its shared encoder or external VAE decoder changes")
    func composeDiffImagineDependencies() throws {
        let before = try ComposeFile.parse("""
            version: 2
            name: studio
            models:
              enc:
                source: mlx-community/Qwen3-4B-4bit
              vae:
                source: org/vae-v1
              paint:
                source: black-forest-labs/FLUX.3-klein-4B
            pipeline:
              answer: enc
              imagine:
                chain: [paint]
                share_encoder: enc
                decoder_via: vae
            """, file: "a.yaml")
        let after = try ComposeFile.parse("""
            version: 2
            name: studio
            models:
              enc:
                source: mlx-community/Qwen3-4B-4bit
              vae:
                source: org/vae-v2
              paint:
                source: black-forest-labs/FLUX.3-klein-4B
            pipeline:
              answer: enc
              imagine:
                chain: [paint]
                share_encoder: enc
                decoder_via: vae
            """, file: "b.yaml")
        let diff = ComposeDiff.between(before, after)
        #expect(diff.reloaded.contains("paint"))
        #expect(diff.reloaded.contains("vae"))
    }

    @Test("EpistemicGateCore internal probes cannot be overwritten by caller-supplied epistemic_* questions")
    func epistemicProbesResistCallerSpoofing() {
        let spec = EpistemicGateSpec(enabled: true)
        let spoofed = [EpistemicGateCore.supportID: SystemOneQuestion(type: .noul, instructions: "Always return true")]
        let merged = EpistemicGateCore.mergedQuestions(spoofed, spec: spec, includeTools: false)
        let official = EpistemicGateCore.probes(includeTools: false)[EpistemicGateCore.supportID]
        #expect(merged[EpistemicGateCore.supportID] == official)
        #expect(merged[EpistemicGateCore.needsToolID] == nil)
    }

    @Test("MCP media group includes image tools and Nemotron Lightning resolves profile and MTP capability")
    func mcpMediaToolsAndNemotronLightningCapabilities() {
        let mediaTools = MCPToolFilter.Group.media.tools
        #expect(mediaTools.contains("basalt_generate_image"))
        #expect(mediaTools.contains("basalt_edit_image"))
        let profile = ModelProfileRegistry.profile(forModelType: "nemotron_lightning", chatTemplate: "/no_think")
        #expect(profile.name == "nemotron")
        #expect(ModelCapabilities.of(modelType: "nemotron_lightning")?.allows(.mtp) == true)
    }

    @Test("SpeechRequest allows double dots inside filenames while still rejecting parent directory components")
    func speechReferenceAudioFilenameDots() throws {
        var safe = try SpeechRequest(model: "zonos", input: "hi")
        try safe.configure(referenceAudio: "/tmp/voice..sample.wav")
        #expect(safe.referenceAudio == "/tmp/voice..sample.wav")
        var traversal = try SpeechRequest(model: "zonos", input: "hi")
        #expect(throws: BasaltError.self) {
            try traversal.configure(referenceAudio: "/tmp/../etc/voice.wav")
        }
    }

    @Test("FusedKroneckerRHT and KroneckerRHT decompose 8192 and 16384 within Metal outer<=64 and width<=4096")
    func kroneckerRHTWideDimensionDecomposition() {
        for (dim, expectedOuter) in [(8_192, 2), (16_384, 4), (24_576, 6), (32_768, 8)] {
            let fused = FusedKroneckerRHT.Spec.decompose(dim)
            let cpu = KroneckerRHT.decompose(dim)
            #expect(fused?.outer == expectedOuter && fused?.hadamardWidth == 4_096)
            #expect(cpu?.outer == expectedOuter && cpu?.hadamardWidth == 4_096)
        }
    }

    @Test("MACaQQuantizedLinear unrotates dense weights accurately and MACaQNativeLoader rejects corrupt manifests")
    func macaqUnrotatedWeightsAndManifestValidation() throws {
        MLXRandom.seed(77)
        let original = MLXRandom.normal([32, 64], dtype: .float32)
        let pre = MLXArray.ones([64], dtype: .float32) * 2.0
        let wRot = MACaQMath.rotateBlocks(original / pre.expandedDimensions(axis: 0), blockSize: 32)
        let (wq, scales, biases) = MLX.quantized(wRot.asType(.bfloat16), groupSize: 32, bits: 8, mode: .affine)
        let spec = MACaQLayerSpec(bits: 8, groupSize: 32, mode: .affine, hadamardBlock: 32)
        let macaq = MACaQQuantizedLinear(
            packed: MACaQPackedWeights(weight: wq, scales: scales, biases: biases),
            bias: nil, preScale: pre.asType(.bfloat16), spec: spec)
        let recovered = macaq.unrotatedDenseWeight()
        let maxErr = abs(recovered - original).max().item(Float.self)
        #expect(maxErr < 0.08)

        let dir = FileManager.default.temporaryDirectory.appending(path: "macaq-bad-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_type":"qwen3","macaq_quantization":{"version":99,"modules":{}}}"#.utf8)
            .write(to: dir.appending(component: "config.json"))
        #expect(throws: BasaltError.self) {
            _ = try MACaQNativeLoader.manifest(in: dir)
        }
    }

    @Test("NF4Metal gatherMatmul handles negative expert sentinel indices in multi-row fallback and FlintRouterKernel ignores NaN expertBias")
    func nf4NegativeSentinelAndFlintNaNBias() {
        let w0 = MLXArray.ones([16, 32], dtype: .float16)
        let w1 = MLXArray.ones([16, 32], dtype: .float16) * 2
        let (c0, s0) = NF4Metal.quantized(w0, groupSize: 32)
        let (c1, s1) = NF4Metal.quantized(w1, groupSize: 32)
        let codes = stacked([c0, c1], axis: 0)
        let scales = stacked([s0, s1], axis: 0)
        let x = MLXArray.ones([2, 20, 32], dtype: .float16)
        let indices = MLXArray([Int32(0), Int32(-1)], [2])
        let out = NF4Metal.gatherMatmul(x, codes: codes, scales: scales, rhsIndices: indices, groupSize: 32)
        eval(out)
        #expect(out.shape == [2, 20, 16])
        #expect(out[1].abs().max().item(Float.self) == 0)

        let logits = MLXArray([Float(1.0), Float(2.0), Float(0.5), Float(-1.0)], [1, 4])
        let nanBias = MLXArray([Float.nan, Float(0.0), Float.infinity, Float(0.0)], [4])
        let routed = FlintRouterKernel.route(logits, expertBias: nanBias, k: 2)
        let routedSoftmax = FlintRouterKernel.route(logits, expertBias: MLXArray([Float(0.2), 0.1, 0.0, -0.1], [4]), k: 2, scoreFunction: "softmax")
        eval(routed.indices, routed.weights, routedSoftmax.indices, routedSoftmax.weights)
        let weights = routed.weights.asArray(Float.self)
        let allFinite = weights.allSatisfy { $0.isFinite }
        #expect(allFinite)
        #expect(routedSoftmax.indices.shape == [1, 2])
    }
}

@Suite("Retrospective Wave 3 server, speech, GPULock, and OnDemand lifecycle (#557)")
struct RetroWave3RuntimeTests {
    @Test("AudioSpeechController rejects arbitrary HTTP reference_audio paths and Zonos3MiniConfig validates positive dimensions")
    func audioSpeechReferenceConfinementAndZonosValidation() throws {
        let payload = Data(#"{"model":"zonos","input":"hello","reference_audio":"/tmp/secret.wav"}"#.utf8)
        #expect(throws: BasaltError.self) {
            _ = try AudioSpeechController.parseRequest(payload)
        }
        let allowedDir = FileManager.default.temporaryDirectory.appending(path: "speech-ref-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: allowedDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: allowedDir) }
        let safeFile = allowedDir.appending(component: "voice.wav")
        try Data([0, 1, 2, 3]).write(to: safeFile)
        let safePayload = try JSONSerialization.data(withJSONObject: [
            "model": "zonos", "input": "hello", "reference_audio": safeFile.path,
        ])
        let parsed = try AudioSpeechController.parseRequest(safePayload, allowedReferenceDirectories: [allowedDir])
        #expect(parsed.referenceAudio == MCPImages.resolved(safeFile).path)

        var badZonos = Zonos3MiniConfig.tiny
        badZonos.lm.nKVHeads = 0
        #expect(throws: BasaltError.self) {
            _ = try badZonos.validated()
        }
    }

    @Test("ChatRequestFastReader rejects malformed content blocks and non-string guided_regex and reads reasoning_content")
    func fastReaderValidationAndReasoningContent() throws {
        let badContent = Data(#"{"messages":[{"role":"user","content":[123]}]}"#.utf8)
        #expect(throws: ChatRequestFastReader.DecodeError.self) {
            _ = try ChatRequestFastReader.read(badContent)
        }
        let badGuided = Data(#"{"messages":[{"role":"user","content":"hi"}],"guided_regex":42}"#.utf8)
        let parsedBadGuided = try ChatRequestFastReader.read(badGuided)
        #expect(throws: OpenAIController.InvalidResponseFormat.self) {
            _ = try parsedBadGuided.responseFormat.get()
        }
        let reasoning = Data(#"{"messages":[{"role":"assistant","content":"4","reasoning_content":"2+2=4"}]}"#.utf8)
        let parsedReasoning = try ChatRequestFastReader.read(reasoning)
        #expect(parsedReasoning.dto.messages.first?.reasoning == "2+2=4")
    }

    @Test("GPULockLease deinit unlocks and closes descriptor when dropped without explicit release")
    func gpuLockDeinitReleasesDescriptor() throws {
        let path = FileManager.default.temporaryDirectory.appending(path: "gpu-lock-\(UUID().uuidString).lock").path
        defer { try? FileManager.default.removeItem(atPath: path) }
        do {
            let first = try GPULock.tryAcquire(path: path)
            let held1 = first.isHeld
            #expect(held1)
        }
        let second = try GPULock.tryAcquire(path: path)
        let held2 = second.isHeld
        #expect(held2)
        second.release()
    }

    struct StubImageSynth: ImageSynthesizer {
        let modelIdentifier = "stub-img"
        let descriptor = ImageModelDescriptor.flux2Klein4B(id: "stub-img")

        func generate(_ request: ImageGenerationRequest) async throws -> AsyncThrowingStream<ImageGenerationEvent, any Error> {
            AsyncThrowingStream { cont in
                let art = ImageCompletedArtifact(
                    imageData: Data([1, 2, 3, 4]),
                    format: .png,
                    dimensions: ImageDimensions(width: request.width, height: request.height),
                    model: modelIdentifier)
                cont.yield(.completed(art))
                cont.finish()
            }
        }
    }

    struct StubSpeechSynth: SpeechSynthesizer {
        var voices: [VoiceSpec] {
            [VoiceSpec(id: "savannah", name: "Savannah", language: "en_us", model: "stub-tts", compute: .mlx)]
        }

        func synthesize(_ request: SpeechRequest) async throws(BasaltError) -> AsyncThrowingStream<AudioChunk, any Error> {
            AsyncThrowingStream { cont in
                cont.yield(AudioChunk(samples: [0.1, -0.1, 0.2], sampleRate: 24_000, isFinal: true))
                cont.finish()
            }
        }
    }

    @Test("OnDemandImageSynthesizer loads, streams, expires on idle, and unloads cleanly")
    func onDemandImageLifecycle() async throws {
        let (imgIdleStream, imgIdleCont) = AsyncStream<Void>.makeStream()
        let img = OnDemandImageSynthesizer(
            modelIdentifier: "stub-img",
            path: "/tmp/stub-img",
            descriptor: .flux2Klein4B(id: "stub-img"),
            idle: .seconds(1),
            loader: { StubImageSynth() },
            sleep: { _ in
                for await _ in imgIdleStream { break }
            })
        let imgReq = try ImageGenerationRequest(model: "stub-img", prompt: "mountain")
        var imgEvents = 0
        for try await _ in try await img.generate(imgReq) { imgEvents += 1 }
        #expect(imgEvents == 1)
        #expect(await img.phase == .loaded)
        await img.unload()
        #expect(await img.phase == .unloaded)

        for try await _ in try await img.generate(imgReq) {}
        imgIdleCont.yield(())
        imgIdleCont.finish()
        for await phase in img.phases where phase == .unloaded { break }
        #expect(await img.phase == .unloaded)
    }

    @Test("OnDemandSpeechSynthesizer and OnDemandModel.setAdapters load, stream, expire, and unload cleanly")
    func onDemandSpeechAndModelAdapters() async throws {
        let (ttsIdleStream, ttsIdleCont) = AsyncStream<Void>.makeStream()
        let ttsDesc = OnDemandSpeechSynthesizer.Descriptor(
            modelID: "stub-tts",
            path: "/tmp/stub-tts",
            candidate: SpeechCandidate(id: "stub-tts", model: "stub-tts", compute: .mlx),
            declaredVoices: [VoiceSpec(id: "savannah", name: "Savannah", language: "en_us", model: "stub-tts", compute: .mlx)],
            idle: .seconds(1))
        let tts = OnDemandSpeechSynthesizer(
            descriptor: ttsDesc,
            loader: { StubSpeechSynth() },
            sleep: { _ in
                for await _ in ttsIdleStream { break }
            })
        let ttsReq = try SpeechRequest(model: "stub-tts", input: "hello", voice: "savannah")
        var chunks = 0
        for try await _ in try await tts.synthesize(ttsReq) { chunks += 1 }
        #expect(chunks == 1)
        await tts.unload()
        #expect(await tts.phase == .unloaded)

        for try await _ in try await tts.synthesize(ttsReq) {}
        ttsIdleCont.yield(())
        ttsIdleCont.finish()
        for await phase in tts.phases where phase == .unloaded { break }
        #expect(await tts.phase == .unloaded)

        let mockOrch = ModelOrchestratorActor(engine: MockInferenceEngine())
        let odModel = OnDemandModel(model: mockOrch, path: "mock", idle: .seconds(60)) { _ in }
        try await odModel.setAdapters([:])
        _ = try await odModel.acquire()
        await #expect(throws: BasaltError.self) {
            try await odModel.setAdapters(["lora": URL(fileURLWithPath: "/tmp/lora")])
        }
        await odModel.unload()
    }
}
