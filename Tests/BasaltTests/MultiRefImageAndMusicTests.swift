import Foundation
import Hummingbird
import HummingbirdTesting
import MLX
import Testing
@testable import BasaltAppCore
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Multi-Reference DiT KV-Cache, RGBA, SeedVR2 1-Step Upscaler, ACE-Step 1.5 & DiT Converter (#576)")
struct MultiRefImageAndMusicTests {
    private static func makeSyntheticRGBAImagePNG(width: Int = 64, height: Int = 64, transparentCorner: Bool = true) throws -> Data {
        var pixels = [Float](repeating: 0, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let idx = (y * width + x) * 4
                pixels[idx] = Float(x) / Float(max(1, width - 1))
                pixels[idx + 1] = Float(y) / Float(max(1, height - 1))
                pixels[idx + 2] = 0.65
                let isCorner = transparentCorner && (x < width / 4 && y < height / 4)
                pixels[idx + 3] = isCorner ? 0.12 : 0.95
            }
        }
        let encoded = try ImagePNGCodec.encodePixels(
            pixels,
            dimensions: ImageDimensions(width: width, height: height),
            channels: 4,
            format: .png,
            forceOpaque: false
        )
        return encoded.data
    }

    @Test("FLUX 3 Klein & FLUX.2 klein-9b-kv reference KV-cache matches uncached velocity within 1e-3 and reuses KV on follow-up edits")
    func referenceKVCacheParityAndReuse() async throws {
        let ref1 = try Self.makeSyntheticRGBAImagePNG(width: 64, height: 64, transparentCorner: true)
        let ref2 = try Self.makeSyntheticRGBAImagePNG(width: 64, height: 64, transparentCorner: false)
        let synth = FluxKleinSynthesizer(
            descriptor: .flux2Klein9BKV(),
            config: .tiny(modelID: "black-forest-labs/FLUX.2-klein-9B-kv", hiddenSize: 64)
        )
        let (cachedVel, uncachedVel) = try await synth.debugReferenceKVCacheParity(
            referenceImages: [ref1, ref2],
            prompt: "Transform basalt column lighting to aurora cyan",
            seed: 42
        )
        #expect(cachedVel.count == uncachedVel.count && !cachedVel.isEmpty)
        let maxAbsDiff = zip(cachedVel, uncachedVel).map { abs($0 - $1) }.max() ?? 1.0
        #expect(maxAbsDiff < 1e-3)

        var cfg1 = ImageGenerationConfig(guidanceScale: 2.5, seed: 7, background: .transparent)
        cfg1.referenceImages = [ref1, ref2]
        cfg1.reuseReferenceKV = true
        let req1 = try ImageGenerationRequest(
            model: synth.modelIdentifier,
            prompt: "First edit pass",
            dimensions: ImageDimensions(width: 64, height: 64),
            steps: 2,
            outputFormat: .png,
            config: cfg1
        )
        var out1: ImageCompletedArtifact?
        for try await ev in try await synth.generate(req1) {
            if case .completed(let art) = ev { out1 = art }
        }
        let firstArtifact = try #require(out1)
        #expect(firstArtifact.referenceKVReused == false)
        #expect(firstArtifact.hasAlpha == true)

        let req2 = try ImageGenerationRequest(
            model: synth.modelIdentifier,
            prompt: "Follow-up edit reusing cached reference KV",
            dimensions: ImageDimensions(width: 64, height: 64),
            steps: 2,
            outputFormat: .png,
            config: cfg1
        )
        var out2: ImageCompletedArtifact?
        for try await ev in try await synth.generate(req2) {
            if case .completed(let art) = ev { out2 = art }
        }
        let secondArtifact = try #require(out2)
        #expect(secondArtifact.referenceKVReused == true)
        let stats = await synth.referenceCacheStats()
        #expect(stats.hits == 1 && stats.misses == 1 && stats.entries == 1)
    }

    @Test("4-channel RGBA PNG output preserves transparency and AppImageCanvas retains up to 10 reference images")
    func nativeRGBAOutputAndTenReferenceCanvas() async throws {
        let qwenSynth = S3DiTSynthesizer(
            descriptor: .qwenImage21(),
            config: .tiny(modelID: "Qwen/Qwen-Image-2.1-Turbo", hiddenSize: 64)
        )
        var cfg = ImageGenerationConfig(guidanceScale: 2.0, seed: 21, background: .transparent)
        cfg.bboxLayout = [ImageBoundingBox(label: "Crystal", x: 0.2, y: 0.2, width: 0.6, height: 0.6)]
        let req = try ImageGenerationRequest(
            model: qwenSynth.modelIdentifier,
            prompt: "Floating basalt crystal on transparent background",
            dimensions: ImageDimensions(width: 64, height: 64),
            steps: 2,
            outputFormat: .png,
            config: cfg
        )
        var completed: ImageCompletedArtifact?
        for try await ev in try await qwenSynth.generate(req) {
            if case .completed(let art) = ev { completed = art }
        }
        let artifact = try #require(completed)
        #expect(artifact.hasAlpha == true)
        let decoded = try ImagePNGCodec.decodePixels(artifact.imageData)
        #expect(decoded.channels == 4 && decoded.width == 64 && decoded.height == 64)
        #expect(decoded.hasTransparency == true)

        var chatState = ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "black-forest-labs/FLUX.3-Image-Klein"))
        for idx in 0 ..< 12 {
            var part = GeneratedImagePart(assetID: "ref-\(idx).png", prompt: "Ref \(idx)", width: 64, height: 64)
            part.filePath = "/tmp/ref-\(idx).png"
            part.previewPNG = artifact.imageData
            _ = ChatReducer.reduce(&chatState, .editGeneratedImage(part))
        }
        let imageAttachments = chatState.attachments.filter { $0.kind == "image" }
        #expect(imageAttachments.count == 10)
        #expect(imageAttachments.first?.sourcePath == "/tmp/ref-2.png")
        #expect(imageAttachments.last?.sourcePath == "/tmp/ref-11.png")
    }

    @Test("SeedVR2OneStepUpscaler stitches overlapping tiles seamlessly and preserves 4-channel RGBA")
    func seedVR2TileStitchingSeamFree() async throws {
        let upscaler = SeedVR2OneStepUpscaler(
            config: SeedVR2Config(hiddenSize: 32, numHeads: 4, numLayers: 1, defaultTileSize: 32, defaultOverlap: 8)
        )
        let inH = 64
        let inW = 64
        var vals = [Float](repeating: 0, count: inH * inW * 4)
        for y in 0 ..< inH {
            for x in 0 ..< inW {
                let base = (y * inW + x) * 4
                vals[base] = 0.25 + 0.5 * Float(x) / Float(inW - 1)
                vals[base + 1] = 0.20 + 0.6 * Float(y) / Float(inH - 1)
                vals[base + 2] = 0.45
                vals[base + 3] = x < 16 ? 0.35 : 0.95
            }
        }
        let tiledOut = try await upscaler.upscalePixels(vals, shape: [1, inH, inW, 4], scaleFactor: 2, tileSize: 32, overlap: 8)
        let singleOut = try await upscaler.upscalePixels(vals, shape: [1, inH, inW, 4], scaleFactor: 2, tileSize: 128, overlap: 8)
        let tiled = MLXArray(tiledOut.pixels, tiledOut.shape)
        let singleTile = MLXArray(singleOut.pixels, singleOut.shape)
        eval(tiled, singleTile)
        #expect(tiled.shape == [1, 128, 128, 4])

        let diff = abs(tiled - singleTile).max()
        eval(diff)
        #expect(diff.item(Float.self) < 0.05)

        let seamColLeft = tiled[0, 0..., 63, 0...].asArray(Float.self)
        let seamColRight = tiled[0, 0..., 64, 0...].asArray(Float.self)
        let maxSeamJump = zip(seamColLeft, seamColRight).map { abs($0 - $1) }.max() ?? 1.0
        #expect(maxSeamJump < 0.08)
    }
}

@Suite("ACE-Step 1.5 Music DiT & Universal DiT Converter Parity (#576)")
struct ACEStepMusicAndDiTConverterTests {
    @Test("ACEStep15MusicSynthesizer and POST /v1/audio/music emit valid WAV and SSE audio events")
    func aceStepMusicSynthesizerAndREST() async throws {
        let musicSynth = ACEStep15MusicSynthesizer(
            config: ACEStep15Config(hiddenSize: 32, numHeads: 4, numLayers: 1, sampleRate: 24_000)
        )
        let req = try MusicGenerationRequest(
            prompt: "Ambient synthwave over volcanic basalt coast",
            durationSeconds: 0.5,
            responseFormat: .wav,
            options: MusicGenerationOptions(lyrics: "Waves of obsidian light", tags: ["synthwave", "ambient"], bpm: 110, steps: 2, seed: 42)
        )
        var samples: [Float] = []
        for try await chunk in try await musicSynth.synthesize(req) {
            samples.append(contentsOf: chunk.samples)
        }
        #expect(samples.count == 12_000)
        let peak = samples.map { abs($0) }.max() ?? 0
        #expect(peak > 0.01 && peak <= 1.0)

        let router = Router(context: BasicRequestContext.self)
        AudioMusicController(synthesizer: musicSynth).add(to: router.group("/v1"))
        try await Application(router: router).test(.router) { client in
            let wavBody = Data("""
            {"prompt":"Lo-fi chill beat","duration_seconds":0.4,"bpm":96,"steps":2,"seed":9,"response_format":"wav"}
            """.utf8)
            try await client.execute(
                uri: "/v1/audio/music",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(data: wavBody)
            ) { response in
                #expect(response.status == .ok)
                #expect(response.headers[.contentType] == "audio/wav")
                let bytes = Data(response.body.readableBytesView)
                #expect(bytes.prefix(4) == Data("RIFF".utf8) && bytes.count > 44)
            }

            let sseBody = Data("""
            {"prompt":"Acoustic guitar motif","duration_seconds":0.3,"steps":2,"seed":5,"stream_format":"sse"}
            """.utf8)
            try await client.execute(
                uri: "/v1/audio/music",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(data: sseBody)
            ) { response in
                #expect(response.status == .ok)
                let text = String(decoding: response.body.readableBytesView, as: UTF8.self)
                #expect(text.contains("event: music.delta") && text.contains("event: music.done"))
            }
        }
    }

    @Test("UniversalModelConverter converts DiT checkpoints to MACaQ MXFP4 (B=32), INT8 SINQ, and back to Dense BF16")
    func universalConverterDiTParity() async throws {
        let tmpRoot = FileManager.default.temporaryDirectory.appending(path: "dit-conv-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmpRoot) }
        let srcDir = tmpRoot.appending(component: "flux3-src")
        let macaqDir = tmpRoot.appending(component: "flux3-macaq")
        let denseDir = tmpRoot.appending(component: "flux3-dense")
        let sinqDir = tmpRoot.appending(component: "seedvr2-sinq")
        try FileManager.default.createDirectory(at: srcDir, withIntermediateDirectories: true)

        let cfgData = try JSONSerialization.data(
            withJSONObject: ["model_type": "flux3_klein", "hidden_size": 64, "num_attention_heads": 4, "num_layers": 1],
            options: [.sortedKeys]
        )
        try cfgData.write(to: srcDir.appending(component: "config.json"))
        let w1 = MLXRandom.normal([64, 64], dtype: .bfloat16, key: MLXRandom.key(11))
        let w2 = MLXRandom.normal([64, 64], dtype: .bfloat16, key: MLXRandom.key(12))
        let normW = MLXArray.ones([64], dtype: .bfloat16)
        eval(w1, w2, normW)
        try MLX.save(
            arrays: [
                "dual_blocks.0.img_qkv.weight": w1,
                "dual_blocks.0.img_out.weight": w2,
                "final_norm.weight": normW,
            ],
            url: srcDir.appending(component: "model.safetensors")
        )
        #expect(DiTBlockQuantizer.isDiTCheckpoint(srcDir))

        var macaqReq = ModelConversionRequest(source: srcDir.path, outputId: "local/flux3-macaq", target: .macaqMXFP4, pvTune: false)
        macaqReq.hadamardBlock = 32
        let macaqRes = try await UniversalModelConverter.convertToDirectory(
            request: macaqReq, sourceURL: srcDir, outputURL: macaqDir
        ) { _ in }
        #expect(macaqRes.shards == 1 && macaqRes.bytesAfter > 0)
        let manifest = try #require(try MACaQNativeLoader.manifest(in: macaqDir))
        #expect(manifest.modules["dual_blocks.0.img_qkv"]?.hadamardBlock == 32)
        #expect(manifest.modules["dual_blocks.0.img_qkv"]?.mode == "mxfp4")

        let denseReq = ModelConversionRequest(source: macaqDir.path, outputId: "local/flux3-dense", target: .denseBF16, pvTune: false)
        let denseRes = try await UniversalModelConverter.convertToDirectory(
            request: denseReq, sourceURL: macaqDir, outputURL: denseDir
        ) { _ in }
        #expect(denseRes.shards == 1)
        let restored = try MLX.loadArrays(url: denseDir.appending(component: "model.safetensors"))
        #expect(restored["dual_blocks.0.img_qkv.weight"]?.shape == [64, 64])

        let sinqReq = ModelConversionRequest(
            source: srcDir.path, outputId: "local/seedvr2-int8",
            target: .affine(bits: 8, groupSize: 64), recipe: "sinq", pvTune: false
        )
        let sinqRes = try await UniversalModelConverter.convertToDirectory(
            request: sinqReq, sourceURL: srcDir, outputURL: sinqDir
        ) { _ in }
        #expect(sinqRes.shards == 1 && sinqRes.bytesAfter > 0)

        let seedDraft = ConversionDraft(sourceId: "ByteDance-Seed/SeedVR2-3B", sourceBytes: 6_000_000_000)
        #expect(seedDraft.resolvedTarget == .affine(bits: 8, groupSize: 64) && seedDraft.recipe == "sinq")
        let fluxDraft = ConversionDraft(sourceId: "black-forest-labs/FLUX.3-Image-Klein", sourceBytes: 11_000_000_000)
        #expect(fluxDraft.resolvedTarget == .macaqMXFP4 && fluxDraft.hadamardBlock == 32)
    }
}
