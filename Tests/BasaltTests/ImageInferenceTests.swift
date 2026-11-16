import Testing
import Foundation
import MLX
@testable import BasaltCore
@testable import BasaltInference

private enum ImageTestFixtures {
    static func collectEvents(
        _ synthesizer: any ImageSynthesizer,
        _ request: ImageGenerationRequest
    ) async throws -> (partials: [ImagePartialFrame], completed: ImageCompletedArtifact) {
        var partials: [ImagePartialFrame] = []
        var completed: ImageCompletedArtifact?
        for try await event in try await synthesizer.generate(request) {
            switch event {
            case .partial(let frame):
                partials.append(frame)
            case .completed(let artifact):
                completed = artifact
            }
        }
        guard let finalArtifact = completed else {
            throw BasaltError.inferenceError("missing completed artifact")
        }
        return (partials, finalArtifact)
    }

    static func maxAbsDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        var peak: Float = 0
        for (a, b) in zip(lhs, rhs) {
            peak = max(peak, abs(a - b))
        }
        return peak
    }

    static func meanL1Diff(_ lhs: [Float], _ rhs: [Float]) -> Float {
        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return 0 }
        var sum: Float = 0
        for idx in 0 ..< count {
            sum += abs(lhs[idx] - rhs[idx])
        }
        return sum / Float(count)
    }

    static func makeSolidOrGradientPNG(width: Int, height: Int, r: Float, g: Float, b: Float, alpha: Float = 1.0) throws -> Data {
        var pixels = [Float](repeating: 0, count: width * height * 4)
        for idx in 0 ..< (width * height) {
            pixels[idx * 4] = r
            pixels[idx * 4 + 1] = g
            pixels[idx * 4 + 2] = b
            pixels[idx * 4 + 3] = alpha
        }
        return try ImagePNGCodec.encodePixels(
            pixels,
            dimensions: ImageDimensions(width: width, height: height),
            channels: 4,
            format: .png
        ).data
    }

    static func makeHalfSplitMaskPNG(width: Int, height: Int) throws -> Data {
        var pixels = [Float](repeating: 1, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let v: Float = x < (width / 2) ? 0.0 : 1.0
                let base = (y * width + x) * 4
                pixels[base] = v
                pixels[base + 1] = v
                pixels[base + 2] = v
                pixels[base + 3] = 1.0
            }
        }
        return try ImagePNGCodec.encodePixels(
            pixels,
            dimensions: ImageDimensions(width: width, height: height),
            channels: 4,
            format: .png
        ).data
    }

    static func makeSyntheticPCMBytes(frequency: Double, amplitude: Double) -> Data {
        var bytes = Data(capacity: 4_800)
        for i in 0 ..< 2_400 {
            let sample = amplitude * sin(2.0 * .pi * frequency * Double(i) / 16_000.0)
            let pcm = Int16(max(-32_767.0, min(32_767.0, (sample * 32_767.0).rounded())))
            let bits = UInt16(bitPattern: pcm)
            bytes.append(UInt8(bits & 0xff))
            bytes.append(UInt8((bits >> 8) & 0xff))
        }
        return bytes
    }
}

@Suite("ImageModelDescriptor license tiers, quantization floor warnings, and codec round-trips (#517)")
struct ImageDescriptorAndCodecTests {
    @Test("ImageModelDescriptor distinguishes Tier 1 Apache-2.0/MIT from Tier 2 non-commercial and 8-bit floor warnings")
    func descriptorTiersAndQuantizationWarnings() {
        let tier1IDs = [
            "black-forest-labs/FLUX.2-klein-4B",
            "Tongyi-MAI/Z-Image-Turbo",
            "Speridlabs/Iris-3B-Turbo",
            "HiDream-ai/HiDream-O1-Image",
            "Efficient-Large-Model/SANA-1.5-1.6B",
            "Bonsai/Bonsai-Image-4B",
            "inclusionAI/Ming-Image-0.1-Design",
        ]
        for modelID in tier1IDs {
            let desc = ImageModelDescriptor.detect(identifier: modelID)
            #expect(desc.licenseTier == .permissiveCommercial)
            #expect(desc.licenseTier.allowsUnrestrictedCommercialUse)
        }

        let klein9B = ImageModelDescriptor.detect(identifier: "black-forest-labs/FLUX.2-klein-9B")
        #expect(klein9B.licenseTier == .nonCommercialResearch && !klein9B.licenseTier.allowsUnrestrictedCommercialUse)

        let flux3 = ImageModelDescriptor.detect(identifier: "black-forest-labs/FLUX.3-Image-Klein")
        #expect(flux3.family == .flux3Klein && flux3.licenseTier == .revenueCappedCommercial)
        #expect(!flux3.licenseTier.allowsUnrestrictedCommercialUse && flux3.minimumRecommendedBits == 8)
        let warn4Bit = flux3.estimateFootprint(quantizationBits: 4)
        let full8Bit = flux3.estimateFootprint(quantizationBits: 8, sharedTextEncoderReused: false)
        let clean8Bit = flux3.estimateFootprint(quantizationBits: 8, sharedTextEncoderReused: true)
        #expect(warn4Bit.qualityWarning != nil)
        #expect(clean8Bit.qualityWarning == nil)
        #expect(clean8Bit.peakResidentBytes == full8Bit.peakResidentBytes - full8Bit.textEncoderBytes)

        let qwen21 = ImageModelDescriptor.detect(identifier: "Qwen/Qwen-Image-2.1-Turbo")
        let fibo2 = ImageModelDescriptor.detect(identifier: "briaai/FIBO-2.0-Edit-Turbo")
        #expect(qwen21.licenseTier == .nonCommercialResearch && qwen21.supportsNativeRGBA)
        #expect(fibo2.licenseTier == .nonCommercialResearch && fibo2.family == .fiboPerceiver)

        let irisFootprint = ImageModelDescriptor.iris3BTurbo().estimateFootprint(quantizationBits: 4)
        #expect(irisFootprint.vaeBytes == 0)
    }

    @Test("ImagePNGCodec encodes and bilinearly resizes PNG, JPEG, WebP fallback, and RGBA alpha tensors")
    func codecRoundTripsFormatsAndAlpha() throws {
        var rgba = [Float](repeating: 0, count: 64 * 64 * 4)
        for y in 0 ..< 64 {
            for x in 0 ..< 64 {
                let idx = (y * 64 + x) * 4
                rgba[idx] = Float(x) / 63.0
                rgba[idx + 1] = Float(y) / 63.0
                rgba[idx + 2] = 0.5
                rgba[idx + 3] = x < 16 ? 0.2 : 1.0
            }
        }
        let tensor = MLXArray(rgba, [64, 64, 4])
        let png = try ImagePNGCodec.encode(tensor, format: .png)
        #expect(png.hasAlpha && png.format == .png && !png.data.isEmpty)
        let decodedPNG = try ImagePNGCodec.decodePixels(png.data)
        #expect(decodedPNG.width == 64 && decodedPNG.height == 64 && decodedPNG.hasTransparency)
        #expect(decodedPNG.rgba(x: 4, y: 32).a < 0.35)
        #expect(decodedPNG.rgba(x: 48, y: 32).a > 0.95)

        let jpeg = try ImagePNGCodec.encode(tensor, format: .jpeg)
        #expect(!jpeg.hasAlpha && jpeg.format == .jpeg)
        let webpOrPNG = try ImagePNGCodec.encode(tensor, format: .webp)
        #expect(!webpOrPNG.data.isEmpty)
        let resized = try ImagePNGCodec.decode(webpOrPNG.data, targetWidth: 128, targetHeight: 96)
        #expect(resized.shape == [96, 128, 4])
    }

    @Test("MLXInferenceEngine.encodePromptForImage provides deterministic unloaded fallback and layout embeddings")
    func sharedTextEncoderBridgeDeterminism() async throws {
        let engine = MLXInferenceEngine()
        let v1 = try await engine.encodePromptForImage("cyberpunk neon alley in rain", dimension: 128)
        let v2 = try await engine.encodePromptForImage("cyberpunk neon alley in rain", dimension: 128)
        let v3 = try await engine.encodePromptForImage("watercolor mountain sunrise", dimension: 128)
        #expect(v1.count == 128 && v1 == v2 && v1 != v3)
    }
}

@Suite("Tiny-fixture #228 reference parity and seed reproducibility across FLUX, S3-DiT, PixelDiT, SANA, and Bonsai (#517)", .serialized)
struct ImageSynthesizerParityTests {
    @Test("FluxKleinSynthesizer matches explicit reference forward pass (#228) and reproduces seeded 64x64 and 128x128 outputs")
    func fluxKleinParityAndReproducibility() async throws {
        let synth = FluxKleinSynthesizer(
            descriptor: .flux2Klein4B(),
            config: .tiny(modelID: "black-forest-labs/FLUX.2-klein-4B", hiddenSize: 64)
        )
        let parity = try await synth.debugVelocityParity(seed: 42)
        #expect(!parity.mlxVelocity.isEmpty && ImageTestFixtures.maxAbsDiff(parity.mlxVelocity, parity.referenceVelocity) < 1e-4)

        let cfg = ImageGenerationConfig(guidanceScale: 2.5, seed: 777)
        let req64 = try ImageGenerationRequest(
            model: synth.modelIdentifier, prompt: "minimalist geometric sculpture",
            dimensions: ImageDimensions(width: 64, height: 64), steps: 3, config: cfg
        )
        let runA = try await ImageTestFixtures.collectEvents(synth, req64)
        let runB = try await ImageTestFixtures.collectEvents(synth, req64)
        #expect(runA.completed.imageData == runB.completed.imageData)
        #expect(runA.completed.seed == 777 && runA.completed.licenseTier == .permissiveCommercial)

        let req128 = try ImageGenerationRequest(
            model: synth.modelIdentifier, prompt: "minimalist geometric sculpture",
            dimensions: ImageDimensions(width: 128, height: 128), steps: 2, config: cfg
        )
        let out128 = try await ImageTestFixtures.collectEvents(synth, req128)
        let decoded128 = try ImagePNGCodec.decodePixels(out128.completed.imageData)
        #expect(decoded128.width == 128 && decoded128.height == 128)
    }

    @Test("S3DiTSynthesizer matches explicit single-stream reference pass (#228) and supports FIBO Perceiver and Step1X")
    func s3DiTParityAndLoaderVariants() async throws {
        let s3 = S3DiTSynthesizer(
            descriptor: .zImageTurbo(),
            config: .tiny(modelID: "Tongyi-MAI/Z-Image-Turbo", hiddenSize: 64)
        )
        let parity = try await s3.debugVelocityParity(seed: 77)
        #expect(!parity.mlxVelocity.isEmpty && ImageTestFixtures.maxAbsDiff(parity.mlxVelocity, parity.referenceVelocity) < 1e-4)

        let dir = FileManager.default.temporaryDirectory.appending(path: "img-fibo-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"hidden_size":48,"num_attention_heads":4,"num_layers":2}"#.utf8)
            .write(to: dir.appending(component: "config.json"))

        let fibo = try await ImageSynthesizerLoader.load(identifier: "briaai/FIBO-2.0-Edit-Turbo", directory: dir)
        var fiboCfg = ImageGenerationConfig(guidanceScale: 2.0, seed: 99)
        fiboCfg.vglSceneJSON = #"{"lighting":"golden_hour","camera":"85mm"}"#
        let fiboReq = try ImageGenerationRequest(
            model: fibo.modelIdentifier, prompt: "studio portrait",
            dimensions: ImageDimensions(width: 64, height: 64), steps: 2, config: fiboCfg
        )
        let fiboOut = try await ImageTestFixtures.collectEvents(fibo, fiboReq)
        #expect(fiboOut.completed.licenseTier == .nonCommercialResearch && !fiboOut.completed.imageData.isEmpty)
    }

    @Test("PixelDiTSynthesizer verifies Iris-3B pixel head, SANA O(N) ReLU linear attention, and Bonsai ternary {-1,0,+1} parity (#228)")
    func pixelDiTSanaAndBonsaiParity() async throws {
        let iris = PixelDiTSynthesizer(descriptor: .iris3BTurbo(), config: .tiny(modelID: "Speridlabs/Iris-3B-Turbo"))
        let irisParity = try await iris.debugPixelDiTParity(seed: 101)
        #expect(ImageTestFixtures.maxAbsDiff(irisParity.mlxVelocity, irisParity.referenceVelocity) < 1e-4)
        let irisProbe = try await ImageSynthesizerLoader.probeImageSynthesizer(iris)
        #expect(irisProbe.width == 64 && !iris.descriptor.usesVAE)

        let sana = PixelDiTSynthesizer(descriptor: .sana15(), config: .tiny(modelID: "Efficient-Large-Model/SANA-1.5-1.6B"))
        let sanaParity = try await sana.debugLinearSanaParity(seed: 202)
        #expect(ImageTestFixtures.maxAbsDiff(sanaParity.linearOutput, sanaParity.quadraticReference) < 1e-4)
        let sanaReq = try ImageGenerationRequest(
            model: sana.modelIdentifier, prompt: "futuristic botanical dome",
            dimensions: ImageDimensions(width: 80, height: 64), steps: 2, config: ImageGenerationConfig(seed: 12)
        )
        let sanaOut = try await ImageTestFixtures.collectEvents(sana, sanaReq)
        #expect(sanaOut.completed.compute == .mlxANEHybrid && sanaOut.completed.width == 80)

        let bonsai = PixelDiTSynthesizer(descriptor: .bonsaiTernary4B(), config: .tiny(modelID: "Bonsai/Bonsai-Image-4B"))
        let ternary = try await bonsai.debugTernaryBitNetParity(seed: 303)
        #expect(!ternary.codes.isEmpty && ternary.codes.allSatisfy { $0 == -1.0 || $0 == 0.0 || $0 == 1.0 })
        #expect(ImageTestFixtures.maxAbsDiff(ternary.mlxOutput, ternary.cpuReference) < 1e-4)
        let bonsaiProbe = try await ImageSynthesizerLoader.probeImageSynthesizer(bonsai)
        #expect(!bonsaiProbe.imageData.isEmpty)
    }
}

@Suite("TeaCache step skipping, TAEF partial preview streaming, and native 4-channel RGBA synthesis (#517)", .serialized)
struct TeaCachePreviewAndAlphaTests {
    @Test("TeaCacheScheduler skips interior steps on 12-step run while preserving step 0 and final step")
    func teacacheSkipsInteriorStepsDeterministically() async throws {
        var cache = TeaCacheState(totalSteps: 12, threshold: 0.2)
        let base = MLXArray((0 ..< 64).map { Float($0) * 0.05 + 1.0 }, [1, 8, 8])
        for s in 0 ..< 12 {
            let current = base * (1.0 + Float(s) * 0.04)
            _ = cache.resolveVelocity(step: s, modulatedInput: current) {
                current * 0.1
            }
        }
        #expect(cache.skippedSteps > 0)
        #expect(cache.executedStepIndices.contains(0))
        #expect(cache.executedStepIndices.contains(11))
        #expect(!cache.skippedStepIndices.contains(0) && !cache.skippedStepIndices.contains(11))

        let synth = FluxKleinSynthesizer(descriptor: .flux2Klein4B(), config: .tiny())
        let noSkipReq = try ImageGenerationRequest(
            model: synth.modelIdentifier, prompt: "architectural concrete pavilion",
            dimensions: ImageDimensions(width: 64, height: 64), steps: 12,
            config: ImageGenerationConfig(guidanceScale: 2.0, seed: 55, teacacheThreshold: 0.0)
        )
        let noSkipOut = try await ImageTestFixtures.collectEvents(synth, noSkipReq)
        #expect(noSkipOut.completed.teacacheSkippedSteps == 0 && noSkipOut.completed.stepsExecuted == 12)

        let skipReq = try ImageGenerationRequest(
            model: synth.modelIdentifier, prompt: "architectural concrete pavilion",
            dimensions: ImageDimensions(width: 64, height: 64), steps: 12,
            config: ImageGenerationConfig(guidanceScale: 2.0, seed: 55, teacacheThreshold: 0.2)
        )
        let skipOut = try await ImageTestFixtures.collectEvents(synth, skipReq)
        #expect(skipOut.completed.teacacheSkippedSteps > 0)
        #expect(skipOut.completed.stepsExecuted + skipOut.completed.teacacheSkippedSteps == 12)
    }

    @Test("previewEverySteps emits TAEF ImagePartialFrame events prior to final ImageCompletedArtifact")
    func previewStreamingEmitsPartialFrames() async throws {
        let synth = S3DiTSynthesizer(descriptor: .zImageTurbo(), config: .tiny())
        let cfg = ImageGenerationConfig(guidanceScale: 2.0, seed: 88, previewEverySteps: 2)
        let req = try ImageGenerationRequest(
            model: synth.modelIdentifier, prompt: "origami crane on wood desk",
            dimensions: ImageDimensions(width: 64, height: 64), steps: 8, config: cfg
        )
        var eventOrder: [String] = []
        var partialSteps: [Int] = []
        for try await event in try await synth.generate(req) {
            switch event {
            case .partial(let frame):
                eventOrder.append("partial")
                partialSteps.append(frame.step)
                let decoded = try ImagePNGCodec.decodePixels(frame.imageData)
                #expect(decoded.width == 64 && decoded.height == 64)
            case .completed:
                eventOrder.append("completed")
            }
        }
        #expect(partialSteps == [2, 4, 6])
        #expect(eventOrder == ["partial", "partial", "partial", "completed"])
    }

    @Test("Qwen-Image-2.1 and Ming-Image-0.1-Design synthesize native 4-channel transparent RGBA PNGs")
    func nativeRGBATransparencyRoundTrip() async throws {
        for desc in [ImageModelDescriptor.qwenImage21(), ImageModelDescriptor.mingImageDesign()] {
            let synth = S3DiTSynthesizer(descriptor: desc, config: .tiny(modelID: desc.id))
            let transCfg = ImageGenerationConfig(guidanceScale: 2.0, seed: 314, background: .transparent)
            let req = try ImageGenerationRequest(
                model: desc.id, prompt: "isolated glossy ceramic teacup sticker",
                dimensions: ImageDimensions(width: 64, height: 64), steps: 3, outputFormat: .png, config: transCfg
            )
            let result = try await ImageTestFixtures.collectEvents(synth, req)
            #expect(result.completed.hasAlpha)
            let decoded = try ImagePNGCodec.decodePixels(result.completed.imageData)
            #expect(decoded.hasTransparency)
            let cornerAlpha = decoded.rgba(x: 1, y: 1).a
            let centerAlpha = decoded.rgba(x: 32, y: 32).a
            #expect(cornerAlpha < 1.0 && cornerAlpha < 0.5)
            #expect(centerAlpha > cornerAlpha)

            let opaqueCfg = ImageGenerationConfig(guidanceScale: 2.0, seed: 314, background: .opaque)
            let opaqueReq = try ImageGenerationRequest(
                model: desc.id, prompt: "isolated glossy ceramic teacup sticker",
                dimensions: ImageDimensions(width: 64, height: 64), steps: 2, outputFormat: .png, config: opaqueCfg
            )
            let opaqueOut = try await ImageTestFixtures.collectEvents(synth, opaqueReq)
            #expect(!opaqueOut.completed.hasAlpha)
        }
    }
}

@Suite("Multi-reference I2I mask inpainting, BBox spatial modulation, and Audio-to-Image FlowEdit guidance (#517)", .serialized)
struct ImageEditingBBoxAndAudioTests {
    @Test("Multi-reference I2I with maskImage preserves unmasked reference pixels while editing masked region")
    func multiReferenceMaskInpaintingPreservesUnmaskedRegion() async throws {
        let refPrimary = try ImageTestFixtures.makeSolidOrGradientPNG(width: 64, height: 64, r: 0.12, g: 0.82, b: 0.24)
        let refSecondary = try ImageTestFixtures.makeSolidOrGradientPNG(width: 64, height: 64, r: 0.85, g: 0.15, b: 0.20)
        let halfMask = try ImageTestFixtures.makeHalfSplitMaskPNG(width: 64, height: 64)

        let synth = FluxKleinSynthesizer(descriptor: .flux3Klein(), config: .tiny(modelID: "black-forest-labs/FLUX.3-Image-Klein"))
        var cfg = ImageGenerationConfig(guidanceScale: 3.0, seed: 404)
        cfg.referenceImages = [refPrimary, refSecondary]
        cfg.maskImage = halfMask
        let req = try ImageGenerationRequest(
            model: synth.modelIdentifier, prompt: "replace right side with crimson sunset",
            dimensions: ImageDimensions(width: 64, height: 64), steps: 4, config: cfg
        )
        let out = try await ImageTestFixtures.collectEvents(synth, req)
        let decoded = try ImagePNGCodec.decodePixels(out.completed.imageData)
        let refDecoded = try ImagePNGCodec.decodePixels(refPrimary)

        let unmaskedOut = decoded.rgba(x: 10, y: 32)
        let unmaskedRef = refDecoded.rgba(x: 10, y: 32)
        #expect(abs(unmaskedOut.r - unmaskedRef.r) < 0.03)
        #expect(abs(unmaskedOut.g - unmaskedRef.g) < 0.03)
        #expect(abs(unmaskedOut.b - unmaskedRef.b) < 0.03)

        var maskedDelta: Float = 0
        for x in 36 ..< 60 {
            let p = decoded.rgba(x: x, y: 32)
            maskedDelta += abs(p.r - unmaskedRef.r) + abs(p.g - unmaskedRef.g) + abs(p.b - unmaskedRef.b)
        }
        #expect(maskedDelta > 0.25)
    }

    @Test("FLUX 3 BBox layout modulation concentrates spatial conditioning inside target bounding box")
    func bboxLayoutModulatesTargetRegion() async throws {
        let synth = FluxKleinSynthesizer(descriptor: .flux3Klein(), config: .tiny(modelID: "black-forest-labs/FLUX.3-Image-Klein"))
        let baseCfg = ImageGenerationConfig(guidanceScale: 2.5, seed: 909)
        var boxCfg = baseCfg
        boxCfg.bboxLayout = [ImageBoundingBox(label: "luminous golden sphere", x: 0.0, y: 0.0, width: 0.5, height: 0.5)]

        let baseReq = try ImageGenerationRequest(
            model: synth.modelIdentifier, prompt: "dark studio table",
            dimensions: ImageDimensions(width: 64, height: 64), steps: 4, config: baseCfg
        )
        let boxReq = try ImageGenerationRequest(
            model: synth.modelIdentifier, prompt: "dark studio table",
            dimensions: ImageDimensions(width: 64, height: 64), steps: 4, config: boxCfg
        )
        let baseImg = try ImagePNGCodec.decodePixels(try await ImageTestFixtures.collectEvents(synth, baseReq).completed.imageData)
        let boxImg = try ImagePNGCodec.decodePixels(try await ImageTestFixtures.collectEvents(synth, boxReq).completed.imageData)

        var insideDiff: Float = 0
        var outsideDiff: Float = 0
        var insideCount: Float = 0
        var outsideCount: Float = 0
        for y in 0 ..< 64 {
            for x in 0 ..< 64 {
                let a = baseImg.rgba(x: x, y: y)
                let b = boxImg.rgba(x: x, y: y)
                let delta = abs(a.r - b.r) + abs(a.g - b.g) + abs(a.b - b.b)
                if x < 28 && y < 28 {
                    insideDiff += delta
                    insideCount += 1
                } else if x >= 36 && y >= 36 {
                    outsideDiff += delta
                    outsideCount += 1
                }
            }
        }
        let insideMean = insideDiff / max(1, insideCount)
        let outsideMean = outsideDiff / max(1, outsideCount)
        #expect(insideMean > outsideMean)
    }

    @Test("Audio-to-Image SonicDiffusion runner shifts generated scene proportionally to audioStrength and supports candidates")
    func audioConditioningScalesWithStrengthAndSelectsBestCandidate() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "img-audio-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let synth = try await ImageSynthesizerLoader.load(identifier: "SonicDiffusion/SonicDiffusion-SD3", directory: dir)
        let audio = ImageTestFixtures.makeSyntheticPCMBytes(frequency: 440.0, amplitude: 0.85)

        func runWithStrength(_ strength: Double, candidates: Int = 1) async throws -> (pixels: [Float], artifact: ImageCompletedArtifact) {
            var cfg = ImageGenerationConfig(guidanceScale: 2.0, seed: 1_337)
            cfg.audioConditioning = audio
            cfg.audioStrength = strength
            cfg.candidates = candidates
            cfg.upsampledPrompt = "vivid acoustic aurora waves"
            let req = try ImageGenerationRequest(
                model: synth.modelIdentifier, prompt: "",
                dimensions: ImageDimensions(width: 64, height: 64), steps: 4, config: cfg
            )
            let res = try await ImageTestFixtures.collectEvents(synth, req)
            let dec = try ImagePNGCodec.decodePixels(res.completed.imageData)
            return (dec.pixels, res.completed)
        }

        let sZero = try await runWithStrength(0.0)
        let sMid = try await runWithStrength(0.45)
        let sHigh = try await runWithStrength(1.6)
        let multiCand = try await runWithStrength(1.0, candidates: 3)

        let deltaMid = ImageTestFixtures.meanL1Diff(sZero.pixels, sMid.pixels)
        let deltaHigh = ImageTestFixtures.meanL1Diff(sZero.pixels, sHigh.pixels)
        #expect(deltaMid > 0)
        #expect(deltaHigh > deltaMid)
        #expect(multiCand.artifact.upsampledPrompt == "vivid acoustic aurora waves")
        #expect(multiCand.artifact.seed >= 1_337 && multiCand.artifact.seed <= 1_339)
    }
}
