import Foundation
import MLX
import MLXNN
import BasaltCore

struct PromptVectorPair: Sendable {
    let positive: [Float]
    let negative: [Float]?
}

struct CandidateOutcome: Sendable {
    let encoded: EncodedImageBuffer
    let seed: UInt64
    let stepsExecuted: Int
    let teacacheSkippedSteps: Int
    let alignmentScore: Float
}

protocol ImageStepRunner: ImageSynthesizer {
    func beginCandidateSession(
        request: ImageGenerationRequest,
        promptVectors: PromptVectorPair,
        seed: UInt64,
        expectedEpoch: Int
    ) async throws(BasaltError) -> Int

    func stepCandidateSession(
        _ sessionID: Int,
        step: Int,
        emitPreview: Bool
    ) async throws(BasaltError) -> ImagePartialFrame?

    func finalizeCandidateSession(_ sessionID: Int) async throws(BasaltError) -> CandidateOutcome

    func releaseCandidateSession(_ sessionID: Int) async
}

enum ImageRunnerCore {
    static func parseDirectoryConfig(
        directory: URL,
        identifier: String,
        externalDecoderDirectory: URL? = nil
    ) throws(BasaltError) -> (descriptor: ImageModelDescriptor, config: ImageDiTConfig) {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDir), isDir.boolValue else {
            throw .modelNotFound(identifier: directory.path)
        }
        let rawJSON = try readFirstJSONObject(
            at: [
                directory.appending(component: "config.json"),
                directory.appending(path: "transformer/config.json"),
            ]
        )
        let vaeJSON = try readFirstJSONObject(
            at: [
                externalDecoderDirectory?.appending(component: "config.json"),
                externalDecoderDirectory?.appending(path: "vae/config.json"),
                directory.appending(path: "vae/config.json"),
            ].compactMap { $0 }
        )
        let descriptor = ImageModelDescriptor.detect(identifier: identifier, configJSON: rawJSON)
        var config = buildDiTConfig(raw: rawJSON, identifier: identifier, descriptor: descriptor)
        config.decoderBaseChannels = resolveDecoderChannels(vaeJSON: vaeJSON ?? rawJSON)
        return (descriptor, config)
    }

    private static func readFirstJSONObject(at urls: [URL]) throws(BasaltError) -> [String: Any]? {
        for configURL in urls where FileManager.default.fileExists(atPath: configURL.path) {
            let data: Data
            do {
                data = try Data(contentsOf: configURL)
            } catch {
                throw .modelConfigurationInvalid(path: configURL.path, reason: error.localizedDescription)
            }
            guard let parsed = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
                throw .modelConfigurationInvalid(path: configURL.path, reason: "config.json must be a JSON object")
            }
            return parsed
        }
        return nil
    }

    private static func resolveDecoderChannels(vaeJSON: [String: Any]?) -> Int {
        if let decOut = vaeJSON?["decoder_block_out_channels"] as? [Int], let first = decOut.first, first < 128 {
            return 18
        }
        return 24
    }

    private static func buildDiTConfig(
        raw: [String: Any]?,
        identifier: String,
        descriptor: ImageModelDescriptor
    ) -> ImageDiTConfig {
        let heads = raw?["num_attention_heads"] as? Int ?? raw?["num_heads"] as? Int ?? 4
        let derivedHidden = (raw?["attention_head_dim"] as? Int).map { $0 * heads }
        let rawHidden = raw?["hidden_size"] as? Int ?? raw?["dim"] as? Int ?? derivedHidden ?? 64
        let cappedHidden = min(256, max(32, rawHidden))
        let baseLayers = raw?["num_hidden_layers"] as? Int ?? raw?["num_layers"] as? Int ?? 2
        let singleLayers = raw?["num_single_layers"] as? Int ?? 0
        let layers = min(6, max(1, baseLayers + singleLayers))
        let latentCh = min(32, max(4, raw?["latent_channels"] as? Int ?? raw?["in_channels"] as? Int ?? 16))
        let defaultPatch: Int = switch descriptor.family {
        case .pixelDiT: 16
        case .linearSana: 1
        default: 2
        }
        let patch = raw?["patch_size"] as? Int ?? defaultPatch
        var cfg = ImageDiTConfig(
            modelID: raw?["model_id"] as? String ?? identifier,
            hiddenSize: cappedHidden,
            numHeads: heads,
            numLayers: layers,
            latentChannels: latentCh,
            patchSize: patch
        )
        cfg.vaeScale = descriptor.family == .linearSana ? 32 : (descriptor.usesVAE ? 8 : 1)
        let macaqDict = raw?["macaq_quantization"] as? [String: Any]
        let moduleBits = (macaqDict?["modules"] as? [String: [String: Any]])?.values.compactMap { $0["bits"] as? Int }.min()
        let macaqBits = (macaqDict?["bits"] as? Int) ?? moduleBits
        cfg.quantizationBits = raw?["bits"] as? Int ?? macaqBits ?? (descriptor.family == .ternaryBitNet ? 2 : 8)
        let idHash = identifier.unicodeScalars.reduce(UInt64(517)) { ($0 &* 131) &+ UInt64($1.value) }
        cfg.initSeed = raw?["seed"] as? UInt64 ?? (idHash == 0 ? 517 : idHash)
        return cfg
    }

    static func loadSafetensorsIfPresent(in directory: URL, into module: Module) throws(BasaltError) {
        let candidates = [directory, directory.appending(component: "transformer"), directory.appending(component: "vae")]
        guard let weightDir = candidates.first(where: { !safetensorsFiles(in: $0).isEmpty }) else { return }
        let files = safetensorsFiles(in: weightDir)
        let localManifest = try MACaQNativeLoader.manifest(in: weightDir)
        let resolvedManifest: MACaQNativeLoader.Manifest?
        if let localManifest {
            resolvedManifest = localManifest
        } else {
            resolvedManifest = try MACaQNativeLoader.manifest(in: directory)
        }
        if let manifest = resolvedManifest {
            try MACaQNativeLoader.install(manifest.modules, in: module)
        }
        let existing = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        var matched: [(String, MLXArray)] = []
        do {
            for file in files {
                let tensors = try MLX.loadArrays(url: file)
                for (key, value) in tensors {
                    if let target = existing[key], target.shape == value.shape {
                        matched.append((key, value.asType(target.dtype)))
                    }
                }
            }
            if !matched.isEmpty {
                module.update(parameters: ModuleParameters.unflattened(matched))
            }
            eval(module)
        } catch {
            throw .weightsCorrupted(path: directory.path, reason: error.localizedDescription)
        }
    }

    private static func safetensorsFiles(in directory: URL) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    static func wantsNativeAlpha(request: ImageGenerationRequest, descriptor: ImageModelDescriptor) -> Bool {
        guard request.outputFormat != .jpeg else { return false }
        return request.config.background == .transparent
            || (descriptor.supportsNativeRGBA && request.config.background == .auto)
    }

    static func resolveBaseSeed(_ request: ImageGenerationRequest) -> UInt64 {
        if let explicit = request.config.seed { return explicit }
        return request.effectivePrompt.unicodeScalars.reduce(UInt64(1_048_573)) {
            ($0 &* 1_099_511_628_211) ^ UInt64($1.value)
        }
    }
}

extension ImageRunnerCore {
    static func resolvePromptVectors(
        request: ImageGenerationRequest,
        embedDim: Int,
        sharedEmbedder: (@Sendable (String) async throws -> [Float])?
    ) async throws -> PromptVectorPair {
        let positive: [Float]
        if let sharedEmbedder {
            let raw = try await sharedEmbedder(request.effectivePrompt)
            positive = FourierTextEmbedder.projectAndNormalize(raw, to: embedDim)
        } else {
            positive = FourierTextEmbedder.embed(request.effectivePrompt, dimension: embedDim)
        }
        let negative: [Float]?
        if let negText = request.negativePrompt, !negText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if let sharedEmbedder {
                let rawNeg = try await sharedEmbedder(negText)
                negative = FourierTextEmbedder.projectAndNormalize(rawNeg, to: embedDim)
            } else {
                negative = FourierTextEmbedder.embed(negText, dimension: embedDim)
            }
        } else {
            negative = nil
        }
        return PromptVectorPair(positive: positive, negative: negative)
    }

    static func decodeReferencesAndMask(
        request: ImageGenerationRequest
    ) throws(BasaltError) -> (references: [MLXArray], mask: MLXArray?) {
        var refs: [MLXArray] = []
        refs.reserveCapacity(request.config.referenceImages.count)
        for data in request.config.referenceImages {
            let decoded = try ImagePNGCodec.decode(
                data,
                targetWidth: request.width,
                targetHeight: request.height
            )
            refs.append(decoded)
        }
        let maskTensor: MLXArray?
        if let maskData = request.config.maskImage {
            let rawMask = try ImagePNGCodec.decode(
                maskData,
                targetWidth: request.width,
                targetHeight: request.height
            )
            let rgbMean = mean(rawMask[0..., 0..., ..<3], axis: -1, keepDims: true)
            let alpha = rawMask[0..., 0..., 3...]
            maskTensor = clip(rgbMean * alpha, min: 0.0, max: 1.0)[.newAxis]
        } else {
            maskTensor = nil
        }
        return (refs, maskTensor)
    }

    static func spatialBBoxModulation(
        boxes: [ImageBoundingBox],
        gridH: Int,
        gridW: Int,
        hiddenDim: Int
    ) -> (mask: MLXArray, bias: MLXArray)? {
        guard !boxes.isEmpty, gridH > 0, gridW > 0 else { return nil }
        let total = gridH * gridW
        var maskFlat = [Float](repeating: 0, count: total)
        var biasFlat = [Float](repeating: 0, count: total * hiddenDim)
        for box in boxes.prefix(16) {
            let x0 = min(1.0, max(0.0, box.x))
            let y0 = min(1.0, max(0.0, box.y))
            let x1 = min(1.0, max(x0, box.x + max(0.0, box.width)))
            let y1 = min(1.0, max(y0, box.y + max(0.0, box.height)))
            let boxVec = FourierTextEmbedder.embedBoundingBox(box, dimension: hiddenDim)
            for row in 0 ..< gridH {
                let cy = (Double(row) + 0.5) / Double(gridH)
                guard cy >= y0 && cy <= y1 else { continue }
                for col in 0 ..< gridW {
                    let cx = (Double(col) + 0.5) / Double(gridW)
                    guard cx >= x0 && cx <= x1 else { continue }
                    let idx = row * gridW + col
                    maskFlat[idx] = 1.0
                    let base = idx * hiddenDim
                    for d in 0 ..< hiddenDim {
                        biasFlat[base + d] += boxVec[d] * 1.35
                    }
                }
            }
        }
        return (
            MLXArray(maskFlat, [1, total]),
            MLXArray(biasFlat, [1, total, hiddenDim])
        )
    }

    static func compositeMaskedPixels(
        generated: MLXArray,
        reference: MLXArray?,
        mask: MLXArray?,
        includeAlpha: Bool
    ) -> MLXArray {
        guard let reference, let mask else { return generated }
        let outCh = includeAlpha ? 4 : 3
        let refNHWC = (reference.ndim == 3 ? reference[.newAxis] : reference)[0..., 0..., 0..., ..<outCh]
        let genNHWC = (generated.ndim == 3 ? generated[.newAxis] : generated)[0..., 0..., 0..., ..<outCh]
        return mask * genNHWC + (1.0 - mask) * refNHWC
    }
}

extension ImageRunnerCore {
    static func extractAudioConditioning(from data: Data?, dimension: Int) -> [Float]? {
        guard let data, !data.isEmpty else { return nil }
        let dim = max(16, dimension)
        let samples = decodeAudioBytesToSamples(data)
        guard !samples.isEmpty else { return nil }
        let bands = min(16, max(4, dim / 4))
        let strideCount = max(1, samples.count / bands)
        var bandStats = [Double](repeating: 0, count: bands * 3)
        for band in 0 ..< bands {
            let start = min(samples.count - 1, band * strideCount)
            let end = min(samples.count, start + strideCount)
            guard end > start else { continue }
            var sq = 0.0
            var peak = 0.0
            var crossings = 0.0
            for idx in start ..< end {
                let v = Double(samples[idx])
                sq += v * v
                peak = max(peak, abs(v))
                if idx > start, (samples[idx] >= 0) != (samples[idx - 1] >= 0) {
                    crossings += 1.0
                }
            }
            let len = Double(end - start)
            bandStats[band * 3] = (sq / len).squareRoot()
            bandStats[band * 3 + 1] = peak
            bandStats[band * 3 + 2] = crossings / len
        }
        let globalRMS = bandStats.enumerated().reduce(0.0) { $1.offset.isMultiple(of: 3) ? $0 + $1.element : $0 } / Double(bands)
        var feat = (0 ..< dim).map { d -> Float in
            let acc = bandStats.enumerated().reduce(0.0) { partial, entry in
                partial + entry.element * cos(Double((d + 1) * (entry.offset + 1)) * 0.29)
            }
            return Float(acc)
        }
        FourierTextEmbedder.normalizeL2(&feat)
        let energyScale = Float(0.5 + min(2.0, globalRMS * 2.5))
        return feat.map { $0 * energyScale }
    }

    private static func decodeAudioBytesToSamples(_ data: Data) -> [Float] {
        let headerOffset = data.count > 44 && data.prefix(4) == Data("RIFF".utf8) ? 44 : 0
        let payload = data.dropFirst(headerOffset)
        if payload.count >= 4 {
            let pairCount = min(24_000, payload.count / 2)
            var pcm = [Float](repeating: 0, count: pairCount)
            var idx = payload.startIndex
            for sampleIdx in 0 ..< pairCount {
                let lo = UInt16(payload[idx])
                let nextIdx = payload.index(after: idx)
                let hi = UInt16(payload[nextIdx])
                pcm[sampleIdx] = Float(Int16(bitPattern: lo | (hi << 8))) / 32_768.0
                idx = payload.index(after: nextIdx)
            }
            return pcm
        }
        return data.prefix(4_096).map { (Float($0) - 128.0) / 128.0 }
    }
}

extension ImageRunnerCore {
    static func makeGenerationStream<Runner: ImageStepRunner>(
        runner: Runner,
        request: ImageGenerationRequest,
        promptVectors: PromptVectorPair,
        expectedEpoch: Int
    ) -> AsyncThrowingStream<ImageGenerationEvent, any Error> {
        let baseSeed = resolveBaseSeed(request)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let clock = ContinuousClock()
                    let started = clock.now
                    var best: CandidateOutcome?
                    for candIdx in 0 ..< request.config.candidates {
                        try Task.checkCancellation()
                        let outcome = try await runSingleCandidate(
                            runner: runner,
                            request: request,
                            promptVectors: promptVectors,
                            seed: baseSeed &+ UInt64(candIdx),
                            expectedEpoch: expectedEpoch,
                            continuation: candIdx == 0 ? continuation : nil
                        )
                        if best == nil || outcome.alignmentScore > best!.alignmentScore {
                            best = outcome
                        }
                    }
                    guard let chosen = best else {
                        throw BasaltError.inferenceError("image synthesis produced zero candidates")
                    }
                    let artifact = buildCompletedArtifact(
                        chosen: chosen,
                        request: request,
                        runner: runner,
                        elapsed: clock.now - started
                    )
                    continuation.yield(.completed(artifact))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error is CancellationError ? BasaltError.cancelled : error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func buildCompletedArtifact<Runner: ImageStepRunner>(
        chosen: CandidateOutcome,
        request: ImageGenerationRequest,
        runner: Runner,
        elapsed: Duration
    ) -> ImageCompletedArtifact {
        let elapsedMs = max(0, Int(elapsed.components.seconds * 1_000 + elapsed.components.attoseconds / 1_000_000_000_000_000))
        var artifact = ImageCompletedArtifact(
            imageData: chosen.encoded.data,
            format: chosen.encoded.format,
            dimensions: ImageDimensions(width: request.width, height: request.height),
            hasAlpha: chosen.encoded.hasAlpha,
            seed: chosen.seed,
            model: runner.modelIdentifier
        )
        artifact.stepsExecuted = chosen.stepsExecuted
        artifact.teacacheSkippedSteps = chosen.teacacheSkippedSteps
        artifact.compute = runner.descriptor.defaultCompute
        artifact.licenseTier = runner.descriptor.licenseTier
        artifact.upsampledPrompt = request.config.upsampledPrompt
        artifact.elapsedMs = elapsedMs
        return artifact
    }

    private static func runSingleCandidate<Runner: ImageStepRunner>(
        runner: Runner,
        request: ImageGenerationRequest,
        promptVectors: PromptVectorPair,
        seed: UInt64,
        expectedEpoch: Int,
        continuation: AsyncThrowingStream<ImageGenerationEvent, any Error>.Continuation?
    ) async throws -> CandidateOutcome {
        let sessionID = try await runner.beginCandidateSession(
            request: request,
            promptVectors: promptVectors,
            seed: seed,
            expectedEpoch: expectedEpoch
        )
        defer {
            await withTaskCancellationShield {
                await runner.releaseCandidateSession(sessionID)
            }
        }
        for step in 0 ..< request.steps {
            try Task.checkCancellation()
            let wantPreview = continuation != nil && TinyAutoencoderPreview.shouldEmitPreview(
                step: step,
                totalSteps: request.steps,
                previewEverySteps: request.config.previewEverySteps
            )
            if let partial = try await runner.stepCandidateSession(sessionID, step: step, emitPreview: wantPreview) {
                continuation?.yield(.partial(partial))
            }
        }
        return try await runner.finalizeCandidateSession(sessionID)
    }
}
