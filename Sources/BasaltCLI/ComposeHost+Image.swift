import Foundation
import BasaltCore
import BasaltServer

extension ComposeHost {
    func loadOrderedTargets(
        _ names: [String],
        in spec: ComposeSpec,
        base: [String: ComposeHeld]
    ) async throws -> [String: ComposeHeld] {
        var current = base
        for name in names where current[name] == nil && !spec.imageGeneratorModels.contains(name) {
            current[name] = try await load(name, in: spec, visibleHeld: current)
        }
        for name in names where current[name] == nil {
            current[name] = try await load(name, in: spec, visibleHeld: current)
        }
        return current
    }

    func loadNewTargets(
        _ names: [String],
        in spec: ComposeSpec,
        baseHeld: [String: ComposeHeld]
    ) async throws -> [String: ComposeHeld] {
        var fresh: [String: ComposeHeld] = [:]
        do {
            for name in names where !spec.imageGeneratorModels.contains(name) {
                fresh[name] = try await load(name, in: spec, visibleHeld: baseHeld.merging(fresh) { $1 })
            }
            for name in names where spec.imageGeneratorModels.contains(name) {
                fresh[name] = try await load(name, in: spec, visibleHeld: baseHeld.merging(fresh) { $1 })
            }
            return fresh
        } catch {
            for model in fresh.values { await Self.unload(model) }
            throw error
        }
    }

    func loadImageDecoder(
        _ name: String,
        source: String,
        model: ComposeSpec.Model
    ) async throws -> ComposeHeld {
        let onDemand = model.keep == .onDemand
        let path = try await services.resolve(source)
        await events.emit(onDemand ? "registered" : "ready", model: name, source)
        return .imageDecoder(source: source, path: path)
    }

    func loadImageGenerator(
        _ name: String,
        source: String,
        model: ComposeSpec.Model,
        in spec: ComposeSpec,
        held: [String: ComposeHeld]
    ) async throws -> ComposeHeld {
        let onDemand = model.keep == .onDemand
        let path = try await services.resolve(source)
        await events.emit(onDemand ? "registered" : "loading", model: name, source)
        let embedder = sharedEmbedder(forGenerator: name, in: spec, held: held)
        let decPath = decoderPath(forGenerator: name, in: spec, held: held)
        let loadSynth = Self.makeSynthesizerLoader(
            services: services,
            source: source,
            path: path,
            decoderPath: decPath,
            embedder: embedder
        )
        guard onDemand else {
            let loaded = try await loadSynth()
            await events.emit("ready", model: name, source)
            return .imageGenerator(loaded)
        }
        let descriptor = ImageModelDescriptor.detect(identifier: source)
        return .onDemandImageGenerator(
            OnDemandImageSynthesizer(
                modelIdentifier: source,
                path: path,
                descriptor: descriptor,
                idle: .seconds(model.idleUnload ?? ComposeSpec.defaultIdleUnload),
                loader: loadSynth
            )
        )
    }

    private static func makeSynthesizerLoader(
        services: ComposeServices,
        source: String,
        path: String,
        decoderPath: String?,
        embedder: (@Sendable (String) async throws -> [Float])?
    ) -> @Sendable () async throws -> any ImageSynthesizer {
        let stdLoader = services.imageGenerator
        let decLoader = services.imageGeneratorWithDecoder
        return {
            if let decoderPath {
                return try await decLoader(source, path, decoderPath, embedder)
            }
            return try await stdLoader(source, path, embedder)
        }
    }

    private func decoderPath(
        forGenerator name: String,
        in spec: ComposeSpec,
        held: [String: ComposeHeld]
    ) -> String? {
        let decName = spec.views.compactMap { view -> String? in
            guard let imagine = view.spec.imagine, imagine.chain.contains(name) else { return nil }
            return imagine.decoderVia
        }.first
        guard let decName, case .imageDecoder(_, let path)? = held[decName] else { return nil }
        return path
    }

    private func sharedEmbedder(
        forGenerator name: String,
        in spec: ComposeSpec,
        held: [String: ComposeHeld]
    ) -> (@Sendable (String) async throws -> [Float])? {
        let encoderName = spec.views.compactMap { view -> String? in
            guard let imagine = view.spec.imagine, imagine.chain.contains(name) else { return nil }
            return imagine.shareEncoder
        }.first
        guard let encoderName else { return nil }
        let fallback = services.sharedTextEmbedder
        switch held[encoderName] {
        case .resident(let orch):
            return { prompt in
                if let vec = try await orch.sharedImagePromptEmbedding(prompt, dimension: 256) { return vec }
                if let fallback { return try await fallback(prompt) }
                return []
            }
        case .onDemand(let onDemand):
            return { prompt in
                _ = try await onDemand.acquire()
                defer {
                    await withTaskCancellationShield {
                        await onDemand.release()
                    }
                }
                if let vec = try await onDemand.model.sharedImagePromptEmbedding(prompt, dimension: 256) { return vec }
                if let fallback { return try await fallback(prompt) }
                return []
            }
        default:
            return fallback
        }
    }
}

extension ComposeHost {
    func imageSynthesizer(
        for view: ComposeSpec,
        of spec: ComposeSpec,
        held: [String: ComposeHeld]
    ) -> (any ImageSynthesizer)? {
        guard let imagine = view.imagine, !imagine.chain.isEmpty else { return nil }
        let entries: [RoutedImageSynthesizer.Entry] = imagine.chain.compactMap { entry in
            switch held[entry] {
            case .imageGenerator(let synth):
                return .init(alias: entry, synthesizer: synth)
            case .onDemandImageGenerator(let synth):
                return .init(alias: entry, synthesizer: synth)
            default:
                return nil
            }
        }
        guard !entries.isEmpty else { return nil }
        let isFlux3 = entries.contains { $0.synthesizer.descriptor.family == .flux3Klein }
        let aux = buildImagineAux(imagine, spec: spec, held: held, flux3SceneCaption: isFlux3)
        let gate = chat.gate
        let gpuBusyProbe: @Sendable () async -> Bool = {
            if await gate.inFlight > 0 { return true }
            return await gate.queued > 0
        }
        return RoutedImageSynthesizer(
            entries: entries,
            onGPUBusy: imagine.onGPUBusy ?? .queue,
            isGPUBusy: gpuBusyProbe,
            aux: aux
        )
    }

    private func buildImagineAux(
        _ imagine: ComposeSpec.Imagine,
        spec: ComposeSpec,
        held: [String: ComposeHeld],
        flux3SceneCaption: Bool
    ) -> RoutedImageSynthesizer.AuxStages {
        var aux = RoutedImageSynthesizer.AuxStages(sharedEncoderModel: imagine.shareEncoder)
        aux.decoderModel = imagine.decoderVia
        aux.defaultCandidates = imagine.candidates
        aux.defaultTeacacheThreshold = imagine.teacacheThreshold
        aux.defaultPreviewEverySteps = imagine.previewEverySteps
        if let upsampleName = imagine.upsampleVia, let answerTarget = try? target(upsampleName, held: held, in: spec) {
            aux.upsamplerModel = upsampleName
            aux.upsampler = Self.makePromptUpsampler(using: answerTarget, flux3SceneCaption: flux3SceneCaption)
        }
        if let judgeName = imagine.judgeVia, let judgeTarget = try? target(judgeName, held: held, in: spec) {
            aux.judgeModel = judgeName
            aux.candidateJudge = Self.makeCandidateJudge(using: judgeTarget)
        }
        return aux
    }

    private static func makePromptUpsampler(
        using target: any AnswerTarget,
        flux3SceneCaption: Bool
    ) -> @Sendable (String) async throws -> String {
        if flux3SceneCaption {
            return { rawPrompt in
                let prompt = """
                    You are the prompt upsampler of a FLUX.3 text-to-image model. Expand the user's request into ONE detailed, self-contained scene caption: open with the medium convention (e.g. Full-color outdoor photograph, Studio photograph, Digital illustration), preserve every explicit subject, count, text and spatial cue, specify concrete lighting, materials and composition, and output ONLY the caption with no quotes or markdown:
                    \(rawPrompt)
                    """
                let reply = try await target.answer(GenerationRequest(prompt: prompt, maxTokens: 240, temperature: 0.35))
                return reply.text.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return { rawPrompt in
            let prompt = "Expand this image prompt into 1-2 vivid visual sentences describing lighting, composition and texture. Reply ONLY with the expanded prompt:\n\(rawPrompt)"
            let reply = try await target.answer(GenerationRequest(prompt: prompt, maxTokens: 160, temperature: 0.4))
            return reply.text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    private static func makeCandidateJudge(
        using target: any AnswerTarget
    ) -> @Sendable (String, [ImageCompletedArtifact]) async throws -> Int {
        { prompt, candidates in
            let images = candidates.prefix(ImageInput.maxCount).compactMap {
                try? ImageInput(data: $0.imageData, mime: $0.format.contentType)
            }
            let instruction = "Pick the best candidate image (index 0 to \(candidates.count - 1)) for prompt: \(prompt). Reply ONLY with the integer index."
            let request = GenerationRequest(
                prompt: instruction,
                maxTokens: 16,
                temperature: 0.0,
                messages: images.isEmpty ? nil : [ChatMessageInput(role: .user, text: instruction, images: images)]
            )
            let reply = try await target.answer(request)
            return parseJudgeIndex(reply.text, count: candidates.count)
        }
    }

    static func parseJudgeIndex(_ text: String, count: Int) -> Int {
        guard count > 0 else { return 0 }
        let digits = text.split { !$0.isNumber }.compactMap { Int($0) }
        if let firstInRange = digits.first(where: { (0 ..< count).contains($0) }) {
            return firstInRange
        }
        if let first = digits.first {
            return min(max(0, first), count - 1)
        }
        return 0
    }
}
