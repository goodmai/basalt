import Foundation
import BasaltCore
import BasaltServer

extension ComposeHost {
    func loadImageGenerator(
        _ name: String,
        source: String,
        model: ComposeSpec.Model,
        in spec: ComposeSpec
    ) async throws -> ComposeHeld {
        let onDemand = model.keep == .onDemand
        let path = try await services.resolve(source)
        await events.emit(onDemand ? "registered" : "loading", model: name, source)
        let embedder = sharedEmbedder(forGenerator: name, in: spec)
        guard onDemand else {
            let loaded = try await services.imageGenerator(source, path, embedder)
            await events.emit("ready", model: name, source)
            return .imageGenerator(loaded)
        }
        let imageGenerator = services.imageGenerator
        let descriptor = ImageModelDescriptor.detect(identifier: source)
        return .onDemandImageGenerator(
            OnDemandImageSynthesizer(
                modelIdentifier: source,
                path: path,
                descriptor: descriptor,
                idle: .seconds(model.idleUnload ?? ComposeSpec.defaultIdleUnload),
                loader: { try await imageGenerator(source, path, embedder) }
            )
        )
    }

    private func sharedEmbedder(
        forGenerator name: String,
        in spec: ComposeSpec
    ) -> (@Sendable (String) async throws -> [Float])? {
        let wantsShare = spec.views.contains {
            guard let imagine = $0.spec.imagine, imagine.chain.contains(name) else { return false }
            return imagine.shareEncoder != nil
        }
        return wantsShare ? services.sharedTextEmbedder : nil
    }

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
        let aux = buildImagineAux(imagine, view: view, spec: spec, held: held)
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
        view: ComposeSpec,
        spec: ComposeSpec,
        held: [String: ComposeHeld]
    ) -> RoutedImageSynthesizer.AuxStages {
        var aux = RoutedImageSynthesizer.AuxStages(sharedEncoderModel: imagine.shareEncoder)
        aux.defaultCandidates = imagine.candidates
        aux.defaultTeacacheThreshold = imagine.teacacheThreshold
        aux.defaultPreviewEverySteps = imagine.previewEverySteps
        if let upsampleName = imagine.upsampleVia, let answerTarget = try? target(upsampleName, held: held, in: spec) {
            aux.upsamplerModel = upsampleName
            aux.upsampler = Self.makePromptUpsampler(using: answerTarget)
        }
        if let judgeName = imagine.judgeVia, let judgeTarget = try? target(judgeName, held: held, in: spec) {
            aux.judgeModel = judgeName
            aux.candidateJudge = Self.makeCandidateJudge(using: judgeTarget)
        }
        return aux
    }

    private static func makePromptUpsampler(
        using target: any AnswerTarget
    ) -> @Sendable (String) async throws -> String {
        { rawPrompt in
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
